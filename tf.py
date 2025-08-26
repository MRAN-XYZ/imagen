# =========================================================
# DCGAN (TensorFlow) on Colab TPU with EMA + Checkpointing
# =========================================================

import os, glob, math, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
IMAGE_SIZE = 64
NZ = 100        # latent dim
NGF = 64        # generator features
NDF = 64        # discriminator features
NC = 3          # channels
BATCH_SIZE = 64 # global batch size (TPU will split across 8 replicas)
NUM_EPOCHS = 200
LR = 2e-4
BETA1, BETA2 = 0.5, 0.999
LABEL_SMOOTH = 0.9          # real label smoothing
SAMPLE_EVERY_EPOCHS = 10
CHECKPOINT_DIR = "./checkpoints_tf"
DATASET_PATH = "/content/imagen/Logos"  # <- point to your folder of images
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------
# TPU init + mixed precision
# ------------------------------
print("TPU init…")
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("Running on TPU:", resolver.cluster_spec().as_dict()['worker'])
except Exception as e:
    print("TPU not found, falling back to CPU/GPU.", e)
    strategy = tf.distribute.get_strategy()

# TPU loves bfloat16
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_bfloat16')

# ------------------------------
# Data pipeline
# ------------------------------
AUTOTUNE = tf.data.AUTOTUNE

def list_image_files(root):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, ext)))
    print(f"Found {len(files)} images.")
    return files

def decode_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method='area')
    img = tf.image.random_flip_left_right(img)
    img = img * 2.0 - 1.0  # [-1, 1]
    return img

files = list_image_files(DATASET_PATH)
if len(files) == 0:
    raise ValueError("No images found. Please check DATASET_PATH.")

ds = tf.data.Dataset.from_tensor_slices(files)
ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
# Important for TPU: drop_remainder=True so each replica gets equal slices
ds = ds.batch(BATCH_SIZE, drop_remainder=True)
ds = ds.prefetch(AUTOTUNE)

dist_ds = strategy.experimental_distribute_dataset(ds)

# ------------------------------
# Models (Keras)
# ------------------------------
from tensorflow.keras import layers, Model

def make_generator():
    # Input z: (None, NZ)
    z = layers.Input(shape=(NZ,), dtype=tf.float32)
    x = layers.Reshape((1, 1, NZ))(z)
    x = layers.Conv2DTranspose(NGF*16, 4, strides=1, padding='valid', use_bias=False)(x)  # 4x4
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(NGF*8, 4, strides=2, padding='same', use_bias=False)(x)   # 8x8
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(NGF*4, 4, strides=2, padding='same', use_bias=False)(x)   # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(NGF*2, 4, strides=2, padding='same', use_bias=False)(x)   # 32x32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(NGF, 4, strides=2, padding='same', use_bias=False)(x)     # 64x64
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(NC, 3, strides=1, padding='same', use_bias=False)(x)
    # Output in [-1, 1]
    out = layers.Activation('tanh', dtype='float32')(x)  # cast to f32 for images
    return Model(z, out, name="Generator")

def make_discriminator():
    inp = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NC), dtype=tf.float32)
    x = layers.Conv2D(NDF, 4, strides=2, padding='same', use_bias=False)(inp)   # 32x32
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(NDF*2, 4, strides=2, padding='same', use_bias=False)(x)   # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(NDF*4, 4, strides=2, padding='same', use_bias=False)(x)   # 8x8
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(NDF*8, 4, strides=2, padding='same', use_bias=False)(x)   # 4x4
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(1, 4, strides=1, padding='valid', use_bias=False)(x)      # 1x1
    x = layers.Flatten()(x)  # logits
    # Keep logits in bf16/fp32 depending on policy, BCE will handle casting
    return Model(inp, x, name="Discriminator")

with strategy.scope():
    netG = make_generator()
    netD = make_discriminator()

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    optG = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA1, beta_2=BETA2)
    optD = tf.keras.optimizers.Adam(learning_rate=LR * 0.5, beta_1=BETA1, beta_2=BETA2)  # match your PyTorch choice

print(netG.summary())
print(netD.summary())

# ------------------------------
# EMA for Generator (manual)
# ------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Create shadow weights as float32 copies
        self.shadow = [tf.Variable(w.read_value(dtype=tf.float32), trainable=False) for w in model.weights]
        self.backup = None

    @tf.function
    def update(self, model):
        for swa, w in zip(self.shadow, model.weights):
            swa.assign(self.decay * swa + (1.0 - self.decay) * tf.cast(w, tf.float32))

    def apply(self, model):
        # backup current weights, then load shadow
        self.backup = [w.read_value() for w in model.weights]
        for w, swa in zip(model.weights, self.shadow):
            w.assign(tf.cast(swa, w.dtype))

    def restore(self, model):
        if self.backup is None:
            return
        for w, b in zip(model.weights, self.backup):
            w.assign(b)
        self.backup = None

with strategy.scope():
    ema_G = EMA(netG, decay=0.999)

# ------------------------------
# Fixed noise for sampling
# ------------------------------
def make_fixed_noise(n=64, nz=NZ, seed=42):
    rng = tf.random.Generator.from_seed(seed)
    return rng.normal([n, nz])

fixed_noise = make_fixed_noise(64, NZ)

# ------------------------------
# Loss functions
# ------------------------------
def d_loss_fn(real_logits, fake_logits):
    # label smoothing on real
    real_labels = tf.ones_like(real_logits) * LABEL_SMOOTH
    fake_labels = tf.zeros_like(fake_logits)

    # Optional: tiny randomization (comment out if you want exact smoothing only)
    # real_labels += tf.random.uniform(tf.shape(real_labels), 0.0, 0.05)
    # fake_labels += tf.random.uniform(tf.shape(fake_labels), 0.0, 0.05)

    real_loss = bce(real_labels, real_logits)
    fake_loss = bce(fake_labels, fake_logits)
    loss = (real_loss + fake_loss) * 0.5
    return tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

def g_loss_fn(fake_logits):
    target = tf.ones_like(fake_logits) * LABEL_SMOOTH
    loss = bce(target, fake_logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

# ------------------------------
# Checkpointing (resume supported)
# ------------------------------
ckpt = tf.train.Checkpoint(
    netG=netG, netD=netD, optG=optG, optD=optD,
    # store EMA shadow as separate variables collection
    ema_vars=ema_G.shadow
)
manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

start_epoch = 0
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    # try to parse epoch index from a sidecar file
    epoch_file = os.path.join(CHECKPOINT_DIR, "epoch.txt")
    if os.path.exists(epoch_file):
        with open(epoch_file, "r") as f:
            start_epoch = int(f.read().strip()) + 1
    print(f"Resumed from: {manager.latest_checkpoint}, start_epoch={start_epoch}")

# ------------------------------
# Per-replica training step (TPU)
# ------------------------------
@tf.function
def train_step(dist_batch):
    def step_fn(real_imgs):
        batch_size = tf.shape(real_imgs)[0]
        z = tf.random.normal([batch_size, NZ])

        # --- Train D ---
        with tf.GradientTape() as tapeD:
            fake_imgs = netG(z, training=True)
            real_logits = netD(real_imgs, training=True)
            fake_logits = netD(fake_imgs, training=True)
            d_loss = d_loss_fn(real_logits, fake_logits)
        gradsD = tapeD.gradient(d_loss, netD.trainable_variables)
        optD.apply_gradients(zip(gradsD, netD.trainable_variables))

        # --- Train G ---
        z2 = tf.random.normal([batch_size, NZ])
        with tf.GradientTape() as tapeG:
            fake_imgs2 = netG(z2, training=True)
            fake_logits2 = netD(fake_imgs2, training=True)
            g_loss = g_loss_fn(fake_logits2)
        gradsG = tapeG.gradient(g_loss, netG.trainable_variables)
        optG.apply_gradients(zip(gradsG, netG.trainable_variables))

        return d_loss, g_loss

    d_loss, g_loss = strategy.run(step_fn, args=(dist_batch,))
    # reduce across replicas
    d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None)

    # EMA update (do once per global step)
    ema_G.update(netG)
    return d_loss, g_loss

# ------------------------------
# Sampling utils
# ------------------------------
def make_grid(imgs, nrow=8):
    """imgs: [N, H, W, 3] in [-1,1]; returns H', W', 3 float32 in [0,1]"""
    imgs = (imgs + 1.0) * 0.5
    imgs = tf.clip_by_value(imgs, 0.0, 1.0)
    N, H, W, C = imgs.shape
    nrow = nrow
    ncol = math.ceil(N / nrow)
    grid = tf.zeros([ncol*H, nrow*W, C], dtype=tf.float32)
    for idx in range(N):
        r = idx // nrow
        c = idx % nrow
        grid[r*H:(r+1)*H, c*W:(c+1)*W, :].assign(imgs[idx])
    return grid.numpy()

def save_samples(epoch, noise=fixed_noise, use_ema=True):
    if use_ema:
        ema_G.apply(netG)
    imgs = netG(noise, training=False).numpy()
    if use_ema:
        ema_G.restore(netG)
    grid = make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title(f"Generated Samples - Epoch {epoch}")
    plt.imshow(grid)
    out_path = f"{CHECKPOINT_DIR}/samples_epoch_{epoch}.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved:", out_path)

# ------------------------------
# Training loop
# ------------------------------
G_losses, D_losses = [], []

print("Starting training…")
for epoch in range(start_epoch, NUM_EPOCHS):
    t0 = time.time()
    d_running, g_running, n_steps = 0.0, 0.0, 0

    for batch in dist_ds:
        d_loss, g_loss = train_step(batch)
        d_running += float(d_loss)
        g_running += float(g_loss)
        n_steps += 1
        # Optional: print every ~50 steps
        if n_steps % 50 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Step {n_steps}  D:{d_loss:.4f}  G:{g_loss:.4f}")

    avg_d = d_running / n_steps
    avg_g = g_running / n_steps
    D_losses.append(avg_d); G_losses.append(avg_g)

    # Save checkpoint
    save_path = manager.save()
    with open(os.path.join(CHECKPOINT_DIR, "epoch.txt"), "w") as f:
        f.write(str(epoch))
    print(f"[Epoch {epoch+1}] D_loss={avg_d:.4f} G_loss={avg_g:.4f} | ckpt: {save_path} | time: {time.time()-t0:.1f}s")

    # Sampling
    if (epoch + 1) % SAMPLE_EVERY_EPOCHS == 0:
        save_samples(epoch+1, fixed_noise, use_ema=True)

print("Training completed!")

# ------------------------------
# Save final EMA generator + curves
# ------------------------------
# Save EMA weights into the actual model, export, then restore
ema_G.apply(netG)
final_gen_path = os.path.join(CHECKPOINT_DIR, "generator_final_ema")
tf.saved_model.save(netG, final_gen_path)
ema_G.restore(netG)
print("Saved EMA Generator to:", final_gen_path)

# Plot curves + final grid
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(G_losses, label='G loss')
plt.plot(D_losses, label='D loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(); plt.title('Training Losses')

# final grid
ema_G.apply(netG)
final_imgs = netG(fixed_noise, training=False).numpy()
ema_G.restore(netG)
grid = make_grid(final_imgs, nrow=8)
plt.subplot(1,2,2)
plt.axis('off'); plt.title('Final Generated Samples')
plt.imshow(grid)
plt.tight_layout()
plt.savefig(f"{CHECKPOINT_DIR}/training_summary.png", bbox_inches='tight', pad_inches=0)
plt.show()

print("Summary saved to:", f"{CHECKPOINT_DIR}/training_summary.png")