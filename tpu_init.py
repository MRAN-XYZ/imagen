# ---------- TPU init (safe) ----------
import tensorflow as tf
try:
    # Use TPU if available in Colab
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # will raise if TPU not available
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU connected:", resolver.cluster_spec().as_dict().get('worker'))
except Exception as e:
    print("TPU not found, falling back to GPU/CPU. Error:", e)
    strategy = tf.distribute.get_strategy()

print("Using strategy:", type(strategy).__name__)

# ---------- Make sure dataset path is OK ----------
import glob, os
DATASET_PATH = "/content/imagen/Logos"  # set this correctly
exts = ("*.jpg", "*.jpeg", "*.png")
files = []
for ext in exts:
    files.extend(glob.glob(os.path.join(DATASET_PATH, ext)))
files = sorted(files)

print("Found files:", len(files))
if len(files) == 0:
    raise ValueError(f"No images found in {DATASET_PATH} — check the path and files.")

# sanity checks: ensure list is flat and elements are strings
print("Type(files):", type(files))
print("Type(files[0]):", type(files[0]), " -> sample:", files[0])
# Optional: check top 5
print("Sample (first 5):", files[:5])

# If any element isn't a str, coerce them:
files = [str(f) for f in files]

# Convert to tf.constant of dtype string to avoid TypeSpec issues
tf_files = tf.constant(files, dtype=tf.string)
print("tf_files shape:", tf_files.shape, tf_files.dtype)

# ---------- Safe tf.data pipeline ----------
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 64
BATCH_SIZE = 64

def decode_and_preprocess(path):
    # path is a scalar tf.string
    img = tf.io.read_file(path)
    # prefer decode_jpeg + fallback decode_png if you want speed, but decode_image is fine
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method='area')
    img = tf.image.random_flip_left_right(img)
    img = img * 2.0 - 1.0  # [-1, 1]
    return img

# Build dataset: from_tensor_slices(tf.constant(...)) avoids TypeSpec weirdness
ds = tf.data.Dataset.from_tensor_slices(tf_files)
ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
# Batch AFTER map (we want decode per-sample), and keep drop_remainder=True for TPU
ds = ds.batch(BATCH_SIZE, drop_remainder=True)
ds = ds.prefetch(AUTOTUNE)

# If using strategy (TPU), distribute
try:
    dist_ds = strategy.experimental_distribute_dataset(ds)
    print("Distributed dataset created.")
except Exception as e:
    dist_ds = ds
    print("Could not distribute dataset (running single device). Error:", e)

# quick check: fetch one batch to make sure pipeline works
for batch in ds.take(1):
    print("Batch shape:", batch.shape, "dtype:", batch.dtype)
    break