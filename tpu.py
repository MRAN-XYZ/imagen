import os
import glob
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, device_put
import flax.linen as nn
from flax.training import train_state
import optax
from functools import partial
import matplotlib.pyplot as plt
from typing import Any

# Check TPU availability
print(f"JAX devices: {jax.devices()}")
print(f"JAX local devices: {jax.local_devices()}")

# ------------------------------
# Simple Configuration
# ------------------------------
# Model params
IMAGE_SIZE = 64
NZ = 100  # latent dim
NGF = 64  # generator features
NDF = 64  # discriminator features
NC = 3    # color channels

# Training params
BATCH_SIZE = 64
NUM_EPOCHS = 200
LR = 0.0002
BETA1, BETA2 = 0.5, 0.999

# Paths
CHECKPOINT_DIR = "./checkpoints"
DATASET_PATH = "/content/imagen/Logos"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------
# Data Loading (NumPy-based for TPU)
# ------------------------------
def load_dataset(root_path, image_size=64):
    """Load and preprocess dataset into memory"""
    extensions = ['*.jpg', '*.png', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root_path, ext)))
    
    print(f"Loading {len(files)} images...")
    
    data = []
    for i, file_path in enumerate(files):
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((image_size, image_size))
            img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
            data.append(img_array)
            
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(files)} images")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    data = np.stack(data, axis=0)
    print(f"Dataset ready: {data.shape}")
    return data

def get_batch(data, batch_size, rng):
    """Get a random batch from the dataset"""
    batch_indices = random.choice(rng, len(data), (batch_size,), replace=False)
    return data[batch_indices]

# Load dataset
dataset = load_dataset(DATASET_PATH, IMAGE_SIZE)

# ------------------------------
# Models (Flax/Linen)
# ------------------------------
class Generator(nn.Module):
    ngf: int = NGF
    nc: int = NC
    
    @nn.compact
    def __call__(self, x, training=True):
        # x shape: (batch, nz, 1, 1)
        
        # 1x1 -> 4x4
        x = nn.ConvTranspose(self.ngf * 16, kernel_size=(4, 4), strides=(1, 1), 
                           padding='VALID', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # 4x4 -> 8x8
        x = nn.ConvTranspose(self.ngf * 8, kernel_size=(4, 4), strides=(2, 2), 
                           padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # 8x8 -> 16x16
        x = nn.ConvTranspose(self.ngf * 4, kernel_size=(4, 4), strides=(2, 2), 
                           padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # 16x16 -> 32x32
        x = nn.ConvTranspose(self.ngf * 2, kernel_size=(4, 4), strides=(2, 2), 
                           padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # 32x32 -> 64x64
        x = nn.ConvTranspose(self.ngf, kernel_size=(4, 4), strides=(2, 2), 
                           padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Final layer
        x = nn.ConvTranspose(self.nc, kernel_size=(3, 3), strides=(1, 1), 
                           padding='SAME', use_bias=False)(x)
        x = nn.tanh(x)
        
        return x

class Discriminator(nn.Module):
    ndf: int = NDF
    nc: int = NC
    
    @nn.compact
    def __call__(self, x, training=True):
        # 64x64 -> 32x32
        x = nn.Conv(self.ndf, kernel_size=(4, 4), strides=(2, 2), 
                   padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # 32x32 -> 16x16
        x = nn.Conv(self.ndf * 2, kernel_size=(4, 4), strides=(2, 2), 
                   padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # 16x16 -> 8x8
        x = nn.Conv(self.ndf * 4, kernel_size=(4, 4), strides=(2, 2), 
                   padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # 8x8 -> 4x4
        x = nn.Conv(self.ndf * 8, kernel_size=(4, 4), strides=(2, 2), 
                   padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # 4x4 -> 1x1
        x = nn.Conv(1, kernel_size=(4, 4), strides=(1, 1), 
                   padding='VALID', use_bias=False)(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        
        return x

# ------------------------------
# Training State and Functions
# ------------------------------
def create_train_state(rng, learning_rate, model, input_shape):
    """Create initial training state"""
    params = model.init(rng, jnp.ones(input_shape), training=True)['params']
    tx = optax.adam(learning_rate, b1=BETA1, b2=BETA2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def bce_with_logits_loss(logits, labels):
    """Binary cross entropy with logits loss"""
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

@jit
def train_discriminator_step(d_state, g_state, real_batch, rng):
    """Single discriminator training step"""
    batch_size = real_batch.shape[0]
    
    def d_loss_fn(d_params):
        # Real images loss
        real_logits = d_state.apply_fn({'params': d_params}, real_batch, training=True)
        real_labels = jnp.ones((batch_size, 1)) * 0.9  # Label smoothing
        real_loss = bce_with_logits_loss(real_logits, real_labels)
        
        # Fake images loss
        noise = random.normal(rng, (batch_size, NZ, 1, 1))
        fake_batch = g_state.apply_fn({'params': g_state.params}, noise, training=False)
        fake_logits = d_state.apply_fn({'params': d_params}, fake_batch, training=True)
        fake_labels = jnp.zeros((batch_size, 1))
        fake_loss = bce_with_logits_loss(fake_logits, fake_labels)
        
        return (real_loss + fake_loss) / 2
    
    loss, grads = value_and_grad(d_loss_fn)(d_state.params)
    d_state = d_state.apply_gradients(grads=grads)
    return d_state, loss

@jit
def train_generator_step(g_state, d_state, rng):
    """Single generator training step"""
    batch_size = BATCH_SIZE
    
    def g_loss_fn(g_params):
        noise = random.normal(rng, (batch_size, NZ, 1, 1))
        fake_batch = g_state.apply_fn({'params': g_params}, noise, training=True)
        fake_logits = d_state.apply_fn({'params': d_state.params}, fake_batch, training=False)
        real_labels = jnp.ones((batch_size, 1)) * 0.9  # Generator wants D to think fakes are real
        return bce_with_logits_loss(fake_logits, real_labels)
    
    loss, grads = value_and_grad(g_loss_fn)(g_state.params)
    g_state = g_state.apply_gradients(grads=grads)
    return g_state, loss

@jit
def generate_samples(g_state, noise):
    """Generate samples using the generator"""
    return g_state.apply_fn({'params': g_state.params}, noise, training=False)

# ------------------------------
# EMA Implementation
# ------------------------------
class EMAState:
    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.shadow = jax.tree_map(lambda x: x.copy(), params)
    
    def update(self, params):
        self.shadow = jax.tree_map(
            lambda shadow, param: (1 - self.decay) * param + self.decay * shadow,
            self.shadow, params
        )
    
    def apply(self, state):
        # Return new state with EMA parameters
        return state.replace(params=self.shadow)

# ------------------------------
# Utility Functions
# ------------------------------
def save_image_grid(images, filename, nrow=8):
    """Save a grid of images"""
    images = np.array(images)
    images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    images = np.clip(images, 0, 1)
    
    batch_size, h, w, c = images.shape
    ncol = int(np.ceil(batch_size / nrow))
    
    grid = np.zeros((nrow * h, ncol * w, c))
    for i in range(batch_size):
        row = i // ncol
        col = i % ncol
        if row < nrow:
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = images[i]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# ------------------------------
# Initialize Everything
# ------------------------------
# Initialize random keys
rng = random.PRNGKey(42)
rng, init_rng_g, init_rng_d = random.split(rng, 3)

# Create models
generator = Generator()
discriminator = Discriminator()

# Create training states
g_state = create_train_state(init_rng_g, LR, generator, (1, NZ, 1, 1))
d_state = create_train_state(init_rng_d, LR * 0.5, discriminator, (1, IMAGE_SIZE, IMAGE_SIZE, NC))

# Create EMA for generator
ema_g = EMAState(g_state.params)

# Fixed noise for visualization
fixed_noise = random.normal(random.PRNGKey(123), (64, NZ, 1, 1))
fixed_noise = device_put(fixed_noise)  # Move to TPU

print(f"Generator parameters: {sum(x.size for x in jax.tree_leaves(g_state.params)):,}")
print(f"Discriminator parameters: {sum(x.size for x in jax.tree_leaves(d_state.params)):,}")

# Training tracking
G_losses, D_losses = [], []

# Move dataset to TPU
dataset = device_put(dataset)

# ------------------------------
# Training Loop
# ------------------------------
print("Starting training...")

for epoch in range(NUM_EPOCHS):
    epoch_G_loss = 0
    epoch_D_loss = 0
    
    # Calculate number of batches
    num_batches = len(dataset) // BATCH_SIZE
    
    for i in range(num_batches):
        # Get batch
        rng, batch_rng = random.split(rng)
        real_batch = get_batch(dataset, BATCH_SIZE, batch_rng)
        real_batch = device_put(real_batch)
        
        # Train discriminator
        rng, d_rng = random.split(rng)
        d_state, d_loss = train_discriminator_step(d_state, g_state, real_batch, d_rng)
        
        # Train generator
        rng, g_rng = random.split(rng)
        g_state, g_loss = train_generator_step(g_state, d_state, g_rng)
        
        # Update EMA
        ema_g.update(g_state.params)
        
        # Track losses
        epoch_G_loss += float(g_loss)
        epoch_D_loss += float(d_loss)
        
        # Print progress
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i}/{num_batches}] "
                  f"D_loss: {float(d_loss):.4f} G_loss: {float(g_loss):.4f}")
    
    # Average losses
    avg_G_loss = epoch_G_loss / num_batches
    avg_D_loss = epoch_D_loss / num_batches
    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] G_loss: {avg_G_loss:.4f} D_loss: {avg_D_loss:.4f}")
    
    # Generate samples every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Use EMA weights for better samples
        ema_state = ema_g.apply(g_state)
        fake_samples = generate_samples(ema_state, fixed_noise)
        fake_samples = np.array(fake_samples)
        
        # Save samples
        save_image_grid(fake_samples, f"{CHECKPOINT_DIR}/samples_epoch_{epoch+1}.png")
        
        # Display samples
        plt.figure(figsize=(8, 8))
        sample_grid = fake_samples[:64]  # Take first 64 samples
        sample_grid = (sample_grid + 1) / 2  # Denormalize
        sample_grid = np.clip(sample_grid, 0, 1)
        
        # Create grid manually for display
        grid_size = 8
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(sample_grid):
                    axes[i, j].imshow(sample_grid[idx])
                axes[i, j].axis('off')
        
        plt.suptitle(f"Generated Samples - Epoch {epoch+1}")
        plt.tight_layout()
        plt.show()

print("Training completed!")

# Plot training curves
if G_losses:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Generate final samples with EMA
    ema_state = ema_g.apply(g_state)
    final_samples = generate_samples(ema_state, fixed_noise)
    final_samples = np.array(final_samples)
    final_samples = (final_samples + 1) / 2
    final_samples = np.clip(final_samples, 0, 1)
    
    # Show a few final samples
    for i in range(min(16, len(final_samples))):
        plt.subplot(4, 4, i+1)
        plt.imshow(final_samples[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/training_summary.png")
    plt.show()

print(f"Training completed! Samples saved in {CHECKPOINT_DIR}/")

# Save final model parameters
np.savez(f"{CHECKPOINT_DIR}/generator_final.npz", **ema_g.shadow)
print(f"Final EMA model saved to {CHECKPOINT_DIR}/generator_final.npz")
