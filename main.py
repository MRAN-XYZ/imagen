# 🚀 Enhanced Balanced DCGAN (64x64) - Fast + Stable on Colab T4
!pip install torch torchvision matplotlib --quiet

import os, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------
# Config
# ----------------------
data_path = "/content/imagen/Logos"   # 👈 your folder with 1400 imgs
latent_dim = 100
batch_size = 64
epochs = 50
lr_g = 0.0002  # Slightly different learning rates can help
lr_d = 0.0002
beta1, beta2 = 0.5, 0.999
image_size = 64
checkpoint_file = "dcgan_checkpoint.pth"

# ----------------------
# Dataset loader (cache into RAM)
# ----------------------
class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = glob.glob(os.path.join(root, "*.jpg")) + \
                    glob.glob(os.path.join(root, "*.png")) + \
                    glob.glob(os.path.join(root, "*.jpeg"))
        self.transform = transform
        print(f"Found {len(self.files)} images")
        
        # preload all images in memory
        self.images = []
        for i, f in enumerate(self.files):
            try:
                img = Image.open(f).convert("RGB")
                self.images.append(img)
                if (i + 1) % 100 == 0:
                    print(f"Loaded {i+1}/{len(self.files)} images")
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        print(f"Successfully loaded {len(self.images)} images into memory")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform: img = self.transform(img)
        return img, 0

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # More explicit resize
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = FlatImageDataset(data_path, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                   num_workers=2, pin_memory=True, persistent_workers=True)

# ----------------------
# Models with slight improvements
# ----------------------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),  # BatchNorm can work well for generator
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )
    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            # Input: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
            # Output: 1 x 1 x 1 (logits)
        )
    def forward(self, x): return self.main(x)

# ----------------------
# Init models + optims
# ----------------------
nz, ngf, ndf, nc = latent_dim, 64, 64, 3
G = Generator(nz, ngf, nc).to(device)
D = Discriminator(nc, ndf).to(device)

# Apply memory format optimization
G = G.to(memory_format=torch.channels_last)
D = D.to(memory_format=torch.channels_last)

G.apply(weights_init)
D.apply(weights_init)

# Print model info
total_params_G = sum(p.numel() for p in G.parameters())
total_params_D = sum(p.numel() for p in D.parameters())
print(f"Generator parameters: {total_params_G:,}")
print(f"Discriminator parameters: {total_params_D:,}")

criterion = nn.BCEWithLogitsLoss()
optD = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
optG = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))

# Learning rate schedulers for stability
schedulerD = optim.lr_scheduler.StepLR(optD, step_size=20, gamma=0.5)
schedulerG = optim.lr_scheduler.StepLR(optG, step_size=20, gamma=0.5)

scaler = GradScaler()

# Resume from checkpoint
start_epoch = 0
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    G.load_state_dict(checkpoint['G'])
    D.load_state_dict(checkpoint['D'])
    optG.load_state_dict(checkpoint['optG'])
    optD.load_state_dict(checkpoint['optD'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Resumed from checkpoint at epoch {start_epoch}")

# ----------------------
# Training loop with improvements
# ----------------------
os.makedirs("samples", exist_ok=True)

# Fixed noise for consistent sample generation
fixed_noise = torch.randn(16, nz, 1, 1, device=device)

# Loss tracking
G_losses = []
D_losses = []

print(f"Starting training from epoch {start_epoch + 1}")

for epoch in range(start_epoch, epochs):
    epoch_G_loss = 0
    epoch_D_loss = 0
    num_batches = 0
    
    for i, (real, _) in enumerate(loader):
        real = real.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        b_size = real.size(0)

        # Create labels with label smoothing for stability
        label_real = torch.ones(b_size, device=device) * 0.9  # Label smoothing
        label_fake = torch.zeros(b_size, device=device)

        # ----------------------
        # Train Discriminator
        # ----------------------
        D.zero_grad(set_to_none=True)
        with autocast():
            # Train on real images
            out_real = D(real).view(-1)
            lossD_real = criterion(out_real, label_real)

            # Train on fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = G(noise)
            out_fake = D(fake.detach()).view(-1)
            lossD_fake = criterion(out_fake, label_fake)
            
            lossD = (lossD_real + lossD_fake) / 2  # Average the losses
        
        scaler.scale(lossD).backward()
        scaler.step(optD)
        scaler.update()

        # ----------------------
        # Train Generator
        # ----------------------
        G.zero_grad(set_to_none=True)
        with autocast():
            # Generator wants discriminator to think fake images are real
            out = D(fake).view(-1)
            lossG = criterion(out, torch.ones(b_size, device=device))  # Use hard labels for G
        
        scaler.scale(lossG).backward()
        scaler.step(optG)
        scaler.update()

        epoch_G_loss += lossG.item()
        epoch_D_loss += lossD.item()
        num_batches += 1

        # Print progress every 10 batches
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(loader)}] "
                  f"D Loss: {lossD.item():.4f} G Loss: {lossG.item():.4f}")

    # Update learning rate schedulers
    schedulerD.step()
    schedulerG.step()
    
    # Calculate average losses for the epoch
    avg_G_loss = epoch_G_loss / num_batches
    avg_D_loss = epoch_D_loss / num_batches
    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)

    print(f"✅ Epoch [{epoch+1}/{epochs}] Complete - "
          f"Avg D Loss: {avg_D_loss:.4f}, Avg G Loss: {avg_G_loss:.4f}")

    # Generate and save sample images
    with torch.no_grad():
        fake_samples = G(fixed_noise).detach().cpu()
        utils.save_image(fake_samples, f"samples/fake_epoch_{epoch+1:03d}.png", 
                        normalize=True, nrow=4, padding=2)

    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'optG': optG.state_dict(),
        'optD': optD.state_dict(),
        'schedulerG': schedulerG.state_dict(),
        'schedulerD': schedulerD.state_dict(),
        'G_losses': G_losses,
        'D_losses': D_losses
    }, checkpoint_file)

    # Plot loss curves every 5 epochs
    if (epoch + 1) % 5 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(G_losses, label='Generator Loss')
        plt.plot(D_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.grid(True)
        plt.savefig(f'samples/losses_epoch_{epoch+1:03d}.png')
        plt.close()

print("🎉 Training complete!")

# Generate final sample grid
print("Generating final samples...")
with torch.no_grad():
    # Generate a larger grid of samples
    final_noise = torch.randn(64, nz, 1, 1, device=device)
    final_samples = G(final_noise).detach().cpu()
    utils.save_image(final_samples, "samples/final_samples.png", 
                    normalize=True, nrow=8, padding=2)

print("✅ Final samples saved to samples/final_samples.png")