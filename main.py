import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------
# Simple Dataset
# ------------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []
        
        # Find images
        extensions = ['*.jpg', '*.png', '*.jpeg']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(root, ext)))
        
        print(f"Loading {len(files)} images...")
        
        # Load to memory
        for i, file_path in enumerate(files):
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                self.data.append(np.array(img, dtype=np.uint8))
                
                if (i + 1) % 100 == 0:
                    print(f"Loaded {i + 1}/{len(files)} images")
            except:
                continue
        
        print(f"Dataset ready: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img

# Simple transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Create dataset
dataset = SimpleImageDataset(DATASET_PATH, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# ------------------------------
# Simple Models
# ------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            
            # Final layer
            nn.ConvTranspose2d(NGF, NC, 3, 1, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            
            # 32x32 -> 16x16
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, True),
            
            # 16x16 -> 8x8
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, True),
            
            # 8x8 -> 4x4
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, True),
            
            # 4x4 -> 1x1
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
        )
    
    def forward(self, x):
        return self.main(x).view(-1)

# ------------------------------
# Simple EMA (Exponential Moving Average)
# ------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = (1 - self.decay) * param.data + self.decay * self.shadow[name]
    
    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            param.data = self.backup[name]

# ------------------------------
# Initialize Everything
# ------------------------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# Create models
netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(init_weights)
netD.apply(init_weights)

# Create EMA for generator
ema_G = EMA(netG)

# Loss and optimizers
criterion = nn.BCEWithLogitsLoss()
optG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
optD = optim.Adam(netD.parameters(), lr=LR * 0.5, betas=(BETA1, BETA2))

# Fixed noise for visualization
fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

# Training tracking
G_losses, D_losses = [], []
start_epoch = 0

print(f"Generator params: {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in netD.parameters()):,}")

# Resume from checkpoint if exists
checkpoint_path = os.path.join(CHECKPOINT_DIR, "last.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['netG'])
    netD.load_state_dict(checkpoint['netD'])
    optG.load_state_dict(checkpoint['optG'])
    optD.load_state_dict(checkpoint['optD'])
    G_losses = checkpoint.get('G_losses', [])
    D_losses = checkpoint.get('D_losses', [])
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    if 'ema_shadow' in checkpoint:
        ema_G.shadow = checkpoint['ema_shadow']
    
    print(f"Resumed from epoch {start_epoch}")

# ------------------------------
# Simple Training Loop
# ------------------------------
print("Starting training...")

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_G_loss = 0
    epoch_D_loss = 0
    
    for i, real_batch in enumerate(dataloader):
        real_batch = real_batch.to(device)
        batch_size = real_batch.size(0)
        
        # Create labels
        real_labels = torch.ones(batch_size, device=device) * 0.9  # Label smoothing
        fake_labels = torch.zeros(batch_size, device=device)
        
        # ------------------------------
        # Train Discriminator
        # ------------------------------
        netD.zero_grad()
        
        # Real images
        real_output = netD(real_batch)
        real_loss = criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, NZ, 1, 1, device=device)
        fake_batch = netG(noise).detach()  # Don't backprop through generator
        fake_output = netD(fake_batch)
        fake_loss = criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optD.step()
        
        # ------------------------------
        # Train Generator
        # ------------------------------
        netG.zero_grad()
        
        # Generate new fake images
        noise = torch.randn(batch_size, NZ, 1, 1, device=device)
        fake_batch = netG(noise)
        output = netD(fake_batch)
        
        # Generator wants discriminator to think fake images are real
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optG.step()
        
        # Update EMA
        ema_G.update(netG)
        
        # Track losses
        epoch_G_loss += g_loss.item()
        epoch_D_loss += d_loss.item()
        
        # Print progress
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i}/{len(dataloader)}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
    
    # Average losses
    avg_G_loss = epoch_G_loss / len(dataloader)
    avg_D_loss = epoch_D_loss / len(dataloader)
    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] G_loss: {avg_G_loss:.4f} D_loss: {avg_D_loss:.4f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optG': optG.state_dict(),
        'optD': optD.state_dict(),
        'G_losses': G_losses,
        'D_losses': D_losses,
        'ema_shadow': ema_G.shadow
    }, checkpoint_path)
    
    # Show samples every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Use EMA weights for better samples
        ema_G.apply(netG)
        with torch.no_grad():
            fake_samples = netG(fixed_noise).cpu()
        ema_G.restore(netG)
        
        # Save and display
        utils.save_image(fake_samples, f"{CHECKPOINT_DIR}/samples_epoch_{epoch+1}.png", 
                        normalize=True, nrow=8)
        
        grid = utils.make_grid(fake_samples, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Samples - Epoch {epoch+1}")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()

print("Training completed!")

# Final model with EMA weights
ema_G.apply(netG)
torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/generator_final.pth")
ema_G.restore(netG)

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
    ema_G.apply(netG)
    with torch.no_grad():
        final_samples = netG(fixed_noise).cpu()
    ema_G.restore(netG)
    
    final_grid = utils.make_grid(final_samples, padding=2, normalize=True)
    plt.axis("off")
    plt.title("Final Generated Samples")
    plt.imshow(np.transpose(final_grid, (1, 2, 0)))
    
    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/training_summary.png")
    plt.show()

print(f"Final model saved to {CHECKPOINT_DIR}/generator_final.pth")