import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from copy import deepcopy

# ------------------------------
# Config
# ------------------------------
image_size = 64   # training resolution
nz = 100          # latent dim
ngf = 64
ndf = 64
nc = 3
batch_size = 64
num_epochs = 900
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
checkpoint_dir = "./checkpoints"
dataset_path = "/content/imagen/Logos"   # <--- Tweak this path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(checkpoint_dir, exist_ok=True)

# ------------------------------
# RAM-based Dataset with uint8
# ------------------------------
class RAMImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []
        
        # Find all image files
        files = glob.glob(os.path.join(root, "*.jpg")) + \
                glob.glob(os.path.join(root, "*.png")) + \
                glob.glob(os.path.join(root, "*.jpeg"))
        
        print(f"Found {len(files)} images in {root}")
        print("Loading images to RAM as uint8...")
        
        # Load all images to RAM as uint8
        for i, file_path in enumerate(files):
            try:
                # Load and resize image
                img = Image.open(file_path).convert("RGB")
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Convert to numpy array (uint8) and store
                img_array = np.array(img, dtype=np.uint8)  # Shape: (H, W, 3)
                self.data.append(img_array)
                
                # Progress indicator
                if (i + 1) % 100 == 0 or (i + 1) == len(files):
                    print(f"Loaded {i + 1}/{len(files)} images to RAM")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.data)} images to RAM")
        
        # Calculate memory usage
        if len(self.data) > 0:
            single_img_size = self.data[0].nbytes
            total_memory = len(self.data) * single_img_size
            print(f"Dataset memory usage: {total_memory / (1024**2):.1f} MB ({total_memory / (1024**3):.2f} GB)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get uint8 image data
        img_array = self.data[idx]
        
        # Convert to PIL Image for transforms
        img = Image.fromarray(img_array)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        return img

# Transform that works with pre-resized images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),  # Add data augmentation
    transforms.CenterCrop(image_size),  # Center crop in case of slight size differences
    transforms.ToTensor(),              # Converts to float32 and scales to [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1,1]
])

# Use the RAM dataset
dataset = RAMImageDataset(dataset_path, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# ------------------------------
# Models
# ------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False),  # Increased capacity
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),  # Changed to 3x3 kernel
            nn.Tanh()
        )

    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Apply spectral norm to all Conv2d layers
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)),  # Reduced capacity
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False)),  # Reduced
            nn.InstanceNorm2d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),   # Reduced
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)), # Reduced
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False))
            # Removed Sigmoid - will use BCEWithLogitsLoss instead
        )

    def forward(self, x): return self.main(x).view(-1)

# ------------------------------
# EMA Helper Class
# ------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ------------------------------
# Init
# ------------------------------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None: nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)

netG, netD = Generator().to(device), Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# Initialize EMA for generator
ema_G = EMA(netG, decay=0.999)

# Print model info
total_params_G = sum(p.numel() for p in netG.parameters())
total_params_D = sum(p.numel() for p in netD.parameters())
print(f"Generator parameters: {total_params_G:,}")
print(f"Discriminator parameters: {total_params_D:,}")
print("✅ Applied Spectral Normalization to Discriminator")
print("✅ Initialized EMA for Generator")

# Use BCEWithLogitsLoss for numerical stability
criterion = nn.BCEWithLogitsLoss()
# Make generator learn faster than discriminator
optG = optim.Adam(netG.parameters(), lr=lr*2, betas=(beta1, beta2))  # G learns faster
optD = optim.Adam(netD.parameters(), lr=lr*0.5, betas=(beta1, beta2))  # D learns slower

schedulerG = optim.lr_scheduler.StepLR(optG, step_size=20, gamma=0.5)
schedulerD = optim.lr_scheduler.StepLR(optD, step_size=20, gamma=0.5)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
G_losses, D_losses = [], []
D_real_probs, D_fake_probs = [], []  # Track discriminator confidence
start_epoch = 0

# Resume
ckpt_path = os.path.join(checkpoint_dir, "last.pth")
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(checkpoint['netG'])
    netD.load_state_dict(checkpoint['netD'])
    optG.load_state_dict(checkpoint['optG'])
    optD.load_state_dict(checkpoint['optD'])
    schedulerG.load_state_dict(checkpoint['schedulerG'])
    schedulerD.load_state_dict(checkpoint['schedulerD'])
    G_losses = checkpoint['G_losses']
    D_losses = checkpoint['D_losses']
    start_epoch = checkpoint['epoch'] + 1
    
    # Load EMA state if available
    if 'ema_shadow' in checkpoint:
        ema_G.shadow = checkpoint['ema_shadow']
        print("✅ Loaded EMA weights from checkpoint")
    else:
        # Reinitialize EMA if not in checkpoint (backward compatibility)
        ema_G = EMA(netG, decay=0.999)
        print("⚠️ EMA weights not found in checkpoint, reinitializing")
    
    print(f"Resumed from epoch {start_epoch}")

# ------------------------------
# Train
# ------------------------------
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    epoch_D_loss = 0
    epoch_G_loss = 0
    epoch_D_real_prob = 0
    epoch_D_fake_prob = 0
    num_batches = 0
    
    for i, real in enumerate(dataloader):
        real = real.to(device, non_blocking=True)
        b_size = real.size(0)

        # Labels with more noise to make discriminator's task harder
        label_real = torch.ones(b_size, device=device) * (0.7 + 0.3 * torch.rand(b_size, device=device))
        label_fake = torch.zeros(b_size, device=device) * (0.3 * torch.rand(b_size, device=device))

        # ---------------- D ----------------
        # Train discriminator less frequently (every other iteration)
        if (i % 2 == 0):  # Only update D every other batch
            netD.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out_real = netD(real)
                    loss_real = criterion(out_real, label_real)
                    
                    # Add more noise to latent space
                    noise = torch.randn(b_size, nz, 1, 1, device=device) * 1.1
                    fake = netG(noise)
                    out_fake = netD(fake.detach())
                    loss_fake = criterion(out_fake, label_fake)
                    
                    # Add gradient penalty
                    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
                    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
                    d_interpolated = netD(interpolated)
                    gradients = torch.autograd.grad(
                        outputs=d_interpolated,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(d_interpolated),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    
                    lossD = (loss_real + loss_fake) / 2 + 10.0 * gradient_penalty
                
                scaler.scale(lossD).backward()
                scaler.unscale_(optD)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
                scaler.step(optD)
            else:
                out_real = netD(real)
                loss_real = criterion(out_real, label_real)
                
                noise = torch.randn(b_size, nz, 1, 1, device=device) * 1.1
                fake = netG(noise)
                out_fake = netD(fake.detach())
                loss_fake = criterion(out_fake, label_fake)
                
                # Add gradient penalty
                alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
                interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
                d_interpolated = netD(interpolated)
                gradients = torch.autograd.grad(
                    outputs=d_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(d_interpolated),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                
                lossD = (loss_real + loss_fake) / 2 + 10.0 * gradient_penalty
                
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
                optD.step()

        # ---------------- G ----------------
        netG.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = netD(fake)
                lossG = criterion(out, label_real)  # G wants D to think fake is real
            
            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()
        else:
            out = netD(fake)
            lossG = criterion(out, label_real)
            
            lossG.backward()
            optG.step()

        # Update EMA after each generator update
        ema_G.update(netG)
        
        # Calculate metrics
        with torch.no_grad():
            D_real_prob = torch.sigmoid(out_real).mean().item()
            D_fake_prob = torch.sigmoid(out_fake).mean().item()
            
        epoch_D_loss += lossD.item() if (i % 2 == 0) else 0
        epoch_G_loss += lossG.item()
        epoch_D_real_prob += D_real_prob
        epoch_D_fake_prob += D_fake_prob
        num_batches += 1
        
        # Print diagnostics every 50 batches
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(dataloader)}]")
            print(f"D real confidence: {D_real_prob:.3f}, D fake confidence: {D_fake_prob:.3f}")
            print(f"D accuracy: {((D_real_prob > 0.5).astype(float) + (D_fake_prob < 0.5).astype(float)) / 2:.3f}")

    # Calculate average metrics for the epoch
    avg_D_loss = epoch_D_loss / (num_batches / 2)  # Account for skipping every other update
    avg_G_loss = epoch_G_loss / num_batches
    avg_D_real_prob = epoch_D_real_prob / num_batches
    avg_D_fake_prob = epoch_D_fake_prob / num_batches
    
    D_losses.append(avg_D_loss)
    G_losses.append(avg_G_loss)
    D_real_probs.append(avg_D_real_prob)
    D_fake_probs.append(avg_D_fake_prob)

    schedulerG.step()
    schedulerD.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] D Loss: {avg_D_loss:.4f} G Loss: {avg_G_loss:.4f}")
    print(f"D real prob: {avg_D_real_prob:.3f}, D fake prob: {avg_D_fake_prob:.3f}")

    # Save checkpoint (including EMA state)
    torch.save({
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "schedulerG": schedulerG.state_dict(),
        "schedulerD": schedulerD.state_dict(),
        "G_losses": G_losses,
        "D_losses": D_losses,
        "D_real_probs": D_real_probs,
        "D_fake_probs": D_fake_probs,
        "ema_shadow": ema_G.shadow  # Save EMA weights
    }, ckpt_path)

    # Preview every 5 epochs to avoid cluttering output
    if (epoch + 1) % 5 == 0:
        # Use EMA weights for sample generation
        ema_G.apply_shadow(netG)
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        ema_G.restore(netG)  # Restore original weights
        
        grid = utils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1} (EMA Generator)")
        plt.imshow(np.transpose(grid,(1,2,0)))
        plt.show()
        
        # Also save to file
        utils.save_image(fake, f"{checkpoint_dir}/samples_epoch_{epoch+1:03d}_ema.png", 
                        normalize=True, nrow=8, padding=2)

print("Training completed!")

# Plot final results
if len(G_losses) > 0:
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    
    # Discriminator confidence
    plt.subplot(2, 2, 2)
    plt.plot(D_real_probs, label='D Real Confidence')
    plt.plot(D_fake_probs, label='D Fake Confidence')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Discriminator Confidence')
    plt.grid(True)
    
    # D accuracy
    plt.subplot(2, 2, 3)
    D_accuracy = [(0.5 * (r > 0.5) + 0.5 * (f < 0.5)) for r, f in zip(D_real_probs, D_fake_probs)]
    plt.plot(D_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Discriminator Accuracy')
    plt.grid(True)
    
    # Final samples
    plt.subplot(2, 2, 4)
    ema_G.apply_shadow(netG)
    with torch.no_grad():
        final_samples = netG(fixed_noise).detach().cpu()
    ema_G.restore(netG)
    
    final_grid = utils.make_grid(final_samples, padding=2, normalize=True)
    plt.axis("off")
    plt.title("Final Generated Samples (EMA)")
    plt.imshow(np.transpose(final_grid,(1,2,0)))
    
    plt.tight_layout()
    plt.savefig(f"{checkpoint_dir}/training_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save final EMA model separately
    ema_G.apply_shadow(netG)
    torch.save(netG.state_dict(), f"{checkpoint_dir}/generator_ema_final.pth")
    ema_G.restore(netG)
    print(f"✅ Final EMA generator saved to {checkpoint_dir}/generator_ema_final.pth")