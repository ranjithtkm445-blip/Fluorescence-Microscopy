"""
step3_unet.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Define and test the U-Net architecture for nuclei segmentation.
    U-Net takes the DAPI channel (nucleus stain) as input and outputs
    a binary mask — 1 = nucleus, 0 = background.

    This file only DEFINES and TESTS the model architecture.
    Training happens in step5_train.py.

RUN:
    python step3_unet.py

OUTPUT:
    - Prints model summary (layers, parameters)
    - Confirms forward pass works (input → output shape)
    - Saves outputs/unet_pseudo_mask.png — visual of pseudo mask on real image
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from step1_dataset import load_metadata, BBBC021Dataset

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = r"D:\fluroscence\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3A — Double Convolution Block
# PURPOSE: The basic building block of U-Net.
#          Two consecutive Conv → BatchNorm → ReLU operations.
#          BatchNorm stabilizes training. ReLU adds non-linearity.
#          Used in both encoder (downsampling) and decoder (upsampling) paths.
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second convolution
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3B — U-Net Encoder (Downsampling Path)
# PURPOSE: Progressively reduce spatial size while increasing feature depth.
#          Each encoder step = DoubleConv + MaxPool.
#          MaxPool halves spatial dimensions (256→128→64→32→16).
#          Skip connections save feature maps for the decoder path.
# ══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, channels=(1, 64, 128, 256, 512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            DoubleConv(channels[i], channels[i+1])
            for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for block in self.enc_blocks:
            x = block(x)
            skip_connections.append(x)   # save for decoder skip connection
            x = self.pool(x)
        return x, skip_connections


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3C — U-Net Decoder (Upsampling Path)
# PURPOSE: Progressively restore spatial size using skip connections.
#          Each decoder step = Upsample + concat skip + DoubleConv.
#          Skip connections bring back fine spatial details lost during
#          downsampling — critical for precise nucleus boundary detection.
#
#          Channel flow:
#          Bottleneck(1024) → upconv → 512 + skip(512) = 1024 → DoubleConv → 512
#          512 → upconv → 256 + skip(256) = 512 → DoubleConv → 256
#          256 → upconv → 128 + skip(128) = 256 → DoubleConv → 128
#          128 → upconv → 64  + skip(64)  = 128 → DoubleConv → 64
# ══════════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels

        # Upsampling convolutions
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1],
                               kernel_size=2, stride=2)
            for i in range(len(channels) - 1)
        ])

        # After concat with skip connection, input = channels[i+1] * 2
        self.dec_blocks = nn.ModuleList([
            DoubleConv(channels[i+1] * 2, channels[i+1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, skip_connections):
        # Reverse skip connections — decoder goes from deep to shallow
        skip_connections = skip_connections[::-1]
        for i in range(len(self.channels) - 1):
            x    = self.upconvs[i](x)
            skip = skip_connections[i]

            # Handle size mismatch from odd input dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            # Concatenate skip connection along channel dimension
            x = torch.cat([skip, x], dim=1)
            x = self.dec_blocks[i](x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3D — Full U-Net Model
# PURPOSE: Combines Encoder + Bottleneck + Decoder + final 1x1 conv.
#          Input  : (B, 1, 256, 256) — single DAPI channel
#          Output : (B, 1, 256, 256) — binary nucleus mask (sigmoid applied)
#
#          Architecture:
#          Input(1) → Enc(64) → Enc(128) → Enc(256) → Enc(512)
#                   → Bottleneck(1024)
#                   → Dec(512) → Dec(256) → Dec(128) → Dec(64)
#                   → Output(1) sigmoid
# ══════════════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    def __init__(self,
                 enc_channels=(1, 64, 128, 256, 512),
                 dec_channels=(1024, 512, 256, 128, 64),
                 num_classes=1):
        super().__init__()

        # Encoder path
        self.encoder    = Encoder(enc_channels)

        # Bottleneck — deepest layer, outputs enc_channels[-1] * 2 = 1024
        self.bottleneck = DoubleConv(enc_channels[-1], enc_channels[-1] * 2)

        # Decoder path — starts at 1024 channels
        self.decoder    = Decoder(dec_channels)

        # Final 1x1 convolution → binary mask output
        self.head       = nn.Conv2d(dec_channels[-1], num_classes,
                                    kernel_size=1)

    def forward(self, x):
        # Encode — save skip connections
        x, skips = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode with skip connections
        x = self.decoder(x, skips)

        # Final mask output with sigmoid
        x = self.head(x)
        return torch.sigmoid(x)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3E — Generate Pseudo Mask
# PURPOSE: We have no ground truth segmentation masks for BBBC021.
#          We generate pseudo masks from the DAPI channel using
#          Otsu thresholding — a classical technique that automatically
#          finds the best intensity threshold to separate bright nuclei
#          from dark background.
#          These pseudo masks are used to train the U-Net.
# ══════════════════════════════════════════════════════════════════════════════

def generate_pseudo_mask(dapi_channel):
    """
    Generate binary nucleus mask from DAPI channel using Otsu threshold.
    Input  : numpy array (H, W) normalized to [0, 1]
    Output : numpy array (H, W) binary — 1=nucleus, 0=background
    """
    import cv2

    # Convert to uint8 for OpenCV
    dapi_uint8 = (dapi_channel * 255).astype(np.uint8)

    # Otsu thresholding — automatically finds best threshold
    _, mask = cv2.threshold(dapi_uint8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening — removes small noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Morphological closing — fills small holes inside nuclei
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return (mask / 255.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3F — Model Summary
# PURPOSE: Print total trainable parameters so we know model size
#          and can estimate memory requirements during training.
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model):
    print("\n" + "="*60)
    print("STEP 3F — U-Net model summary")
    print("="*60)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Model size (approx)  : {total_params * 4 / 1024 / 1024:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3G — Forward Pass Test
# PURPOSE: Run one dummy input through the model to confirm
#          input and output shapes are correct before training.
# ══════════════════════════════════════════════════════════════════════════════

def test_forward_pass(model, device):
    print("\n" + "="*60)
    print("STEP 3G — Forward pass test")
    print("="*60)

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 1, 256, 256).to(device)
        out   = model(dummy)

    print(f"\n  Input shape  : {dummy.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Output min   : {out.min():.4f}")
    print(f"  Output max   : {out.max():.4f}")
    print(f"  Expected     : (2, 1, 256, 256) with values in [0, 1]")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3H — Visualize Pseudo Mask on Real Image
# PURPOSE: Load one real DAPI image, generate its pseudo mask using
#          Otsu thresholding, and save side-by-side comparison.
#          Confirms pseudo masks look reasonable before U-Net training.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_pseudo_mask(dataset):
    print("\n" + "="*60)
    print("STEP 3H — Visualizing pseudo mask on real DAPI image")
    print("="*60)

    # Load one real sample
    img, label = dataset[10]
    dapi       = img[0].numpy()   # DAPI channel only

    # Generate pseudo mask using Otsu
    mask = generate_pseudo_mask(dapi)

    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(dapi, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('DAPI channel (input)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Pseudo mask (Otsu threshold)', fontsize=12)
    axes[1].axis('off')

    # Overlay — DAPI in blue, mask in red
    overlay        = np.zeros((256, 256, 3))
    overlay[:,:,2] = dapi           # blue channel = DAPI
    overlay[:,:,0] = mask * 0.5     # red channel  = mask overlay
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title('Overlay (DAPI + mask)', fontsize=12)
    axes[2].axis('off')

    class_name = dataset.get_class_names()[label.item()]
    fig.suptitle(f"Pseudo mask — MoA: {class_name}", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "unet_pseudo_mask.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Device — use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Build model
    print("\n" + "="*60)
    print("STEP 3A-D — Building U-Net model")
    print("="*60)
    model = UNet().to(device)
    print("  U-Net built successfully")

    # Model summary
    print_model_summary(model)

    # Forward pass test
    test_forward_pass(model, device)

    # Load dataset for visualization
    meta    = load_metadata()
    dataset = BBBC021Dataset(meta, augment=False)

    # Visualize pseudo mask on real image
    visualize_pseudo_mask(dataset)

    print("\n" + "="*60)
    print("  Step 3 complete. Ready for Step 4.")
    print("="*60)