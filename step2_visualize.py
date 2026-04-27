"""
step2_visualize.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Visually inspect the loaded dataset before training.
    Loads sample images from each MoA class and saves them as PNG files
    so you can confirm channels loaded correctly and normalization looks right.

RUN:
    python step2_visualize.py

OUTPUT:
    - outputs/sample_grid.png          → 3 channel view of one site
    - outputs/class_samples.png        → one site per MoA class
    - outputs/class_distribution.png   → bar chart of class counts
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from step1_dataset import load_metadata, BBBC021Dataset, resolve_path

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = r"D:\fluroscence\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2A — Visualize One Site (3 Channels)
# PURPOSE: Show DAPI, Tubulin, Actin channels side by side for one site.
#          Confirms each channel loaded and normalized correctly.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_one_site(dataset, idx=0):
    print("\n" + "="*60)
    print("STEP 2A — Visualizing one site (3 channels)")
    print("="*60)

    print(f"  Loading site index {idx}...")
    img, label = dataset[idx]
    img        = img.numpy()
    class_name = dataset.get_class_names()[label.item()]

    print(f"  Class : {class_name}")
    print(f"  Shape : {img.shape}")
    print(f"  Min   : {img.min():.4f}  Max: {img.max():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    channels  = ['DAPI (nucleus)', 'Tubulin (cytoskeleton)', 'Actin (cell body)']
    cmaps     = ['Blues', 'Greens', 'Reds']

    for i, (ax, name, cmap) in enumerate(zip(axes, channels, cmaps)):
        ax.imshow(img[i], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name, fontsize=12)
        ax.axis('off')

    fig.suptitle(f"Site {idx} — MoA: {class_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "sample_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2B — Visualize One Site Per MoA Class
# PURPOSE: Pick one representative site from each class and display
#          all 3 channels. Lets you see what each drug MoA looks like
#          visually — e.g. actin disruptors show different cell shape
#          compared to microtubule stabilizers.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_class_samples(dataset):
    print("\n" + "="*60)
    print("STEP 2B — Visualizing one sample per MoA class")
    print("="*60)

    class_names = dataset.get_class_names()
    num_classes = len(class_names)

    # Find first site index for each class
    print("  Finding one site per class...")
    class_indices = {}
    for idx in tqdm(range(len(dataset)), desc="  Scanning"):
        label = dataset.labels[idx]
        if label not in class_indices:
            class_indices[label] = idx
        if len(class_indices) == num_classes:
            break

    print("  Loading and plotting...")
    fig, axes = plt.subplots(num_classes, 3, figsize=(14, num_classes * 3))
    channels  = ['DAPI', 'Tubulin', 'Actin']
    cmaps     = ['Blues', 'Greens', 'Reds']

    for row, (label_idx, site_idx) in enumerate(
        tqdm(sorted(class_indices.items()), desc="  Plotting classes")
    ):
        img, label = dataset[site_idx]
        img        = img.numpy()
        class_name = class_names[label_idx]

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(img[col], cmap=cmaps[col], vmin=0, vmax=1)
            if col == 0:
                ax.set_ylabel(class_name, fontsize=9,
                              rotation=45, ha='right', va='center')
            if row == 0:
                ax.set_title(channels[col], fontsize=11)
            ax.axis('off')

    fig.suptitle("One sample per MoA class — DAPI / Tubulin / Actin",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "class_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2C — Class Distribution Bar Chart
# PURPOSE: Visualize class imbalance so we know which classes are
#          underrepresented. We will use this info in Step 5 to apply
#          weighted loss during training.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_class_distribution(dataset):
    print("\n" + "="*60)
    print("STEP 2C — Visualizing class distribution")
    print("="*60)

    class_names = dataset.get_class_names()
    counts      = np.bincount(dataset.labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, counts, color='steelblue', edgecolor='white')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(count),
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("MoA Class", fontsize=11)
    ax.set_ylabel("Number of Sites", fontsize=11)
    ax.set_title("Class distribution — Week1 dataset", fontsize=13)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

    print(f"\n  Class counts:")
    for name, count in zip(class_names, counts):
        print(f"    {name:<35} {count:>4} sites")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2D — DataLoader Batch Check
# PURPOSE: Confirm PyTorch DataLoader batches images correctly.
#          This is the exact same loader used during training.
# ══════════════════════════════════════════════════════════════════════════════

def check_dataloader(dataset):
    print("\n" + "="*60)
    print("STEP 2D — DataLoader batch check")
    print("="*60)

    loader           = DataLoader(dataset, batch_size=8,
                                  shuffle=True, num_workers=0)
    imgs, labels     = next(iter(loader))

    print(f"  Batch image shape : {imgs.shape}")
    print(f"  Batch label shape : {labels.shape}")
    print(f"  Batch labels      : {labels.tolist()}")
    print(f"  Image min / max   : {imgs.min():.4f} / {imgs.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2E — Pre-check All Paths
# PURPOSE: Scan all 540 sites and report how many files are missing
#          before we attempt any training. Catches broken paths early.
# ══════════════════════════════════════════════════════════════════════════════

def precheck_paths(dataset):
    print("\n" + "="*60)
    print("STEP 2E — Pre-checking all image paths")
    print("="*60)

    missing = 0
    for i in tqdm(range(len(dataset)), desc="  Scanning all sites"):
        row = dataset.meta.iloc[i]
        p   = resolve_path(row['Image_PathName_DAPI'], row['Image_FileName_DAPI'])
        if p is None:
            missing += 1

    print(f"\n  Total sites  : {len(dataset)}")
    print(f"  Found        : {len(dataset) - missing}")
    print(f"  Missing      : {missing}")

    if missing == 0:
        print("  All files accounted for.")
    else:
        print(f"  WARNING: {missing} sites have missing DAPI files.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Load metadata and build dataset
    meta    = load_metadata()
    dataset = BBBC021Dataset(meta, augment=False)

    # Step 2E — check all paths first
    precheck_paths(dataset)

    # Step 2A — visualize one site
    visualize_one_site(dataset, idx=10)

    # Step 2B — one sample per class
    visualize_class_samples(dataset)

    # Step 2C — class distribution chart
    visualize_class_distribution(dataset)

    # Step 2D — dataloader batch check
    check_dataloader(dataset)

    print("\n" + "="*60)
    print("  Step 2 complete. Check D:\\fluroscence\\outputs\\")
    print("="*60)