"""
step1_dataset.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Load the BBBC021 metadata CSVs, merge them to connect every imaging
    site to its MoA label, filter strictly to the 6 Week1 plates we have
    on disk, and build a PyTorch Dataset that returns a 3-channel
    fluorescence image tensor + MoA class label.

RUN:
    python step1_dataset.py

OUTPUT:
    - Prints dataset statistics (site count, class distribution)
    - Prints unique plate paths to confirm no other weeks are included
    - Confirms one image loads correctly (shape, value range, label)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import tifffile
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = r"D:\fluroscence"
IMAGE_CSV  = r"D:\fluroscence\BBBC021_v1_image.csv"
MOA_CSV    = r"D:\fluroscence\BBBC021_v1_moa.csv"
IMAGE_SIZE = 256

# ── Only these 6 plates are downloaded on disk ────────────────────────────────
AVAILABLE_PLATES = [
    'Week1/Week1_22123',
    'Week1/Week1_22141',
    'Week1/Week1_22161',
    'Week1/Week1_22361',
    'Week1/Week1_22381',
    'Week1/Week1_22401',
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1A — Load and Merge Metadata
# PURPOSE: Join image.csv + moa.csv on compound + concentration.
#          This gives every imaging site a MoA class label.
#          Drop DMSO (negative control). Filter strictly to the 6
#          Week1 plates we have downloaded on disk.
# ══════════════════════════════════════════════════════════════════════════════

def load_metadata():
    print("\n" + "="*60)
    print("STEP 1A — Loading and merging metadata")
    print("="*60)

    img_df = pd.read_csv(IMAGE_CSV)
    moa_df = pd.read_csv(MOA_CSV)

    print(f"  image.csv rows loaded : {len(img_df)}")
    print(f"  moa.csv rows loaded   : {len(moa_df)}")

    # Clean column names — remove stray quotes
    img_df.columns = img_df.columns.str.strip().str.replace('"', '')
    moa_df.columns = moa_df.columns.str.strip().str.replace('"', '')

    # Normalize compound names and concentrations for clean joining
    moa_df['compound']      = moa_df['compound'].str.strip().str.lower()
    moa_df['concentration'] = moa_df['concentration'].astype(float).round(6)

    img_df['Image_Metadata_Compound']      = img_df['Image_Metadata_Compound'].str.strip().str.lower()
    img_df['Image_Metadata_Concentration'] = img_df['Image_Metadata_Concentration'].astype(float).round(6)

    # Join image metadata with MoA labels on compound + concentration
    merged = img_df.merge(
        moa_df,
        left_on  = ['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
        right_on = ['compound', 'concentration'],
        how      = 'inner'
    )

    print(f"  Rows after merge      : {len(merged)}")

    # Drop DMSO — negative control, not a drug MoA class
    merged = merged[merged['moa'] != 'DMSO'].reset_index(drop=True)
    print(f"  Rows after DMSO drop  : {len(merged)}")

    # Strict filter — only rows whose path exactly matches our 6 plates
    merged = merged[
        merged['Image_PathName_DAPI'].isin(AVAILABLE_PLATES)
    ].reset_index(drop=True)

    print(f"  Rows after plate filter : {len(merged)}")

    # Print unique plate paths — confirm no other weeks present
    print(f"\n  Plate paths in dataset:")
    for p in sorted(merged['Image_PathName_DAPI'].unique()):
        count = (merged['Image_PathName_DAPI'] == p).sum()
        print(f"    {p:<30} {count:>4} sites")

    print(f"\n  Total sites            : {len(merged)}")
    print(f"  Unique MoA classes     : {merged['moa'].nunique()}")
    print(f"  Unique compounds       : {merged['Image_Metadata_Compound'].nunique()}")

    print(f"\n  MoA class distribution:")
    for moa in sorted(merged['moa'].unique()):
        count = (merged['moa'] == moa).sum()
        print(f"    {moa:<35} {count:>4} sites")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1B — Resolve Absolute File Path
# PURPOSE: image.csv stores partial paths like 'Week1/Week1_22123' and
#          filenames like 'C10_s1_w1XXXX.tif' without the date prefix.
#          This function finds the actual file on disk by suffix match.
# ══════════════════════════════════════════════════════════════════════════════

def resolve_path(path_val, fname_val):
    """
    Converts image CSV path + filename into absolute Windows path.
    Uses suffix match because image CSV filenames are missing the date
    segment (e.g. 150607) that is present in the actual filename on disk.
    """
    week_sub    = path_val.replace('\\', '/').split('/')[-1]  # Week1_22123
    folder      = f"BBBC021_v1_images_{week_sub}"
    dir_path    = os.path.join(ROOT, folder, week_sub)

    if not os.path.isdir(dir_path):
        return None

    fname_lower = fname_val.lower()
    for f in os.listdir(dir_path):
        if f.lower().endswith(fname_lower):
            return os.path.join(dir_path, f)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1C — Per-Channel Normalization
# PURPOSE: Each fluorescence channel (DAPI, Tubulin, Actin) has a very
#          different intensity range. Percentile normalization clips
#          outlier bright pixels and scales each channel independently
#          to [0, 1]. This is standard practice for fluorescence microscopy.
# ══════════════════════════════════════════════════════════════════════════════

def percentile_normalize(channel, low=1, high=99):
    p_low  = np.percentile(channel, low)
    p_high = np.percentile(channel, high)
    denom  = (p_high - p_low) if (p_high - p_low) > 0 else 1.0
    return np.clip((channel - p_low) / denom, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1D — PyTorch Dataset Class
# PURPOSE: Wraps the metadata into a PyTorch Dataset so DataLoader can
#          batch, shuffle, and feed images to the model during training.
#          Each __getitem__ call loads 3 .tif files, normalizes them,
#          resizes to 256x256, and returns (image_tensor, label).
# ══════════════════════════════════════════════════════════════════════════════

class BBBC021Dataset(Dataset):
    def __init__(self, metadata, label_encoder=None, augment=False):
        """
        metadata      : merged DataFrame from load_metadata()
        label_encoder : pass fitted encoder for val set to share class mapping
        augment       : apply random flips and rotations during training
        """
        self.meta    = metadata.reset_index(drop=True)
        self.augment = augment

        # Fit label encoder on training set
        # Pass existing encoder for validation set so classes stay consistent
        if label_encoder is None:
            self.le = LabelEncoder()
            self.le.fit(self.meta['moa'])
        else:
            self.le = label_encoder

        self.labels = self.le.transform(self.meta['moa'])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        # Load all 3 fluorescence channels
        dapi    = self._load_channel(row, 'Image_PathName_DAPI',    'Image_FileName_DAPI')
        tubulin = self._load_channel(row, 'Image_PathName_Tubulin', 'Image_FileName_Tubulin')
        actin   = self._load_channel(row, 'Image_PathName_Actin',   'Image_FileName_Actin')

        # Stack into (3, H, W) array
        img = np.stack([dapi, tubulin, actin], axis=0).astype(np.float32)

        # Resize each channel to IMAGE_SIZE x IMAGE_SIZE
        img = self._resize(img)

        # Normalize each channel independently to [0, 1]
        for i in range(3):
            img[i] = percentile_normalize(img[i])

        # Apply augmentation during training only
        if self.augment:
            img = self._augment(img)

        label = int(self.labels[idx])
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _load_channel(self, row, path_col, fname_col):
        """Load one .tif channel. Returns zeros if file is missing."""
        fpath = resolve_path(row[path_col], row[fname_col])
        if fpath is None or not os.path.exists(fpath):
            return np.zeros((1024, 1280), dtype=np.float32)
        return tifffile.imread(fpath).astype(np.float32)

    def _resize(self, img):
        """Resize all 3 channels to IMAGE_SIZE x IMAGE_SIZE."""
        out = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for i in range(3):
            out[i] = cv2.resize(img[i], (IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_AREA)
        return out

    def _augment(self, img):
        """Random horizontal flip, vertical flip, and 90° rotation."""
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
        k = np.random.randint(0, 4)
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        return img

    def get_class_names(self):
        return list(self.le.classes_)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1E — Sanity Check
# PURPOSE: Run this file directly to confirm everything works before
#          moving to Step 2. Loads metadata, builds dataset, scans all
#          paths, loads one sample, prints shape + label.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Load and merge metadata
    meta = load_metadata()

    # Build dataset
    print("\n" + "="*60)
    print("STEP 1D — Building PyTorch dataset")
    print("="*60)
    ds = BBBC021Dataset(meta, augment=False)
    print(f"\n  Dataset size : {len(ds)} sites")
    print(f"  Class names  : {ds.get_class_names()}")

    # Scan all paths with progress bar
    print("\n" + "="*60)
    print("STEP 1E — Scanning all image paths")
    print("="*60)
    missing = 0
    for i in tqdm(range(len(ds)), desc="  Checking paths"):
        row = ds.meta.iloc[i]
        p   = resolve_path(row['Image_PathName_DAPI'], row['Image_FileName_DAPI'])
        if p is None:
            missing += 1

    print(f"\n  Total sites : {len(ds)}")
    print(f"  Found       : {len(ds) - missing}")
    print(f"  Missing     : {missing}")

    # Load one sample and confirm
    print("\n" + "="*60)
    print("STEP 1F — Loading one sample (sanity check)")
    print("="*60)

    # Find first non-missing sample
    good_idx = 0
    for i in range(len(ds)):
        row = ds.meta.iloc[i]
        p   = resolve_path(row['Image_PathName_DAPI'], row['Image_FileName_DAPI'])
        if p is not None:
            good_idx = i
            break

    img, label = ds[good_idx]
    print(f"\n  Sample index : {good_idx}")
    print(f"  Image shape  : {img.shape}")
    print(f"  Image min    : {img.min():.4f}")
    print(f"  Image max    : {img.max():.4f}")
    print(f"  Label index  : {label.item()}")
    print(f"  Label name   : {ds.get_class_names()[label.item()]}")

    print("\n" + "="*60)
    print("  Step 1 complete. Ready for Step 2.")
    print("="*60)