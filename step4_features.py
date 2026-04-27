"""
step4_features.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Extract biological features from each imaging site using the pseudo
    masks generated in Step 3. Features include:
    - Morphological features (cell count, size, shape, roundness)
    - Per-channel intensity statistics (mean, std, median, max)
    - Synthetic omics vector (simulated gene expression per compound)

    These features feed into the multi-modal fusion model in Step 5.

RUN:
    python step4_features.py

OUTPUT:
    - outputs/features.csv        → all extracted features per site
    - outputs/feature_preview.png → feature distributions per class
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from step1_dataset import load_metadata, BBBC021Dataset, resolve_path
from step3_unet import generate_pseudo_mask
import tifffile

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = r"D:\fluroscence"
OUTPUT_DIR = r"D:\fluroscence\outputs"
IMAGE_SIZE = 256
OMICS_DIM  = 64    # synthetic omics vector dimension
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4A — Load Raw Channel Image
# PURPOSE: Load one channel directly from disk at full resolution
#          before resizing, so morphological measurements are accurate.
#          Resizing before measurement would distort cell size estimates.
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_channel(row, path_col, fname_col):
    """Load raw .tif channel and normalize to [0,1]."""
    fpath = resolve_path(row[path_col], row[fname_col])
    if fpath is None or not os.path.exists(fpath):
        return None
    img = tifffile.imread(fpath).astype(np.float32)
    # Percentile normalize
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    denom   = (p99 - p1) if (p99 - p1) > 0 else 1.0
    return np.clip((img - p1) / denom, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4B — Morphological Feature Extraction
# PURPOSE: Use the pseudo mask to find individual nucleus regions and
#          measure their physical properties.
#
#          Features extracted per site:
#          - cell_count        : number of detected nuclei
#          - mean_area         : average nucleus area in pixels
#          - std_area          : variation in nucleus size
#          - mean_perimeter    : average nucleus boundary length
#          - mean_roundness    : how circular nuclei are (1.0 = perfect circle)
#          - mean_solidity     : how solid/convex nuclei are
#          - nuclear_area_frac : fraction of image covered by nuclei
# ══════════════════════════════════════════════════════════════════════════════

def extract_morphological_features(mask):
    """
    Extract morphological features from binary nucleus mask.
    Input  : numpy array (H, W) binary mask
    Output : dict of morphological features
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find connected components — each component = one nucleus
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA
    # Label 0 = background, skip it
    areas      = stats[1:, cv2.CC_STAT_AREA]
    cell_count = len(areas)

    if cell_count == 0:
        return {
            'cell_count'       : 0,
            'mean_area'        : 0.0,
            'std_area'         : 0.0,
            'mean_perimeter'   : 0.0,
            'mean_roundness'   : 0.0,
            'mean_solidity'    : 0.0,
            'nuclear_area_frac': 0.0,
        }

    # Filter out very small regions (noise) — min 20 pixels
    valid_mask = areas >= 20
    areas      = areas[valid_mask]
    cell_count = len(areas)

    if cell_count == 0:
        return {
            'cell_count'       : 0,
            'mean_area'        : 0.0,
            'std_area'         : 0.0,
            'mean_perimeter'   : 0.0,
            'mean_roundness'   : 0.0,
            'mean_solidity'    : 0.0,
            'nuclear_area_frac': 0.0,
        }

    # Measure perimeter and roundness per nucleus
    perimeters = []
    roundness  = []
    solidity   = []

    # Get valid label indices (skip background label 0)
    valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= 20)[0] + 1

    for label_idx in valid_labels:
        # Extract single nucleus mask
        nucleus = (labels == label_idx).astype(np.uint8) * 255

        # Find contour
        contours, _ = cv2.findContours(nucleus,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        cnt  = contours[0]
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)

        if peri > 0 and area > 0:
            perimeters.append(peri)
            # Roundness = 4π × area / perimeter²
            # Perfect circle = 1.0
            roundness.append(4 * np.pi * area / (peri ** 2))

            # Solidity = area / convex hull area
            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity.append(area / hull_area)

    h, w               = mask.shape
    nuclear_area_frac  = mask.sum() / (h * w)

    return {
        'cell_count'       : cell_count,
        'mean_area'        : float(np.mean(areas)),
        'std_area'         : float(np.std(areas)),
        'mean_perimeter'   : float(np.mean(perimeters)) if perimeters else 0.0,
        'mean_roundness'   : float(np.mean(roundness))  if roundness  else 0.0,
        'mean_solidity'    : float(np.mean(solidity))   if solidity   else 0.0,
        'nuclear_area_frac': float(nuclear_area_frac),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4C — Intensity Feature Extraction
# PURPOSE: Measure pixel intensity statistics for each channel.
#          These capture how bright/dim cells are in each stain —
#          which changes depending on drug mechanism.
#
#          Features per channel (DAPI, Tubulin, Actin):
#          - mean   : average brightness
#          - std    : variation in brightness
#          - median : robust central brightness
#          - max    : brightest pixel (outlier structures)
#          - p25    : 25th percentile intensity
#          - p75    : 75th percentile intensity
# ══════════════════════════════════════════════════════════════════════════════

def extract_intensity_features(dapi, tubulin, actin):
    """
    Extract intensity statistics from all 3 channels.
    Input  : 3 numpy arrays (H, W) normalized to [0, 1]
    Output : dict of intensity features
    """
    features = {}
    channels = {'dapi': dapi, 'tubulin': tubulin, 'actin': actin}

    for name, ch in channels.items():
        if ch is None:
            features[f'{name}_mean']   = 0.0
            features[f'{name}_std']    = 0.0
            features[f'{name}_median'] = 0.0
            features[f'{name}_max']    = 0.0
            features[f'{name}_p25']    = 0.0
            features[f'{name}_p75']    = 0.0
        else:
            features[f'{name}_mean']   = float(np.mean(ch))
            features[f'{name}_std']    = float(np.std(ch))
            features[f'{name}_median'] = float(np.median(ch))
            features[f'{name}_max']    = float(np.max(ch))
            features[f'{name}_p25']    = float(np.percentile(ch, 25))
            features[f'{name}_p75']    = float(np.percentile(ch, 75))

    return features


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4D — Synthetic Omics Vector
# PURPOSE: Real drug discovery pipelines combine imaging data with omics
#          data (gene expression, proteomics). We don't have real omics
#          data for BBBC021, so we simulate it.
#
#          Each compound gets a unique but reproducible 64-dimensional
#          vector generated from a seeded random number generator.
#          Same compound always gets the same vector.
#          This lets us demonstrate multi-modal fusion without real omics.
# ══════════════════════════════════════════════════════════════════════════════

def generate_omics_vectors(compounds):
    """
    Generate synthetic omics vectors for each unique compound.
    Input  : list of compound names
    Output : dict mapping compound name → 64-dim numpy vector
    """
    np.random.seed(RANDOM_SEED)
    unique_compounds = list(set(compounds))
    omics_dict       = {}

    for compound in unique_compounds:
        # Seed based on compound name hash for reproducibility
        seed = abs(hash(compound)) % (2**31)
        rng  = np.random.RandomState(seed)
        omics_dict[compound] = rng.randn(OMICS_DIM).astype(np.float32)

    return omics_dict


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4E — Extract Features for All Sites
# PURPOSE: Loop over all 372 sites, extract morphological + intensity +
#          omics features, and collect into a single DataFrame.
#          This is the feature matrix that feeds into Step 5.
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(metadata):
    print("\n" + "="*60)
    print("STEP 4E — Extracting features for all sites")
    print("="*60)

    # Generate omics vectors for all compounds
    compounds    = metadata['Image_Metadata_Compound'].tolist()
    omics_dict   = generate_omics_vectors(compounds)
    print(f"  Omics vectors generated for {len(omics_dict)} compounds")

    all_features = []

    for idx in tqdm(range(len(metadata)), desc="  Extracting features"):
        row = metadata.iloc[idx]

        # Load raw channels at full resolution
        dapi    = load_raw_channel(row, 'Image_PathName_DAPI',    'Image_FileName_DAPI')
        tubulin = load_raw_channel(row, 'Image_PathName_Tubulin', 'Image_FileName_Tubulin')
        actin   = load_raw_channel(row, 'Image_PathName_Actin',   'Image_FileName_Actin')

        if dapi is None:
            continue

        # Resize DAPI to 256x256 for mask generation
        dapi_resized = cv2.resize(dapi, (IMAGE_SIZE, IMAGE_SIZE),
                                  interpolation=cv2.INTER_AREA)

        # Generate pseudo mask from DAPI
        mask = generate_pseudo_mask(dapi_resized)

        # Resize all channels to 256x256 for intensity features
        if tubulin is not None:
            tubulin = cv2.resize(tubulin, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_AREA)
        if actin is not None:
            actin = cv2.resize(actin, (IMAGE_SIZE, IMAGE_SIZE),
                               interpolation=cv2.INTER_AREA)

        # Extract features
        morph_feats     = extract_morphological_features(mask)
        intensity_feats = extract_intensity_features(dapi_resized,
                                                      tubulin, actin)

        # Get omics vector for this compound
        compound     = row['Image_Metadata_Compound']
        omics_vector = omics_dict[compound]

        # Build omics feature dict
        omics_feats  = {f'omics_{i}': float(omics_vector[i])
                        for i in range(OMICS_DIM)}

        # Combine all features
        site_features = {
            'site_idx'    : idx,
            'compound'    : compound,
            'concentration': row['Image_Metadata_Concentration'],
            'moa'         : row['moa'],
            'plate'       : row['Image_PathName_DAPI'],
        }
        site_features.update(morph_feats)
        site_features.update(intensity_feats)
        site_features.update(omics_feats)

        all_features.append(site_features)

    df = pd.DataFrame(all_features)
    print(f"\n  Total sites processed : {len(df)}")
    print(f"  Total features        : {len(df.columns)}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4F — Save Features to CSV
# PURPOSE: Save extracted features to disk so they can be loaded
#          directly in Step 5 without re-extracting every time.
# ══════════════════════════════════════════════════════════════════════════════

def save_features(df):
    print("\n" + "="*60)
    print("STEP 4F — Saving features to CSV")
    print("="*60)

    save_path = os.path.join(OUTPUT_DIR, "features.csv")
    df.to_csv(save_path, index=False)
    print(f"  Saved → {save_path}")
    print(f"  Shape : {df.shape} (rows × columns)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4G — Visualize Feature Distributions
# PURPOSE: Plot key morphological features grouped by MoA class.
#          This shows whether features are discriminative —
#          i.e. whether different drug classes produce visually
#          different feature distributions.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_features(df):
    print("\n" + "="*60)
    print("STEP 4G — Visualizing feature distributions")
    print("="*60)

    features_to_plot = [
        'cell_count',
        'mean_area',
        'mean_roundness',
        'nuclear_area_frac',
        'tubulin_mean',
        'actin_mean',
    ]

    classes     = sorted(df['moa'].unique())
    colors      = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    n_features  = len(features_to_plot)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes      = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        for j, cls in enumerate(classes):
            vals = df[df['moa'] == cls][feat].values
            ax.hist(vals, bins=20, alpha=0.6,
                    color=colors[j % len(colors)],
                    label=cls)
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Feature distributions by MoA class", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "feature_preview.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4H — Feature Summary
# PURPOSE: Print mean value of key features per MoA class so we can
#          confirm features are actually different across classes.
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_summary(df):
    print("\n" + "="*60)
    print("STEP 4H — Feature summary per MoA class")
    print("="*60)

    key_features = ['cell_count', 'mean_area',
                    'mean_roundness', 'tubulin_mean', 'actin_mean']

    summary = df.groupby('moa')[key_features].mean().round(3)
    print(f"\n{summary.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Load metadata
    meta = load_metadata()

    # Extract all features
    df = extract_all_features(meta)

    # Save to CSV
    save_features(df)

    # Print summary
    print_feature_summary(df)

    # Visualize distributions
    visualize_features(df)

    print("\n" + "="*60)
    print("  Step 4 complete. Ready for Step 5.")
    print("="*60)