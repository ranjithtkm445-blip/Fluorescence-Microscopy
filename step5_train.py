"""
step5_train.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Build and train the multi-modal fusion model for MoA classification.
    Two parallel branches:
      - Image branch  : ResNet18 CNN processes 3-channel fluorescence images
      - Omics branch  : MLP processes 64-dim synthetic omics vector
    Both branches are concatenated and passed through FC layers to predict
    the MoA class. Weighted loss handles class imbalance.

    IMPORTANT: Train/val split is done by COMPOUND not by site.
    One compound per MoA class is held out for validation.
    This ensures all classes appear in both sets and prevents
    the model from memorizing omics vectors.

RUN:
    python step5_train.py

OUTPUT:
    - outputs/training_curves.png     → loss and accuracy per epoch
    - outputs/confusion_matrix.png    → per-class prediction results
    - checkpoints/best_model.pth      → best model weights
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, accuracy_score)
import torchvision.models as models
from step1_dataset import load_metadata, BBBC021Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = r"D:\fluroscence"
OUTPUT_DIR     = r"D:\fluroscence\outputs"
CHECKPOINT_DIR = r"D:\fluroscence\checkpoints"
FEATURES_CSV   = r"D:\fluroscence\outputs\features.csv"
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Training settings ─────────────────────────────────────────────────────────
IMAGE_SIZE    = 256
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
RANDOM_SEED   = 42
NUM_WORKERS   = 0
OMICS_DIM     = 64
NUM_CLASSES   = 4

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5A — Image Branch (ResNet18 CNN)
# PURPOSE: Extract visual features from 3-channel fluorescence images.
#          ResNet18 pretrained on ImageNet used as feature extractor.
#          Final FC layer replaced to output 256-dim embedding.
#          Pretrained low-level filters transfer well to cell morphology.
# ══════════════════════════════════════════════════════════════════════════════

class ImageBranch(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        resnet    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5B — Omics Branch (MLP)
# PURPOSE: Process the 64-dim synthetic omics vector.
#          Simple MLP with BatchNorm for stable training.
#          Outputs 128-dim embedding to be fused with image branch.
# ══════════════════════════════════════════════════════════════════════════════

class OmicsBranch(nn.Module):
    def __init__(self, input_dim=OMICS_DIM, embedding_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5C — Fusion Model
# PURPOSE: Concatenate image + omics embeddings and classify MoA.
#          ImageBranch(256) + OmicsBranch(128) → concat(384)
#          → FC(256) → FC(128) → FC(num_classes)
# ══════════════════════════════════════════════════════════════════════════════

class FusionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.image_branch = ImageBranch(embedding_dim=256)
        self.omics_branch = OmicsBranch(embedding_dim=128)

        # Input = 256 (image) + 128 (omics) = 384
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, omics):
        img_emb   = self.image_branch(image)
        omics_emb = self.omics_branch(omics)
        fused     = torch.cat([img_emb, omics_emb], dim=1)
        return self.classifier(fused)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5D — Multi-Modal Dataset
# PURPOSE: Extends BBBC021Dataset to also return the omics vector
#          for each site alongside the image tensor and label.
#          Loads omics vectors from features.csv.
# ══════════════════════════════════════════════════════════════════════════════

class MultiModalDataset(BBBC021Dataset):
    def __init__(self, metadata, features_df,
                 label_encoder=None, augment=False):
        super().__init__(metadata, label_encoder, augment)

        omics_cols      = [c for c in features_df.columns
                           if c.startswith('omics_')]
        self.omics_data = features_df[omics_cols].values.astype(np.float32)
        self.feat_index = features_df['site_idx'].values

        self.idx_to_feat = {
            int(self.feat_index[i]): i
            for i in range(len(self.feat_index))
        }

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        feat_row = self.idx_to_feat.get(idx, None)
        if feat_row is not None:
            omics = torch.tensor(self.omics_data[feat_row],
                                 dtype=torch.float32)
        else:
            omics = torch.zeros(OMICS_DIM, dtype=torch.float32)

        return image, omics, label


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5E — Weighted Sampler
# PURPOSE: Handle class imbalance by oversampling minority classes.
#          WeightedRandomSampler gives each class equal expected frequency.
# ══════════════════════════════════════════════════════════════════════════════

def make_weighted_sampler(labels):
    class_counts   = np.bincount(labels)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(sample_weights),
        replacement = True
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5F — Compound-Level Train/Val Split
# PURPOSE: Split by compound not by site to prevent data leakage.
#          One compound per MoA class goes to validation.
#          This ensures all 4 classes appear in both train and val,
#          and the model must generalize to unseen compounds.
# ══════════════════════════════════════════════════════════════════════════════

def compound_split(meta):
    print("\n" + "="*60)
    print("STEP 5F — Compound-level train/val split")
    print("="*60)

    moa_to_compounds = meta.groupby('moa')['Image_Metadata_Compound'].unique()

    train_cpds = []
    val_cpds   = []

    np.random.seed(RANDOM_SEED)
    for moa, cpds in moa_to_compounds.items():
        cpds = list(cpds)
        np.random.shuffle(cpds)
        if len(cpds) == 1:
            # Only one compound for this class — keep in train
            train_cpds.extend(cpds)
        else:
            # One compound to val, rest to train
            val_cpds.append(cpds[0])
            train_cpds.extend(cpds[1:])

    train_meta = meta[
        meta['Image_Metadata_Compound'].isin(train_cpds)
    ].reset_index(drop=True)
    val_meta   = meta[
        meta['Image_Metadata_Compound'].isin(val_cpds)
    ].reset_index(drop=True)

    print(f"\n  Train compounds : {sorted(train_cpds)}")
    print(f"  Val compounds   : {sorted(val_cpds)}")
    print(f"\n  Train sites     : {len(train_meta)}")
    print(f"  Val sites       : {len(val_meta)}")

    print(f"\n  Train class distribution:")
    for moa in sorted(train_meta['moa'].unique()):
        count = (train_meta['moa'] == moa).sum()
        print(f"    {moa:<35} {count:>4} sites")

    print(f"\n  Val class distribution:")
    for moa in sorted(val_meta['moa'].unique()):
        count = (val_meta['moa'] == moa).sum()
        print(f"    {moa:<35} {count:>4} sites")

    return train_meta, val_meta


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5G — Training Loop
# PURPOSE: Train fusion model for NUM_EPOCHS.
#          Weighted cross-entropy handles class imbalance.
#          Saves best model based on validation accuracy.
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader,
                class_weights, device, num_epochs=NUM_EPOCHS):
    print("\n" + "="*60)
    print("STEP 5G — Training fusion model")
    print("="*60)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10, gamma=0.5)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : []
    }
    best_val_acc    = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(num_epochs):

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, omics, labels in tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1:02d}/{num_epochs} [train]",
            leave=False
        ):
            images = images.to(device)
            omics  = omics.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, omics)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, omics, labels in val_loader:
                images  = images.to(device)
                omics   = omics.to(device)
                labels  = labels.to(device)
                outputs = model(images, omics)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        # ── Metrics ───────────────────────────────────────────────────────
        train_loss /= train_total
        val_loss   /= val_total
        train_acc   = train_correct / train_total
        val_acc     = val_correct   / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step()

        print(f"  Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    Saved best model (val_acc={val_acc:.3f})")

    print(f"\n  Best validation accuracy : {best_val_acc:.3f}")
    print(f"  Saved → {best_model_path}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5H — Evaluate Model
# PURPOSE: Load best weights, run on val set, print per-class F1 scores.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, val_loader, class_names, device):
    print("\n" + "="*60)
    print("STEP 5H — Evaluating best model")
    print("="*60)

    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path,
                                     map_location=device,
                                     weights_only=True))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, omics, labels in val_loader:
            images  = images.to(device)
            omics   = omics.to(device)
            outputs = model(images, omics)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n  Overall accuracy : {acc:.4f}")
    print(f"  Weighted F1      : {f1:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names))

    return all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5I — Plot Training Curves
# PURPOSE: Visualize loss and accuracy over epochs.
#          Diagnose overfitting or underfitting.
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history):
    print("\n" + "="*60)
    print("STEP 5I — Plotting training curves")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train loss')
    axes[0].plot(history['val_loss'],   label='Val loss')
    axes[0].set_title('Loss per epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train acc')
    axes[1].plot(history['val_acc'],   label='Val acc')
    axes[1].set_title('Accuracy per epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5J — Plot Confusion Matrix
# PURPOSE: Show which classes are confused with each other.
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(all_preds, all_labels, class_names):
    print("\n" + "="*60)
    print("STEP 5J — Plotting confusion matrix")
    print("="*60)

    cm      = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im      = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10)

    ax.set_title('Confusion matrix — validation set', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Load metadata and features
    meta        = load_metadata()
    features_df = pd.read_csv(FEATURES_CSV)

    # Compound-level split
    train_meta, val_meta = compound_split(meta)

    # Build datasets
    print("\n" + "="*60)
    print("STEP 5D — Building multi-modal datasets")
    print("="*60)

    train_ds = MultiModalDataset(train_meta, features_df,
                                  augment=True)
    val_ds   = MultiModalDataset(val_meta,   features_df,
                                  label_encoder=train_ds.le,
                                  augment=False)

    class_names = train_ds.get_class_names()
    print(f"  Classes : {class_names}")

    # Weighted sampler
    sampler      = make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

    # Class weights for loss
    class_counts  = np.bincount(train_ds.labels)
    class_weights = torch.tensor(1.0 / class_counts,
                                  dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    print(f"\n  Class weights for loss:")
    for name, w in zip(class_names, class_weights):
        print(f"    {name:<35} {w:.4f}")

    # Build model
    print("\n" + "="*60)
    print("STEP 5A-C — Building fusion model")
    print("="*60)
    model        = FusionModel(num_classes=len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # Train
    history = train_model(model, train_loader, val_loader,
                          class_weights, device, NUM_EPOCHS)

    # Plot training curves
    plot_training_curves(history)

    # Evaluate
    all_preds, all_labels = evaluate_model(
        model, val_loader, class_names, device
    )

    # Confusion matrix
    plot_confusion_matrix(all_preds, all_labels, class_names)

    print("\n" + "="*60)
    print("  Step 5 complete. Ready for Step 6.")
    print("="*60)