"""
DAY 1 — Data Preparation
========================
- Organizes your dataset into train/val/test splits
- Preprocesses and normalizes images
- Runs EDA: class counts, imbalance check, sample grid

Run: python day1_data_preparation.py
"""

import argparse
import os
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURATION — edit these paths
# ─────────────────────────────────────────
RAW_DATASET_DIR  = "./Masked-fer2013/train"       # your original dataset folder
OUTPUT_DIR       = "./dataset"           # organized output
IMG_SIZE         = (96, 96)
TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
TEST_RATIO       = 0.10
RANDOM_SEED      = 42

DEFAULT_EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def infer_emotions(raw_dir):
    """Infer emotion/class names from top-level subdirectories."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    return sorted([d.name for d in raw_path.iterdir() if d.is_dir()])


# ─────────────────────────────────────────
# STEP 1 — Organize into splits
# ─────────────────────────────────────────
def organize_dataset(raw_dir, out_dir, emotions):
    """
    Expects raw_dir structure:
        Masked-fer2013/train/
            angry/  img1.jpg img2.jpg ...
            happy/  ...
    Produces:
        dataset/train/angry/ ...
        dataset/val/angry/   ...
        dataset/test/angry/  ...
    """
    print("\n[1/4] Organizing dataset into train/val/test splits...")
    stats = {}

    for emotion in emotions:
        src_folder = Path(raw_dir) / emotion
        if not src_folder.exists():
            print(f"  WARNING: folder not found — {src_folder}")
            continue

        images = [f for f in src_folder.iterdir()
                  if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:]
        }

        stats[emotion] = {s: len(imgs) for s, imgs in splits.items()}
        stats[emotion]['total'] = n

        for split, imgs in splits.items():
            dest = Path(out_dir) / split / emotion
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, dest / img_path.name)

        print(f"  {emotion:10s}  total={n:5d}  "
              f"train={splits['train'].__len__():4d}  "
              f"val={splits['val'].__len__():4d}  "
              f"test={splits['test'].__len__():4d}")

    return stats


# ─────────────────────────────────────────
# STEP 2 — Preprocess dataset
# ─────────────────────────────────────────
def preprocess_dataset(dataset_dir, emotions):
    """Resize and perform normalize check."""
    print("\n[2/4] Preprocessing images...")
    total = 0
    errors = 0

    for split in ['train', 'val', 'test']:
        for emotion in emotions:
            folder = Path(dataset_dir) / split / emotion
            if not folder.exists():
                continue
            for img_path in folder.iterdir():
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        errors += 1
                        continue

                    # Resize
                    img = cv2.resize(img, IMG_SIZE)

                    # Save back
                    cv2.imwrite(str(img_path), img)
                    total += 1

                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    errors += 1

    print(f"  Processed {total} images, {errors} errors")


# ─────────────────────────────────────────
# STEP 3 — EDA: class distribution
# ─────────────────────────────────────────
def run_eda(dataset_dir, stats, emotions):
    print("\n[3/4] Running EDA...")

    # Count actual files on disk
    counts = {'train': {}, 'val': {}, 'test': {}}
    for split in counts:
        for emotion in emotions:
            folder = Path(dataset_dir) / split / emotion
            if folder.exists():
                counts[split][emotion] = len(list(folder.glob('*.*')))
            else:
                counts[split][emotion] = 0

    train_counts = [counts['train'].get(e, 0) for e in emotions]
    val_counts   = [counts['val'].get(e, 0)   for e in emotions]
    test_counts  = [counts['test'].get(e, 0)  for e in emotions]

    # Class balance chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dataset EDA — Masked Face Emotion Detection', fontsize=14, fontweight='bold')

    x = np.arange(len(emotions))
    w = 0.28
    colors = ['#4C9BE8', '#F0A500', '#E05C5C']

    axes[0].bar(x - w, train_counts, w, label='Train', color=colors[0], alpha=0.85)
    axes[0].bar(x,     val_counts,   w, label='Val',   color=colors[1], alpha=0.85)
    axes[0].bar(x + w, test_counts,  w, label='Test',  color=colors[2], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(emotions, rotation=30, ha='right')
    axes[0].set_title('Class distribution per split')
    axes[0].set_ylabel('Image count')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Imbalance ratio
    total_per_class = [sum([counts[s].get(e, 0) for s in counts]) for e in emotions]
    max_c = max(total_per_class) if max(total_per_class) > 0 else 1
    imbalance = [c / max_c for c in total_per_class]
    bar_colors = ['#E05C5C' if r < 0.5 else '#F0A500' if r < 0.8 else '#4CAF50'
                  for r in imbalance]

    axes[1].barh(emotions, imbalance, color=bar_colors, alpha=0.85)
    axes[1].set_xlim(0, 1.2)
    axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Imbalance threshold')
    axes[1].axvline(0.8, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_title('Class imbalance ratio (1.0 = most common class)')
    axes[1].set_xlabel('Ratio relative to majority class')
    red_p   = mpatches.Patch(color='#E05C5C', label='Severe (<0.5)')
    orange_p= mpatches.Patch(color='#F0A500', label='Moderate (0.5–0.8)')
    green_p = mpatches.Patch(color='#4CAF50', label='Balanced (>0.8)')
    axes[1].legend(handles=[green_p, orange_p, red_p], loc='lower right')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: eda_class_distribution.png")

    # Print summary
    print("\n  Class summary (train split):")
    for e, c in zip(emotions, train_counts):
        bar = '█' * (c // 50)
        print(f"    {e:10s}  {c:5d}  {bar}")

    total_train = sum(train_counts)
    print(f"\n  Total training images: {total_train}")
    if min(total_per_class) > 0 and max_c / min(total_per_class) > 3:
        print("  WARNING: Severe class imbalance detected.")
        print("  Recommendation: use class_weight='balanced' in model.fit()")
    else:
        print("  Class balance looks acceptable.")


# ─────────────────────────────────────────
# STEP 4 — Show sample images
# ─────────────────────────────────────────
def show_sample_grid(dataset_dir, emotions):
    print("\n[4/4] Generating sample image grid...")
    fig, axes = plt.subplots(len(emotions), 5, figsize=(12, 3 * len(emotions)))
    fig.suptitle('Sample images per emotion (after masking)', fontsize=13, fontweight='bold')

    for row, emotion in enumerate(emotions):
        folder = Path(dataset_dir) / 'train' / emotion
        if not folder.exists():
            continue
        images = list(folder.glob('*.*'))[:5]
        for col in range(5):
            ax = axes[row][col]
            ax.axis('off')
            if col < len(images):
                img = cv2.imread(str(images[col]))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
            if col == 0:
                ax.set_ylabel(emotion, rotation=0, labelpad=50,
                              fontsize=11, fontweight='bold', va='center')

    plt.tight_layout()
    plt.savefig('eda_sample_grid.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("  Saved: eda_sample_grid.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print(" DAY 1 — Data Preparation")
    print("=" * 50)

    parser = argparse.ArgumentParser(description='Prepare and analyze masked emotion dataset')
    parser.add_argument('--raw_dataset_dir', type=str, default=RAW_DATASET_DIR,
                        help='Path to raw dataset dir (default ./raw_dataset)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Path to output dataset dir (default ./dataset)')
    parser.add_argument('--emotions', type=str, default='',
                        help='Comma-separated emotion folder names. If empty, infer from raw dir.')
    args = parser.parse_args()

    raw_dir = args.raw_dataset_dir
    out_dir = args.output_dir

    if not Path(raw_dir).exists():
        print(f"\nERROR: RAW_DATASET_DIR not found: {raw_dir}")
        print("Please use --raw_dataset_dir to provide the correct path.")
        exit(1)

    if args.emotions.strip():
        emotions = [e.strip() for e in args.emotions.split(',') if e.strip()]
    else:
        emotions = infer_emotions(raw_dir)

    if not emotions:
        print(f"\nERROR: No emotion subfolders found in {raw_dir}")
        print("Expected structure like: <raw_dir>/<emotion>/image.png")
        print(f"Example with default emotions: {raw_dir}/")
        for e in DEFAULT_EMOTIONS:
            print(f"    {e}/  img001.jpg  img002.jpg ...")
        exit(1)

    print(f"Using emotions: {emotions}")


    stats = organize_dataset(raw_dir, out_dir, emotions)
    preprocess_dataset(out_dir, emotions)
    run_eda(out_dir, stats, emotions)
    show_sample_grid(out_dir, emotions)

    print("\nDay 1 complete!")
    print(f"Dataset ready at: {OUTPUT_DIR}/")
    print("Next: run day2_train_model.py")
