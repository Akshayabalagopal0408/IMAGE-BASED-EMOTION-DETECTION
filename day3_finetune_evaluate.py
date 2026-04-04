"""
DAY 3 — Fine-tuning & Evaluation
=================================
- Loads Phase 1 model from Day 2
- Unfreezes top layers of MobileNetV2 (fine-tuning)
- Trains Phase 2 with low learning rate
- Full evaluation: confusion matrix, F1, per-class accuracy
- Compares eye-region vs full-face approach

Run: python day3_finetune_evaluate.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import cv2

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATASET_DIR   = "./dataset"
MODEL_DIR     = "./models"
RESULTS_DIR   = "./results"
IMG_SIZE      = (96, 96)
BATCH_SIZE    = 32
EPOCHS_P2     = 50       # Phase 2: fine-tuning
UNFREEZE_LAST = 80       # how many layers of MobileNetV2 to unfreeze
RANDOM_SEED   = 42

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)


# ─────────────────────────────────────────
# STEP 1 — Rebuild data pipeline
# ─────────────────────────────────────────
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    return x, y

def preprocess_augment(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_crop(x, size=[*IMG_SIZE, 3]) \
        if False else x   # disabled — resize already done
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    return x, y

def load_datasets():
    print("\n[1/5] Loading datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        labels='inferred', label_mode='categorical',
        class_names=EMOTIONS, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED
    ).map(preprocess_augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        labels='inferred', label_mode='categorical',
        class_names=EMOTIONS, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=False
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'test'),
        labels='inferred', label_mode='categorical',
        class_names=EMOTIONS, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=False
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def compute_weights():
    labels = []
    for i, e in enumerate(EMOTIONS):
        folder = Path(DATASET_DIR) / 'train' / e
        if folder.exists():
            labels.extend([i] * len(list(folder.glob('*.*'))))
    labels = np.array(labels)
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {i: w for i, w in enumerate(weights)}


# ─────────────────────────────────────────
# STEP 2 — Fine-tune: unfreeze top layers
# ─────────────────────────────────────────
def finetune_model(train_ds, val_ds, class_weights):
    print(f"\n[2/5] Loading Phase 1 model and fine-tuning top {UNFREEZE_LAST} layers...")

    model_path = os.path.join(MODEL_DIR, 'best_model_p1.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun day2_train_model.py first.")

    model = tf.keras.models.load_model(model_path)

    # Find the MobileNetV2 base layer
    base_model = None
    for layer in model.layers:
        if 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        # Freeze all except last UNFREEZE_LAST layers
        for layer in base_model.layers[:-UNFREEZE_LAST]:
            layer.trainable = False

        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        print(f"  Unfroze {trainable_count} layers in MobileNetV2")
    else:
        print("  WARNING: MobileNetV2 base not found, training all layers")
        model.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model_final.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=4, min_lr=1e-8, verbose=1),
        CSVLogger(os.path.join(MODEL_DIR, 'training_log_p2.csv'))
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P2,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    with open(os.path.join(MODEL_DIR, 'history_p2.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f)

    best_val = max(history.history['val_accuracy'])
    print(f"\n  Phase 2 best val accuracy: {best_val:.4f} ({best_val*100:.1f}%)")
    return model, history


# ─────────────────────────────────────────
# STEP 3 — Full evaluation on test set
# ─────────────────────────────────────────
def evaluate_model(model, test_ds):
    print("\n[3/5] Evaluating on test set...")

    y_true, y_pred_prob = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_prob.extend(preds)
        y_true.extend(y_batch.numpy())

    y_true      = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_true_idx  = np.argmax(y_true, axis=1)
    y_pred_idx  = np.argmax(y_pred_prob, axis=1)

    acc     = np.mean(y_true_idx == y_pred_idx)
    f1_mac  = f1_score(y_true_idx, y_pred_idx, average='macro')
    f1_w    = f1_score(y_true_idx, y_pred_idx, average='weighted')

    try:
        auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    print(f"\n  Test Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1       : {f1_mac:.4f}")
    print(f"  Weighted F1    : {f1_w:.4f}")
    print(f"  ROC-AUC (macro): {auc:.4f}")

    print("\n  Per-class report:")
    print(classification_report(y_true_idx, y_pred_idx, target_names=EMOTIONS))

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(f1_mac),
        'weighted_f1': float(f1_w),
        'roc_auc': float(auc)
    }
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return y_true_idx, y_pred_idx, y_pred_prob, metrics


# ─────────────────────────────────────────
# STEP 4 — Confusion matrix
# ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    print("\n[4/5] Plotting confusion matrix...")

    cm      = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix — Masked Face Emotion Detection', fontsize=13, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].tick_params(axis='x', rotation=30)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalized (row = actual class)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")

    # Per-class accuracy bar
    per_class_acc = cm_norm.diagonal()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#4CAF50' if a >= 0.75 else '#F0A500' if a >= 0.55 else '#E05C5C'
              for a in per_class_acc]
    bars = ax.bar(EMOTIONS, per_class_acc, color=colors, alpha=0.87, edgecolor='white')
    ax.set_ylim(0, 1.15)
    ax.axhline(0.75, color='green',  linestyle='--', alpha=0.5, label='75% target')
    ax.axhline(0.55, color='orange', linestyle='--', alpha=0.5, label='55% threshold')
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Per-class accuracy on test set')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, 'per_class_accuracy.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path2}")


# ─────────────────────────────────────────
# STEP 5 — Combined training curve (P1+P2)
# ─────────────────────────────────────────
def plot_combined_curves():
    print("\n[5/5] Plotting combined training curves...")
    try:
        with open(os.path.join(MODEL_DIR, 'history_p1.json')) as f:
            h1 = json.load(f)
        with open(os.path.join(MODEL_DIR, 'history_p2.json')) as f:
            h2 = json.load(f)
    except FileNotFoundError:
        print("  History files not found, skipping.")
        return

    acc  = h1['accuracy']     + h2['accuracy']
    vacc = h1['val_accuracy'] + h2['val_accuracy']
    loss = h1['loss']         + h2['loss']
    vloss= h1['val_loss']     + h2['val_loss']
    ep   = range(1, len(acc) + 1)
    p1_end = len(h1['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Full Training History — Phase 1 + Phase 2', fontsize=13, fontweight='bold')

    for ax, train, val, ylabel, title in [
        (axes[0], acc,  vacc,  'Accuracy', 'Accuracy'),
        (axes[1], loss, vloss, 'Loss',     'Loss'),
    ]:
        ax.plot(ep, train, '#4C9BE8', label='Train', linewidth=2)
        ax.plot(ep, val,   '#E05C5C', label='Val',   linewidth=2, linestyle='--')
        ax.axvline(p1_end + 0.5, color='gray', linestyle=':', alpha=0.7)
        ax.text(p1_end * 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.02,
                'Phase 1', ha='center', fontsize=9, color='gray')
        ax.text(p1_end + (len(ep) - p1_end) * 0.5,
                ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.02,
                'Phase 2', ha='center', fontsize=9, color='gray')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'full_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print(" DAY 3 — Fine-tuning & Evaluation")
    print("=" * 50)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_ds, val_ds, test_ds = load_datasets()
    class_weights = compute_weights()
    model, history = finetune_model(train_ds, val_ds, class_weights)

    y_true, y_pred, y_prob, metrics = evaluate_model(model, test_ds)
    plot_confusion_matrix(y_true, y_pred)
    plot_combined_curves()

    print("\n" + "=" * 50)
    print(f" FINAL RESULTS")
    print("=" * 50)
    print(f"  Accuracy    : {metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"\n  All results saved to: {RESULTS_DIR}/")
    print("\nDay 3 complete!")
    print("Next: run day4_demo_report.py")
