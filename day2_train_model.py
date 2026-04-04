"""
DAY 2 — Model Training
======================
- Loads prepared dataset from Day 1
- Builds MobileNetV2 transfer learning model
- Trains with frozen base (Phase 1)
- Saves best model + training history

Run: python day2_train_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import json

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATASET_DIR  = "./dataset"
MODEL_DIR    = "./models"
IMG_SIZE     = (96, 96)
BATCH_SIZE   = 32
EPOCHS_P1    = 20        # Phase 1: train head only
RANDOM_SEED  = 42

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)


# ─────────────────────────────────────────
# STEP 1 — Data pipeline
# ─────────────────────────────────────────
def build_data_pipeline():
    print("\n[1/4] Building data pipeline...")

    # Augmentation for training
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.12),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

    def preprocess_train(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = augmentation(x, training=True)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        return x, y

    def preprocess_eval(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        return x, y

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        labels='inferred',
        label_mode='categorical',
        class_names=EMOTIONS,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        labels='inferred',
        label_mode='categorical',
        class_names=EMOTIONS,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_ds = train_ds.map(preprocess_train,  num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(preprocess_eval,     num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    n_train = sum(1 for _ in Path(DATASET_DIR + '/train').rglob('*.*')
                  if _.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    n_val   = sum(1 for _ in Path(DATASET_DIR + '/val').rglob('*.*')
                  if _.suffix.lower() in ('.jpg', '.jpeg', '.png'))

    print(f"  Training images : {n_train}")
    print(f"  Validation images: {n_val}")

    return train_ds, val_ds, n_train


# ─────────────────────────────────────────
# STEP 2 — Compute class weights
# ─────────────────────────────────────────
def compute_weights():
    labels = []
    for emotion_idx, emotion in enumerate(EMOTIONS):
        folder = Path(DATASET_DIR) / 'train' / emotion
        if folder.exists():
            n = len(list(folder.glob('*.*')))
            labels.extend([emotion_idx] * n)

    labels = np.array(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    weight_dict = {i: w for i, w in enumerate(weights)}
    print("\n[2/4] Class weights:")
    for i, (emotion, w) in enumerate(zip(EMOTIONS, weights)):
        print(f"  {emotion:10s}  weight={w:.3f}")
    return weight_dict


# ─────────────────────────────────────────
# STEP 3 — Build model
# ─────────────────────────────────────────
def build_model():
    print("\n[3/4] Building MobileNetV2 transfer learning model...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False   # freeze for Phase 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name='input_image')

    # Eye-region attention: spatial attention to upper face
    # (focuses model on eyes/eyebrows — the visible emotion signal when masked)
    x = base_model(inputs, training=False)

    # Spatial attention gate
    attn = layers.Conv2D(1, (1, 1), activation='sigmoid', name='spatial_attn')(x)
    x    = layers.Multiply(name='attended_features')([x, attn])

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.4, name='drop1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='MaskedEmotionNet')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.summary()
    print(f"\n  Trainable params: {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    return model


# ─────────────────────────────────────────
# STEP 4 — Train Phase 1
# ─────────────────────────────────────────
def train_phase1(model, train_ds, val_ds, class_weights):
    print(f"\n[4/4] Training Phase 1 — head only ({EPOCHS_P1} epochs max)...")

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model_p1.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(os.path.join(MODEL_DIR, 'training_log_p1.csv'))
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P1,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save history
    with open(os.path.join(MODEL_DIR, 'history_p1.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f)

    return history


# ─────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────
def plot_history(history, tag='p1'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Training History — Phase {tag[-1]}', fontsize=13, fontweight='bold')

    h = history.history
    epochs = range(1, len(h['accuracy']) + 1)

    axes[0].plot(epochs, h['accuracy'],     '#4C9BE8', label='Train acc', linewidth=2)
    axes[0].plot(epochs, h['val_accuracy'], '#E05C5C', label='Val acc',   linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    best_epoch = np.argmax(h['val_accuracy']) + 1
    best_val   = max(h['val_accuracy'])
    axes[0].axvline(best_epoch, color='green', linestyle=':', alpha=0.7,
                    label=f'Best: {best_val:.3f} @ ep{best_epoch}')
    axes[0].legend()

    axes[1].plot(epochs, h['loss'],     '#4C9BE8', label='Train loss', linewidth=2)
    axes[1].plot(epochs, h['val_loss'], '#E05C5C', label='Val loss',   linewidth=2, linestyle='--')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = f'training_curves_{tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")

    best_acc = max(h['val_accuracy'])
    print(f"\n  Best validation accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    return best_acc


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print(" DAY 2 — Model Training (Phase 1)")
    print("=" * 50)

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n  GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  No GPU detected — training on CPU (will be slower)")

    train_ds, val_ds, n_train = build_data_pipeline()
    class_weights = compute_weights()
    model = build_model()
    history = train_phase1(model, train_ds, val_ds, class_weights)
    best_acc = plot_history(history, tag='p1')

    print("\nDay 2 complete!")
    print(f"Model saved to: {MODEL_DIR}/best_model_p1.keras")
    print("Next: run day3_finetune_evaluate.py")
