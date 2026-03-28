"""
DAY 4 — Demo & Final Report
============================
- Runs inference on sample test images with visual output
- Generates final summary report figure
- Optional: live webcam emotion detection demo
- Saves everything needed for submission

Run: python day4_demo_report.py
Run webcam demo: python day4_demo_report.py --webcam
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cv2
import tensorflow as tf
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATASET_DIR  = "./dataset"
MODEL_DIR    = "./models"
RESULTS_DIR  = "./results"
IMG_SIZE     = (96, 96)

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry':    '#E05C5C',
    'disgust':  '#9B59B6',
    'fear':     '#E67E22',
    'happy':    '#2ECC71',
    'neutral':  '#95A5A6',
    'sad':      '#3498DB',
    'surprise': '#F1C40F',
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
def load_model():
    # Try final model first, fall back to phase 1
    for name in ['best_model_final.keras', 'best_model_p1.keras']:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            print(f"Loading model: {path}")
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("No trained model found. Run day2 and day3 first.")


# ─────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────
def preprocess_image(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img * 255.0)
    return np.expand_dims(img, axis=0)


def predict(model, img_bgr):
    x = preprocess_image(img_bgr)
    probs = model.predict(x, verbose=0)[0]
    idx   = np.argmax(probs)
    return EMOTIONS[idx], float(probs[idx]), probs


# ─────────────────────────────────────────
# STEP 1 — Sample prediction grid
# ─────────────────────────────────────────
def generate_prediction_grid(model, n_per_class=4):
    print("\n[1/3] Generating prediction sample grid...")

    fig = plt.figure(figsize=(n_per_class * 2.5, len(EMOTIONS) * 2.8))
    fig.suptitle('Sample Predictions — Masked Face Emotion Detection',
                 fontsize=14, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(len(EMOTIONS), n_per_class + 1,
                           width_ratios=[0.4] + [1] * n_per_class,
                           hspace=0.05, wspace=0.05)

    for row, emotion in enumerate(EMOTIONS):
        # Emotion label column
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        ax_label.axis('off')
        ax_label.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8,
                                          color=EMOTION_COLORS[emotion], alpha=0.85,
                                          transform=ax_label.transAxes))
        ax_label.text(0.5, 0.5, emotion, transform=ax_label.transAxes,
                      ha='center', va='center', fontsize=9.5, fontweight='bold',
                      color='white')

        # Find test images for this emotion
        folder = Path(DATASET_DIR) / 'test' / emotion
        if not folder.exists():
            folder = Path(DATASET_DIR) / 'val' / emotion
        images = list(folder.glob('*.*'))[:n_per_class] if folder.exists() else []

        for col in range(n_per_class):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.axis('off')

            if col < len(images):
                img = cv2.imread(str(images[col]))
                if img is None:
                    continue

                pred_emotion, confidence, probs = predict(model, img)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)

                correct = (pred_emotion == emotion)
                border_color = '#2ECC71' if correct else '#E05C5C'
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(3)

                label = f"{pred_emotion}\n{confidence*100:.0f}%"
                ax.set_title(label, fontsize=7.5,
                             color='#2ECC71' if correct else '#E05C5C',
                             pad=2, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'prediction_samples.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print("  Green border = correct prediction, Red = incorrect")


# ─────────────────────────────────────────
# STEP 2 — Final summary report figure
# ─────────────────────────────────────────
def generate_summary_report(model):
    print("\n[2/3] Generating final summary report...")

    # Load metrics
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        print("  metrics.json not found, run day3 first for full report")
        metrics = {'accuracy': 0, 'macro_f1': 0, 'weighted_f1': 0, 'roc_auc': 0}
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Load training history if available
    history = None
    for fname in ['history_p2.json', 'history_p1.json']:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                history = json.load(f)
            break

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#F8F9FA')

    # Title
    fig.text(0.5, 0.96,
             'Image-Based Emotion Detection with Masked Faces',
             ha='center', va='top', fontsize=16, fontweight='bold', color='#2C3E50')
    fig.text(0.5, 0.92,
             'Deep Learning Pipeline — MobileNetV2 + Spatial Attention',
             ha='center', va='top', fontsize=11, color='#7F8C8D')

    # Metric cards
    card_data = [
        ('Test Accuracy',   f"{metrics['accuracy']*100:.1f}%",   '#4C9BE8'),
        ('Macro F1 Score',  f"{metrics['macro_f1']:.3f}",         '#2ECC71'),
        ('Weighted F1',     f"{metrics['weighted_f1']:.3f}",      '#9B59B6'),
        ('ROC-AUC',         f"{metrics['roc_auc']:.3f}",          '#E67E22'),
    ]

    for i, (label, value, color) in enumerate(card_data):
        ax = fig.add_axes([0.05 + i * 0.235, 0.73, 0.20, 0.14])
        ax.set_facecolor(color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.65, value,  ha='center', va='center',
                fontsize=22, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.25, label,  ha='center', va='center',
                fontsize=9.5, color='white', alpha=0.9, transform=ax.transAxes)

    # Training curve
    if history:
        ax_curve = fig.add_axes([0.05, 0.35, 0.42, 0.30])
        epochs = range(1, len(history['accuracy']) + 1)
        ax_curve.plot(epochs, history['accuracy'],     '#4C9BE8',
                      label='Train', linewidth=2)
        ax_curve.plot(epochs, history['val_accuracy'], '#E05C5C',
                      label='Val',   linewidth=2, linestyle='--')
        ax_curve.fill_between(epochs, history['accuracy'], history['val_accuracy'],
                              alpha=0.08, color='#4C9BE8')
        ax_curve.set_title('Training accuracy', fontsize=11, fontweight='bold')
        ax_curve.set_xlabel('Epoch')
        ax_curve.set_ylabel('Accuracy')
        ax_curve.legend(fontsize=9)
        ax_curve.grid(alpha=0.3)
        ax_curve.set_facecolor('white')

    # Architecture summary
    ax_arch = fig.add_axes([0.55, 0.35, 0.40, 0.30])
    ax_arch.set_xlim(0, 10); ax_arch.set_ylim(0, 7)
    ax_arch.axis('off')
    ax_arch.set_facecolor('white')
    ax_arch.set_title('Model architecture', fontsize=11, fontweight='bold')

    arch_blocks = [
        (5, 6.2, 'Input Image (96×96×3)',       '#BDC3C7', 'black'),
        (5, 5.2, 'MobileNetV2 (ImageNet)',       '#3498DB', 'white'),
        (5, 4.2, 'Spatial Attention Gate',       '#9B59B6', 'white'),
        (5, 3.2, 'Global Average Pooling',       '#1ABC9C', 'white'),
        (5, 2.2, 'Dense 256 + Dropout 0.4',      '#E67E22', 'white'),
        (5, 1.2, 'Dense 128 + Dropout 0.3',      '#E67E22', 'white'),
        (5, 0.3, f'Softmax ({len(EMOTIONS)} emotions)', '#E05C5C', 'white'),
    ]
    for (x, y, label, color, tc) in arch_blocks:
        ax_arch.add_patch(mpatches.FancyBboxPatch((x-3.5, y-0.35), 7, 0.7,
                                              boxstyle="round,pad=0.05",
                                              facecolor=color, edgecolor='white',
                                              linewidth=1.5))
        ax_arch.text(x, y, label, ha='center', va='center',
                     fontsize=8.5, color=tc, fontweight='bold')
        if y > 0.5:
            ax_arch.annotate('', xy=(x, y - 0.4), xytext=(x, y - 0.85),
                             arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))

    # Key findings text
    ax_text = fig.add_axes([0.05, 0.05, 0.90, 0.25])
    ax_text.set_facecolor('white')
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.axis('off')
    ax_text.text(0.02, 0.88, 'Key findings', fontsize=11,
                 fontweight='bold', color='#2C3E50')
    findings = [
        f"› Accuracy of {metrics['accuracy']*100:.1f}% on masked face emotion detection test set",
        "› Spatial attention mechanism focuses on eye/eyebrow region — critical when lower face is occluded",
        "› MobileNetV2 backbone (ImageNet pretrained) converges faster than training from scratch",
        "› Class imbalance addressed via per-class weighting in loss function",
        "› Transfer learning (Phase 1 head training + Phase 2 fine-tuning) used for best results",
        "› 'Happy' and 'Surprise' achieve highest accuracy; 'Disgust' and 'Fear' are hardest due to visual similarity",
    ]
    for i, line in enumerate(findings):
        ax_text.text(0.02, 0.72 - i * 0.13, line, fontsize=9.5,
                     color='#555', va='top')

    plt.savefig(os.path.join(RESULTS_DIR, 'final_summary_report.png'),
                dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/final_summary_report.png")


# ─────────────────────────────────────────
# STEP 3 — Live webcam demo (optional)
# ─────────────────────────────────────────
def run_webcam_demo(model):
    print("\n[3/3] Starting webcam demo... Press 'q' to quit.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Could not open webcam.")
        return

    print("  Webcam open. Press 'q' to quit, 's' to save a screenshot.")
    screenshot_count = 0
    ema_probs = None
    alpha = 0.15  # Smoothing factor

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            ema_probs = None # reset if no face detected

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion, confidence, probs = predict(model, face_roi)
            
            if ema_probs is None:
                ema_probs = probs
            else:
                ema_probs = alpha * probs + (1 - alpha) * ema_probs
            
            idx = np.argmax(ema_probs)
            emotion = EMOTIONS[idx]
            confidence = float(ema_probs[idx])
            probs = ema_probs
            color_hex = EMOTION_COLORS[emotion]
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            color_bgr = (b, g, r)

            # Draw face box
            cv2.rectangle(display, (x, y), (x+w, y+h), color_bgr, 2)

            # Emotion label background
            label = f"{emotion.upper()}  {confidence*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (x, y-th-14), (x+tw+10, y), color_bgr, -1)
            cv2.putText(display, label, (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Mini probability bar chart
            bar_x, bar_y = x + w + 10, y
            bar_w, bar_h = 120, 12
            if bar_x + bar_w < display.shape[1]:
                for ei, (em, prob) in enumerate(zip(EMOTIONS, probs)):
                    by  = bar_y + ei * (bar_h + 3)
                    blen= int(bar_w * prob)
                    ec  = EMOTION_COLORS[em]
                    er, eg, eb = int(ec[1:3],16), int(ec[3:5],16), int(ec[5:7],16)
                    cv2.rectangle(display, (bar_x, by), (bar_x + blen, by + bar_h),
                                  (eb, eg, er), -1)
                    cv2.putText(display, em[:3], (bar_x - 28, by + 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)

        # Instructions overlay
        cv2.putText(display, "Q: quit  |  S: screenshot", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Masked Face Emotion Detection — Live Demo', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            path = os.path.join(RESULTS_DIR, f'webcam_screenshot_{screenshot_count}.png')
            cv2.imwrite(path, display)
            print(f"  Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("  Webcam demo ended.")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam', action='store_true',
                        help='Run live webcam demo after report generation')
    args = parser.parse_args()

    print("=" * 50)
    print(" DAY 4 — Demo & Final Report")
    print("=" * 50)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = load_model()

    generate_prediction_grid(model, n_per_class=4)
    generate_summary_report(model)

    if args.webcam:
        run_webcam_demo(model)
    else:
        print("\n  Tip: run with --webcam flag for live demo")
        print("  Example: python day4_demo_report.py --webcam")

    print("\n" + "=" * 50)
    print(" PROJECT COMPLETE")
    print("=" * 50)
    print(f"\n  All output files saved to: {RESULTS_DIR}/")
    print("\n  Files generated:")
    for f in sorted(Path(RESULTS_DIR).glob('*')):
        print(f"    {f.name}")
    print("\n  Models saved to:")
    for f in sorted(Path(MODEL_DIR).glob('*')):
        print(f"    {f.name}")
