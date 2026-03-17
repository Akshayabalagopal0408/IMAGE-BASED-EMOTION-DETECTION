import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import get_data, EMOTIONS
import os

def evaluate_model():
    # Load Model
    try:
        model = tf.keras.models.load_model('saved_models/emotion_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Test Data
    print("Loading test data...")
    (_, _, _), (X_test_masked, X_test_orig, y_test) = get_data(num_samples=2000)

    # Predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(X_test_orig)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Emotion Detection (ResNet)')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS)
    print("\nClassification Report:")
    print(report)
    
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    print("Saved evaluation_report.txt")

if __name__ == "__main__":
    evaluate_model()
