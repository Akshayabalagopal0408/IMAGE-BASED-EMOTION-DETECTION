import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_data, EMOTIONS
import random
import os
import cv2

def predict_pipeline():
    # Load Models
    try:
        classifier = tf.keras.models.load_model('saved_models/emotion_model.h5')
        print("Emotion model loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train.py first!")
        return

    validation_root = 'dataset/extracted/masked_dataset/validation'
    if not os.path.exists(validation_root):
        print(f"Error: Validation root {validation_root} not found.")
        return

    for folder_name in os.listdir(validation_root):
        folder_path = os.path.join(validation_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        emotion_label = folder_name.capitalize()
        print(f"\nProcessing emotion: {emotion_label}")
        
        # Pick 2 samples
        all_images = os.listdir(folder_path)
        if len(all_images) < 2:
            samples = all_images
        else:
            samples = random.sample(all_images, 2)
            
        for img_name in samples:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (64, 64))
                img_input = img_resized.astype('float32') / 255.0
                img_input = np.expand_dims(img_input, axis=(0, -1)) # Batch of 1
                
                # Detect Emotion
                emotion_probs = classifier.predict(img_input)
                pred_emotion_idx = np.argmax(emotion_probs)
                pred_label = EMOTIONS[pred_emotion_idx]
                
                # Display/Save
                plt.figure(figsize=(5, 5))
                plt.imshow(img_resized, cmap='gray')
                plt.title(f"True: {emotion_label} | Pred: {pred_label}")
                plt.axis('off')
                
                safe_name = f"{emotion_label}_{img_name.split('.')[0]}"
                plt.savefig(f"prediction_{safe_name}.png")
                print(f"Saved prediction_{safe_name}.png -> Predicted: {pred_label}")
                plt.close()
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

if __name__ == "__main__":
    predict_pipeline()
