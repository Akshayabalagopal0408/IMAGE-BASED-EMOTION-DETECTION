import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

# Standard FER2013 order from data_loader.py
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_on_new_dataset():
    # Load Model
    try:
        classifier = tf.keras.models.load_model('saved_models/emotion_model.h5')
        print("Emotion model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    dataset_root = 'dataset/extracted/masked_fer2013/Masked-fer2013/validation'
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root {dataset_root} not found.")
        return

    # Categories in the NEW dataset
    categories = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    
    for category in categories:
        folder_path = os.path.join(dataset_root, category)
        if not os.path.exists(folder_path):
            continue
            
        print(f"\nProcessing category: {category}")
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
                img_input = np.expand_dims(img_input, axis=(0, -1))
                
                # Predict
                probs = classifier.predict(img_input)
                pred_idx = np.argmax(probs)
                pred_label = EMOTIONS[pred_idx]
                
                # Save result plot
                plt.figure(figsize=(5, 5))
                plt.imshow(img_resized, cmap='gray')
                plt.title(f"New DS | True: {category.capitalize()} | Pred: {pred_label}")
                plt.axis('off')
                
                save_path = f"new_ds_prediction_{category}_{img_name.split('.')[0]}.png"
                plt.savefig(save_path)
                print(f"Saved {save_path} -> Predicted: {pred_label}")
                plt.close()
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

if __name__ == "__main__":
    predict_on_new_dataset()
