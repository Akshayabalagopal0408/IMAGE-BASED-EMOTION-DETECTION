import numpy as np
import cv2
import os
import requests
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Configuration ---
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
# Emotion labels (Standard FER2013 order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 

# Folder mapping for the masked dataset
FOLDER_MAPPING = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sad': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

def load_from_directory(base_path, max_samples_per_class=500):
    """
    Loads images from a directory structure like:
    base_path/emotion_folder/*.jpg
    """
    X = []
    y = []
    print(f"Loading data from {base_path}...")
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        emotion_label = FOLDER_MAPPING.get(folder_name.lower())
        if emotion_label not in EMOTIONS:
            print(f"Skipping unknown folder: {folder_name}")
            continue
            
        emotion_idx = EMOTIONS.index(emotion_label)
        count = 0
        for img_name in os.listdir(folder_path):
            if count >= max_samples_per_class:
                break
            
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(img.astype('float32') / 255.0)
                    
                    label = np.zeros(len(EMOTIONS))
                    label[emotion_idx] = 1
                    y.append(label)
                    count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return np.array(X), np.array(y)

def get_data(num_samples=5000, csv_path='fer2013.csv'):
    # 1. Try FER2013 CSV
    if os.path.exists(csv_path):
        print(f"Found {csv_path}. Using REAL dataset (FER2013 CSV).")
        X_original, y_emotion = load_fer2013(csv_path, max_samples=num_samples)
        if X_original is not None:
             X_original = X_original[..., np.newaxis]
             return prepare_splits(X_original, y_emotion)

    # 2. Try Masked Dataset Directory
    masked_root = 'dataset/extracted/masked_dataset'
    if os.path.exists(masked_root):
        print("Found masked_dataset directory. Loading train/validation folders...")
        X_train, y_train = load_from_directory(os.path.join(masked_root, 'train'))
        X_test, y_test = load_from_directory(os.path.join(masked_root, 'validation'))
        
        if len(X_train) > 0:
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
            
            # Since we only have MASKED images in this dataset, 
            # we will set masked=orig for the autoencoder training 
            # (which will learn identity if it's already masked).
            # This is a fallback to allow the code to run even if original images are missing.
            # In a real scenario, you'd want the original unmasked images for the target.
            print("WARNING: Using Masked images as both input and target for autoencoder (Fallback).")
            return (X_train, X_train, y_train), (X_test, X_test, y_test)

    # 3. Fallback to Sample Faces
    if os.path.exists('sample_faces') and len(os.listdir('sample_faces')) > 0:
        print("Using sample faces...")
        X_original = load_samples_from_folder('sample_faces', num_samples)
        if X_original is not None:
             X_original = X_original[..., np.newaxis]
             y_emotion = np.zeros((len(X_original), len(EMOTIONS)))
             for i in range(len(X_original)):
                y_emotion[i, np.random.randint(0, len(EMOTIONS))] = 1
             return prepare_splits(X_original, y_emotion)

    # 4. Fallback to Synthetic
    print("All real data sources failed. Generating synthetic faces...")
    X_original, y_emotion = generate_synthetic_faces(num_samples)
    X_original = X_original[..., np.newaxis]
    return prepare_splits(X_original, y_emotion)

def prepare_splits(X_original, y_emotion):
    print("Applying synthetic masks...")
    X_masked = np.array([apply_random_mask(img.squeeze()) for img in X_original])
    X_masked = X_masked[..., np.newaxis]
    
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y_emotion, test_size=0.2, random_state=42)
    X_train_masked, X_test_masked, _, _ = train_test_split(X_masked, y_emotion, test_size=0.2, random_state=42)
    
    return (X_train_masked, X_train_orig, y_train), (X_test_masked, X_test_orig, y_test)

if __name__ == "__main__":
    # Test the loader
    (xm, xo, y), _ = get_data(10)
    print("Data loaded shape:", xm.shape, xo.shape, y.shape)
    
    # Save a sample to verify
    debug_img = np.hstack((xm[0].squeeze(), xo[0].squeeze()))
    debug_imp_pil = Image.fromarray((debug_img * 255).astype(np.uint8))
    debug_imp_pil.save("debug_data_loader.png")
    print("Saved debug_data_loader.png")
