from datasets import load_dataset
import pandas as pd
import numpy as np
import cv2

def download_fer2013_hf():
    print("Attempting to download FER2013 via Hugging Face Datasets...")
    try:
        # 'fer2013' is available on HF
        dataset = load_dataset("fer2013", split="train")
        
        print("Download successful! Converting to CSV format for consistency...")
        
        # Convert to format matching original CSV: emotion, pixels
        # HF dataset has 'emotion' (int) and 'pixels' (list of integers or image)
        # Let's inspect one item
        example = dataset[0]
        print("Example keys:", example.keys())
        
        # If 'pixels' is not a string, we need to convert
        # Check if 'pixel_values' or 'image' exists
        
        # We will iterate and save to CSV
        data = []
        for item in dataset:
            emotion = item['emotion']
            # HF fer2013 typically has 'pixels' as a list of 2304 integers
            pixels = item['pixels'] 
            pixel_str = " ".join(map(str, pixels))
            data.append([emotion, pixel_str, "Training"])
            
        df = pd.DataFrame(data, columns=['emotion', 'pixels', 'Usage'])
        df.to_csv('fer2013.csv', index=False)
        print("Saved fer2013.csv!")
        return True
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
        return False

if __name__ == "__main__":
    download_fer2013_hf()
