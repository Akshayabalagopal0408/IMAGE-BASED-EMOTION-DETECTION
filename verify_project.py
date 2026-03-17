import sys
import os

def check_imports():
    print("Checking imports...")
    try:
        import tensorflow as tf
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        print("Imports successful!")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def check_files():
    print("Checking files...")
    files = ['data_loader.py', 'models.py', 'train.py', 'predict.py']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Missing files: {missing}")
        return False
    print("All script files present.")
    return True

if __name__ == "__main__":
    if check_imports() and check_files():
        print("\nSUCCESS: Project environment looks good.")
        print("You can now run 'python train.py' to train the models.")
    else:
        print("\nFAILURE: Please setup your environment or check missing files.")
