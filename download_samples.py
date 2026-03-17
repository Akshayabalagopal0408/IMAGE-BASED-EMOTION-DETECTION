import requests
import os
import cv2
import numpy as np
try:
    from skimage import data
except ImportError:
    data = None

SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golshifteh_Farahani_at_2013_Cannes_Film_Festival.jpg/320px-Golshifteh_Farahani_at_2013_Cannes_Film_Festival.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg/320px-Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/George_Clooney_66%C3%A8me_Festival_de_Venise_%28Mostra%29.jpg/320px-George_Clooney_66%C3%A8me_Festival_de_Venise_%28Mostra%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Donald_Trump_official_portrait.jpg/320px-Donald_Trump_official_portrait.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Dwayne_Johnson_2014_%28cropped%29.jpg/320px-Dwayne_Johnson_2014_%28cropped%29.jpg"
]

def download_samples():
    if not os.path.exists('sample_faces'):
        os.makedirs('sample_faces')
        
    print("Downloading sample faces...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    count = 0
    for i, url in enumerate(SAMPLE_URLS):
        try:
            r = requests.get(url, stream=True, headers=headers, timeout=10)
            if r.status_code == 200:
                with open(f"sample_faces/face_{i}.jpg", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"Downloaded face_{i}.jpg")
                count += 1
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    # Fallback to skimage if we have very few images
    if count < 5 and data is not None:
        print("Using skimage built-in samples as fallback...")
        try:
            # Astronaut is a clear face
            img = data.astronaut()
            cv2.imwrite("sample_faces/astronaut.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("Saved astronaut.jpg")
            
            # Chelsea is a cat, but has 'facial' features (eyes/nose) good for testing texture
            img2 = data.chelsea()
            cv2.imwrite("sample_faces/chelsea.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            print("Saved chelsea.jpg")
        except Exception as e:
            print(f"Could not save skimage samples: {e}")

if __name__ == "__main__":
    download_samples()
