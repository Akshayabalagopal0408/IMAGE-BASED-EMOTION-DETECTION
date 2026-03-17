import requests
import os

MIRRORS = [
    "https://raw.githubusercontent.com/badriadhikari/FER-2013/master/fer2013.csv",
    "https://svn.opensciencegrid.org/public/fer2013/fer2013.csv",
    "https://docs.google.com/uc?export=download&id=0B9K_H0J0L8t4WjV4S1Z4Ym45ZVU", # Google Drive often fails with confirm
    "https://raw.githubusercontent.com/Ariel5/fer2013/master/fer2013.csv"
]

def try_download():
    for url in MIRRORS:
        print(f"Trying {url}...")
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200 and 'text/html' not in r.headers.get('content-type', ''):
                print("Download started!")
                with open('fer2013.csv', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024): 
                        if chunk: 
                            f.write(chunk)
                print(f"Successfully downloaded from {url}")
                # Check size
                if os.path.getsize('fer2013.csv') < 10000: # Too small, probably error page
                     print("File too small, likely not the dataset.")
                     continue
                return True
            else:
                print(f"Failed: Status {r.status_code} or Content-Type {r.headers.get('content-type')}")
        except Exception as e:
            print(f"Error: {e}")
            
    return False

if __name__ == "__main__":
    if try_download():
        print("Dataset ready.")
    else:
        print("All mirrors failed.")
