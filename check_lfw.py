from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

def check_lfw():
    print("Fetching LFW...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    print(f"Loaded {lfw_people.images.shape} images")
    
    # Save one to see
    plt.imshow(lfw_people.images[0], cmap='gray')
    plt.savefig("lfw_sample.png")
    print("Saved lfw_sample.png")

if __name__ == "__main__":
    check_lfw()
