import numpy as np
import os

from kmeans import KMeans
from dataList import DataList
from PIL import Image

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def main():
    """ dim = int(input("How many dimensions: "))
    n_centroids = int(input("How many centroids: "))
    iterations = int(input("How many iterations: "))
    n_dataP = int(input("How many points: ")) """

    img = Image.open("e02\\rose.jpg")
    image_array = np.array(img)
    # Extract the RGB values from the image array
    rgb_values = image_array.reshape((-1, 3))
    
    # Define the number of clusters and the maximum number of iterations
    num_clusters = 256
    max_iterations = 100

    kMeans = KMeans(num_clusters, max_iterations)

    new_img_flat = kMeans.calc(rgb_values)
        
    # Reshape the flat array back to the original shape of the image
    new_img_array = new_img_flat.reshape(image_array.shape)
    
    # Create a new image from the array
    new_img = Image.fromarray(np.uint8(new_img_array))
    
    # Get the path to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the filename of the new image
    new_img_filename = f'rose-256.jpg'

    # Join the script directory path with the filename to get the full path of the new image
    new_img_path = os.path.join(script_dir, new_img_filename)

    # Save the new image
    new_img.save(new_img_path)

main()
