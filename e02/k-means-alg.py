import numpy as np

from kmeans import KMeans
from dataList import DataList
from PIL import Image

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def main():
    """ dim = int(input("How many dimensions: "))
    n_centroids = int(input("How many centroids: "))
    iterations = int(input("How many iterations: "))
    n_dataP = int(input("How many points: ")) """
    
    colorCluster = ["4", "16", "256"]
    
    img = np.asarray((Image.open("C:\\Users\\david\\Desktop\\Github_Repos\\MedVer\\e02\\rose.jpg").convert("RGB")))
    
    kMeans = KMeans(3, 4, 2)
    
    """ dataPoints = DataList(3)
    # create data points for clustering
    for elem in img:
        dataPoints.add(elem)
    dataPoints = dataPoints.npList() """
    
    print(f"List\n {img}")
    
    dataPoints_Of_Centroids = kMeans.calc(img)
    centroids = kMeans.get_centroids().astype(int)
    print(f"Centroids\n {centroids}")
    
    
    input("press any key to close the programm\n")
            
main()