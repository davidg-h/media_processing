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

    img = np.array(Image.open("..\\e02\\rose.jpg"))
    # rgb values after kmeans
    img_kmeans = np.zeros(img.shape)

    kMeans = KMeans(3, 4, 2)

    """ dataPoints = DataList(3)
    # create data points for clustering
    for elem in img:
        dataPoints.add(elem)
    dataPoints = dataPoints.npList() """

    dataPoints_Of_Centroids = kMeans.calc(img)
    centroids = kMeans.get_centroids().astype(int)

    # create a flattened version of dataPoints
    flat_dataPoints = [pixel for sublist in dataPoints_Of_Centroids for pixel in sublist]

    # convert flat_dataPoints to a numpy array
    flat_dataPoints = np.array(flat_dataPoints)

    # use in1d to get indices of img_array in flat_dataPoints
    indices = np.where(np.in1d(flat_dataPoints, img))[0]

    new_Image(indices, centroids, img_kmeans)

    input("press any key to close the programm\n")


def new_Image(indices, centroids, image):
    temp = []
    for i in range(len(indices)):
        temp[i] = np.array[centroids[indices[i]]]
    print(temp)


main()
