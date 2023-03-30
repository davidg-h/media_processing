import numpy as np

from kmeans import KMeans
from dataList import DataList

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def main():
    dim = int(input("How many dimensions: "))
    n_centroids = int(input("How many centroids: "))
    iterations = int(input("How many iterations: "))
    n_dataP = int(input("How many points: "))
    
    kMeans = KMeans(dim, n_centroids, iterations);
    dataPoints = DataList(dim)
    
    # create data points for clustering
    for _ in range(n_dataP):
        dataPoints.add(DataList.createDP(dim))
    dataPoints = dataPoints.npList()
    
    print(f"List\n {dataPoints}")
    
    kMeans.calc(dataPoints)
    
    input("press any key to close the programm\n")
            
main()