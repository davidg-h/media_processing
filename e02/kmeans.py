import random
from matplotlib import pyplot as plt
import numpy as np

class KMeans:
    def __init__(self, dim, n_centroids = 2, max_iter = 10):
        self.n_centroids = n_centroids
        self.max_iter = max_iter
        self.dim = dim
        self.centroids = self.createCentroids()
    
    def get_centroids(self):
        return np.array(self.centroids)
    
    def createCentroids(self):
        centroids = [[] for _ in range(self.n_centroids)]
    
        for i in range(self.n_centroids): 
            for _ in range(self.dim):
                centroids[i].append(random.uniform(0, 255))
        return np.array(centroids)
        
    
    def calc(self, dataPoints):
        iteration = 0
        prev_centroids = None
        
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # sort data points to centroid with min euclidean distance
            sorted_dataPoints = [[] for _ in range(len(self.centroids))]
            
            for points in dataPoints:
                for point in points:
                    dists = KMeans.euclideanDistance(point, self.centroids)
                    centroid_id = np.argmin(dists)
                    sorted_dataPoints[centroid_id].append(point)
                
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_dataPoints]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
                
                
            #KMeans.plotLists(self.sorted_dataPoints , self.centroids, dim =3)
            iteration += 1
        return sorted_dataPoints
    
    @staticmethod
    def euclideanDistance(point, data): #TODO sorting to right center indices
        """
        Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data)**2, axis=1))
    
    @staticmethod
    def plotLists(arr, centers, dim = 2):
       # create a flat list of all points and their respective group index
        flat_arr = []
        for i, sublist in enumerate(arr):
            if isinstance(sublist, list):
                for point in sublist:
                    if dim == 2:
                        flat_arr.append([point[0], point[1], i])
                    else:
                        flat_arr.append([point[0], point[1], point[2], i])
            else:
                if dim == 2:
                    flat_arr.append([sublist[0], sublist[1], -1])
                else:
                    flat_arr.append([sublist[0], sublist[1], sublist[2], -1])
        flat_arr = np.array(flat_arr)

        # split the flat array into subarrays based on the number of points in each sublist
        splits = np.cumsum([len(sublist) if isinstance(sublist, list) else 1 for sublist in arr])
        groups = np.split(flat_arr, splits[:-1])

        # create a dictionary of unique group indices and their respective color
        group_color = {}
        num_groups = len(groups)
        for i in range(num_groups):
            group_color[i] = np.random.rand(3,)
        group_color[num_groups] = np.random.rand(3,) # add color for cluster centers

        # plot the points with different colors for groups and single points
        fig = plt.figure(figsize=(8, 8))
        if dim == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        for group in groups:
            if len(group) > 0:
                group_index = group[0, -1]
                color = group_color[group_index]
                if group_index == -1:
                    ax.scatter(group[0, 0], group[0, 1], c=color, alpha=0.8)
                else:
                    if dim == 2:
                        ax.scatter(group[:, 0], group[:, 1], c=color, alpha=0.8)
                    else:
                        ax.scatter(group[:, 0], group[:, 1], group[:, 2], c=color, alpha=0.8)

        # plot the cluster centers with a different color and marker
        for i, center in enumerate(centers):
            if dim == 2:
                ax.scatter(center[0], center[1], c=group_color[i], marker='x', s=200)
            else:
                ax.scatter(center[0], center[1], center[2], c=group_color[i], marker='x', s=200)

        # update the plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.show(block=False)
        input("for next iteration press enter")
        plt.close()          
            
