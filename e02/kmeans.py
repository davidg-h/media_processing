import random
from matplotlib import pyplot as plt
import numpy as np

class KMeans:
    def __init__(self, number_clusters = 6, max_iter = 100):
        self.K = number_clusters
        self.max_iter = max_iter
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []
    
    def get_centroids(self):
        return np.array(self.centroids, dtype=np.int32)
    
    def createCentroids(self):
        # take random indices of the number of rows/rgb-vectors for self.K times
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        # save the vectors as centroids
        return [self.X[idx] for idx in random_sample_idxs]
    
    def calc_newCentroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples, dtype = int)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels
    
    def create_Clusters(self, centroids):
        # Assign samples(=rgb value of pixel) to closest centroids (create clusters)
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_Centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def closest_Centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [KMeans.euclideanDistance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index
    
    def calc(self, dataPoints):
        
        self.X = dataPoints
        # n_samples = rows; n_features = columns
        self.n_samples, self.n_features = dataPoints.shape
        print(f"Shape of data after extracting rgb values: {dataPoints.shape}")
        print(f"{self.n_samples} pixels need to be processed")
        # init centers for clusters
        self.centroids = self.createCentroids()
        
        iteration = 0
        centroid_old = None
        
        while np.not_equal(self.centroids, centroid_old).any() and iteration < self.max_iter:
            print("Iteration percentage: {:.0%}".format(iteration/self.max_iter))
            self.clusters = self.create_Clusters(self.centroids)
            
            # Calculate new centroids from the clusters
            centroid_old = self.centroids
            self.centroids = self.calc_newCentroids(self.clusters)
            
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = centroid_old[i]
            
            iteration += 1
        labels = self.get_cluster_labels(self.clusters)
        print("Calculation done, image ready")
        return self.data_Map(labels, self.n_features)
    
    def data_Map(self,mappedLabels, dim):
        # shaping back to original rgb_values shape to regenerate the image
        centroids = self.get_centroids()
        map = np.zeros((len(mappedLabels), dim))
        for i in range(len(mappedLabels)):
            map[i] = centroids[mappedLabels[i]]
        return map
    
    def plotLists(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)
        plt.show()   
            
    @staticmethod
    def euclideanDistance(p1, p2):
        """
        Euclidean distance between two points
        """
        return np.sqrt(np.sum((p1 - p2)**2))