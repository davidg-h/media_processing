import numpy as np
import matplotlib.pyplot as plt
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        
        for i in range(self.max_iter):
            # Assign each point to the nearest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(x)
                
            # Update centroids to be the average of the points in each cluster
            for j in range(self.n_clusters):
                self.centroids[j] = np.mean(clusters[j], axis=0)
                
            # Plot the data and the clusters
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot the data points
            ax.scatter(X[:, 0], X[:, 1], s=50)

            # Plot the cluster centroids
            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=200, linewidths=3, color='r')

            # Plot the cluster assignments
            for i in range(self.n_clusters):
                ax.scatter(np.array(clusters[i])[:, 0], np.array(clusters[i])[:, 1], s=50)

            # Set the axis labels and title
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('K-Means Clustering')

            # Show the plot
            plt.show()
                
    def predict(self, X):
        # Assign each point to the nearest centroid
        clusters = [[] for _ in range(self.n_clusters)]
        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.centroids]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(x)
            
        # Return the assigned cluster labels
        labels = np.zeros(len(X), dtype=int)
        for j in range(self.n_clusters):
            for x in clusters[j]:
                labels[np.where(X == x)[0][0]] = j
        return labels



# Generate some sample data
np.random.seed(0)
X = np.random.randn(100, 2)

# Initialize the KMeans algorithm with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the KMeans algorithm to the data
kmeans.fit(X)

