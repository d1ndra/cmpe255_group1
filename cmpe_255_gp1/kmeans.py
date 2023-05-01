from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Ob_fns(Enum):
    EUCLIDEAN = 1
    MANHATTAN = 2

def euclidean(data, centroid):
    return np.sqrt(((data - centroid)**2).sum(axis=1))

def manhattan(data, centroid):
    return abs(data - centroid).sum(axis=1)

def plot_centroid_data(data, centroids, labels, iteration=-1, title=None):
    columns = data.columns
    plt.scatter(data[columns[0]], data[columns[1]], c=labels)
    plt.scatter(centroids[columns[0]], centroids[columns[1]], marker='x', s=200, linewidths=3, color='r')
    if iteration != -1:
        plt.title(f"Iteration: {iteration}")
    if title:
        plt.title(title)
    plt.show()

class KMeans():
  def __init__(self, n_clusters=8, max_iter=300, objective=Ob_fns.EUCLIDEAN, verbose=False):
    objective_fns = {
        Ob_fns.EUCLIDEAN: euclidean,
        Ob_fns.MANHATTAN: manhattan
    }
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.obfn = objective_fns[objective]
    self.verbose = verbose

  def fit(self, X):
    centroids = X.sample(self.n_clusters)
    iterations = 0
    while iterations < self.max_iter:
        prev_centroids = centroids.copy()
        # Assign each point to the nearest centroid
        #  distances = np.sqrt(((data - centroids.iloc[0])**2).sum(axis=1))
        distances = self.obfn(X, centroids.iloc[0])
        labels = np.zeros(len(X))
        for i in range(1, self.n_clusters):
            # temp_distances = np.sqrt(((data - centroids.iloc[i])**2).sum(axis=1))
            temp_distances = self.obfn(X, centroids.iloc[i])
            # update centroids where the distance to current centroid is lower than before
            labels[temp_distances < distances] = i
            # update distance to centroid
            distances[temp_distances < distances] = temp_distances[temp_distances < distances]
        if self.verbose and len(X.columns) == 2:
            plot_centroid_data(X, centroids, labels, iterations)
        # Update the centroids
        for i in range(self.n_clusters):
            centroids.iloc[i] = X[labels == i].mean()
        iterations += 1
        # Check if centroid has changed, break if the same centroids
        if np.array_equal(prev_centroids.values,centroids.values):
            break
    if self.verbose and len(X.columns) == 2:
        plot_centroid_data(X, centroids, labels, iterations)
    print(f"K means algorithm converged in {iterations} iterations for {self.n_clusters} clusters")
    return labels, centroids

def calculate_wcss(data, labels, centroids):
    wcss = 0
    for i in range(len(centroids)):
        cluster_data = data[labels == i]
        centroid = centroids.iloc[i]
        cluster_distances = euclidean(cluster_data, centroid)
        wcss += (cluster_distances**2).sum()
    return wcss
