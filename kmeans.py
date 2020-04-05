#### CS 4641 Group 2 Project K-Means Implementation
#### Contributors: Nick Cich (ncich3) and Zack Vogel (dvogel3)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from utils import load_csv

## Maximum K value to test when visualizing the loss values of K-Means algorithm on data
MAX_K = 10

## Ideal K values for each dataset, in order when pulled from world_indices.csv; found by plotting loss values
## for different values of K in visualize_kmeans() and then using elbow method to determine ideal K value
IDEAL_K_1D = np.array(
    [3, 3, 3, 3, 4, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 4, 4,
     2])


## Parameters are boolean variables to determine what code should run; if `visualize_elbow` is True, the loss values of each
## dataset will be plotted. Otherwise, the data will be assigned clusters using the K-Means algorithm with the number of
## clusters taken from `IDEAL_K_1D`. If `visualize_ideal_k` is True, these clusters will be plotted.
def run_1D_kmeans(visualize_elbow=False, visualize_ideal_k=False):
    # Loads data from world_indices.csv
    df = load_csv('world_indices.csv')
    data = df.to_numpy()

    # Scales all data to be between 0 and 1
    data[:, 2:] = (data[:, 2:] - np.amin(data[:, 2:], axis=0)[None, :]) / (np.amax(data[:, 2:], axis=0) - np.amin(
        data[:, 2:], axis=0))[None, :]

    if visualize_elbow:
        for x in range(2, np.shape(data)[1]):
            visualize_kmeans(data[:, x].reshape(-1, 1), df.columns[x])
    else:
        loss = np.zeros(np.shape(data)[1] - 2)
        for x in range(2, np.shape(data)[1]):
            loss[x - 2] = perform_ideal_kmeans_1D(data[:, x], df.columns[x], IDEAL_K_1D[x - 2], visualize_ideal_k)
        print('Average Loss For Scaled Data: ' + str(np.mean(loss)))


def visualize_kmeans(data, name):
    loss = np.zeros(MAX_K)
    loss[0] = float("inf")
    for K in range(1, MAX_K):
        kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
        loss[K] = kmeans.inertia_
    plt.plot(loss)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Loss Value')
    plt.title(name)
    plt.show()


def perform_ideal_kmeans_1D(data, x_name, y_name, ideal_k, visualize_ideal_k, col1, col2):
    kmeans = KMeans(n_clusters=ideal_k, random_state=0).fit(data[[col1, col2]])
    print(data[col1])
    print(data[col2])
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print(centroids)
    if visualize_ideal_k:
        plt.scatter(data[col1], data[col2], c=clusters, cmap='rainbow')
        plt.title('Visualization of K = ' + str(ideal_k) + " for Indices " + x_name + " and " + y_name, fontsize=15)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.show()
    return kmeans.inertia_