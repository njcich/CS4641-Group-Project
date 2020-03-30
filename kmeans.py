#### CS 4641 Group 2 Project K-Means Implementation
#### Contributor: Nick Cich (ncich3)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from utils import load_csv

## Maximum K value to test when visualizing the loss values of K-Means algorithm on data
MAX_K = 10

## Ideal K values for each dataset, in order when pulled from world_indices.csv; found by plotting loss values
## for different values of K in visualize_kmeans() and then using elbow method to determine ideal K value
IDEAL_K_1D = np.array([3, 3, 3, 3, 4, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 4, 4, 2])

## Boolean variables to determine what code should run; if `VISUALIZE_ELBOW` is True, the loss values of each dataset will be
## plotted.  If `RUN_IDEAL_KMEANS` is True, the data will be assigned clusters using the K-Means algorithm with the number of
## clusters taken from `IDEAL_K_1D`. If VISUALIZE_IDEAL_K is True, these clusters will be plotted.
VISUALIZE_ELBOW = False
RUN_IDEAL_KMEANS = True
VISUALIZE_IDEAL_K = False

def main():
    # Loads data from world_indices.csv
    df = load_csv('world_indices.csv')
    data = df.to_numpy()

    # Scales all data to be between 0 and 1
    data[:, 2:] = (data[:, 2:] - np.amin(data[:, 2:], axis=0)[None, :]) / (np.amax(data[:, 2:], axis=0) - np.amin(data[:, 2:], axis=0))[None, :]
    
    if VISUALIZE_ELBOW:
        for x in range(2, np.shape(data)[1]):
            visualize_kmeans(data[:, x].reshape(-1, 1), df.columns[x])
    elif RUN_IDEAL_KMEANS:
        loss = np.zeros(np.shape(data)[1] - 2)
        for x in range(2, np.shape(data)[1]):
            loss[x - 2] = perform_ideal_kmeans_1D(data[:, x], df.columns[x], IDEAL_K_1D[x - 2])
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

def perform_ideal_kmeans_1D(data, name, ideal_k):
    kmeans = KMeans(n_clusters=ideal_k, random_state=0).fit(data[:, None])
    clusters = kmeans.labels_
    if VISUALIZE_IDEAL_K:
        plt.scatter(data.ravel(), np.zeros_like(data).ravel(), c=clusters, cmap='rainbow')
        plt.title('Visualization of K = ' + str(ideal_k), fontsize=15)
        plt.xlabel(name)
        plt.show()
    return kmeans.inertia_

if __name__ == '__main__':
    main()