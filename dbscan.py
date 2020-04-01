#### CS 4641 Group 2 Project DBSCAN Implementation
#### Contributor: Nick Cich (ncich3)

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from utils import load_csv

## Minimum number of points in a neighborhood for a point to be considered as a core point, including the point itself
MIN_PTS = 10

## The maximum distance between two samples for one to be considered as in the neighborhood of the other; found by plotting
## distance of `MIN_PTS` + 1 nearest neighbors of each dataset in visualize_DBSCAN() and then using elbow method to determine
## ideal epsilon value                                                                                                         15                                                     23                                                                                  35
EPS_1D = np.array([0.034, 0.035, 0.046, 0.033, 0.060, 0.054, 0.034, 0.035, 0.025, 0.055, 0.027, 0.035, 0.035, 0.034, 0.063, 0.039, 0.042, 0.059, 0.047, 0.030, 0.040, 0.041, 0.040, 0.047, 0.040, 0.031, 0.042, 0.037, 0.032, 0.033, 0.028, 0.047, 0.037, 0.029, 0.022, 0.032, 0.038, 0.050, 0.057])

## Parameters are boolean variables to determine what code should run; if `visualize_elbow` is True, the loss values of each
## dataset will be plotted. Otherwise, the data will be assigned clusters using the DBSCAN algorithm with the value of epsilon
## clusters taken from `EPS_1D`. Then, these clusters will be plotted.
def run_1D_DBSCAN(visualize_elbow=False):
    # Loads data from world_indices.csv
    df = load_csv('world_indices.csv')
    data = df.to_numpy()

    # Scales all data to be between 0 and 1
    data[:, 2:] = (data[:, 2:] - np.amin(data[:, 2:], axis=0)[None, :]) / (np.amax(data[:, 2:], axis=0) - np.amin(data[:, 2:], axis=0))[None, :]
    
    if visualize_elbow:
        for x in range(2, np.shape(data)[1]):
            visualize_DBSCAN(data[:, x].reshape(-1, 1), df.columns[x])
    else:
        for x in range(2, np.shape(data)[1]):
            print(x - 2)
            perform_DBSCAN_1D(data[:, x], df.columns[x], EPS_1D[x - 2])
        perform_DBSCAN_1D(data[:, x], df.columns[x], EPS_1D[x - 2])

def visualize_DBSCAN(data, name):
    neighbors = NearestNeighbors(n_neighbors=(MIN_PTS + 1))
    neighbors.fit(data)
    k_neighbor_distance = neighbors.kneighbors(data)[0][:, MIN_PTS]
    plt.plot(np.sort(k_neighbor_distance))
    plt.xlabel('Points Sorted According to Distance of K-th Nearest Neighbor')
    plt.ylabel('K-th Nearest Neighbor Distance')
    plt.title(name)
    plt.show()

def perform_DBSCAN_1D(data, name, epsilon):
    dbscan = DBSCAN(eps=epsilon, min_samples=MIN_PTS).fit(data[:, None])
    clusters = dbscan.labels_
    plt.scatter(data.ravel(), np.zeros_like(data).ravel(), c=clusters, cmap='rainbow')
    plt.title('Visualization of ' + name, fontsize=15)
    plt.xlabel(name)
    plt.show()