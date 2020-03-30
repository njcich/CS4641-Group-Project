from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from utils import load_csv

MAX_K = 10
IDEAL_K = np.array([])

def main():
    df = load_csv('world_indices.csv')
    data = df.to_numpy()
    for x in range(2, np.shape(data)[1]):
        visualize_kmeans(data[:, x].reshape(-1, 1), df.columns[x])

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

if __name__ == '__main__':
    main()