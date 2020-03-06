from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_csv

MAX_K = 30

def main():
    df = load_csv('world_indices.csv')
    data = df.to_numpy()
    print(data)
    for x in range(np.shape(data)[1]):
        kmeans(data[x, 1:].reshape(-1, 1), df.columns[x])

def kmeans(data, col_name):
    loss = np.zeros(MAX_K)
    for K in range(MAX_K - 1):
        kmeans = KMeans(n_clusters=(K + 1), random_state=0).fit(data)
        loss[K] = kmeans.inertia_
    plt.plot(loss)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Loss Value')
    plt.title(col_name)
    plt.show()

if __name__ == '__main__':
    main()