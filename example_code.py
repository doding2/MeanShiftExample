import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA


def main():
    # Load dataset (https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering)
    dataset = pd.read_csv('wine_quality.csv')
    print(dataset.head())

    # Fit PCA for feature reduction
    # and plotting multidimensional data in 2-dimension with centroids
    pca = PCA(n_components=2)
    dataset_pca = pca.fit_transform(dataset)

    # Hyperparameter Tuning: find best bandwidth value
    bandwidth = estimate_bandwidth(dataset)
    print('bandwidth: ', bandwidth)

    # Clustering with Mean Shift
    mean_shift = MeanShift(bandwidth=bandwidth)
    cluster_labels = mean_shift.fit_predict(dataset_pca)
    unique_cluster_labels = np.unique(cluster_labels)
    centroids = mean_shift.cluster_centers_
    print('cluster labels: ', unique_cluster_labels)
    print('centroids: ', centroids)

    # Make cluster dataframe with dataset and cluster labels columns for plotting easily
    cluster_df = pd.DataFrame(data=dataset_pca, columns=['principal component1', 'principal component2'])
    cluster_df['cluster_labels'] = cluster_labels
    print(cluster_df.head())

    # Visualize clustering results with scatter plot
    markers = ['o', 's', '^', 'x', '*']
    for label in unique_cluster_labels:
        # Visualize data in cluster
        cluster_data = cluster_df[cluster_df['cluster_labels'] == label]
        plt.scatter(
            x=cluster_data['principal component1'],
            y=cluster_data['principal component2'],
            edgecolors='k',
            marker=markers[label])

        # Visualize centroid of cluster with label text
        center = centroids[label]
        plt.scatter(x=center[0], y=center[1], s=300, color='white', edgecolor='k', alpha=0.7, marker=markers[label])
        plt.scatter(x=center[0], y=center[1], s=70, color='k', edgecolor='k', marker='$%d$' % label)

    plt.show()


if __name__ == '__main__':
    main()
