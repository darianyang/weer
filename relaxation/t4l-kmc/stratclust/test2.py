from sklearn.cluster import KMeans
#from mr_toolkit.clustering import StratifiedClusters
from stratified_clustering import StratifiedClusters
import numpy as np
import matplotlib.pyplot as plt

colors = np.array(['tab:red', 'tab:orange', 'tab:cyan', 'tab:blue', 'tab:pink', 'tab:purple'])

# Number of clusters to place in each stratum
n_clusters = 2

sample_data = np.array([
    [3.0, 23],
    [3.5, 27],
    [4.5, 87],
    [6.0, 14],
    [6.2, 8],
    [5.3, 91],
    [8.4, 33],
    [8.7, 32],
    [8.9, 80],
])

# Generate random sample data
np.random.seed(42)  # For reproducibility
sample_data = np.column_stack((
    np.random.uniform(3.0, 9.0, 300),  # First column in the range [3.0, 9.0]
    np.random.uniform(8, 91, 300)      # Second column in the range [8, 91]
))

# plt.scatter(*sample_data.T)

vertical_bounds = np.array([5, 7])
horizontal_bounds = np.array([25, 50])

#clusterer = StratifiedClusters(n_clusters, bin_bounds=vertical_bounds)
#clusterer = StratifiedClusters(n_clusters, bin_bounds=horizontal_bounds)
clusterer = StratifiedClusters(n_clusters, bin_bounds=[vertical_bounds, horizontal_bounds], is_2d=True)

#clusterer.fit(sample_data, coords_to_stratify=[0])
#clusterer.fit(sample_data, coords_to_stratify=[1])
clusterer.fit(sample_data, coords_to_stratify=[0, 1])

assignments = clusterer.predict(sample_data)

#plt.scatter(*sample_data.T, color=colors[assignments])
plt.scatter(*sample_data.T, cmap='hsv', c=assignments)

for bound in vertical_bounds:
    plt.axvline(bound, color='gray')

for bound in horizontal_bounds:
    plt.axhline(bound, color='gray')

plt.show()