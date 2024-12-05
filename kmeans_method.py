'''
KMEANS CLUSTERING:
:K-means method:
Uses scikit-learn's KMeans implementation
Clusters based on y and z values only
Finds the minimum z value within each cluster
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_minima(df, n_clusters=10):
    """
    Find local minima using k-means clustering.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    n_clusters: Number of clusters to create
    
    Returns:
    DataFrame with minimal z points from each cluster
    """
    # Extract y and z columns for clustering
    yz_data = df[['y', 'z']].values
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(yz_data)
    
    # Find minimum z value point in each cluster
    minima = []
    for cluster in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster]
        min_point = cluster_data.loc[cluster_data['z'].idxmin()]
        minima.append({'y': min_point['y'], 'z': min_point['z']})
    
    result_df = pd.DataFrame(minima)
    
    # Ensure global minimum is included
    global_min = df.loc[df['z'].idxmin()]
    if not any((result_df['y'] == global_min['y']) & (result_df['z'] == global_min['z'])):
        result_df = pd.concat([result_df, 
                             pd.DataFrame([{'y': global_min['y'], 'z': global_min['z']}])],
                             ignore_index=True)
    
    # Sort by y value
    result_df = result_df.sort_values('y')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(df['y'], df['z'], alpha=0.5, label='Original Data')
    plt.scatter(result_df['y'], result_df['z'], color='red', s=100, label='Cluster Minima')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('K-Means Clustering Minima')
    plt.legend()
    plt.show()
    
    return result_df''''''