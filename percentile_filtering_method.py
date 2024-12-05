'''PErcentile filtering:
Filters data below specified z-value percentile
Samples evenly across y-range
Helps focus on the lowest z values
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def percentile_filtering_minima(df, percentile_threshold=10, n_samples=10):
    """
    Find minima using percentile filtering.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    percentile_threshold: Keep points below this z-value percentile
    n_samples: Number of evenly spaced samples to take across y-range
    
    Returns:
    DataFrame with minimal z points from filtered data
    """
    # Calculate z-value percentile threshold
    z_threshold = np.percentile(df['z'], percentile_threshold)
    
    # Filter points below threshold
    filtered_df = df[df['z'] <= z_threshold].copy()
    
    # Calculate y-range boundaries for sampling
    y_min, y_max = filtered_df['y'].min(), filtered_df['y'].max()
    y_ranges = np.linspace(y_min, y_max, n_samples + 1)
    
    # Sample points across y-range
    selected_points = []
    for i in range(len(y_ranges) - 1):
        mask = (filtered_df['y'] >= y_ranges[i]) & (filtered_df['y'] < y_ranges[i + 1])
        if mask.any():
            region_df = filtered_df[mask]
            min_idx = region_df['z'].idxmin()
            selected_points.append({
                'y': filtered_df.loc[min_idx, 'y'],
                'z': filtered_df.loc[min_idx, 'z']
            })
    
    result_df = pd.DataFrame(selected_points)
    
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
    plt.scatter(filtered_df['y'], filtered_df['z'], 
               alpha=0.5, color='green', label=f'Below {percentile_threshold}th percentile')
    plt.scatter(result_df['y'], result_df['z'], 
               color='red', s=100, label='Selected Minima')
    plt.axhline(z_threshold, color='gray', linestyle='--', 
               label=f'{percentile_threshold}th percentile')
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Percentile Filtering Minima')
    plt.legend()
    plt.show()
    
    return result_df