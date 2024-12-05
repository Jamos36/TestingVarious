'''
Dynamic Programing:
Uses smoothed data to reduce noise
Optimally segments y-range
Finds minimum z value in each segment
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def dynamic_programming_minima(df, n_segments=10, smoothing_window=5):
    """
    Find minima using dynamic programming approach.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    n_segments: Number of segments to divide the y-range into
    smoothing_window: Window size for moving average smoothing
    
    Returns:
    DataFrame with minimal z points from optimal segments
    """
    # Sort by y value and reset index
    df_sorted = df.sort_values('y').reset_index(drop=True)
    
    # Smooth z values using moving average
    df_sorted['z_smooth'] = df_sorted['z'].rolling(window=smoothing_window, 
                                                  center=True).mean()
    df_sorted['z_smooth'].fillna(df_sorted['z'], inplace=True)
    
    # Calculate cost matrix
    n_points = len(df_sorted)
    segment_size = n_points // n_segments
    cost_matrix = np.zeros((n_segments, n_points))
    split_points = np.zeros((n_segments, n_points), dtype=int)
    
    # Initialize first row of cost matrix
    for j in range(segment_size, n_points):
        segment_data = df_sorted.iloc[:j+1]
        cost_matrix[0, j] = segment_data['z_smooth'].min()
    
    # Fill rest of cost matrix using dynamic programming
    for i in range(1, n_segments):
        for j in range(i * segment_size, n_points):
            min_cost = float('inf')
            min_split = 0
            
            # Try different split points
            for k in range((i-1) * segment_size, j - segment_size + 1):
                segment_data = df_sorted.iloc[k+1:j+1]
                if len(segment_data) == 0:
                    continue
                    
                cost = cost_matrix[i-1, k] + segment_data['z_smooth'].min()
                
                if cost < min_cost:
                    min_cost = cost
                    min_split = k
            
            cost_matrix[i, j] = min_cost
            split_points[i, j] = min_split
    
    # Backtrack to find optimal split points
    selected_points = []
    current_pos = n_points - 1
    
    for i in range(n_segments-1, -1, -1):
        if i > 0:
            split = split_points[i, current_pos]
            segment_data = df_sorted.iloc[split+1:current_pos+1]
        else:
            segment_data = df_sorted.iloc[:current_pos+1]
            
        if len(segment_data) > 0:
            min_idx = segment_data['z'].idxmin()
            selected_points.append({
                'y': df_sorted.loc[min_idx, 'y'],
                'z': df_sorted.loc[min_idx, 'z']
            })
            
        current_pos = split
    
    result_df = pd.DataFrame(selected_points)
    
    # Ensure global minimum is included
    global_min = df_sorted.loc[df_sorted['z'].idxmin()]
    if not any((result_df['y'] == global_min['y']) & (result_df['z'] == global_min['z'])):
        result_df = pd.concat([result_df, 
                             pd.DataFrame([{'y': global_min['y'], 'z': global_min['z']}])],
                             ignore_index=True)
    
    # Sort by y value
    result_df = result_df.sort_values('y')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sorted['y'], df_sorted['z'], alpha=0.5, label='Original Data')
    plt.plot(df_sorted['y'], df_sorted['z_smooth'], 'g-', alpha=0.5, label='Smoothed Data')
    plt.scatter(result_df['y'], result_df['z'], color='red', s=100, label='DP Minima')
    
    # Plot segment boundaries
    y_bounds = np.array_split(df_sorted['y'], n_segments)
    for bound in y_bounds:
        plt.axvline(bound.iloc[-1], color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Dynamic Programming Minima')
    plt.legend()
    plt.show()
    
    return result_df
	