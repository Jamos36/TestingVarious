'''
distance-based sampling:
Starts from global minimum and works outward
Maintains minimum delta_y spacing between points
Finds local minima in each interval
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def distance_based_minima(df, delta_y=1.0):
    """
    Find local minima using distance-based sampling.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    delta_y: Minimum distance between points in y-direction
    
    Returns:
    DataFrame with minimal z points separated by at least delta_y
    """
    # Sort by y value
    df_sorted = df.sort_values('y').reset_index(drop=True)
    
    # Start with global minimum
    global_min_idx = df_sorted['z'].idxmin()
    current_y = df_sorted.loc[global_min_idx, 'y']
    
    # Initialize results with global minimum
    selected_points = [{'y': df_sorted.loc[global_min_idx, 'y'],
                       'z': df_sorted.loc[global_min_idx, 'z']}]
    
    # Look forward
    current_y = df_sorted.loc[global_min_idx, 'y']
    idx = global_min_idx
    while idx < len(df_sorted):
        # Find next region starting at delta_y distance
        next_region = df_sorted[df_sorted['y'] >= current_y + delta_y]
        if len(next_region) == 0:
            break
            
        # Find minimum in next region
        min_idx = next_region['z'].idxmin()
        selected_points.append({
            'y': df_sorted.loc[min_idx, 'y'],
            'z': df_sorted.loc[min_idx, 'z']
        })
        current_y = df_sorted.loc[min_idx, 'y']
        idx = min_idx
    
    # Look backward
    current_y = df_sorted.loc[global_min_idx, 'y']
    idx = global_min_idx
    while idx > 0:
        # Find previous region ending at delta_y distance
        prev_region = df_sorted[df_sorted['y'] <= current_y - delta_y]
        if len(prev_region) == 0:
            break
            
        # Find minimum in previous region
        min_idx = prev_region['z'].idxmin()
        selected_points.append({
            'y': df_sorted.loc[min_idx, 'y'],
            'z': df_sorted.loc[min_idx, 'z']
        })
        current_y = df_sorted.loc[min_idx, 'y']
        idx = min_idx
    
    result_df = pd.DataFrame(selected_points).sort_values('y')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sorted['y'], df_sorted['z'], alpha=0.5, label='Original Data')
    plt.scatter(result_df['y'], result_df['z'], color='red', s=100, label='Distance-Based Minima')
    
    # Plot delta_y intervals
    for y in result_df['y']:
        plt.axvline(y, color='gray', linestyle='--', alpha=0.3)
        
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Distance-Based Minima')
    plt.legend()
    plt.show()
    
    return result_df