'''
Gradient Based
Uses Savitzky-Golay filter for smoothing
Calculates gradient to find local minima
More sensitive to local variations in the data
'''

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def gradient_minima(df, window_length=11, poly_order=3):
    """
    Find local minima using gradient analysis.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    window_length: Window length for Savitzky-Golay filter
    poly_order: Polynomial order for Savitzky-Golay filter
    
    Returns:
    DataFrame with points where gradient changes from negative to positive
    """
    # Sort by y value
    df_sorted = df.sort_values('y').reset_index(drop=True)
    
    # Smooth the data using Savitzky-Golay filter
    z_smooth = savgol_filter(df_sorted['z'], window_length, poly_order)
    
    # Calculate gradient
    gradient = np.gradient(z_smooth)
    
    # Find where gradient changes from negative to positive
    minima_idx = []
    for i in range(1, len(gradient)):
        if gradient[i-1] < 0 and gradient[i] > 0:
            minima_idx.append(i)
    
    # Create result dataframe
    result_df = df_sorted.iloc[minima_idx][['y', 'z']].copy()
    
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
    plt.plot(df_sorted['y'], z_smooth, 'g-', alpha=0.5, label='Smoothed Data')
    plt.scatter(result_df['y'], result_df['z'], color='red', s=100, label='Local Minima')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Gradient-Based Minima')
    plt.legend()
    plt.show()
    
    return result_df