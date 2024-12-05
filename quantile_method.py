'''
Quantile-based
Divides the y-axis into equal-sized quantiles
Finds minimum z value within each quantile
Visualizes quantile boundaries

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quantile_minima(df, n_quantiles=10):
    """
    Find local minima using quantile-based sampling.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    n_quantiles: Number of quantiles to create
    
    Returns:
    DataFrame with minimal z points from each quantile
    """
    # Calculate quantile boundaries
    quantiles = np.linspace(0, 1, n_quantiles+1)
    y_bounds = df['y'].quantile(quantiles)
    
    # Find minimum z value in each quantile
    minima = []
    for i in range(len(y_bounds)-1):
        mask = (df['y'] >= y_bounds[i]) & (df['y'] < y_bounds[i+1])
        if mask.any():
            quantile_data = df[mask]
            min_point = quantile_data.loc[quantile_data['z'].idxmin()]
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
    plt.scatter(result_df['y'], result_df['z'], color='red', s=100, label='Quantile Minima')
    for bound in y_bounds:
        plt.axvline(bound, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Quantile-Based Minima')
    plt.legend()
    plt.show()
    
    return result_df