''''
Monotonic Trend Minimum Finder:
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def find_monotonic_minimum(df, tolerance=0.05, smoothing_window=11, poly_order=3):
    """
    Find minimum z value with monotonic trends on both sides within tolerance.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    tolerance: Allowed deviation from monotonicity (as fraction of z range)
    smoothing_window: Window length for Savitzky-Golay filter
    poly_order: Polynomial order for smoothing
    
    Returns:
    DataFrame with selected points showing monotonic trend
    """
    # Sort by y value
    df_sorted = df.sort_values('y').reset_index(drop=True)
    
    # Apply Savitzky-Golay filter to smooth the data
    z_smooth = savgol_filter(df_sorted['z'], smoothing_window, poly_order)
    
    # Find global minimum
    min_idx = np.argmin(z_smooth)
    min_y = df_sorted.iloc[min_idx]['y']
    min_z = df_sorted.iloc[min_idx]['z']
    
    # Calculate tolerance band
    z_range = df_sorted['z'].max() - df_sorted['z'].min()
    tolerance_value = z_range * tolerance
    
    # Function to check if trend is monotonic within tolerance
    def is_monotonic(values, increasing=True):
        if increasing:
            diffs = np.diff(values)
            return np.all(diffs > -tolerance_value)
        else:
            diffs = np.diff(values)
            return np.all(diffs < tolerance_value)
    
    # Find longest monotonic sequence to the left of minimum
    left_points = []
    current_idx = min_idx
    while current_idx > 0:
        sequence = z_smooth[current_idx-1:min_idx+1]
        if is_monotonic(sequence, increasing=False):
            left_points.append(current_idx-1)
            current_idx -= 1
        else:
            break
    
    # Find longest monotonic sequence to the right of minimum
    right_points = []
    current_idx = min_idx
    while current_idx < len(z_smooth)-1:
        sequence = z_smooth[min_idx:current_idx+2]
        if is_monotonic(sequence, increasing=True):
            right_points.append(current_idx+1)
            current_idx += 1
        else:
            break
    
    # Combine all points
    selected_indices = sorted(left_points + [min_idx] + right_points)
    result_df = df_sorted.iloc[selected_indices][['y', 'z']].copy()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot all data points in light gray
    plt.scatter(df_sorted['y'], df_sorted['z'], alpha=0.2, color='gray', label='Original Data')
    
    # Plot smoothed line
    plt.plot(df_sorted['y'], z_smooth, 'g-', alpha=0.5, label='Smoothed Data')
    
    # Plot selected points
    plt.scatter(result_df['y'], result_df['z'], color='blue', s=50, label='Selected Points')
    
    # Highlight minimum point
    plt.scatter(min_y, min_z, color='red', s=200, label='Global Minimum', zorder=5)
    
    # Plot tolerance bands
    y_selected = result_df['y'].values
    z_selected = result_df['z'].values
    plt.fill_between(y_selected, 
                    z_selected - tolerance_value,
                    z_selected + tolerance_value,
                    alpha=0.2, color='blue', label='Tolerance Band')
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Monotonic Trend Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (
        f'Tolerance: {tolerance*100:.1f}% of z range\n'
        f'Points before minimum: {len(left_points)}\n'
        f'Points after minimum: {len(right_points)}\n'
        f'Total points: {len(selected_indices)}'
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return result_df

def analyze_monotonicity(df):
    """
    Analyze the monotonicity of the selected points.
    
    Parameters:
    df: DataFrame with selected points
    
    Returns:
    Dictionary with monotonicity analysis
    """
    diffs_left = np.diff(df['z'].values[:df['z'].idxmin()])
    diffs_right = np.diff(df['z'].values[df['z'].idxmin():])
    
    analysis = {
        'left_trend': {
            'strictly_decreasing': np.all(diffs_left < 0),
            'average_change': np.mean(diffs_left) if len(diffs_left) > 0 else 0,
            'max_increase': np.max(diffs_left) if len(diffs_left) > 0 else 0
        },
        'right_trend': {
            'strictly_increasing': np.all(diffs_right > 0),
            'average_change': np.mean(diffs_right) if len(diffs_right) > 0 else 0,
            'max_decrease': np.min(diffs_right) if len(diffs_right) > 0 else 0
        }
    }
    
    return analysis
	
	# Assuming your data is in a DataFrame called 'data' with columns x, y, z
result = find_monotonic_minimum(data, tolerance=0.05)

# If you want to check how well it followed the monotonic trends
analysis = analyze_monotonicity(result)