import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def rolling_monotonic_analysis(df, window_size=None, window_overlap=0.5, tolerance=0.05, 
                             smoothing_window=11, poly_order=3):
    """
    Analyze data using rolling windows and find monotonic minima in each window.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    window_size: Size of rolling window in y units (if None, auto-calculated)
    window_overlap: Fraction of window overlap (0 to 1)
    tolerance: Allowed deviation from monotonicity (as fraction of z range)
    smoothing_window: Window length for Savitzky-Golay filter
    poly_order: Polynomial order for smoothing
    
    Returns:
    DataFrame with selected minima points and their monotonic regions
    """
    # Sort by y value
    df_sorted = df.sort_values('y').reset_index(drop=True)
    
    # Auto-calculate window size if not provided
    if window_size is None:
        y_range = df_sorted['y'].max() - df_sorted['y'].min()
        window_size = y_range / 10  # Default to 10 windows
    
    # Calculate window step size
    step_size = window_size * (1 - window_overlap)
    
    # Initialize results storage
    all_minima = []
    window_bounds = []
    
    # Calculate total z range for consistent tolerance
    total_z_range = df_sorted['z'].max() - df_sorted['z'].min()
    tolerance_value = total_z_range * tolerance
    
    # Function to check if trend is monotonic within tolerance
    def is_monotonic(values, increasing=True):
        if len(values) < 2:
            return True
        diffs = np.diff(values)
        if increasing:
            return np.all(diffs > -tolerance_value)
        else:
            return np.all(diffs < tolerance_value)
    
    # Process each window
    current_y = df_sorted['y'].min()
    while current_y <= df_sorted['y'].max() - window_size:
        # Get data in current window
        mask = (df_sorted['y'] >= current_y) & (df_sorted['y'] < current_y + window_size)
        window_data = df_sorted[mask].copy()
        
        if len(window_data) > smoothing_window:  # Only process if enough points
            # Smooth the window data
            z_smooth = savgol_filter(window_data['z'], smoothing_window, poly_order)
            window_data['z_smooth'] = z_smooth
            
            # Find minimum in window
            min_idx = window_data['z_smooth'].idxmin()
            min_point = window_data.loc[min_idx]
            
            # Look for monotonic regions around minimum
            left_points = []
            right_points = []
            
            # Check left side
            for i in range(min_idx - 1, window_data.index[0] - 1, -1):
                sequence = window_data.loc[i:min_idx, 'z_smooth']
                if is_monotonic(sequence, increasing=False):
                    left_points.append(i)
                else:
                    break
            
            # Check right side
            for i in range(min_idx + 1, window_data.index[-1] + 1):
                sequence = window_data.loc[min_idx:i, 'z_smooth']
                if is_monotonic(sequence, increasing=True):
                    right_points.append(i)
                else:
                    break
            
            # If we found monotonic regions on both sides
            if left_points or right_points:
                selected_indices = sorted(left_points + [min_idx] + right_points)
                selected_points = df_sorted.loc[selected_indices][['y', 'z']].copy()
                
                all_minima.append({
                    'window_start': current_y,
                    'window_end': current_y + window_size,
                    'min_y': min_point['y'],
                    'min_z': min_point['z'],
                    'points': selected_points,
                    'n_left_points': len(left_points),
                    'n_right_points': len(right_points)
                })
                
                window_bounds.append((current_y, current_y + window_size))
        
        current_y += step_size
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot all data points
    plt.scatter(df_sorted['y'], df_sorted['z'], alpha=0.2, color='gray', label='Original Data')
    
    # Plot each window's results
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_minima)))
    for minima, color in zip(all_minima, colors):
        # Plot selected points in this window
        plt.scatter(minima['points']['y'], minima['points']['z'], 
                   color=color, s=50, alpha=0.7)
        
        # Highlight minimum point
        plt.scatter(minima['min_y'], minima['min_z'], 
                   color=color, s=200, marker='*', 
                   label=f'Minimum at y={minima["min_y"]:.2f}')
        
        # Show window bounds
        plt.axvline(minima['window_start'], color=color, linestyle='--', alpha=0.2)
        plt.axvline(minima['window_end'], color=color, linestyle='--', alpha=0.2)
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Rolling Window Monotonic Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add text box with analysis statistics
    stats_text = (
        f'Number of windows: {len(all_minima)}\n'
        f'Window size: {window_size:.2f}\n'
        f'Overlap: {window_overlap*100:.0f}%\n'
        f'Tolerance: {tolerance*100:.1f}% of z range'
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(all_minima)