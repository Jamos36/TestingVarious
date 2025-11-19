import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def remove_spikes_from_envelope(df, x_col, y_col, threshold=0.5):
    """
    Remove spikes based on curvature analysis.
    
    A spike is identified as a point with high curvature (abrupt change in direction).
    Uses signed area of triangle formed by three consecutive points.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with envelope points
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    threshold : float
        Curvature threshold for spike detection
    
    Returns
    -------
    pd.DataFrame
        Dataframe with spikes removed
    """
    if len(df) <= 2:
        return df.copy()
    
    x = df[x_col].values
    y = df[y_col].values
    
    mask = np.ones(len(df), dtype=bool)
    mask[0] = True
    mask[-1] = True
    
    for i in range(1, len(df) - 1):
        x0, y0 = x[i - 1], y[i - 1]
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        
        dx1 = x1 - x0
        dy1 = y1 - y0
        dx2 = x2 - x1
        dy2 = y2 - y1
        
        dist1 = np.sqrt(dx1**2 + dy1**2)
        dist2 = np.sqrt(dx2**2 + dy2**2)
        
        if dist1 > 0 and dist2 > 0:
            cos_angle = (dx1 * dx2 + dy1 * dy2) / (dist1 * dist2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            if angle > threshold:
                mask[i] = False
    
    return df[mask].reset_index(drop=True)


def process_with_spike_removal(df, x_col, y_col, thresholds=[0.3, 0.5, 0.7, 1.0], 
                               output_dir=None, plot_output=None):
    """
    Process envelope with multiple curvature thresholds and generate comparison plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with envelope points
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    thresholds : list
        List of angle threshold values (in radians)
    output_dir : str, optional
        Directory to save cleaned dataframes as CSVs
    plot_output : str, optional
        Path to save comparison plot
    
    Returns
    -------
    dict
        Dictionary with threshold as key and cleaned DataFrame as value
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for threshold in thresholds:
        cleaned_df = remove_spikes_from_envelope(df, x_col, y_col, threshold)
        results[threshold] = cleaned_df
        
        if output_dir:
            csv_file = os.path.join(output_dir, f'spike_removed_threshold_{threshold:.2f}.csv')
            cleaned_df.to_csv(csv_file, index=False)
    
    if plot_output:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        x_orig = df[x_col].values
        y_orig = df[y_col].values
        
        for idx, (threshold, ax) in enumerate(zip(thresholds, axes)):
            cleaned_df = results[threshold]
            x_cleaned = cleaned_df[x_col].values
            y_cleaned = cleaned_df[y_col].values
            
            spike_count = len(df) - len(cleaned_df)
            
            ax.scatter(x_orig, y_orig, alpha=0.3, s=20, color='lightblue', label='Original data', zorder=1)
            ax.plot(x_orig, y_orig, alpha=0.5, color='gray', linewidth=1, label='Original envelope', zorder=2)
            ax.scatter(x_orig, y_orig, s=40, color='red', edgecolor='darkred', alpha=0.7, zorder=3)
            
            ax.plot(x_cleaned, y_cleaned, color='orange', linewidth=2, label='Spike removed', zorder=4)
            ax.scatter(x_cleaned, y_cleaned, s=50, color='orange', edgecolor='darkorange', 
                      linewidth=1, marker='s', alpha=0.8, zorder=5)
            
            ax.set_xlabel(x_col, fontsize=11)
            ax.set_ylabel(y_col, fontsize=11)
            ax.set_title(f'Curvature threshold = {threshold:.2f} | Spikes removed: {spike_count}', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_output, dpi=150, bbox_inches='tight')
        plt.close()
    
    return results


if __name__ == '__main__':
    np.random.seed(42)
    test_x = np.array([1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10])
    test_y = np.array([1, 2, 0.5, 3, 2, 1.5, 1, 2, 1, 0.5, 1])
    test_df = pd.DataFrame({'x': test_x, 'y': test_y})
    
    results = process_with_spike_removal(test_df, 'x', 'y', 
                                        output_dir='spike_removal_test_1',
                                        plot_output='spike_removal_test_1/comparison.png')
    
    print("Spike removal results (Curvature-based):")
    for threshold, df in results.items():
        print(f"  Threshold {threshold:.2f}: {len(df)} points (removed {len(test_df) - len(df)} spikes)")
