import numpy as np
import pandas as pd
import os


def method2_multi_tier_density_bins(df, x_col, y_col, baseline_bins=50, output_dir=None):
    """
    Multi-tier density bins: 3 density levels with different subdivision factors.
    Sparse: 1x, Medium: 2x, Dense: 4x
    Guarantees first and last points and global minimum are included.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    baseline_bins : int
        Number of baseline uniform bins
    output_dir : str, optional
        Directory to save output CSVs
    
    Returns
    -------
    pd.DataFrame
        Dataframe with envelope points (x_col, y_col columns)
    """
    x = df[x_col].values
    y = df[y_col].values
    
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    x_min, x_max = x_sorted.min(), x_sorted.max()
    baseline_edges = np.linspace(x_min, x_max, baseline_bins + 1)
    
    bin_counts = np.zeros(baseline_bins)
    for i in range(baseline_bins):
        mask = (x_sorted >= baseline_edges[i]) & (x_sorted <= baseline_edges[i + 1])
        bin_counts[i] = np.sum(mask)
    
    mean_count = np.mean(bin_counts)
    std_count = np.std(bin_counts)
    sparse_threshold = mean_count - 0.5 * std_count
    dense_threshold = mean_count + 0.5 * std_count
    
    refined_edges = []
    for i in range(baseline_bins):
        x_low = baseline_edges[i]
        x_high = baseline_edges[i + 1] if i < baseline_bins - 1 else x_max
        count = bin_counts[i]
        
        if count > dense_threshold:
            subdivisions = 4
        elif count > sparse_threshold:
            subdivisions = 2
        else:
            subdivisions = 1
        
        sub_edges = np.linspace(x_low, x_high, subdivisions + 1)
        refined_edges.extend(sub_edges[:-1])
    
    refined_edges.append(x_max)
    refined_edges = np.unique(np.sort(refined_edges))
    
    envelope_x = [x_sorted[0]]
    envelope_y = [y_sorted[0]]
    
    for i in range(len(refined_edges) - 1):
        mask = (x_sorted >= refined_edges[i]) & (x_sorted <= refined_edges[i + 1])
        if np.sum(mask) > 0:
            y_bin = y_sorted[mask]
            x_bin = x_sorted[mask]
            min_idx = np.argmin(y_bin)
            new_x = x_bin[min_idx]
            new_y = y_bin[min_idx]
            
            if not np.isclose(new_x, envelope_x[-1]):
                envelope_x.append(new_x)
                envelope_y.append(new_y)
    
    if not np.isclose(envelope_x[-1], x_sorted[-1]):
        envelope_x.append(x_sorted[-1])
        envelope_y.append(y_sorted[-1])
    
    envelope_x = np.array(envelope_x)
    envelope_y = np.array(envelope_y)
    
    global_min_idx = np.argmin(y_sorted)
    global_min_x = x_sorted[global_min_idx]
    global_min_y = y_sorted[global_min_idx]
    
    if not any(np.isclose(envelope_x, global_min_x)):
        idx = np.searchsorted(envelope_x, global_min_x)
        envelope_x = np.insert(envelope_x, idx, global_min_x)
        envelope_y = np.insert(envelope_y, idx, global_min_y)
    
    result_df = pd.DataFrame({x_col: envelope_x, y_col: envelope_y})
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, 'envelope_points.csv'), index=False)
        bin_edges_df = pd.DataFrame({x_col: refined_edges})
        bin_edges_df.to_csv(os.path.join(output_dir, 'bin_edges.csv'), index=False)
    
    return result_df


if __name__ == '__main__':
    np.random.seed(42)
    test_x = np.sort(np.random.uniform(0, 10, 100))
    test_y = np.sin(test_x) + np.random.normal(0, 0.2, 100)
    test_df = pd.DataFrame({'x': test_x, 'y': test_y})
    
    result = method2_multi_tier_density_bins(test_df, 'x', 'y', output_dir='test_output')
    print("Result dataframe:")
    print(result)
