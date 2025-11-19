import numpy as np
import pandas as pd
import os


def method4_dynamic_local_density_bins(df, x_col, y_col, target_bins=50, window_fraction=0.15, edge_bins=7, output_dir=None):
    """
    Dynamic local density bins: generates ~target_bins adaptive bins.
    - Creates many candidate bin edges initially
    - Uses local density to keep more edges in dense regions
    - Merges edges in sparse regions
    - Adds extra bins at the edges for better coverage
    - Result: adaptive bin density matching data density
    Guarantees first and last points and global minimum are included.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    target_bins : int
        Target number of adaptive bins
    window_fraction : float
        Fraction of x_range for density window
    edge_bins : int
        Number of extra bins to add at each edge
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
    x_range = x_max - x_min
    n_points = len(x_sorted)
    
    window_size = window_fraction * x_range
    
    candidate_edges = np.linspace(x_min, x_max, target_bins * 3)
    
    density_at_edges = []
    for edge_x in candidate_edges:
        count = np.sum(np.abs(x_sorted - edge_x) <= window_size / 2)
        density_at_edges.append(count)
    
    density_at_edges = np.array(density_at_edges)
    mean_density = np.mean(density_at_edges)
    std_density = np.std(density_at_edges)
    
    threshold_sparse = mean_density - 0.5 * std_density
    threshold_dense = mean_density + 0.5 * std_density
    
    refined_edges = [x_min]
    for i, (edge_x, density) in enumerate(zip(candidate_edges[1:-1], density_at_edges[1:-1])):
        if density > threshold_dense:
            refined_edges.append(edge_x)
        elif density > threshold_sparse:
            if i % 2 == 0:
                refined_edges.append(edge_x)
    
    refined_edges.append(x_max)
    refined_edges = np.unique(refined_edges)
    
    edge_width = 0.15 * x_range
    left_edge_bins = np.linspace(x_min, x_min + edge_width, edge_bins + 1)[1:]
    right_edge_bins = np.linspace(x_max - edge_width, x_max, edge_bins + 1)[:-1]
    
    refined_edges = np.concatenate([refined_edges, left_edge_bins, right_edge_bins])
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
    
    result = method4_dynamic_local_density_bins(test_df, 'x', 'y', output_dir='test_output')
    print("Result dataframe:")
    print(result)
