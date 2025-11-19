import numpy as np
import pandas as pd
import os


def method3_point_count_bins(df, x_col, y_col, points_per_bin=10, output_dir=None):
    """
    Point-count bins: each bin contains approximately points_per_bin points.
    Guarantees first and last points and global minimum are included.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    points_per_bin : int
        Target number of points per bin
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
    
    n_points = len(x_sorted)
    n_bins = max(1, n_points // points_per_bin)
    
    bin_indices = np.array_split(np.arange(n_points), n_bins)
    
    bin_edges = [x_sorted[0]]
    envelope_x = [x_sorted[0]]
    envelope_y = [y_sorted[0]]
    
    for indices in bin_indices:
        if len(indices) > 0:
            x_bin = x_sorted[indices]
            y_bin = y_sorted[indices]
            
            min_idx = np.argmin(y_bin)
            new_x = x_bin[min_idx]
            new_y = y_bin[min_idx]
            
            if not np.isclose(new_x, envelope_x[-1]):
                envelope_x.append(new_x)
                envelope_y.append(new_y)
            
            if len(indices) > 1:
                bin_edges.append(x_bin[-1])
    
    if not np.isclose(envelope_x[-1], x_sorted[-1]):
        envelope_x.append(x_sorted[-1])
        envelope_y.append(y_sorted[-1])
    
    bin_edges.append(x_sorted[-1])
    bin_edges = np.unique(bin_edges)
    
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
        bin_edges_df = pd.DataFrame({x_col: bin_edges})
        bin_edges_df.to_csv(os.path.join(output_dir, 'bin_edges.csv'), index=False)
    
    return result_df


if __name__ == '__main__':
    np.random.seed(42)
    test_x = np.sort(np.random.uniform(0, 10, 100))
    test_y = np.sin(test_x) + np.random.normal(0, 0.2, 100)
    test_df = pd.DataFrame({'x': test_x, 'y': test_y})
    
    result = method3_point_count_bins(test_df, 'x', 'y', output_dir='test_output')
    print("Result dataframe:")
    print(result)
