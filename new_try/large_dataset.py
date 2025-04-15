import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull

def find_minimum_bounding_line(data, x_col='x', y_col='y', sample_size=None):
    """
    Find the minimum bounding line for a set of 2D points in a DataFrame.
    The line follows the lower boundary of the points.
    
    Args:
        data: pandas DataFrame containing the points
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        sample_size: Optional, number of points to sample if dataset is very large
                    Set to None to use all points
    
    Returns:
        tuple: (left_hull, right_hull) where each hull is a list of points forming the lower boundary
    """
    # Convert DataFrame to numpy array
    if isinstance(data, pd.DataFrame):
        # For large datasets, we can work directly with the DataFrame for efficiency
        # Find the point with minimum y value directly from DataFrame
        min_y_idx = data[y_col].idxmin()
        min_y_point = data.loc[min_y_idx, [x_col, y_col]].values
        min_x = min_y_point[0]
        
        # Use efficient DataFrame operations for large datasets
        # Split into left and right segments based on the x value of the min y point
        left_df = data[data[x_col] <= min_x].sort_values(by=x_col)
        right_df = data[data[x_col] >= min_x].sort_values(by=x_col)
        
        # If the dataset is very large, optionally sample it
        # This helps avoid memory issues while still capturing the boundary
        if sample_size is not None and len(data) > sample_size:
            print(f"Sampling {sample_size} points from {len(data)} total points")
            
            # Always include the minimum y point and points near the boundaries
            essential_points = pd.concat([
                left_df.head(int(sample_size/4)),  # Beginning points
                left_df.tail(int(sample_size/4)),  # Points near minimum
                right_df.head(int(sample_size/4)), # Points near minimum
                right_df.tail(int(sample_size/4))  # End points
            ])
            
            # Add the min point if it's not already included
            if not any((essential_points[x_col] == min_x) & (essential_points[y_col] == min_y_point[1])):
                essential_points = pd.concat([essential_points, pd.DataFrame({
                    x_col: [min_x], 
                    y_col: [min_y_point[1]]
                })])
            
            left_points = essential_points[essential_points[x_col] <= min_x][[x_col, y_col]].values
            right_points = essential_points[essential_points[x_col] >= min_x][[x_col, y_col]].values
        else:
            left_points = left_df[[x_col, y_col]].values
            right_points = right_df[[x_col, y_col]].values
    else:
        # Assume it's already a numpy array or list of tuples
        points = np.array(data)
        
        # Find the point with minimum y value
        min_y_idx = np.argmin(points[:, 1])
        min_y_point = points[min_y_idx]
        min_x = min_y_point[0]
        
        # Sort points by x coordinate
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        # Split points into left and right segments
        left_points = sorted_points[sorted_points[:, 0] <= min_x]
        right_points = sorted_points[sorted_points[:, 0] >= min_x]
        
        # If the dataset is very large, optionally sample it
        if sample_size is not None and len(points) > sample_size:
            print(f"Sampling {sample_size} points from {len(points)} total points")
            left_size = len(left_points)
            right_size = len(right_points)
            
            # Select a subset of points, ensuring we include boundaries
            if left_size > sample_size // 2:
                left_indices = np.concatenate([
                    np.linspace(0, left_size//2, sample_size//4, dtype=int),
                    np.linspace(left_size//2, left_size-1, sample_size//4, dtype=int)
                ])
                left_points = left_points[left_indices]
            
            if right_size > sample_size // 2:
                right_indices = np.concatenate([
                    np.linspace(0, right_size//2, sample_size//4, dtype=int),
                    np.linspace(right_size//2, right_size-1, sample_size//4, dtype=int)
                ])
                right_points = right_points[right_indices]
    
    # Find the lower convex hull for left segment
    left_hull = find_lower_hull(left_points)
    
    # Find the lower convex hull for right segment
    right_hull = find_lower_hull(right_points)
    
    return left_hull, right_hull

def find_lower_hull(points):
    """
    Find the lower convex hull of a set of points.
    This implementation is optimized for large datasets.
    
    Args:
        points: Numpy array of shape (n, 2) sorted by x-coordinate
    
    Returns:
        numpy.ndarray: Points forming the lower hull
    """
    n = len(points)
    if n <= 2:
        return points
    
    # For very large datasets, we can optimize by pre-filtering
    # points that are obviously not part of the lower hull
    if n > 10000:
        # Divide x-range into bins and keep only the minimum y point in each bin
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        num_bins = min(n // 100, 1000)  # Number of bins to use
        bin_edges = np.linspace(x_min, x_max, num_bins + 1)
        
        filtered_points = []
        for i in range(num_bins):
            bin_mask = (points[:, 0] >= bin_edges[i]) & (points[:, 0] < bin_edges[i+1])
            bin_points = points[bin_mask]
            if len(bin_points) > 0:
                # Keep the minimum y point in this bin
                min_idx = np.argmin(bin_points[:, 1])
                filtered_points.append(bin_points[min_idx])
                
                # Also keep some other low points to ensure we don't miss hull points
                if len(bin_points) > 10:
                    # Sort by y and keep a few of the lowest
                    sorted_idx = np.argsort(bin_points[:, 1])
                    for j in range(1, min(5, len(sorted_idx))):
                        if min_idx != sorted_idx[j]:  # Don't duplicate the min point
                            filtered_points.append(bin_points[sorted_idx[j]])
        
        if filtered_points:
            points = np.vstack(filtered_points)
            # Re-sort by x coordinate
            sort_idx = np.argsort(points[:, 0])
            points = points[sort_idx]
    
    hull = []
    
    # Process points from left to right
    for i in range(len(points)):
        # Remove points that would create a concave segment
        while len(hull) >= 2 and not is_lower_hull(hull[-2], hull[-1], points[i]):
            hull.pop()
        
        hull.append(points[i])
    
    return np.array(hull)

def is_lower_hull(p1, p2, p3):
    """
    Check if three points form a lower hull.
    Returns True if the middle point is above or on the line connecting p1 and p3.
    
    Args:
        p1, p2, p3: Three consecutive points
    
    Returns:
        bool: True if p2 is below or on the line from p1 to p3
    """
    # Calculate the cross product (p2 - p1) × (p3 - p1)
    # If positive, p2 is above the line; if negative, p2 is below the line
    cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    # We want p2 to be below the line (cross product > 0)
    return cross_product <= 0

def visualize_result(data, left_hull, right_hull, x_col='x', y_col='y'):
    """
    Visualize the points and the minimum bounding line.
    
    Args:
        data: pandas DataFrame or numpy array of points
        left_hull, right_hull: The output of find_minimum_bounding_line
        x_col, y_col: Column names if data is a DataFrame
    """
    plt.figure(figsize=(10, 6))
    
    # Get the points as a numpy array
    if isinstance(data, pd.DataFrame):
        points = data[[x_col, y_col]].values
        x_values = data[x_col].values
        y_values = data[y_col].values
    else:
        points = np.array(data)
        x_values = points[:, 0]
        y_values = points[:, 1]
    
    # Plot all points
    plt.scatter(x_values, y_values, color='blue', label='Data Points')
    
    # Plot the minimum point
    min_y_idx = np.argmin(y_values)
    min_y_point = points[min_y_idx]
    plt.scatter(min_y_point[0], min_y_point[1], color='red', s=100, label='Min Y Point')
    
    # Plot the left hull
    plt.plot(left_hull[:, 0], left_hull[:, 1], 'g-', linewidth=2, label='Left Hull')
    
    # Plot the right hull
    plt.plot(right_hull[:, 0], right_hull[:, 1], 'm-', linewidth=2, label='Right Hull')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Minimum Bounding Line')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Test with standard small dataset
    print("Testing with small dataset (30 points):")
    n_points = 30
    x = np.linspace(0, 1, n_points)
    
    # Create a V-shaped pattern with minimum at x=0.6
    min_x = 0.6
    y_base = 5 * np.abs(x - min_x) + 1
    
    # Add some noise
    y = y_base + np.random.rand(n_points) * 2
    
    # Ensure all x and y values are unique
    while len(np.unique(x)) < len(x) or len(np.unique(y)) < len(y):
        # If duplicates exist, add small random perturbations
        x += np.random.rand(n_points) * 0.01
        y += np.random.rand(n_points) * 0.01
        
        # Normalize x back to [0, 1] range
        x = (x - min(x)) / (max(x) - min(x))
    
    # Create a pandas DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    
    print("Sample DataFrame:")
    print(df.head())
    
    # Method 1: Using the DataFrame directly
    left_hull, right_hull = find_minimum_bounding_line(df, x_col='x', y_col='y')
    
    # Visualize the result
    visualize_result(df, left_hull, right_hull, x_col='x', y_col='y')
    
    # Test with a large dataset (simulating 60,000 points)
    print("\nTesting with large dataset (60,000 points):")
    
    # Generate a large dataset
    n_large = 60000
    x_large = np.random.rand(n_large)
    
    # Sort x to ensure uniqueness and to facilitate the example
    x_large.sort()
    
    # Add small random perturbations to ensure uniqueness
    x_large += np.random.rand(n_large) * 0.0001
    x_large = np.clip(x_large, 0, 1)  # Keep within [0,1]
    
    # Create a V-shaped pattern with minimum at x=0.6
    min_x_large = 0.6
    y_base_large = 5 * np.abs(x_large - min_x_large) + 1
    
    # Add noise but ensure the overall pattern is maintained
    y_large = y_base_large + np.random.rand(n_large) * 3
    
    # Add some random perturbations to ensure uniqueness
    y_large += np.random.rand(n_large) * 0.0001
    
    # Create a DataFrame
    df_large = pd.DataFrame({
        'x': x_large,
        'y': y_large
    })
    
    print(f"Large DataFrame shape: {df_large.shape}")
    print(df_large.head())
    
    # Time the execution for the large dataset
    import time
    start_time = time.time()
    
    # Use sampling for very large datasets
    left_hull_large, right_hull_large = find_minimum_bounding_line(
        df_large, x_col='x', y_col='y', sample_size=5000
    )
    
    end_time = time.time()
    print(f"Processing time for 60,000 points: {end_time - start_time:.2f} seconds")
    
    # Display number of points in the hull
    print(f"Number of points in left hull: {len(left_hull_large)}")
    print(f"Number of points in right hull: {len(right_hull_large)}")
    
    # Create a smaller sample for visualization
    sample_idx = np.random.choice(n_large, 1000, replace=False)
    df_sample = df_large.iloc[sample_idx].copy()
    
    # Add the hull points to ensure they're displayed
    hull_points_x = np.concatenate([left_hull_large[:, 0], right_hull_large[:, 0]])
    hull_points_y = np.concatenate([left_hull_large[:, 1], right_hull_large[:, 1]])
    
    for i in range(len(hull_points_x)):
        if i not in df_sample.index:
            df_sample = pd.concat([df_sample, pd.DataFrame({
                'x': [hull_points_x[i]],
                'y': [hull_points_y[i]]
            })])
    
    # Visualize the result with the sampled points
    visualize_result(df_sample, left_hull_large, right_hull_large, x_col='x', y_col='y')
    
    print("\nLeft hull points (first 5):")
    for point in left_hull_large[:5]:
        print(f"  ({point[0]:.4f}, {point[1]:.4f})")
    
    print("\nRight hull points (first 5):")
    for point in right_hull_large[:5]:
        print(f"  ({point[0]:.4f}, {point[1]:.4f})")
