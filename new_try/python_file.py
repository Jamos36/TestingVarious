# left_hull, right_hull = find_minimum_bounding_line(your_dataframe, x_col='your_x_column', y_col='your_y_column')
# visualize_result(your_dataframe, left_hull, right_hull, x_col='your_x_column', y_col='your_y_column')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull

def find_minimum_bounding_line(data, x_col='x', y_col='y'):
    """
    Find the minimum bounding line for a set of 2D points in a DataFrame.
    The line follows the lower boundary of the points.
    
    Args:
        data: pandas DataFrame containing the points
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
    
    Returns:
        tuple: (left_hull, right_hull) where each hull is a list of points forming the lower boundary
    """
    # Convert DataFrame to numpy array
    if isinstance(data, pd.DataFrame):
        points = data[[x_col, y_col]].values
    else:
        # Assume it's already a numpy array or list of tuples
        points = np.array(data)
    
    # Find the point with minimum y value
    min_y_idx = np.argmin(points[:, 1])
    min_y_point = points[min_y_idx]
    
    # Sort points by x coordinate
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]
    
    # Find the index of min_y_point in sorted points
    min_point_idx = np.where((sorted_points == min_y_point).all(axis=1))[0][0]
    
    # Split points into left and right segments
    left_points = sorted_points[:min_point_idx+1]  # Include the min point in both segments
    right_points = sorted_points[min_point_idx:]
    
    # Find the lower convex hull for left segment
    left_hull = find_lower_hull(left_points)
    
    # Find the lower convex hull for right segment
    right_hull = find_lower_hull(right_points)
    
    return left_hull, right_hull

def find_lower_hull(points):
    """
    Find the lower convex hull of a set of points.
    
    Args:
        points: Numpy array of shape (n, 2) sorted by x-coordinate
    
    Returns:
        list: Points forming the lower hull
    """
    n = len(points)
    if n <= 2:
        return points
    
    hull = []
    
    # Process points from left to right
    for i in range(n):
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
    
    # Create a dataset with characteristics as described
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
    
    # Method 2: If we had different column names
    # Renaming columns to demonstrate flexibility
    renamed_df = df.rename(columns={'x': 'position', 'y': 'height'})
    
    print("\nRenamed DataFrame:")
    print(renamed_df.head())
    
    # Find the minimum bounding line with different column names
    left_hull2, right_hull2 = find_minimum_bounding_line(renamed_df, x_col='position', y_col='height')
    
    # Visualize the result
    visualize_result(renamed_df, left_hull2, right_hull2, x_col='position', y_col='height')
    
    print("\nLeft hull points:")
    for point in left_hull:
        print(f"  ({point[0]:.4f}, {point[1]:.4f})")
    
    print("\nRight hull points:")
    for point in right_hull:
        print(f"  ({point[0]:.4f}, {point[1]:.4f})")
