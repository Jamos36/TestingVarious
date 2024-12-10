import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def calculate_subplot_dimensions(n_coords):
    """
    Calculate the number of rows and columns needed for the upper triangular plots.
    
    Parameters:
    n_coords: int, number of coordinate columns
    
    Returns:
    tuple: (rows, cols) for subplot layout
    """
    n_plots = (n_coords * (n_coords - 1)) // 2  # Number of unique pairs
    if n_plots == 0:  # Handle single dimension case
        return 1, 1
    
    # Calculate minimum number of rows needed
    rows = n_coords - 1
    # Calculate minimum number of columns needed
    cols = n_coords - 1
    
    return rows, cols

def plot_upper_triangle_dimensions(df, n_quantiles=10):
    """
    Create scatter plots for unique pairs of dimensions with intensity coloring.
    
    Parameters:
    df: DataFrame where last column is intensity and others are coordinates
    n_quantiles: Number of quantiles for minima calculation
    """
    intensity_col = df.columns[-1]
    coord_cols = df.columns[:-1]
    n_coords = len(coord_cols)
    
    if n_coords < 2:
        raise ValueError("Need at least two coordinate columns for paired plotting")
    
    # Calculate subplot layout
    rows, cols = calculate_subplot_dimensions(n_coords)
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    # Counter for subplot positioning
    plot_idx = 1
    
    # Create plots for unique pairs
    for i in range(n_coords-1):  # Row index
        for j in range(i+1, n_coords):  # Column index
            x_col = coord_cols[i]
            y_col = coord_cols[j]
            
            # Calculate position in subplot grid
            ax = plt.subplot(rows, cols, plot_idx)
            
            # Create scatter plot
            scatter = ax.scatter(df[x_col], df[y_col], 
                               c=df[intensity_col], 
                               cmap='viridis', 
                               alpha=0.6)
            
            # Add labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{y_col} vs {x_col}')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label=intensity_col)
            
            # Calculate and plot quantile minima
            minima = calculate_2d_quantile_minima(
                df, x_col, y_col, intensity_col, n_quantiles)
            
            if not minima.empty:
                ax.scatter(minima[x_col], minima[y_col], 
                          color='red', s=100, alpha=0.8,
                          label='Quantile Minima')
                
            ax.legend()
            
            plot_idx += 1
    
    plt.tight_layout()
    return fig

def calculate_2d_quantile_minima(df, x_col, y_col, intensity_col, n_quantiles):
    """
    Calculate minima for 2D quantile regions.
    
    Parameters:
    df: DataFrame containing the data
    x_col: Name of x-axis column
    y_col: Name of y-axis column
    intensity_col: Name of intensity column
    n_quantiles: Number of quantiles to use
    
    Returns:
    DataFrame containing minima points
    """
    # Calculate quantile boundaries for both dimensions
    x_quantiles = np.linspace(0, 1, n_quantiles+1)
    y_quantiles = np.linspace(0, 1, n_quantiles+1)
    
    x_bounds = df[x_col].quantile(x_quantiles)
    y_bounds = df[y_col].quantile(y_quantiles)
    
    # Initialize empty DataFrame for minima
    minima_list = []
    
    # Find minima in each 2D quantile region
    for i in range(len(x_bounds)-1):
        for j in range(len(y_bounds)-1):
            # Create mask for current region
            mask = (
                (df[x_col] >= x_bounds[i]) & 
                (df[x_col] < x_bounds[i+1]) &
                (df[y_col] >= y_bounds[j]) & 
                (df[y_col] < y_bounds[j+1])
            )
            
            if mask.any():
                # Find point with minimum intensity in this region
                region_data = df[mask]
                min_idx = region_data[intensity_col].idxmin()
                min_point = region_data.loc[min_idx, [x_col, y_col, intensity_col]]
                minima_list.append(min_point)
    
    # Create DataFrame from minima list
    if minima_list:
        minima_df = pd.DataFrame(minima_list)
        
        # Add global minimum if not already included
        global_min_idx = df[intensity_col].idxmin()
        global_min = df.loc[global_min_idx, [x_col, y_col, intensity_col]]
        
        # Check if global minimum is already in minima_df
        global_min_mask = (
            (minima_df[x_col] == global_min[x_col]) & 
            (minima_df[y_col] == global_min[y_col]) &
            (minima_df[intensity_col] == global_min[intensity_col])
        )
        
        if not global_min_mask.any():
            minima_df = pd.concat([minima_df, 
                                 pd.DataFrame([global_min])], 
                                ignore_index=True)
        
        return minima_df
    
    return pd.DataFrame(columns=[x_col, y_col, intensity_col])

# Example usage:
if __name__ == "__main__":
    # Create sample data with multiple dimensions
    np.random.seed(42)
    n_points = 1000
    
    # Create sample data with 4 dimensions plus intensity
    data = {
        'w': np.random.normal(0, 1, n_points),
        'x': np.random.normal(0, 1, n_points),
        'y': np.random.normal(0, 1, n_points),
        'z': np.random.normal(0, 1, n_points),
        'intensity': np.random.normal(0, 1, n_points)
    }
    
    df = pd.DataFrame(data)
    
    # Add some structure to the intensity
    df['intensity'] += (df['w']**2 + df['x']**2 + 
                       df['y']**2 + df['z']**2)
    
    # Create plots
    fig = plot_upper_triangle_dimensions(df, n_quantiles=5)
    plt.show()