import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quantile_minima_multidim(df, n_quantiles=10):
    """
    Find local minima using quantile-based sampling across multiple dimensions.
    
    Parameters:
    df: DataFrame where the last column is the intensity value
        and all other columns are coordinates
    n_quantiles: Number of quantiles to create
    
    Returns:
    Dictionary of DataFrames containing minimal intensity points for each dimension
    """
    # Get intensity column name (last column)
    intensity_col = df.columns[-1]
    coord_cols = df.columns[:-1]
    
    # Calculate quantile boundaries for each dimension
    quantiles = np.linspace(0, 1, n_quantiles+1)
    
    # Create figure with subplots for each dimension
    n_dims = len(coord_cols)
    fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 5))
    if n_dims == 1:
        axes = [axes]
    
    # Dictionary to store results for each dimension
    results = {}
    
    for idx, coord in enumerate(coord_cols):
        # Calculate quantile boundaries for this dimension
        bounds = df[coord].quantile(quantiles)
        
        # Initialize DataFrame for minima in this dimension
        minima_df = pd.DataFrame(columns=[coord, intensity_col])
        
        # Find minimum intensity value in each quantile
        for i in range(len(bounds)-1):
            mask = (df[coord] >= bounds[i]) & (df[coord] < bounds[i+1])
            if mask.any():
                quantile_data = df[mask]
                min_point_idx = quantile_data[intensity_col].idxmin()
                min_point = quantile_data.loc[min_point_idx, [coord, intensity_col]]
                minima_df = pd.concat([minima_df, 
                                     pd.DataFrame([min_point])],
                                     ignore_index=True)
        
        # Ensure global minimum is included
        global_min_idx = df[intensity_col].idxmin()
        global_min = df.loc[global_min_idx, [coord, intensity_col]]
        if not any((minima_df[coord] == global_min[coord]) & 
                  (minima_df[intensity_col] == global_min[intensity_col])):
            minima_df = pd.concat([minima_df,
                                 pd.DataFrame([global_min])],
                                 ignore_index=True)
        
        # Sort by coordinate value
        minima_df = minima_df.sort_values(coord)
        
        # Store results
        results[coord] = minima_df
        
        # Create scatter plot
        ax = axes[idx]
        scatter = ax.scatter(df[coord], df[intensity_col], 
                           c=df[intensity_col], cmap='viridis',
                           alpha=0.5, label='Original Data')
        ax.scatter(minima_df[coord], minima_df[intensity_col], 
                  color='red', s=100, label='Quantile Minima')
        
        # Add quantile boundaries
        for bound in bounds:
            ax.axvline(bound, color='gray', linestyle='--', alpha=0.3)
            
        ax.set_xlabel(f'{coord} Coordinate')
        ax.set_ylabel(f'{intensity_col}')
        ax.set_title(f'{intensity_col} vs {coord}')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Example usage:
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    
    data = {
        'x': np.random.normal(0, 1, n_points),
        'y': np.random.normal(0, 1, n_points),
        'intensity': np.random.normal(0, 1, n_points)
    }
    df = pd.DataFrame(data)
    
    # Add some structure to the data
    df['intensity'] += df['x']**2 + df['y']**2
    
    # Run analysis
    results = quantile_minima_multidim(df, n_quantiles=10)

