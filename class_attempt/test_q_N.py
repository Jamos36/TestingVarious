import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import re

def clean_dataframe(df):
    """
    Remove columns containing 'Blah Blah' followed by a number.
    
    Parameters:
    df: Input DataFrame
    
    Returns:
    DataFrame with filtered columns
    """
    # Create a pattern to match 'Blah Blah' followed by numbers
    pattern = r'Blah Blah \d+'
    
    # Get columns to drop
    cols_to_drop = [col for col in df.columns if re.search(pattern, col)]
    
    # Return cleaned dataframe
    return df.drop(columns=cols_to_drop)

def get_analysis_columns(df):
    """
    Get all possible column combinations for analysis.
    Assumes the last column is the z-variable to minimize.
    
    Parameters:
    df: Input DataFrame
    
    Returns:
    List of tuples containing column combinations (x, y, z)
    """
    # Get all columns except the last one
    feature_cols = list(df.columns[:-1])
    z_col = df.columns[-1]
    
    # Generate all possible pairs of feature columns
    column_pairs = list(combinations(feature_cols, 2))
    
    # Create analysis combinations (x, y, z)
    return [(x, y, z_col) for x, y in column_pairs]

def quantile_minima_3d(df, x_col, y_col, z_col, n_quantiles=70):
    """
    Find local minima using quantile-based sampling for 3D data.
    
    Parameters:
    df: DataFrame
    x_col, y_col: Column names for x and y coordinates
    z_col: Column name for z values to minimize
    n_quantiles: Number of quantiles to create
    
    Returns:
    DataFrame with minimal z points from each quantile
    """
    # Calculate quantile boundaries for both x and y
    quantiles = np.linspace(0, 1, n_quantiles+1)
    x_bounds = df[x_col].quantile(quantiles)
    y_bounds = df[y_col].quantile(quantiles)
    
    # Find minimum z value in each quantile grid
    minima = []
    for i in range(len(x_bounds)-1):
        for j in range(len(y_bounds)-1):
            mask = (
                (df[x_col] >= x_bounds[i]) & 
                (df[x_col] < x_bounds[i+1]) &
                (df[y_col] >= y_bounds[j]) & 
                (df[y_col] < y_bounds[j+1])
            )
            if mask.any():
                quantile_data = df[mask]
                min_point = quantile_data.loc[quantile_data[z_col].idxmin()]
                minima.append({
                    x_col: min_point[x_col],
                    y_col: min_point[y_col],
                    z_col: min_point[z_col]
                })
    
    result_df = pd.DataFrame(minima)
    
    # Ensure global minimum is included
    global_min = df.loc[df[z_col].idxmin()]
    min_point_dict = {
        x_col: global_min[x_col],
        y_col: global_min[y_col],
        z_col: global_min[z_col]
    }
    
    if not any((result_df[x_col] == global_min[x_col]) & 
               (result_df[y_col] == global_min[y_col]) & 
               (result_df[z_col] == global_min[z_col])):
        result_df = pd.concat([result_df, pd.DataFrame([min_point_dict])], 
                            ignore_index=True)
    
    return result_df

def analyze_and_save(input_df, output_dir='results'):
    """
    Perform comprehensive analysis on the dataset and save results.
    
    Parameters:
    input_df: Input DataFrame
    output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the dataframe
    cleaned_df = clean_dataframe(input_df)
    
    # Get all possible column combinations
    combinations = get_analysis_columns(cleaned_df)
    
    # Analyze each combination
    for x_col, y_col, z_col in combinations:
        # Perform quantile analysis
        result_df = quantile_minima_3d(cleaned_df, x_col, y_col, z_col)
        
        # Save results
        filename = f"{x_col}_{y_col}_{z_col}_minima.csv"
        filepath = os.path.join(output_dir, filename)
        result_df.to_csv(filepath, index=False)
        
        # Create 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original data
        ax.scatter(cleaned_df[x_col], cleaned_df[y_col], c=cleaned_df[z_col], 
                  cmap ='viridis',
                  alpha=0.5, label='Original Data')
        
        # Plot minima points
        ax.scatter(result_df[x_col], result_df[y_col], 
                  color='red', s=100, label='Quantile Minima')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'Quantile-Based Minima: {x_col} vs {y_col} vs {z_col}')
        ax.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"{x_col}_{y_col}_{z_col}_plot.png"))
        plt.close()

# Example usage:
"""
# Create sample data
data = {
    'a': np.random.rand(100),
    'b': np.random.rand(100),
    'c': np.random.rand(100),
    'Blah Blah': np.random.rand(100),
    'Blah Blah 52': np.random.rand(100),
    'z': np.random.rand(100)
}
df = pd.DataFrame(data)

# Run analysis
analyze_and_save(df)
"""