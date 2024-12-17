import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import re

def load_and_clean_data(filepath):
    """
    Load CSV and remove columns containing 'Something Something' followed by numbers.
    
    Parameters:
    filepath: Path to CSV file
    
    Returns:
    cleaned DataFrame
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Create pattern to match 'Something Something' followed by numbers
    pattern = r'Something Something \d+'
    
    # Get columns to drop
    cols_to_drop = [col for col in df.columns if re.search(pattern, col)]
    
    # Return cleaned dataframe
    return df.drop(columns=cols_to_drop)

def create_scatter_plots(df, base_output_dir='results'):
    """
    Create scatter plots for all column combinations vs last column intensity.
    
    Parameters:
    df: Input DataFrame
    base_output_dir: Base directory for outputs
    """
    # Get all columns except the last one
    columns = df.columns[:-1]
    intensity_col = df.columns[-1]
    
    # Generate all possible pairs of columns
    column_pairs = list(combinations(columns, 2))
    
    # Create plots for each combination
    for col1, col2 in column_pairs:
        # Create directory for this combination
        dir_name = f"{col1}_{col2}"
        os.makedirs(os.path.join(base_output_dir, dir_name), exist_ok=True)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df[col1], df[col2], 
                            c=np.log10(df[intensity_col]),
                            cmap='viridis',
                            alpha=0.6)
        
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'{col1} vs {col2}\nColor: log10({intensity_col})')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'log10({intensity_col})')
        
        # Save plot
        plt.savefig(os.path.join(base_output_dir, dir_name, f'{col1}_{col2}_scatter.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        subset_df = df[[col1, col2, intensity_col]]
        subset_df.to_csv(os.path.join(base_output_dir, dir_name, f'{col1}_{col2}_data.csv'),
                        index=False)

def quantile_minima(df, x_col, z_col, n_quantiles=70):
    """
    Find local minima using quantile-based sampling.
    
    Parameters:
    df: DataFrame
    x_col: Column for x-axis
    z_col: Column for intensity (z-axis)
    n_quantiles: Number of quantiles
    
    Returns:
    DataFrame with minimal z points from each quantile
    """
    # Calculate quantile boundaries
    quantiles = np.linspace(0, 1, n_quantiles+1)
    x_bounds = df[x_col].quantile(quantiles)
    
    # Find minimum z value in each quantile
    minima = []
    for i in range(len(x_bounds)-1):
        mask = (df[x_col] >= x_bounds[i]) & (df[x_col] < x_bounds[i+1])
        if mask.any():
            quantile_data = df[mask]
            min_point = quantile_data.loc[quantile_data[z_col].idxmin()]
            minima.append({
                x_col: min_point[x_col],
                z_col: min_point[z_col]
            })
    
    result_df = pd.DataFrame(minima)
    
    # Ensure global minimum is included
    global_min = df.loc[df[z_col].idxmin()]
    min_point_dict = {
        x_col: global_min[x_col],
        z_col: global_min[z_col]
    }
    
    if not any((result_df[x_col] == global_min[x_col]) & 
               (result_df[z_col] == global_min[z_col])):
        result_df = pd.concat([result_df, pd.DataFrame([min_point_dict])], 
                            ignore_index=True)
    
    # Sort by x_col
    return result_df.sort_values(x_col)

def create_minima_plots(df, output_dir='minima_results'):
    """
    Create plots of each column vs intensity with quantile minima curves.
    
    Parameters:
    df: Input DataFrame
    output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all columns except the last one
    columns = df.columns[:-1]
    intensity_col = df.columns[-1]
    
    # Create plots for each column vs intensity
    for col in columns:
        # Calculate quantile minima
        minima_df = quantile_minima(df, col, intensity_col)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot original data
        plt.scatter(df[col], df[intensity_col], alpha=0.3, label='Original Data')
        
        # Plot minima curve
        plt.plot(minima_df[col], minima_df[intensity_col], 'r-', 
                label='Quantile Minima', linewidth=2)
        plt.scatter(minima_df[col], minima_df[intensity_col], 
                   color='red', s=50)
        
        plt.xlabel(col)
        plt.ylabel(intensity_col)
        plt.title(f'{col} vs {intensity_col} with Quantile Minima')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{col}_minima.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save minima data
        minima_df.to_csv(os.path.join(output_dir, f'{col}_minima.csv'),
                        index=False)

def main(input_filepath):
    """
    Main function to run the entire analysis pipeline.
    
    Parameters:
    input_filepath: Path to input CSV file
    """
    # Load and clean data
    df = load_and_clean_data(input_filepath)
    
    # Create scatter plots for all combinations
    create_scatter_plots(df)
    
    # Create minima plots
    create_minima_plots(df)

# Example usage:
"""
# Run the analysis
main('data.csv')
"""