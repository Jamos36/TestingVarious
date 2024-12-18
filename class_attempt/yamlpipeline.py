import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from itertools import combinations
import re
from typing import List, Tuple, Dict

def load_yaml_data(yaml_path: str) -> Tuple[List[List[int]], List[str]]:
    """
    Load coordinate list and string list from YAML file.
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['coordinate_list'], data['string_list']

def group_strings(strings: List[str]) -> Dict[str, List[str]]:
    """
    Group strings based on common prefixes and suffixes.
    Returns dictionary with pattern as key and list of matching strings as value.
    """
    groups = {}
    # Find all unique prefix_suffix combinations
    for s in strings:
        # Extract prefix and suffix using regex
        match = re.match(r'([a-zA-Z]+_).*?(_[a-zA-Z]+)$', s)
        if match:
            prefix, suffix = match.groups()
            pattern = f"{prefix}*{suffix}"
            if pattern not in groups:
                groups[pattern] = []
            groups[pattern].append(s)
    return groups

def coordinate_to_string(coord: List[int]) -> str:
    """
    Convert coordinate list to string with "Troublesome" prefix.
    [0,1] -> "Troublesome 01"
    """
    coord_str = ''.join(str(x) for x in coord)
    return f"Troublesome {coord_str}"

def match_groups_to_coordinates(string_groups: Dict[str, List[str]], 
                              coordinates: List[List[int]]) -> List[Tuple[List[str], str]]:
    """
    Match string groups to coordinates in order.
    Returns list of tuples: (string_group, troublesome_coord_string)
    """
    matched_pairs = []
    coord_strings = [coordinate_to_string(coord) for coord in coordinates]
    
    # Convert dictionary values to list for ordered matching
    grouped_strings = list(string_groups.values())
    
    # Match groups with coordinates if lengths match
    if len(grouped_strings) == len(coordinates):
        matched_pairs = list(zip(grouped_strings, coord_strings))
    
    return matched_pairs

def create_scatter_plots(df: pd.DataFrame, str_cols: List[str], trouble_col: str, 
                        output_dir: str = 'results'):
    """
    Create scatter plots for string columns vs trouble column.
    """
    # Create directory
    dir_name = f"{str_cols[0]}_{str_cols[1]}"
    os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df[str_cols[0]], df[str_cols[1]], 
                        c=np.log10(df[trouble_col]),
                        cmap='viridis',
                        alpha=0.6)
    
    plt.xlabel(str_cols[0])
    plt.ylabel(str_cols[1])
    plt.title(f'{str_cols[0]} vs {str_cols[1]}\nColor: log10({trouble_col})')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'log10({trouble_col})')
    
    # Save plot
    plt.savefig(os.path.join(output_dir, dir_name, f'{dir_name}_scatter.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    subset_df = df[str_cols + [trouble_col]]
    subset_df.to_csv(os.path.join(output_dir, dir_name, f'{dir_name}_data.csv'),
                    index=False)

def quantile_minima(df: pd.DataFrame, x_col: str, z_col: str, 
                    n_quantiles: int = 70) -> pd.DataFrame:
    """
    Calculate quantile minima for given columns.
    """
    quantiles = np.linspace(0, 1, n_quantiles+1)
    x_bounds = df[x_col].quantile(quantiles)
    
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
    
    return pd.DataFrame(minima).sort_values(x_col)

def create_minima_plots(df: pd.DataFrame, str_cols: List[str], trouble_col: str, 
                       output_dir: str = 'minima_results'):
    """
    Create minima plots for each string column vs trouble column.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for col in str_cols:
        # Calculate minima
        minima_df = quantile_minima(df, col, trouble_col)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot original data
        plt.scatter(df[col], df[trouble_col], alpha=0.3, label='Original Data')
        
        # Plot minima curve
        plt.plot(minima_df[col], minima_df[trouble_col], 'r-', 
                label='Quantile Minima', linewidth=2)
        
        plt.xlabel(col)
        plt.ylabel(trouble_col)
        plt.title(f'{col} vs {trouble_col} with Quantile Minima')
        plt.legend()
        
        # Save plot and data
        plt.savefig(os.path.join(output_dir, f'{col}_minima.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        minima_df.to_csv(os.path.join(output_dir, f'{col}_minima.csv'),
                        index=False)

def main(yaml_path: str, data_path: str):
    """
    Main function to process YAML and data files and create visualizations.
    """
    # Load YAML data
    coordinates, strings = load_yaml_data(yaml_path)
    
    # Group strings
    string_groups = group_strings(strings)
    
    # Match groups with coordinates
    matched_pairs = match_groups_to_coordinates(string_groups, coordinates)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Process each matched pair
    for string_group, trouble_col in matched_pairs:
        if len(string_group) == 2:  # Ensure we have pairs
            create_scatter_plots(df, string_group, trouble_col)
            create_minima_plots(df, string_group, trouble_col)

# Example usage:
"""
# Example YAML file structure:
# coordinate_list:
#   - [0, 1]
#   - [2, 3]
#   - [7, 9]
# string_list:
#   - some_string_thing
#   - some_other_thing
#   - purple_string_head
#   - purple_other_head
#   - keyboard_other_black
#   - keyboard_string_black

main('config.yaml', 'data.csv')
"""