import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import re
import itertools
from itertools import combinations
from typing import List, Tuple, Dict

def load_yaml(yaml_path: str) -> Tuple[List[List[int]], List[str]]:
    os.chdir('/path/to/main/folder')
    with open(yaml_path,'r') as file:
        data = yaml.safe_load(file)
        return data['coordinate_list'], data['test_list']
    
def group_strings(strings: List[str]) -> Dict[str, List[str]]:
    groups={}
    for s in strings:
        if len(strings)>2:
            match = re.match(r'([a-zA-Z]+_).*?(_[a-zA-Z]+)$', s)
            if match:
                prefix, suffix = match.groups()
                pattern = f"{prefix}*{suffix}"
                if pattern not in groups:
                    groups[pattern] = []
                groups[pattern].append(s)
        else:
            for i in range(len(strings)):
                groups=[]
                groups.append(strings[i])
        return groups
    
def coordinate_to_string(coord: List[int]) -> str:
    coord_str = ''.join(str(x) for x in coord)
    return f"Troublesome {coord_str}"

def match_groups_to_coordinates(string_groups: Dict[str, List[str]], 
                              coordinates: List[List[int]]) -> List[Tuple[List[str], str]]:
    """
    Match string groups to coordinates in order.
    """
    matched_pairs = []
    if len(coordinates) > 1:
        coord_strings = [coordinate_to_string(coord) for coord in coordinates]
        grouped_strings = list(string_groups.values())
    else:
        coord_strings = ['Desired Symb']
        grouped_strings = list(string_groups)
    
    if len(grouped_strings) == len(coordinates):
        matched_pairs = list(zip(grouped_strings, coord_strings))
    
    return matched_pairs

def create_multi_scatter_plots(df: pd.DataFrame, str_cols: List[str], trouble_col: str, 
                        output_dir: str = 'results'):
    """
    Create scatter plots for string columns vs trouble column.
    """
    dir_name = f"{str_cols[0]}_{str_cols[1]}" #if len(str_cols) > 1 else str_cols[0]
    os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    

    scatter = plt.scatter(df[str_cols[0]], df[str_cols[1]], 
                            c=np.log10(df[trouble_col]),
                            cmap='viridis',
                            alpha=0.6)
        
    plt.xlabel(str_cols[0])
    plt.ylabel(str_cols[1])
    plt.title(f'{str_cols[0]} vs {str_cols[1]}\nColor: log10({trouble_col})')

    
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'log10({trouble_col})')
    
    plt.savefig(os.path.join(output_dir, dir_name, f'{dir_name}_scatter.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    cols_to_save = str_cols + [trouble_col]
    subset_df = df[cols_to_save]
    subset_df.to_csv(os.path.join(output_dir, dir_name, f'{dir_name}_data.csv'),
                    index=False)
    

def create_single_scatter_plots(df: pd.DataFrame, str_cols: List[str], trouble_col: str, 
                        output_dir: str = 'results'):
    """
    Create scatter plots for string columns vs trouble column.
    """
    dir_name = f"{str_cols[0]}_{trouble_col}}" #if len(str_cols) > 1 else str_cols[0]
    os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
 

    scatter = plt.scatter(df.iloc[,:0], df.ilc[,:1], 
                        c=np.log10(df[trouble_col]),
                        cmap='viridis',
                        alpha=0.6)
    
    plt.xlabel(str_cols[0])
    plt.ylabel(str_cols[1])
    plt.title(f'{str_cols[0]} vs {str_cols[1]}\nColor: log10({trouble_col})')


    cbar = plt.colorbar(scatter)
    cbar.set_label(f'log10({trouble_col})')
    
    plt.savefig(os.path.join(output_dir, dir_name, f'{dir_name}_scatter.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    cols_to_save = str_cols + [trouble_col]
    subset_df = df[cols_to_save]
    subset_df.to_csv(os.path.join(output_dir, dir_name, f'{dir_name}_data.csv'),
                    index=False)





def create_nonuniform_quantiles(n_quantiles: int, edge_density: float = 0.3) -> np.ndarray:
    """
    Create non-uniform quantile spacing with higher density at edges.
    
    Parameters:
    n_quantiles: Number of quantiles to create
    edge_density: Proportion of points to concentrate at edges (0-1)
    
    Returns:
    Array of quantile values between 0 and 1
    """
    # Create three sets of points:
    # 1. Dense points at the start
    # 2. Regular spacing in the middle
    # 3. Dense points at the end
    
    n_edge = int(n_quantiles * edge_density)  # points for each edge
    n_middle = n_quantiles - (2 * n_edge)     # points for the middle
    
    # Create edge points with exponential spacing
    start_points = np.exp(np.linspace(np.log(1e-6), np.log(0.1), n_edge))
    end_points = 1 - np.exp(np.linspace(np.log(0.1), np.log(1e-6), n_edge))
    
    # Create middle points with linear spacing
    middle_points = np.linspace(0.1, 0.9, n_middle)
    
    # Combine all points and ensure boundaries are included
    quantiles = np.concatenate([[0], start_points, middle_points, end_points, [1]])
    
    # Sort and remove any duplicates
    return np.unique(quantiles)

def quantile_minima(df: pd.DataFrame, x_col: str, z_col: str, 
                   n_quantiles: int = 70, 
                   edge_density: float = 0.3,
                   min_points: int = 3) -> pd.DataFrame:
    """
    Calculate quantile minima with increased density at the extremes.
    
    Parameters:
    df: DataFrame containing the data
    x_col: Column name for x-axis
    z_col: Column name for z-axis (values to minimize)
    n_quantiles: Number of quantiles to create
    edge_density: Proportion of points to concentrate at edges (0-1)
    min_points: Minimum number of points required in a quantile
    
    Returns:
    DataFrame with minimal z points from each quantile
    """
    # Create non-uniform quantile boundaries
    quantiles = create_nonuniform_quantiles(n_quantiles, edge_density)
    x_bounds = df[x_col].quantile(quantiles)
    
    minima = []
    for i in range(len(x_bounds)-1):
        mask = (df[x_col] >= x_bounds[i]) & (df[x_col] < x_bounds[i+1])
        if mask.sum() >= min_points:  # Only process if enough points in quantile
            quantile_data = df[mask]
            min_point = quantile_data.loc[quantile_data[z_col].idxmin()]
            minima.append({
                x_col: min_point[x_col],
                z_col: min_point[z_col],
                'n_points': mask.sum()  # Track number of points in quantile
            })
    
    result_df = pd.DataFrame(minima)
    
    # Add global minimum if not already included
    global_min = df.loc[df[z_col].idxmin()]
    min_point_dict = {
        x_col: global_min[x_col],
        z_col: global_min[z_col],
        'n_points': 1
    }
    
    if len(result_df) == 0 or not any((result_df[x_col] == global_min[x_col]) & 
                                     (result_df[z_col] == global_min[z_col])):
        result_df = pd.concat([result_df, pd.DataFrame([min_point_dict])], 
                            ignore_index=True)
    
    # Sort by x_col
    result_df = result_df.sort_values(x_col)
    
    return result_df





def create_minima_plots(df: pd.DataFrame, str_cols: List[str], trouble_col: str, 
                       output_dir: str = 'minima_results'):
    """
    Create minima plots with enhanced quantile analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(str_cols, str):
        str_cols = [str_cols] #force to array - otherwise takes first char in string in next loop
    
    for col in str_cols:
        minima_df = quantile_minima(df, col, trouble_col, n_quantiles=70, edge_density=0.3)
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(df[col], df[trouble_col], alpha=0.3, label='Original Data')
        
        plt.plot(minima_df[col], minima_df[trouble_col], 'r-', 
                label='Quantile Minima', linewidth=2)
        
        sizes = np.clip(minima_df['n_points'] * 5, 50, 200)
        plt.scatter(minima_df[col], minima_df[trouble_col], 
                   s=sizes, color='red', alpha=0.5,
                   label='Minima Points')
        
        plt.xlabel(col)
        plt.ylabel(trouble_col)
        plt.title(f'{col} vs {trouble_col} with Enhanced Quantile Minima')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'{col}_minima.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        minima_df.to_csv(os.path.join(output_dir, f'{col}_minima.csv'),
                        index=False)


def main(yaml_path: str, data_path: str):
    """
    Main function to process YAML and data files and create visualizations.
    Handles both single and multi-node cases.
    """
    # Load YAML data
    coordinates, strings = load_yaml(yaml_path)
    string_groups = group_strings(strings)
    matched_pairs = match_groups_to_coordinates(string_groups, coordinates)
    # Load data
    df = pd.read_csv(data_path)
    for string_group, trouble_col in matched_pairs:
        if len(string_group) >= 2:
            create_multi_scatter_plots(df, string_group, trouble_col)
            create_minima_plots(df, string_group, trouble_col)
        else:
            create_single_scatter_plots(df, string_group, trouble_col)
            create_minima_plots(df, string_group, trouble_col)
