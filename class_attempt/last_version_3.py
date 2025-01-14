import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataVisualizer:
    """Class for creating and managing data visualizations with quantile analysis."""
    
    output_dir: str = 'results'
    n_quantiles: int = 70
    edge_density: float = 0.3
    min_points: int = 3
    
    def __post_init__(self):
        """Create output directory after initialization."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    @staticmethod
    def load_yaml(yaml_path: str) -> Tuple[List[List[int]], List[str]]:
        """Load and parse YAML configuration file."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            return data['coordinate_list'], data['test_list']
    
    @staticmethod
    def group_strings(strings: List[str]) -> Dict[str, List[str]]:
        """Group strings based on common prefix and suffix patterns."""
        groups = {}
        if len(strings) <= 2:
            return {strings[0]: strings}
            
        for s in strings:
            match = re.match(r'([a-zA-Z]+_).*?(_[a-zA-Z]+)$', s)
            if match:
                prefix, suffix = match.groups()
                pattern = f"{prefix}*{suffix}"
                if pattern not in groups:
                    groups[pattern] = []
                groups[pattern].append(s)
        return groups
    
    @staticmethod
    def coordinate_to_string(coord: List[int]) -> str:
        """Convert coordinate list to string representation."""
        return f"Troublesome {''.join(map(str, coord))}"
    
    def match_groups_to_coordinates(self, string_groups: Dict[str, List[str]], 
                                  coordinates: List[List[int]]) -> List[Tuple[List[str], str]]:
        """Match string groups to coordinates."""
        if len(coordinates) == 1:
            return [(list(string_groups.values())[0], 'Desired_Symbol')]
            
        coord_strings = [self.coordinate_to_string(coord) for coord in coordinates]
        grouped_strings = list(string_groups.values())
        
        if len(grouped_strings) == len(coordinates):
            return list(zip(grouped_strings, coord_strings))
        return []
    
    def create_nonuniform_quantiles(self) -> np.ndarray:
        """Create non-uniform quantile spacing with higher density at edges."""
        n_edge = int(self.n_quantiles * self.edge_density)
        n_middle = self.n_quantiles - (2 * n_edge)
        
        start_points = np.exp(np.linspace(np.log(1e-6), np.log(0.1), n_edge))
        end_points = 1 - np.exp(np.linspace(np.log(0.1), np.log(1e-6), n_edge))
        middle_points = np.linspace(0.1, 0.9, n_middle)
        
        return np.unique(np.concatenate([[0], start_points, middle_points, end_points, [1]]))
    
    def calculate_quantile_minima(self, df: pd.DataFrame, x_col: str, 
                                z_col: str) -> pd.DataFrame:
        """Calculate quantile minima with increased density at extremes."""
        quantiles = self.create_nonuniform_quantiles()
        x_bounds = df[x_col].quantile(quantiles)
        
        minima = []
        for i in range(len(x_bounds)-1):
            mask = (df[x_col] >= x_bounds[i]) & (df[x_col] < x_bounds[i+1])
            if mask.sum() >= self.min_points:
                quantile_data = df[mask]
                min_point = quantile_data.loc[quantile_data[z_col].idxmin()]
                minima.append({
                    x_col: min_point[x_col],
                    z_col: min_point[z_col],
                    'n_points': mask.sum()
                })
        
        result_df = pd.DataFrame(minima)
        
        # Add global minimum if not already included
        global_min = df.loc[df[z_col].idxmin()]
        min_point = {
            x_col: global_min[x_col],
            z_col: global_min[z_col],
            'n_points': 1
        }
        
        if len(result_df) == 0 or not any((result_df[x_col] == global_min[x_col]) & 
                                         (result_df[z_col] == global_min[z_col])):
            result_df = pd.concat([result_df, pd.DataFrame([min_point])], 
                                ignore_index=True)
        
        return result_df.sort_values(x_col)
    
    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                    trouble_col: str, output_path: Path):
        """Create scatter plot with colorbar."""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df[x_col], df[y_col], 
                            c=np.log10(df[trouble_col]),
                            cmap='viridis', alpha=0.6)
        
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col}\nColor: log10({trouble_col})')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'log10({trouble_col})')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_minima(self, df: pd.DataFrame, x_col: str, trouble_col: str):
        """Create minima plots with enhanced quantile analysis."""
        minima_df = self.calculate_quantile_minima(df, x_col, trouble_col)
        output_dir = Path(self.output_dir) / 'minima'
        output_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(df[x_col], df[trouble_col], alpha=0.3, label='Original Data')
        plt.plot(minima_df[x_col], minima_df[trouble_col], 'r-', 
                label='Quantile Minima', linewidth=2)
        
        sizes = np.clip(minima_df['n_points'] * 5, 50, 200)
        plt.scatter(minima_df[x_col], minima_df[trouble_col], 
                   s=sizes, color='red', alpha=0.5, label='Minima Points')
        
        plt.xlabel(x_col)
        plt.ylabel(trouble_col)
        plt.title(f'{x_col} vs {trouble_col} with Enhanced Quantile Minima')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / f'{x_col}_minima.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        minima_df.to_csv(output_dir / f'{x_col}_minima.csv', index=False)
    
    def process_data(self, yaml_path: str, data_path: str):
        """Process YAML and data files to create visualizations."""
        coordinates, strings = self.load_yaml(yaml_path)
        string_groups = self.group_strings(strings)
        matched_pairs = self.match_groups_to_coordinates(string_groups, coordinates)
        
        df = pd.read_csv(data_path)
        
        for string_group, trouble_col in matched_pairs:
            if len(string_group) >= 2:
                for x_col, y_col in zip(string_group[:-1], string_group[1:]):
                    output_dir = Path(self.output_dir) / f"{x_col}_{y_col}"
                    output_dir.mkdir(exist_ok=True)
                    
                    self.plot_scatter(df, x_col, y_col, trouble_col, 
                                    output_dir / f'{x_col}_{y_col}_scatter.png')
                    
                    subset_df = df[[x_col, y_col, trouble_col]]
                    subset_df.to_csv(output_dir / f'{x_col}_{y_col}_data.csv', 
                                   index=False)
            
            # Create minima plots for all columns
            for col in string_group:
                self.plot_minima(df, col, trouble_col)