import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import re
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    n_quantiles: int = 70
    edge_density: float = 0.3
    min_points: int = 3
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300
    alpha: float = 0.6

class DataVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize DataVisualizer with optional configuration."""
        self.config = config or VisualizationConfig()
        self.df: Optional[pd.DataFrame] = None
    
    @staticmethod
    def load_yaml(yaml_path: str) -> Tuple[List[List[int]], List[str]]:
        """Load and parse YAML configuration file."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            return data['coordinate_list'], data['test_list']
    
    @staticmethod
    def group_strings(strings: List[str]) -> Dict[str, List[str]]:
        """Group strings based on prefix and suffix patterns."""
        groups = {}
        if len(strings) <= 2:
            return {s: [s] for s in strings}
            
        for s in strings:
            match = re.match(r'([a-zA-Z]+_).*?(_[a-zA-Z]+)$', s)
            if match:
                prefix, suffix = match.groups()
                pattern = f"{prefix}*{suffix}"
                if pattern not in groups:
                    groups[pattern] = []
                groups[pattern].append(s)
        
        return groups if groups else {s: [s] for s in strings}
    
    @staticmethod
    def coordinate_to_string(coord: List[int]) -> str:
        """Convert coordinate list to string representation."""
        return f"Troublesome {''.join(map(str, coord))}"
    
    def match_groups_to_coordinates(
        self,
        string_groups: Dict[str, List[str]],
        coordinates: List[List[int]]
    ) -> List[Tuple[List[str], str]]:
        """Match string groups to coordinates."""
        if len(coordinates) == 1:
            return [(group, 'Desired Symb') for group in string_groups.values()]
            
        coord_strings = [self.coordinate_to_string(coord) for coord in coordinates]
        grouped_strings = list(string_groups.values())
        
        return list(zip(grouped_strings, coord_strings)) if len(grouped_strings) == len(coordinates) else []

    def _create_output_directory(self, base_dir: str, subdirs: List[str]) -> str:
        """Create nested output directory structure."""
        path = os.path.join(base_dir, *subdirs)
        os.makedirs(path, exist_ok=True)
        return path
    
    def _setup_plot(self, xlabel: str, ylabel: str, title: str) -> None:
        """Set up common plot parameters."""
        plt.figure(figsize=self.config.figure_size)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    
    def create_scatter_plot(
        self,
        x_col: str,
        y_col: str,
        color_col: str,
        output_dir: str = 'results'
    ) -> None:
        """Create and save scatter plot with color mapping."""
        if self.df is None:
            raise ValueError("DataFrame not loaded. Call load_data first.")
            
        dir_name = f"{x_col}_{y_col}"
        output_path = self._create_output_directory(output_dir, [dir_name])
        
        self._setup_plot(
            xlabel=x_col,
            ylabel=y_col,
            title=f'{x_col} vs {y_col}\nColor: log10({color_col})'
        )
        
        scatter = plt.scatter(
            self.df[x_col],
            self.df[y_col],
            c=np.log10(self.df[color_col]),
            cmap='viridis',
            alpha=self.config.alpha
        )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'log10({color_col})')
        
        plt.savefig(
            os.path.join(output_path, f'{dir_name}_scatter.png'),
            dpi=self.config.dpi,
            bbox_inches='tight'
        )
        plt.close()
        
        # Save associated data
        self.df[[x_col, y_col, color_col]].to_csv(
            os.path.join(output_path, f'{dir_name}_data.csv'),
            index=False
        )
    
    def _create_nonuniform_quantiles(self) -> np.ndarray:
        """Create non-uniform quantile spacing with higher density at edges."""
        n_edge = int(self.config.n_quantiles * self.config.edge_density)
        n_middle = self.config.n_quantiles - (2 * n_edge)
        
        start_points = np.exp(np.linspace(np.log(1e-6), np.log(0.1), n_edge))
        end_points = 1 - np.exp(np.linspace(np.log(0.1), np.log(1e-6), n_edge))
        middle_points = np.linspace(0.1, 0.9, n_middle)
        
        quantiles = np.concatenate([[0], start_points, middle_points, end_points, [1]])
        return np.unique(quantiles)
    
    def calculate_quantile_minima(
        self,
        x_col: str,
        z_col: str
    ) -> pd.DataFrame:
        """Calculate quantile minima with increased density at extremes."""
        if self.df is None:
            raise ValueError("DataFrame not loaded. Call load_data first.")
            
        quantiles = self._create_nonuniform_quantiles()
        x_bounds = self.df[x_col].quantile(quantiles)
        
        minima = []
        for i in range(len(x_bounds)-1):
            mask = (self.df[x_col] >= x_bounds[i]) & (self.df[x_col] < x_bounds[i+1])
            if mask.sum() >= self.config.min_points:
                quantile_data = self.df[mask]
                min_idx = quantile_data[z_col].idxmin()
                min_point = quantile_data.loc[min_idx]
                minima.append({
                    x_col: min_point[x_col],
                    z_col: min_point[z_col],
                    'n_points': mask.sum()
                })
        
        result_df = pd.DataFrame(minima)
        
        # Add global minimum if not already included
        global_min = self.df.loc[self.df[z_col].idxmin()]
        min_point_dict = {
            x_col: global_min[x_col],
            z_col: global_min[z_col],
            'n_points': 1
        }
        
        if len(result_df) == 0 or not any(
            (result_df[x_col] == global_min[x_col]) & 
            (result_df[z_col] == global_min[z_col])
        ):
            result_df = pd.concat(
                [result_df, pd.DataFrame([min_point_dict])],
                ignore_index=True
            )
        
        return result_df.sort_values(x_col)
    
    def create_minima_plot(
        self,
        x_col: str,
        z_col: str,
        output_dir: str = 'minima_results'
    ) -> None:
        """Create and save minima plot with enhanced quantile analysis."""
        if self.df is None:
            raise ValueError("DataFrame not loaded. Call load_data first.")
            
        output_path = self._create_output_directory(output_dir, [])
        minima_df = self.calculate_quantile_minima(x_col, z_col)
        
        self._setup_plot(
            xlabel=x_col,
            ylabel=z_col,
            title=f'{x_col} vs {z_col} with Enhanced Quantile Minima'
        )
        
        plt.scatter(
            self.df[x_col],
            self.df[z_col],
            alpha=0.3,
            label='Original Data'
        )
        
        plt.plot(
            minima_df[x_col],
            minima_df[z_col],
            'r-',
            label='Quantile Minima',
            linewidth=2
        )
        
        sizes = np.clip(minima_df['n_points'] * 5, 50, 200)
        plt.scatter(
            minima_df[x_col],
            minima_df[z_col],
            s=sizes,
            color='red',
            alpha=0.5,
            label='Minima Points'
        )
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(
            os.path.join(output_path, f'{x_col}_minima.png'),
            dpi=self.config.dpi,
            bbox_inches='tight'
        )
        plt.close()
        
        minima_df.to_csv(
            os.path.join(output_path, f'{x_col}_minima.csv'),
            index=False
        )
    
    def process_data(
        self,
        yaml_path: str,
        data_path: str
    ) -> None:
        """Process YAML and data files to create visualizations."""
        coordinates, strings = self.load_yaml(yaml_path)
        string_groups = self.group_strings(strings)
        matched_pairs = self.match_groups_to_coordinates(string_groups, coordinates)
        
        self.df = pd.read_csv(data_path)
        
        for string_group, trouble_col in matched_pairs:
            if isinstance(string_group, list) and len(string_group) >= 2:
                # Multi-column case
                for col1, col2 in zip(string_group[:-1], string_group[1:]):
                    self.create_scatter_plot(col1, col2, trouble_col)
                    self.create_minima_plot(col1, trouble_col)
                    self.create_minima_plot(col2, trouble_col)
            else:
                # Single column case
                col = string_group[0] if isinstance(string_group, list) else string_group
                self.create_scatter_plot(col, trouble_col, trouble_col)
                self.create_minima_plot(col, trouble_col)

# Example usage:
if __name__ == "__main__":
    config = VisualizationConfig(
        n_quantiles=70,
        edge_density=0.3,
        min_points=3,
        figure_size=(10, 8),
        dpi=300,
        alpha=0.6
    )
    
    visualizer = DataVisualizer(config)
    visualizer.process_data('config.yaml', 'data.csv')