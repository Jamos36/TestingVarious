import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
from scipy.stats import pearsonr
from scipy.spatial.distance import directed_hausdorff
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging

@dataclass
class AnalysisResult:
    """Data class to store analysis results."""
    minima: Dict
    metrics: Dict
    optimal_points: Dict
    convergence_analysis: Dict
    raw_data: Dict

class QuantileAnalyzer:
    """Class for analyzing multidimensional data using quantile-based methods."""
    
    def __init__(self, n_quantiles: int = 10, threshold: float = 0.01):
        """
        Initialize the analyzer.
        
        Args:
            n_quantiles: Number of quantiles for analysis
            threshold: Convergence threshold
        """
        self.n_quantiles = n_quantiles
        self.threshold = threshold
        self.logger = self._setup_logger()
        self.results = None
        
    @staticmethod
    def _setup_logger():
        """Set up logging configuration."""
        logger = logging.getLogger('QuantileAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_single_dataset(self, df: pd.DataFrame, show_plot: bool = True) -> Dict:
        """
        Analyze a single dataset using quantile-based method.
        
        Args:
            df: DataFrame with coordinates and intensity
            show_plot: Whether to display plots
            
        Returns:
            Dictionary of results per dimension
        """
        intensity_col = df.columns[-1]
        coord_cols = df.columns[:-1]
        quantiles = np.linspace(0, 1, self.n_quantiles + 1)
        results = {}
        
        if show_plot:
            n_dims = len(coord_cols)
            fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 5))
            if n_dims == 1:
                axes = [axes]
        
        for idx, coord in enumerate(coord_cols):
            results[coord] = self._analyze_dimension(df, coord, intensity_col, quantiles)
            
            if show_plot:
                self._plot_dimension(df, results[coord], coord, intensity_col, 
                                   quantiles, axes[idx])
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return results

    def _analyze_dimension(self, df: pd.DataFrame, coord: str, 
                         intensity_col: str, quantiles: np.ndarray) -> pd.DataFrame:
        """Analyze a single dimension of the dataset."""
        bounds = df[coord].quantile(quantiles)
        minima_df = pd.DataFrame(columns=[coord, intensity_col])
        
        # Find minima in each quantile
        for i in range(len(bounds)-1):
            mask = (df[coord] >= bounds[i]) & (df[coord] < bounds[i+1])
            if mask.any():
                quantile_data = df[mask]
                min_point_idx = quantile_data[intensity_col].idxmin()
                min_point = quantile_data.loc[min_point_idx, [coord, intensity_col]]
                minima_df = pd.concat([minima_df, pd.DataFrame([min_point])],
                                    ignore_index=True)
        
        # Add global minimum
        global_min_idx = df[intensity_col].idxmin()
        global_min = df.loc[global_min_idx, [coord, intensity_col]]
        if not any((minima_df[coord] == global_min[coord]) & 
                  (minima_df[intensity_col] == global_min[intensity_col])):
            minima_df = pd.concat([minima_df, pd.DataFrame([global_min])],
                                ignore_index=True)
        
        return minima_df.sort_values(coord)

    def _plot_dimension(self, df: pd.DataFrame, minima_df: pd.DataFrame, 
                       coord: str, intensity_col: str, quantiles: np.ndarray, 
                       ax: plt.Axes) -> None:
        """Plot analysis results for a single dimension."""
        scatter = ax.scatter(df[coord], df[intensity_col], 
                           c=df[intensity_col], cmap='viridis',
                           alpha=0.5, label='Original Data')
        ax.scatter(minima_df[coord], minima_df[intensity_col], 
                  color='red', s=100, label='Quantile Minima')
        
        # Add quantile boundaries
        bounds = df[coord].quantile(quantiles)
        for bound in bounds:
            ax.axvline(bound, color='gray', linestyle='--', alpha=0.3)
            
        ax.set_xlabel(f'{coord} Coordinate')
        ax.set_ylabel(f'{intensity_col}')
        ax.set_title(f'{intensity_col} vs {coord}')
        ax.legend()
        plt.colorbar(scatter, ax=ax)

    def analyze_multiple_runs(self, base_path: str) -> AnalysisResult:
        """
        Analyze multiple runs from different datasets.
        
        Args:
            base_path: Path to directory containing run folders
            
        Returns:
            AnalysisResult object containing all analysis results
        """
        self.logger.info(f"Starting analysis of multiple runs in {base_path}")
        results = self._load_runs(base_path)
        
        if not results:
            raise ValueError("No valid data files found")
        
        # Analyze convergence
        metrics = self._calculate_convergence_metrics(results)
        optimal_points, convergence_analysis = self._find_optimal_points(results)
        
        # Store results
        self.results = AnalysisResult(
            minima={n: data['minima'] for n, data in results.items()},
            metrics=metrics,
            optimal_points=optimal_points,
            convergence_analysis=convergence_analysis,
            raw_data=results
        )
        
        return self.results

    def _load_runs(self, base_path: str) -> Dict:
        """Load and analyze multiple run directories."""
        results = {}
        base_path = Path(base_path)
        
        run_dirs = glob.glob(str(base_path / "**" / "somerun_*"), recursive=True)
        for run_dir in run_dirs:
            try:
                num_points = int(re.search(r'somerun_(\d+)', run_dir).group(1))
                csv_path = Path(run_dir) / f"data_{num_points}.csv"
                
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    minima = self.analyze_single_dataset(df, show_plot=False)
                    results[num_points] = {'df': df, 'minima': minima}
                    self.logger.info(f"Successfully processed {csv_path}")
            except Exception as e:
                self.logger.error(f"Error processing {run_dir}: {e}")
                
        return results

    def _calculate_convergence_metrics(self, results: Dict) -> Dict:
        """Calculate convergence metrics between consecutive runs."""
        metrics = {}
        sorted_nums = sorted(results.keys())
        
        for i in range(len(sorted_nums)-1):
            current_n = sorted_nums[i]
            next_n = sorted_nums[i+1]
            metrics[f"{current_n}_to_{next_n}"] = {}
            
            for dim in results[current_n]['minima'].keys():
                metrics[f"{current_n}_to_{next_n}"][dim] = self._calculate_dimension_metrics(
                    results[current_n]['minima'][dim],
                    results[next_n]['minima'][dim]
                )
                
        return metrics

    def _calculate_dimension_metrics(self, current_minima: pd.DataFrame, 
                                   next_minima: pd.DataFrame) -> Dict:
        """Calculate metrics for a single dimension."""
        dim = current_minima.columns[0]
        intensity_col = current_minima.columns[-1]
        
        common_x = np.linspace(
            max(current_minima[dim].min(), next_minima[dim].min()),
            min(current_minima[dim].max(), next_minima[dim].max()),
            1000
        )
        
        current_interp = np.interp(common_x, current_minima[dim], 
                                 current_minima[intensity_col])
        next_interp = np.interp(common_x, next_minima[dim], 
                               next_minima[intensity_col])
        
        return {
            'correlation': pearsonr(current_interp, next_interp)[0],
            'mse': np.mean((current_interp - next_interp)**2),
            'hausdorff': directed_hausdorff(
                np.column_stack((common_x, current_interp)),
                np.column_stack((common_x, next_interp))
            )[0],
            'max_diff': np.max(np.abs(current_interp - next_interp)),
            'relative_change': np.mean(np.abs(current_interp - next_interp)) / 
                             np.mean(np.abs(current_interp))
        }

    def _find_optimal_points(self, results: Dict) -> Tuple[Dict, Dict]:
        """Find optimal number of points for each dimension."""
        convergence_analysis = {}
        optimal_points = {}
        
        for dim in next(iter(results.values()))['minima'].keys():
            convergence_analysis[dim] = self._analyze_dimension_convergence(
                results, dim)
            optimal_points[dim] = self._find_dimension_optimal_points(
                convergence_analysis[dim])
            
        return optimal_points, convergence_analysis

    def plot_results(self) -> None:
        """Plot all analysis results."""
        if self.results is None:
            raise ValueError("No results available. Run analysis first.")
            
        self._plot_combined_results()
        self._plot_convergence_analysis()
        self._print_summary()

    def _plot_combined_results(self) -> None:
        """Plot combined results for all runs."""
        for dim in next(iter(self.results.raw_data.values()))['minima'].keys():
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.results.raw_data)))
            
            for (num_points, data), color in zip(
                sorted(self.results.raw_data.items()), colors):
                df = data['df']
                minima = data['minima'][dim]
                
                plt.scatter(df[dim], df[df.columns[-1]], alpha=0.1, color=color)
                plt.plot(minima[dim], minima[df.columns[-1]], 
                        label=f'n={num_points}', color=color, linewidth=2)
            
            plt.xlabel(f'{dim} Coordinate')
            plt.ylabel('Intensity')
            plt.title(f'Comparison of Minima Lines for {dim}')
            plt.legend()
            plt.show()

    def _plot_convergence_analysis(self) -> None:
        """Plot convergence analysis for each dimension."""
        for dim, analysis in self.results.convergence_analysis.items():
            points = sorted(analysis.keys())
            relative_changes = [analysis[n]['relative_change'] for n in points]
            
            plt.figure(figsize=(10, 6))
            plt.plot(points, relative_changes, 'o-', label=dim)
            plt.axhline(y=self.threshold, color='r', linestyle='--', 
                       label='Convergence Threshold')
            plt.xlabel('Number of Points')
            plt.ylabel('Relative Change')
            plt.title(f'Convergence Analysis for {dim}')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            plt.show()

    def _print_summary(self) -> None:
        """Print summary of analysis results."""
        print("\nAnalysis Summary")
        print("================")
        print("\nOptimal number of points per dimension:")
        for dim, points in self.results.optimal_points.items():
            print(f"{dim}: {points}")
            
        print("\nConvergence analysis:")
        for dim, analysis in self.results.convergence_analysis.items():
            print(f"\nDimension: {dim}")
            for n, data in analysis.items():
                print(f"Points: {n} -> {data['next_n']}")
                print(f"Relative change: {data['relative_change']:.6f}")
                print(f"Converged: {data['converged']}")

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
    df['intensity'] += df['x']**2 + df['y']**2
    
    # Create analyzer instance
    analyzer = QuantileAnalyzer(n_quantiles=10, threshold=0.01)
    
    # Single dataset analysis
    single_results = analyzer.analyze_single_dataset(df)
    
    # Multi-run analysis
    base_path = r"\user\runs"  # Replace with actual path
    multi_results = analyzer.analyze_multiple_runs(base_path)
    analyzer.plot_results()