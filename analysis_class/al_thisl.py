import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
from scipy.stats import pearsonr
from scipy.spatial.distance import directed_hausdorff

def quantile_minima_multidim(df, n_quantiles=10, show_plot=True):
    """
    Find local minima using quantile-based sampling across multiple dimensions.
    
    Parameters:
    df: DataFrame where the last column is the intensity value
        and all other columns are coordinates
    n_quantiles: Number of quantiles to create
    show_plot: Boolean to control whether to display the plot
    
    Returns:
    Dictionary of DataFrames containing minimal intensity points for each dimension
    """
    # Get intensity column name (last column)
    intensity_col = df.columns[-1]
    coord_cols = df.columns[:-1]
    
    # Calculate quantile boundaries for each dimension
    quantiles = np.linspace(0, 1, n_quantiles+1)
    
    # Create figure with subplots for each dimension if showing plot
    if show_plot:
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
        
        if show_plot:
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
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    return results

def load_and_analyze_runs(base_path, n_quantiles=10):
    """
    Load and analyze data from multiple runs with different numbers of points.
    
    Parameters:
    base_path: str, path to the root directory containing run folders
    n_quantiles: Number of quantiles for analysis
    
    Returns:
    dict: Dictionary containing analyzed data for each run
    """
    results = {}
    base_path = Path(base_path)
    
    # Find all run directories
    run_dirs = glob.glob(str(base_path / "**" / "somerun_*"), recursive=True)
    
    for run_dir in run_dirs:
        # Extract the number from the directory name
        num_match = re.search(r'somerun_(\d+)', run_dir)
        if not num_match:
            continue
            
        num_points = int(num_match.group(1))
        csv_path = Path(run_dir) / f"data_{num_points}.csv"
        
        # Check if CSV exists
        if not csv_path.exists():
            continue
            
        try:
            # Load and analyze the data
            df = pd.read_csv(csv_path)
            minima = quantile_minima_multidim(df, n_quantiles=n_quantiles, show_plot=False)
            
            results[num_points] = {
                'df': df,
                'minima': minima
            }
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue
    
    return results

def plot_combined_results(results):
    """
    Plot results from multiple runs on one graph for each dimension.
    
    Parameters:
    results: dict, output from load_and_analyze_runs
    """
    # Get all dimensions from the first result
    first_result = next(iter(results.values()))
    dimensions = list(first_result['minima'].keys())
    
    # Create a plot for each dimension
    for dim in dimensions:
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        for (num_points, data), color in zip(sorted(results.items()), colors):
            df = data['df']
            minima = data['minima'][dim]
            
            # Plot original data with low alpha
            plt.scatter(df[dim], df[df.columns[-1]], alpha=0.1, color=color)
            
            # Plot minima line
            plt.plot(minima[dim], minima[df.columns[-1]], 
                    label=f'n={num_points}', 
                    color=color, 
                    linewidth=2)
        
        plt.xlabel(f'{dim} Coordinate')
        plt.ylabel('Intensity')
        plt.title(f'Comparison of Minima Lines for {dim} vs Intensity')
        plt.legend()
        plt.show()

def calculate_convergence_metrics(results):
    """Calculate metrics to assess convergence between different sample sizes."""
    metrics = {}
    sorted_nums = sorted(results.keys())
    
    for i in range(len(sorted_nums)-1):
        current_n = sorted_nums[i]
        next_n = sorted_nums[i+1]
        
        metrics[f"{current_n}_to_{next_n}"] = {}
        
        # Calculate metrics for each dimension
        for dim in results[current_n]['minima'].keys():
            current_minima = results[current_n]['minima'][dim]
            next_minima = results[next_n]['minima'][dim]
            
            # Interpolate to common x-points for comparison
            common_x = np.linspace(
                max(current_minima[dim].min(), next_minima[dim].min()),
                min(current_minima[dim].max(), next_minima[dim].max()),
                1000
            )
            
            intensity_col = current_minima.columns[-1]
            current_interp = np.interp(common_x, 
                                     current_minima[dim], 
                                     current_minima[intensity_col])
            next_interp = np.interp(common_x, 
                                  next_minima[dim], 
                                  next_minima[intensity_col])
            
            metrics[f"{current_n}_to_{next_n}"][dim] = {
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
    
    return metrics

def find_optimal_points(results, threshold=0.01):
    """Find the optimal number of points based on convergence criteria."""
    metrics = calculate_convergence_metrics(results)
    
    # Analyze convergence for each dimension
    convergence_analysis = {}
    sorted_nums = sorted(results.keys())
    
    for dim in next(iter(results.values()))['minima'].keys():
        convergence_analysis[dim] = {}
        
        for i in range(len(sorted_nums)-1):
            current_n = sorted_nums[i]
            next_n = sorted_nums[i+1]
            
            metric_key = f"{current_n}_to_{next_n}"
            relative_change = metrics[metric_key][dim]['relative_change']
            
            convergence_analysis[dim][current_n] = {
                'next_n': next_n,
                'relative_change': relative_change,
                'converged': relative_change < threshold,
                'metrics': metrics[metric_key][dim]
            }
            
            # If we've reached convergence for this dimension
            if relative_change < threshold:
                break
    
    # Find the optimal points considering all dimensions
    optimal_points = {}
    for dim, analysis in convergence_analysis.items():
        # Find the first convergence point for this dimension
        for n, data in analysis.items():
            if data['converged']:
                optimal_points[dim] = data['next_n']
                break
        else:
            optimal_points[dim] = sorted_nums[-1]
    
    return optimal_points, convergence_analysis

def plot_convergence_analysis(convergence_analysis):
    """Plot the convergence analysis results for each dimension."""
    for dim, analysis in convergence_analysis.items():
        points = sorted(analysis.keys())
        relative_changes = [analysis[n]['relative_change'] for n in points]
        
        plt.figure(figsize=(10, 6))
        plt.plot(points, relative_changes, 'o-', label=dim)
        plt.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold')
        plt.xlabel('Number of Points')
        plt.ylabel('Relative Change')
        plt.title(f'Convergence Analysis for {dim}')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.show()

def analyze_data_convergence(base_path, n_quantiles=10, threshold=0.01):
    """Main function to analyze convergence of multiple runs."""
    # Load and analyze all runs
    results = load_and_analyze_runs(base_path, n_quantiles)
    
    if not results:
        raise ValueError("No valid data files found")
    
    # Plot combined results
    plot_combined_results(results)
    
    # Find optimal number of points
    optimal_points, convergence_analysis = find_optimal_points(results, threshold)
    
    # Plot convergence analysis
    plot_convergence_analysis(convergence_analysis)
    
    print("\nOptimal number of points per dimension:")
    for dim, points in optimal_points.items():
        print(f"{dim}: {points}")
    
    print("\nConvergence analysis:")
    for dim, analysis in convergence_analysis.items():
        print(f"\nDimension: {dim}")
        for n, data in analysis.items():
            print(f"Points: {n} -> {data['next_n']}")
            print(f"Relative change: {data['relative_change']:.6f}")
            print(f"Converged: {data['converged']}")
    
    return optimal_points, results, convergence_analysis

# Example usage:
if __name__ == "__main__":
    # Example 1: Single run analysis
    np.random.seed(42)
    n_points = 1000
    
    data = {
        'x': np.random.normal(0, 1, n_points),
        'y': np.random.normal(0, 1, n_points),
        'intensity': np.random.normal(0, 1, n_points)
    }
    df = pd.DataFrame(data)
    df['intensity'] += df['x']**2 + df['y']**2
    
    # Run single analysis
    results_single = quantile_minima_multidim(df, n_quantiles=10)
    
    # Example 2: Multi-run analysis
    base_path = r"\user\runs"  # Replace with your actual path
    optimal_points, results_multi, convergence = analyze_data_convergence(
        base_path, n_quantiles=10, threshold=0.01
    )