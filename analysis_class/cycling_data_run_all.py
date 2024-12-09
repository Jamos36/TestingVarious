import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
from scipy.stats import pearsonr
from scipy.spatial.distance import directed_hausdorff

def load_and_analyze_runs(base_path):
    """
    Load and analyze data from multiple runs with different numbers of points.
    
    Parameters:
    base_path: str, path to the root directory containing run folders
    
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
            minima = quantile_minima(df)  # Using your original function
            
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
    Plot results from multiple runs on one graph.
    
    Parameters:
    results: dict, output from load_and_analyze_runs
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (num_points, data), color in zip(sorted(results.items()), colors):
        df = data['df']
        minima = data['minima']
        
        # Plot original data with low alpha
        plt.scatter(df['y'], df['z'], alpha=0.1, color=color)
        
        # Plot minima line
        plt.plot(minima['y'], minima['z'], 
                label=f'n={num_points}', 
                color=color, 
                linewidth=2)
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Intensity')
    plt.title('Comparison of Minima Lines for Different Sample Sizes')
    plt.legend()
    plt.show()

def calculate_convergence_metrics(results):
    """
    Calculate metrics to assess convergence between different sample sizes.
    
    Parameters:
    results: dict, output from load_and_analyze_runs
    
    Returns:
    dict: Dictionary containing convergence metrics
    """
    metrics = {}
    sorted_nums = sorted(results.keys())
    
    for i in range(len(sorted_nums)-1):
        current_n = sorted_nums[i]
        next_n = sorted_nums[i+1]
        
        current_minima = results[current_n]['minima']
        next_minima = results[next_n]['minima']
        
        # Interpolate to common x-points for comparison
        common_y = np.linspace(
            max(current_minima['y'].min(), next_minima['y'].min()),
            min(current_minima['y'].max(), next_minima['y'].max()),
            1000
        )
        
        current_interp = np.interp(common_y, current_minima['y'], current_minima['z'])
        next_interp = np.interp(common_y, next_minima['y'], next_minima['z'])
        
        # Calculate various metrics
        metrics[f"{current_n}_to_{next_n}"] = {
            'correlation': pearsonr(current_interp, next_interp)[0],
            'mse': np.mean((current_interp - next_interp)**2),
            'hausdorff': directed_hausdorff(
                np.column_stack((common_y, current_interp)),
                np.column_stack((common_y, next_interp))
            )[0],
            'max_diff': np.max(np.abs(current_interp - next_interp)),
            'relative_change': np.mean(np.abs(current_interp - next_interp)) / np.mean(np.abs(current_interp))
        }
    
    return metrics

def find_optimal_points(results, threshold=0.01):
    """
    Find the optimal number of points based on convergence criteria.
    
    Parameters:
    results: dict, output from load_and_analyze_runs
    threshold: float, relative change threshold for convergence
    
    Returns:
    int: Optimal number of points
    dict: Detailed analysis of the optimization
    """
    metrics = calculate_convergence_metrics(results)
    
    # Analyze convergence
    convergence_analysis = {}
    sorted_nums = sorted(results.keys())
    
    for i in range(len(sorted_nums)-1):
        current_n = sorted_nums[i]
        next_n = sorted_nums[i+1]
        
        metric_key = f"{current_n}_to_{next_n}"
        relative_change = metrics[metric_key]['relative_change']
        
        convergence_analysis[current_n] = {
            'next_n': next_n,
            'relative_change': relative_change,
            'converged': relative_change < threshold,
            'metrics': metrics[metric_key]
        }
        
        # If we've reached convergence, this is our optimal point
        if relative_change < threshold:
            return next_n, convergence_analysis
    
    # If we haven't converged, return the largest number we have
    return sorted_nums[-1], convergence_analysis

def plot_convergence_analysis(convergence_analysis):
    """
    Plot the convergence analysis results.
    
    Parameters:
    convergence_analysis: dict, output from find_optimal_points
    """
    points = sorted(convergence_analysis.keys())
    relative_changes = [convergence_analysis[n]['relative_change'] for n in points]
    
    plt.figure(figsize=(10, 6))
    plt.plot(points, relative_changes, 'o-')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold')
    plt.xlabel('Number of Points')
    plt.ylabel('Relative Change')
    plt.title('Convergence Analysis')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_data_convergence(base_path, threshold=0.01):
    """
    Main function to analyze convergence of multiple runs.
    
    Parameters:
    base_path: str, path to the root directory containing run folders
    threshold: float, relative change threshold for convergence
    
    Returns:
    tuple: (optimal points, full results, convergence analysis)
    """
    # Load and analyze all runs
    results = load_and_analyze_runs(base_path)
    
    if not results:
        raise ValueError("No valid data files found")
    
    # Plot combined results
    plot_combined_results(results)
    
    # Find optimal number of points
    optimal_points, convergence_analysis = find_optimal_points(results, threshold)
    
    # Plot convergence analysis
    plot_convergence_analysis(convergence_analysis)
    
    print(f"\nOptimal number of points: {optimal_points}")
    print("\nConvergence analysis:")
    for n, analysis in convergence_analysis.items():
        print(f"\nPoints: {n} -> {analysis['next_n']}")
        print(f"Relative change: {analysis['relative_change']:.6f}")
        print(f"Converged: {analysis['converged']}")
    
    return optimal_points, results, convergence_analysis

# Example usage:
if __name__ == "__main__":
    base_path = r"\user\runs"  # Replace with actual path
    optimal_points, results, convergence = analyze_data_convergence(base_path)