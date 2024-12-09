import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec

def quantile_minima_multidim(df, n_quantiles=10):
    """Original function from before - unchanged"""
    # [Previous implementation remains the same]
    pass

def interpolate_minima(original_df, minima_results, n_points=1000):
    """
    Interpolate between minima points to create a smooth curve.
    
    Parameters:
    original_df: Original DataFrame with all data points
    minima_results: Dictionary of DataFrames from quantile_minima_multidim
    n_points: Number of points to interpolate
    
    Returns:
    Dictionary of DataFrames with interpolated values for each dimension
    """
    interpolated_results = {}
    intensity_col = original_df.columns[-1]
    
    for coord in minima_results.keys():
        minima_df = minima_results[coord]
        
        # Sort by coordinate value
        sorted_df = minima_df.sort_values(coord)
        
        # Create interpolation function
        f_linear = interp1d(sorted_df[coord], sorted_df[intensity_col], 
                           kind='cubic', fill_value='extrapolate')
        
        # Create new coordinate points
        new_coords = np.linspace(sorted_df[coord].min(), 
                               sorted_df[coord].max(), 
                               n_points)
        
        # Interpolate intensity values
        new_intensities = f_linear(new_coords)
        
        # Create DataFrame with interpolated values
        interpolated_results[coord] = pd.DataFrame({
            coord: new_coords,
            intensity_col: new_intensities
        })
    
    return interpolated_results

def find_feature_points(df, window_length=11):
    """
    Find important features in the data including:
    - Global minimum
    - Turning points (inflection points)
    - Local minima and maxima
    
    Parameters:
    df: DataFrame with coordinate and intensity columns
    window_length: Window length for smoothing (must be odd)
    
    Returns:
    Dictionary containing feature points
    """
    coord_col = df.columns[0]
    intensity_col = df.columns[1]
    
    # Smooth the data using Savitzky-Golay filter
    y_smooth = savgol_filter(df[intensity_col], window_length, 3)
    
    # Calculate first and second derivatives
    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)
    
    # Find global minimum
    global_min_idx = df[intensity_col].idxmin()
    global_min = df.loc[global_min_idx]
    
    # Find turning points (where second derivative changes sign)
    turning_points = df[np.where(np.diff(np.signbit(d2y)))[0]]
    
    # Find local minima and maxima
    local_min_idx = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0]
    local_max_idx = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0]
    
    local_minima = df.iloc[local_min_idx]
    local_maxima = df.iloc[local_max_idx]
    
    return {
        'global_minimum': global_min,
        'turning_points': turning_points,
        'local_minima': local_minima,
        'local_maxima': local_maxima
    }

def create_advanced_plots(original_df, minima_results, interpolated_results, feature_points):
    """
    Create a comprehensive set of plots analyzing the data.
    
    Parameters:
    original_df: Original DataFrame with all data points
    minima_results: Dictionary of DataFrames from quantile_minima_multidim
    interpolated_results: Dictionary of DataFrames from interpolate_minima
    feature_points: Dictionary from find_feature_points
    """
    intensity_col = original_df.columns[-1]
    coord_cols = original_df.columns[:-1]
    
    for coord in coord_cols:
        # Create figure with GridSpec for flexible subplot layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Main scatter plot with minima and interpolation
        ax1 = fig.add_subplot(gs[0, :])
        scatter = ax1.scatter(original_df[coord], original_df[intensity_col], 
                            c=original_df[intensity_col], cmap='viridis', 
                            alpha=0.5, label='Original Data')
        ax1.scatter(minima_results[coord][coord], 
                   minima_results[coord][intensity_col], 
                   color='red', s=100, label='Quantile Minima')
        ax1.plot(interpolated_results[coord][coord], 
                interpolated_results[coord][intensity_col], 
                'g-', label='Interpolated Curve')
        
        # Add feature points
        ax1.scatter(feature_points['global_minimum'][coord], 
                   feature_points['global_minimum'][intensity_col], 
                   color='yellow', s=200, label='Global Minimum', 
                   edgecolor='black')
        ax1.scatter(feature_points['turning_points'][coord], 
                   feature_points['turning_points'][intensity_col], 
                   color='purple', s=150, label='Turning Points', 
                   edgecolor='black')
        
        ax1.set_title(f'{intensity_col} vs {coord} - Main Analysis')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Kernel Density Estimation
        ax2 = fig.add_subplot(gs[1, 0])
        xy = np.vstack([original_df[coord], original_df[intensity_col]])
        z = gaussian_kde(xy)(xy)
        ax2.scatter(original_df[coord], original_df[intensity_col], 
                   c=z, s=50, alpha=0.5)
        ax2.set_title('Density Estimation')
        
        # 3. Histogram with KDE
        ax3 = fig.add_subplot(gs[1, 1])
        original_df[intensity_col].hist(bins=50, ax=ax3, density=True, 
                                      alpha=0.5, label='Histogram')
        original_df[intensity_col].plot(kind='kde', ax=ax3, 
                                      label='KDE')
        ax3.set_title('Intensity Distribution')
        ax3.legend()
        
        # 4. Residuals plot
        ax4 = fig.add_subplot(gs[2, 0])
        interp_func = interp1d(interpolated_results[coord][coord], 
                             interpolated_results[coord][intensity_col], 
                             kind='cubic', fill_value='extrapolate')
        residuals = original_df[intensity_col] - interp_func(original_df[coord])
        ax4.scatter(original_df[coord], residuals, alpha=0.5)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title('Residuals from Interpolated Curve')
        
        # 5. QQ plot of residuals
        ax5 = fig.add_subplot(gs[2, 1])
        residuals_sorted = np.sort(residuals)
        theoretical_quantiles = np.random.normal(0, np.std(residuals), 
                                               len(residuals))
        theoretical_quantiles.sort()
        ax5.scatter(theoretical_quantiles, residuals_sorted, alpha=0.5)
        ax5.plot([-3*np.std(residuals), 3*np.std(residuals)], 
                [-3*np.std(residuals), 3*np.std(residuals)], 
                'r--')
        ax5.set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.show()

def analyze_data(df, n_quantiles=10, n_interp_points=1000):
    """
    Main function to run all analyses.
    
    Parameters:
    df: DataFrame where last column is intensity
    n_quantiles: Number of quantiles for minima detection
    n_interp_points: Number of interpolation points
    
    Returns:
    Dictionary containing all results
    """
    # Run quantile minima analysis
    minima_results = quantile_minima_multidim(df, n_quantiles)
    
    # Interpolate minima
    interpolated_results = interpolate_minima(df, minima_results, n_interp_points)
    
    # Find feature points for each dimension
    feature_points = {}
    intensity_col = df.columns[-1]
    for coord in df.columns[:-1]:
        feature_points[coord] = find_feature_points(
            minima_results[coord][[coord, intensity_col]]
        )
    
    # Create plots
    create_advanced_plots(df, minima_results, interpolated_results, feature_points)
    
    return {
        'minima_results': minima_results,
        'interpolated_results': interpolated_results,
        'feature_points': feature_points
    }

# Example usage:
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    
    # Create a more complex intensity function
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) + np.exp(-(X**2 + Y**2)/10)
    
    data = {
        'x': X.flatten(),
        'y': Y.flatten(),
        'intensity': Z.flatten()
    }
    df = pd.DataFrame(data)
    
    # Run analysis
    results = analyze_data(df, n_quantiles=10, n_interp_points=1000)