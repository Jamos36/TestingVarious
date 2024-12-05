import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def logarithmic_density_analyzer(df, min_points_per_decade=20, distance_threshold=0.1):
    """
    Analyzes data density in logarithmic space, particularly for small intensity values.
    Recommends interpolation or additional data points where needed.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    min_points_per_decade: Minimum number of points required per order of magnitude
    distance_threshold: Maximum allowed distance between points in log space
    
    Returns:
    dict with analysis results and recommendations
    """
    # Convert to log space for analysis
    z_log = np.log10(df['z'])
    z_decades = np.floor(z_log)
    
    # Analyze point density per decade
    density_analysis = {}
    interpolation_needed = []
    
    for decade in np.unique(z_decades):
        mask = (z_decades == decade)
        points_in_decade = np.sum(mask)
        density_analysis[decade] = points_in_decade
        
        if points_in_decade < min_points_per_decade:
            interpolation_needed.append(decade)
    
    # Find gaps in data
    sorted_indices = np.argsort(df['y'])
    y_sorted = df['y'].values[sorted_indices]
    z_sorted = df['z'].values[sorted_indices]
    y_gaps = np.diff(y_sorted)
    z_ratios = np.diff(np.log10(z_sorted))
    
    large_gaps = np.where((y_gaps > distance_threshold) | (np.abs(z_ratios) > 1))[0]
    
    # Visualize analysis
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data distribution
    plt.subplot(2, 1, 1)
    plt.scatter(df['y'], np.log10(df['z']), alpha=0.5)
    plt.ylabel('log10(Intensity)')
    plt.title('Data Distribution in Log Space')
    
    # Highlight gaps
    for gap_idx in large_gaps:
        plt.axvspan(y_sorted[gap_idx], y_sorted[gap_idx + 1], 
                   color='red', alpha=0.2)
    
    # Plot 2: Points per decade
    plt.subplot(2, 1, 2)
    decades = list(density_analysis.keys())
    points = list(density_analysis.values())
    plt.bar(decades, points)
    plt.axhline(y=min_points_per_decade, color='r', linestyle='--', 
                label=f'Minimum required ({min_points_per_decade} points)')
    plt.xlabel('log10(Intensity) Decade')
    plt.ylabel('Number of Points')
    plt.title('Points per Decade of Intensity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'density_analysis': density_analysis,
        'interpolation_needed': interpolation_needed,
        'large_gaps': list(zip(y_sorted[large_gaps], y_sorted[large_gaps + 1])),
        'recommendations': {
            'total_points_needed': sum(
                max(0, min_points_per_decade - density_analysis[d])
                for d in density_analysis
            ),
            'decades_needing_attention': interpolation_needed
        }
    }

def adaptive_interpolation_analyzer(df, target_resolution=0.01, intensity_scaling=True):
    """
    Analyzes need for interpolation based on local intensity gradients.
    Provides higher resolution near the minimum and in regions of rapid change.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    target_resolution: Desired minimum distance between points
    intensity_scaling: Whether to adjust resolution based on intensity magnitude
    
    Returns:
    dict with analysis and interpolation recommendations
    """
    # Sort by y coordinate
    df_sorted = df.sort_values('y').copy()
    
    # Calculate local gradients
    df_sorted['z_log'] = np.log10(df_sorted['z'])
    gradients = np.gradient(df_sorted['z_log'], df_sorted['y'])
    
    # Calculate local resolution requirements
    if intensity_scaling:
        local_resolution = target_resolution * (
            1 + 10 * np.abs(gradients)
        ) * np.sqrt(df_sorted['z'])
    else:
        local_resolution = target_resolution * (1 + 10 * np.abs(gradients))
    
    # Analyze where interpolation is needed
    y_gaps = np.diff(df_sorted['y'])
    required_points = []
    
    for i in range(len(y_gaps)):
        current_gap = y_gaps[i]
        local_req = local_resolution[i]
        if current_gap > local_req:
            n_points = int(np.ceil(current_gap / local_req)) - 1
            required_points.append({
                'start_y': df_sorted['y'].iloc[i],
                'end_y': df_sorted['y'].iloc[i + 1],
                'points_needed': n_points,
                'local_gradient': gradients[i]
            })
    
    # Visualize analysis
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and gradients
    plt.subplot(2, 1, 1)
    plt.scatter(df_sorted['y'], df_sorted['z_log'], alpha=0.5, label='Data Points')
    plt.plot(df_sorted['y'], gradients, 'r-', alpha=0.5, label='Local Gradient')
    plt.ylabel('log10(Intensity) / Gradient')
    plt.legend()
    plt.title('Data Points and Local Gradients')
    
    # Plot 2: Required resolution
    plt.subplot(2, 1, 2)
    plt.plot(df_sorted['y'][:-1], y_gaps, 'b-', label='Current Spacing')
    plt.plot(df_sorted['y'], local_resolution, 'r--', label='Required Resolution')
    plt.yscale('log')
    plt.ylabel('Point Spacing')
    plt.xlabel('Y Coordinate')
    plt.legend()
    plt.title('Current vs Required Point Spacing')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'required_points': required_points,
        'total_points_needed': sum(p['points_needed'] for p in required_points),
        'max_gradient': np.max(np.abs(gradients)),
        'min_gradient': np.min(np.abs(gradients)),
    }

def nearest_neighbor_density_analyzer(df, radius_factor=0.1, min_neighbors=5):
    """
    Analyzes data density using nearest neighbor statistics.
    Identifies regions needing more data points based on local density.
    
    Parameters:
    df: DataFrame with columns ['x', 'y', 'z']
    radius_factor: Factor to determine search radius (relative to data range)
    min_neighbors: Minimum number of neighbors expected within radius
    
    Returns:
    dict with analysis results and recommendations
    """
    # Normalize coordinates for density analysis
    y_range = df['y'].max() - df['y'].min()
    z_range = np.log10(df['z'].max()) - np.log10(df['z'].min())
    
    y_norm = (df['y'] - df['y'].min()) / y_range
    z_norm = (np.log10(df['z']) - np.log10(df['z'].min())) / z_range
    
    # Create KD-tree for efficient neighbor search
    points = np.column_stack([y_norm, z_norm])
    tree = cKDTree(points)
    
    # Calculate search radius
    search_radius = radius_factor
    
    # Find number of neighbors for each point
    neighbors = tree.query_ball_point(points, search_radius)
    n_neighbors = np.array([len(n) - 1 for n in neighbors])  # -1 to exclude self
    
    # Identify sparse regions
    sparse_mask = n_neighbors < min_neighbors
    sparse_points = df[sparse_mask]
    
    # Calculate local density
    density = n_neighbors / (np.pi * search_radius**2)
    
    # Visualize results
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Point density
    plt.subplot(3, 1, 1)
    scatter = plt.scatter(df['y'], np.log10(df['z']), c=density, 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Local Density')
    plt.ylabel('log10(Intensity)')
    plt.title('Local Point Density')
    
    # Plot 2: Neighbor count
    plt.subplot(3, 1, 2)
    scatter = plt.scatter(df['y'], np.log10(df['z']), c=n_neighbors, 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Number of Neighbors')
    plt.ylabel('log10(Intensity)')
    plt.title('Number of Neighbors within Radius')
    
    # Plot 3: Sparse regions
    plt.subplot(3, 1, 3)
    plt.scatter(df['y'], np.log10(df['z']), alpha=0.2, label='All Points')
    plt.scatter(sparse_points['y'], np.log10(sparse_points['z']), 
                color='red', alpha=0.6, label='Sparse Regions')
    plt.ylabel('log10(Intensity)')
    plt.xlabel('Y Coordinate')
    plt.title('Identified Sparse Regions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'density_stats': {
            'mean_density': np.mean(density),
            'std_density': np.std(density),
            'min_density': np.min(density),
            'max_density': np.max(density)
        },
        'neighbor_stats': {
            'mean_neighbors': np.mean(n_neighbors),
            'std_neighbors': np.std(n_neighbors),
            'min_neighbors': np.min(n_neighbors),
            'max_neighbors': np.max(n_neighbors)
        },
        'sparse_regions': len(sparse_points),
        'sparse_points': sparse_points,
        'recommendations': {
            'additional_points_needed': sum(min_neighbors - n_neighbors[sparse_mask]),
            'average_density_target': np.mean(density[~sparse_mask])
        }
    }