import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

@dataclass
class AnalysisResults:
    """Store results from various analysis methods"""
    kmeans_minima: Dict = field(default_factory=dict)
    quantile_minima: Dict = field(default_factory=dict)
    gradient_minima: Dict = field(default_factory=dict)
    distance_minima: Dict = field(default_factory=dict)
    percentile_minima: Dict = field(default_factory=dict)
    dp_minima: Dict = field(default_factory=dict)
    density_analysis: Dict = field(default_factory=dict)
    monotonic_regions: Dict = field(default_factory=dict)
    interpolation_needs: Dict = field(default_factory=dict)

class ComprehensiveMinima:
    """
    Comprehensive class for minima analysis using multiple methods.
    Includes data density analysis and interpolation recommendations.
    """
    def __init__(self, data: pd.DataFrame, 
                 tolerance: float = 0.05,
                 min_points_per_decade: int = 20,
                 window_size: Optional[float] = None):
        """
        Initialize with data and general parameters.
        
        Parameters:
        data: DataFrame with columns ['x', 'y', 'z']
        tolerance: Allowed deviation from monotonicity
        min_points_per_decade: Minimum points required per order of magnitude
        window_size: Size of rolling window (None for auto-calculation)
        """
        if not all(col in data.columns for col in ['x', 'y', 'z']):
            raise ValueError("Data must contain 'x', 'y', and 'z' columns")
            
        self.df = data.copy().sort_values('y').reset_index(drop=True)
        self.tolerance = tolerance
        self.min_points_per_decade = min_points_per_decade
        self.window_size = window_size or (self.df['y'].max() - self.df['y'].min()) / 10
        
        # Initialize results container
        self.results = AnalysisResults()
        
        # Pre-compute log-space values
        self.df['z_log'] = np.log10(self.df['z'])
        
    def run_all_analyses(self, n_clusters: int = 10, 
                        n_quantiles: int = 10,
                        percentile_threshold: float = 10) -> AnalysisResults:
        """Run all available analysis methods"""
        # Traditional minima finding methods
        self._kmeans_analysis(n_clusters)
        self._quantile_analysis(n_quantiles)
        self._gradient_analysis()
        self._distance_based_analysis()
        self._percentile_analysis(percentile_threshold)
        self._dynamic_programming_analysis()
        
        # Density and interpolation analysis
        self._analyze_data_density()
        self._find_monotonic_regions()
        self._analyze_interpolation_needs()
        
        return self.results
    
    def _kmeans_analysis(self, n_clusters: int):
        """K-means clustering method (Method 4)"""
        yz_data = self.df[['y', 'z_log']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(yz_data)
        
        minima = []
        for cluster in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == cluster]
            min_point = cluster_data.loc[cluster_data['z'].idxmin()]
            minima.append({
                'y': min_point['y'],
                'z': min_point['z'],
                'cluster': cluster
            })
        
        self.results.kmeans_minima = {
            'minima': pd.DataFrame(minima),
            'clusters': self.df['cluster'].copy()
        }
    
    def _quantile_analysis(self, n_quantiles: int):
        """Quantile-based method (Method 5)"""
        quantiles = np.linspace(0, 1, n_quantiles+1)
        y_bounds = self.df['y'].quantile(quantiles)
        
        minima = []
        for i in range(len(y_bounds)-1):
            mask = (self.df['y'] >= y_bounds[i]) & (self.df['y'] < y_bounds[i+1])
            if mask.any():
                quantile_data = self.df[mask]
                min_point = quantile_data.loc[quantile_data['z'].idxmin()]
                minima.append({
                    'y': min_point['y'],
                    'z': min_point['z'],
                    'quantile': i
                })
        
        self.results.quantile_minima = {
            'minima': pd.DataFrame(minima),
            'bounds': y_bounds
        }
    
    def _gradient_analysis(self):
        """Gradient-based method (Method 6)"""
        z_smooth = savgol_filter(self.df['z_log'], 
                               min(11, len(self.df) - 1 if len(self.df) % 2 == 0 
                                   else len(self.df)), 3)
        gradient = np.gradient(z_smooth)
        
        minima_idx = []
        for i in range(1, len(gradient)):
            if gradient[i-1] < 0 and gradient[i] > 0:
                minima_idx.append(i)
        
        self.results.gradient_minima = {
            'minima': self.df.iloc[minima_idx][['y', 'z']].copy(),
            'gradient': gradient,
            'smoothed': z_smooth
        }
    
    def _distance_based_analysis(self):
        """Distance-based method (Method 7)"""
        delta_y = self.window_size / 10
        min_idx = self.df['z'].idxmin()
        current_y = self.df.loc[min_idx, 'y']
        
        selected_points = [{'y': self.df.loc[min_idx, 'y'],
                          'z': self.df.loc[min_idx, 'z']}]
        
        # Look forward and backward
        for direction in [1, -1]:
            idx = min_idx
            while 0 <= idx < len(self.df):
                next_region = self.df[
                    (self.df['y'] >= current_y + delta_y * direction)
                    if direction > 0 else
                    (self.df['y'] <= current_y + delta_y * direction)
                ]
                
                if len(next_region) == 0:
                    break
                    
                min_idx_region = next_region['z'].idxmin()
                selected_points.append({
                    'y': self.df.loc[min_idx_region, 'y'],
                    'z': self.df.loc[min_idx_region, 'z']
                })
                current_y = self.df.loc[min_idx_region, 'y']
                idx = min_idx_region
        
        self.results.distance_minima = {
            'minima': pd.DataFrame(selected_points),
            'delta_y': delta_y
        }
    
    def _percentile_analysis(self, percentile_threshold: float):
        """Percentile filtering method (Method 8)"""
        z_threshold = np.percentile(self.df['z'], percentile_threshold)
        filtered_df = self.df[self.df['z'] <= z_threshold].copy()
        
        y_ranges = np.linspace(filtered_df['y'].min(),
                             filtered_df['y'].max(),
                             int(len(filtered_df) / self.min_points_per_decade))
        
        minima = []
        for i in range(len(y_ranges) - 1):
            mask = (filtered_df['y'] >= y_ranges[i]) & (filtered_df['y'] < y_ranges[i + 1])
            if mask.any():
                region_df = filtered_df[mask]
                min_idx = region_df['z'].idxmin()
                minima.append({
                    'y': filtered_df.loc[min_idx, 'y'],
                    'z': filtered_df.loc[min_idx, 'z']
                })
        
        self.results.percentile_minima = {
            'minima': pd.DataFrame(minima),
            'threshold': z_threshold,
            'filtered_data': filtered_df
        }
    
    def _dynamic_programming_analysis(self):
        """Dynamic programming method (Method 9)"""
        n_segments = int(len(self.df) / self.min_points_per_decade)
        z_smooth = savgol_filter(self.df['z_log'], 
                               min(11, len(self.df) - 1 if len(self.df) % 2 == 0 
                                   else len(self.df)), 3)
        
        # DP matrix
        n_points = len(self.df)
        cost = np.zeros((n_segments, n_points))
        split = np.zeros((n_segments, n_points), dtype=int)
        
        # Initialize
        for j in range(n_points):
            cost[0, j] = np.min(z_smooth[:j+1])
            
        # Fill DP matrix
        for i in range(1, n_segments):
            for j in range(i, n_points):
                costs = [cost[i-1, k] + np.min(z_smooth[k+1:j+1])
                        for k in range(i-1, j)]
                cost[i, j] = np.min(costs)
                split[i, j] = np.argmin(costs) + i-1
        
        # Backtrack
        boundaries = []
        pos = n_points - 1
        for i in range(n_segments-1, -1, -1):
            if i > 0:
                boundaries.append(split[i, pos])
            pos = split[i, pos]
        
        boundaries = sorted(boundaries)
        
        # Find minima in segments
        minima = []
        start_idx = 0
        for end_idx in boundaries + [n_points]:
            segment = self.df.iloc[start_idx:end_idx]
            min_idx = segment['z'].idxmin()
            minima.append({
                'y': self.df.loc[min_idx, 'y'],
                'z': self.df.loc[min_idx, 'z']
            })
            start_idx = end_idx
        
        self.results.dp_minima = {
            'minima': pd.DataFrame(minima),
            'boundaries': boundaries,
            'smoothed': z_smooth
        }
    
    def _analyze_data_density(self):
        """Analyze data density in log space"""
        self.results.density_analysis = {
            'points_per_decade': {
                decade: np.sum(np.floor(self.df['z_log']) == decade)
                for decade in np.unique(np.floor(self.df['z_log']))
            },
            'gaps': self._find_gaps(),
            'density_score': self._calculate_density_score()
        }
    
    def _find_gaps(self) -> List[Dict]:
        """Find significant gaps in data"""
        y_gaps = np.diff(self.df['y'])
        z_ratios = np.diff(self.df['z_log'])
        significant_gaps = np.where((y_gaps > self.tolerance) | 
                                  (np.abs(z_ratios) > 1))[0]
        
        return [{'start_y': self.df['y'].iloc[i],
                'end_y': self.df['y'].iloc[i+1],
                'gap_size': y_gaps[i]}
               for i in significant_gaps]
    
    def _calculate_density_score(self) -> float:
        """Calculate overall density score"""
        tree = cKDTree(self.df[['y', 'z_log']])
        distances, _ = tree.query(self.df[['y', 'z_log']], k=2)
        return float(np.mean(distances[:, 1]))
    
    def _find_monotonic_regions(self):
        """Find regions with monotonic behavior"""
        z_smooth = savgol_filter(self.df['z_log'], 
                               min(11, len(self.df) - 1 if len(self.df) % 2 == 0 
                                   else len(self.df)), 3)
        gradient = np.gradient(z_smooth)
        
        # Find potential minima
        minima_idx = []
        for i in range(1, len(gradient)):
            if gradient[i-1] < 0 and gradient[i] > 0:
                minima_idx.append(i)
        
        # Analyze monotonicity around each minimum
        monotonic_regions = []
        for idx in minima_idx:
            region = self._analyze_monotonic_region(idx, z_smooth)
            if region:
                monotonic_regions.append(region)
        
        self.results.monotonic_regions = {
            'regions': monotonic_regions,
            'smoothed': z_smooth,
            'gradient': gradient
        }
    
    def _analyze_monotonic_region(self, center_idx: int, 
                                smoothed_values: np.ndarray) -> Optional[Dict]:
        """Analyze monotonic behavior around a point"""
        tolerance_value = (np.max(smoothed_values) - 
                         np.min(smoothed_values)) * self.tolerance
        
        left_idx = right_idx = center_idx
        
        # Expand left while monotonic decreasing
        while left_idx > 0:
            if smoothed_values[left_idx-1] > smoothed_values[left_idx] + tolerance_value:
                break
            left_idx -= 1
            
        # Expand right while monotonic increasing
        while right_idx < len(smoothed_values) - 1:
            if smoothed_values[right_idx+1] < smoothed_values[right_idx] - tolerance_value:
                break
            right_idx += 1
        
        if left_idx != right_idx:
            return {
                'center_idx': center_idx,
                'left_idx': left_idx,
                'right_idx': right_idx,
                'points': self.df.iloc[left_idx:right_idx+1].copy()
            }
        return None
    
    def _analyze_interpolation_needs(self):
        """Analyze where interpolation is needed"""
        # Calculate local density requirements
        y_gaps = np.diff(self.df['y'])
        z_changes = np.diff(self.df['z_log'])
        local_density = np.abs(z_changes) / y_gaps
        median_density = np.median(local_density)
        
        interpolation_points = []
        for i in range(len(y_gaps)):
            if local_density[i] < median_density / 2:
                n_points = int(y_gaps[i] * median_density) - 1
                if n_points > 0:
                    interpolation_points.append({
                        'start_y': self.df['y'].iloc[i],
                        'end_y': self.df['y'].iloc[i + 1],
                        'points_needed': n_points,
                        'current_density': local_density[i],
                        'target_density': median_density
                    })
        
        self.results.interpolation_needs = {
            'points_needed': interpolation_points,
            'total_points': sum(p['points_needed'] for p in interpolation_points),
            'median_density': median_density
        }
        interpolation_points.append({
                        'start_y': self.df['y'].iloc[i],
                        'end_y': self.df['y'].iloc[i + 1],
                        'points_needed': n_points,
                        'current_density': local_density[i],
                        'target_density': median_density
                    })
        
        self.results.interpolation_needs = {
            'points_needed': interpolation_points,
            'total_points': sum(p['points_needed'] for p in interpolation_points),
            'median_density': median_density
        }
    
    def plot_all_results(self):
        """Generate comprehensive visualization of all analyses"""
        fig = plt.figure(figsize=(20, 25))
        gs = plt.GridSpec(5, 2, figure=fig)
        
        # Plot 1: Original data with all identified minima
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_all_minima(ax1)
        
        # Plot 2: K-means results
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_kmeans_results(ax2)
        
        # Plot 3: Quantile results
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_quantile_results(ax3)
        
        # Plot 4: Gradient analysis
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_gradient_results(ax4)
        
        # Plot 5: Distance-based results
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_distance_results(ax5)
        
        # Plot 6: Percentile filtering results
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_percentile_results(ax6)
        
        # Plot 7: Dynamic programming results
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_dp_results(ax7)
        
        # Plot 8: Density analysis
        ax8 = fig.add_subplot(gs[4, 0])
        self._plot_density_analysis(ax8)
        
        # Plot 9: Interpolation needs
        ax9 = fig.add_subplot(gs[4, 1])
        self._plot_interpolation_needs(ax9)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_all_minima(self, ax):
        """Plot original data with minima from all methods"""
        ax.scatter(self.df['y'], self.df['z_log'], alpha=0.2, 
                  color='gray', label='Original Data')
        
        methods = {
            'K-means': self.results.kmeans_minima.get('minima', None),
            'Quantile': self.results.quantile_minima.get('minima', None),
            'Gradient': self.results.gradient_minima.get('minima', None),
            'Distance': self.results.distance_minima.get('minima', None),
            'Percentile': self.results.percentile_minima.get('minima', None),
            'DP': self.results.dp_minima.get('minima', None)
        }
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for (method, minima), color in zip(methods.items()):
            if minima is not None:
                ax.scatter(minima['y'], np.log10(minima['z']), 
                         label=method, color=color, s=100)
        
        ax.set_title('Comparison of All Minima Detection Methods')
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('log10(Intensity)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def get_consensus_minimum(self) -> Dict:
        """
        Find the consensus minimum across all methods.
        Returns the point most consistently identified as a minimum.
        """
        all_minima = []
        
        # Collect minima from all methods
        if self.results.kmeans_minima.get('minima') is not None:
            all_minima.extend(self.results.kmeans_minima['minima'][['y', 'z']].values)
        if self.results.quantile_minima.get('minima') is not None:
            all_minima.extend(self.results.quantile_minima['minima'][['y', 'z']].values)
        if self.results.gradient_minima.get('minima') is not None:
            all_minima.extend(self.results.gradient_minima['minima'][['y', 'z']].values)
        if self.results.distance_minima.get('minima') is not None:
            all_minima.extend(self.results.distance_minima['minima'][['y', 'z']].values)
        if self.results.percentile_minima.get('minima') is not None:
            all_minima.extend(self.results.percentile_minima['minima'][['y', 'z']].values)
        if self.results.dp_minima.get('minima') is not None:
            all_minima.extend(self.results.dp_minima['minima'][['y', 'z']].values)
        
        all_minima = np.array(all_minima)
        
        # Use DBSCAN to cluster the minima points
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=self.tolerance, min_samples=2).fit(all_minima)
        
        # Find the largest cluster
        if len(all_minima) > 0:
            labels = clustering.labels_
            largest_cluster = max(set(labels), key=list(labels).count)
            cluster_points = all_minima[labels == largest_cluster]
            
            # Return the median point in the largest cluster
            consensus_point = np.median(cluster_points, axis=0)
            
            return {
                'y': consensus_point[0],
                'z': consensus_point[1],
                'confidence': len(cluster_points) / len(all_minima)
            }
        
        return None
    
    def get_interpolation_recommendations(self) -> Dict:
        """
        Get detailed recommendations for additional data points.
        """
        if self.results.interpolation_needs is None:
            self._analyze_interpolation_needs()
            
        recommendations = {
            'total_points_needed': self.results.interpolation_needs['total_points'],
            'regions': self.results.interpolation_needs['points_needed'],
            'density_statistics': {
                'median_density': self.results.interpolation_needs['median_density'],
                'points_per_decade': self.results.density_analysis['points_per_decade']
            },
            'suggestions': []
        }
        
        # Add specific suggestions
        if recommendations['total_points_needed'] > 0:
            recommendations['suggestions'].append(
                f"Add {recommendations['total_points_needed']} points across "
                f"{len(recommendations['regions'])} regions"
            )
            
            for region in recommendations['regions']:
                recommendations['suggestions'].append(
                    f"Region {region['start_y']:.3f} to {region['end_y']:.3f}: "
                    f"add {region['points_needed']} points"
                )
        
        return recommendations

# Usage example:
"""
# Initialize analyzer
analyzer = ComprehensiveMinima(data, tolerance=0.05, min_points_per_decade=20)

# Run all analyses
results = analyzer.run_all_analyses()

# Visualize results
analyzer.plot_all_results()

# Get consensus minimum
consensus = analyzer.get_consensus_minimum()

# Get interpolation recommendations
recommendations = analyzer.get_interpolation_recommendations()
"""