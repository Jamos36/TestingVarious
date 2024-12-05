import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class MinimaAnalyzer:
    def __init__(self, data, min_points_per_decade=20, tolerance=0.05, window_size=None):
        """
        Initialize the minima analyzer.
        
        Parameters:
        data: DataFrame with columns ['x', 'y', 'z']
        min_points_per_decade: Minimum points required per order of magnitude
        tolerance: Allowed deviation from monotonicity
        window_size: Size of rolling window (None for auto-calculation)
        """
        self.df = data.copy().sort_values('y').reset_index(drop=True)
        self.min_points_per_decade = min_points_per_decade
        self.tolerance = tolerance
        self.window_size = window_size or (self.df['y'].max() - self.df['y'].min()) / 10
        
        # Store analysis results
        self.density_analysis = None
        self.minima_analysis = None
        self.interpolation_needs = None
    
    def analyze(self):
        """Run all analyses and store results."""
        self.density_analysis = self._analyze_data_density()
        self.minima_analysis = self._find_monotonic_minima()
        self.interpolation_needs = self._analyze_interpolation_needs()
        return self
    
    def _analyze_data_density(self):
        """Analyze data density in log space."""
        z_log = np.log10(self.df['z'])
        z_decades = np.floor(z_log)
        
        # Points per decade analysis
        density = {decade: np.sum(z_decades == decade) 
                  for decade in np.unique(z_decades)}
        
        # Gap analysis
        y_gaps = np.diff(self.df['y'])
        z_ratios = np.diff(np.log10(self.df['z']))
        large_gaps = np.where((y_gaps > self.tolerance) | 
                            (np.abs(z_ratios) > 1))[0]
        
        return {
            'density_per_decade': density,
            'gaps': list(zip(self.df['y'].iloc[large_gaps], 
                           self.df['y'].iloc[large_gaps + 1])),
            'insufficient_decades': [d for d, count in density.items() 
                                   if count < self.min_points_per_decade]
        }
    
    def _find_monotonic_minima(self):
        """Find minima with monotonic behavior."""
        # Smooth data
        z_smooth = savgol_filter(self.df['z'], 
                               min(11, len(self.df) - 1 if len(self.df) % 2 == 0 
                                   else len(self.df)), 3)
        
        # Find potential minima
        gradient = np.gradient(z_smooth)
        potential_minima = []
        
        for i in range(1, len(gradient)):
            if gradient[i-1] < 0 and gradient[i] > 0:
                window = self._analyze_window(i, z_smooth)
                if window:
                    potential_minima.append(window)
        
        return potential_minima
    
    def _analyze_window(self, center_idx, smoothed_values):
        """Analyze monotonic behavior around a potential minimum."""
        # Check monotonicity with tolerance
        z_range = self.df['z'].max() - self.df['z'].min()
        tolerance_value = z_range * self.tolerance
        
        left_idx = center_idx
        right_idx = center_idx
        
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
                'min_idx': center_idx,
                'min_y': self.df['y'].iloc[center_idx],
                'min_z': self.df['z'].iloc[center_idx],
                'left_idx': left_idx,
                'right_idx': right_idx,
                'points': self.df.iloc[left_idx:right_idx+1]
            }
        return None
    
    def _analyze_interpolation_needs(self):
        """Analyze where interpolation is needed."""
        y_gaps = np.diff(self.df['y'])
        z_ratios = np.diff(np.log10(self.df['z']))
        
        # Calculate local density requirements
        local_resolution = np.abs(z_ratios) / y_gaps
        median_resolution = np.median(local_resolution)
        
        needed_points = []
        for i in range(len(y_gaps)):
            if local_resolution[i] < median_resolution / 2:
                n_points = int(y_gaps[i] * median_resolution) - 1
                if n_points > 0:
                    needed_points.append({
                        'start_y': self.df['y'].iloc[i],
                        'end_y': self.df['y'].iloc[i + 1],
                        'points_needed': n_points
                    })
        
        return needed_points
    
    def plot_analysis(self):
        """Plot comprehensive analysis results."""
        if not all([self.density_analysis, self.minima_analysis, 
                   self.interpolation_needs]):
            self.analyze()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Data and minima
        ax1.scatter(self.df['y'], np.log10(self.df['z']), alpha=0.5, 
                   label='Original Data')
        
        for minimum in self.minima_analysis:
            points = minimum['points']
            ax1.scatter(points['y'], np.log10(points['z']), alpha=0.7)
            ax1.scatter(minimum['min_y'], np.log10(minimum['min_z']), 
                       color='red', s=100, marker='*')
            
        ax1.set_ylabel('log10(Intensity)')
        ax1.set_title('Identified Minima and Monotonic Regions')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Density analysis
        decades = list(self.density_analysis['density_per_decade'].keys())
        counts = list(self.density_analysis['density_per_decade'].values())
        ax2.bar(decades, counts)
        ax2.axhline(y=self.min_points_per_decade, color='r', linestyle='--',
                   label=f'Min Required ({self.min_points_per_decade})')
        ax2.set_xlabel('log10(Intensity) Decade')
        ax2.set_ylabel('Points per Decade')
        ax2.set_title('Data Density Analysis')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_interpolation_suggestions(self):
        """Get suggestions for additional data points."""
        if not self.interpolation_needs:
            self.analyze()
            
        total_points = sum(p['points_needed'] for p in self.interpolation_needs)
        
        return {
            'total_points_needed': total_points,
            'regions': self.interpolation_needs,
            'density_issues': self.density_analysis['insufficient_decades'],
            'recommendations': [
                f"Add {total_points} points across {len(self.interpolation_needs)} regions",
                f"Insufficient density in decades: {self.density_analysis['insufficient_decades']}"
            ]
        }