import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize, interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks, peak_prominences
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, df):
        """
        Initialize the analyzer with a DataFrame where the last column is intensity.
        """
        self.df = df
        self.intensity_col = df.columns[-1]
        self.coord_cols = df.columns[:-1]
        self.results = {}
        
    def run_all_analyses(self):
        """Run all available analyses."""
        self.basic_statistics()
        self.find_peaks_and_valleys()
        self.analyze_topology()
        self.cluster_analysis()
        self.perform_dimensionality_analysis()
        self.analyze_periodicity()
        self.detect_anomalies()
        self.create_visualization_suite()
        return self.results
        
    def basic_statistics(self):
        """Calculate basic statistical measures and distributions."""
        stats_dict = {}
        
        # Global statistics
        stats_dict['global'] = {
            'mean': self.df[self.intensity_col].mean(),
            'median': self.df[self.intensity_col].median(),
            'std': self.df[self.intensity_col].std(),
            'skew': stats.skew(self.df[self.intensity_col]),
            'kurtosis': stats.kurtosis(self.df[self.intensity_col]),
            'iqr': stats.iqr(self.df[self.intensity_col])
        }
        
        # Calculate statistics by dimension
        for coord in self.coord_cols:
            # Bin the data
            bins = np.linspace(self.df[coord].min(), self.df[coord].max(), 20)
            digitized = np.digitize(self.df[coord], bins)
            
            # Calculate statistics per bin
            bin_stats = []
            for bin_idx in range(1, len(bins)):
                bin_data = self.df[self.intensity_col][digitized == bin_idx]
                if len(bin_data) > 0:
                    bin_stats.append({
                        'bin_center': (bins[bin_idx-1] + bins[bin_idx])/2,
                        'mean': bin_data.mean(),
                        'median': bin_data.median(),
                        'std': bin_data.std() if len(bin_data) > 1 else 0,
                        'count': len(bin_data)
                    })
            
            stats_dict[coord] = pd.DataFrame(bin_stats)
            
        self.results['statistics'] = stats_dict
        
    def find_peaks_and_valleys(self, prominence=1, width=None):
        """
        Identify peaks, valleys, and inflection points using various methods.
        """
        peak_data = {}
        
        for coord in self.coord_cols:
            # Sort data by coordinate
            sorted_data = self.df.sort_values(coord)
            intensity = sorted_data[self.intensity_col].values
            coords = sorted_data[coord].values
            
            # Smooth the data
            smoothed = gaussian_filter(intensity, sigma=3)
            
            # Find peaks and valleys
            peaks, peak_props = find_peaks(smoothed, prominence=prominence, width=width)
            valleys, valley_props = find_peaks(-smoothed, prominence=prominence, width=width)
            
            # Calculate derivatives
            first_derivative = np.gradient(smoothed)
            second_derivative = np.gradient(first_derivative)
            
            # Find inflection points
            inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]
            
            peak_data[coord] = {
                'peaks': {'coordinates': coords[peaks], 'intensities': intensity[peaks],
                         'prominences': peak_props['prominences']},
                'valleys': {'coordinates': coords[valleys], 'intensities': intensity[valleys],
                          'prominences': valley_props['prominences']},
                'inflection_points': {'coordinates': coords[inflection_points],
                                    'intensities': intensity[inflection_points]},
                'derivatives': {'first': first_derivative, 'second': second_derivative}
            }
            
        self.results['peak_analysis'] = peak_data
        
    def analyze_topology(self):
        """
        Analyze the topological features of the data.
        """
        topology_data = {}
        
        for coord in self.coord_cols:
            # Create 2D representation
            points = np.column_stack((self.df[coord], self.df[self.intensity_col]))
            
            # Calculate convex hull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Estimate surface complexity
            complexity = {
                'hull_area': hull.area,
                'hull_perimeter': hull.area,  # For 2D, area gives perimeter
                'point_density': len(points) / hull.area,
                'fractal_dimension': self._estimate_fractal_dimension(points)
            }
            
            # Calculate local curvature
            curvature = self._estimate_curvature(points)
            
            topology_data[coord] = {
                'hull_points': hull_points,
                'complexity_metrics': complexity,
                'curvature': curvature
            }
            
        self.results['topology'] = topology_data
        
    def cluster_analysis(self):
        """
        Perform various clustering analyses.
        """
        cluster_data = {}
        
        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        # K-means clustering with elbow method
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Choose optimal k using elbow method
        optimal_k = self._find_elbow_point(k_range, inertias)
        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans_optimal.fit_predict(scaled_data)
        
        cluster_data['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        }
        
        cluster_data['kmeans'] = {
            'labels': kmeans_labels,
            'n_clusters': optimal_k,
            'centroids': kmeans_optimal.cluster_centers_,
            'inertias': inertias
        }
        
        self.results['clustering'] = cluster_data
        
    def perform_dimensionality_analysis(self):
        """
        Analyze dimensionality and correlations.
        """
        dim_data = {}
        
        # PCA analysis
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        dim_data['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'transformed_data': pca_result
        }
        
        # Correlation analysis
        correlation_matrix = self.df.corr()
        dim_data['correlations'] = correlation_matrix
        
        # Calculate mutual information
        mutual_info = {}
        for col1 in self.df.columns:
            mutual_info[col1] = {}
            for col2 in self.df.columns:
                if col1 != col2:
                    mutual_info[col1][col2] = self._mutual_information(
                        self.df[col1], self.df[col2]
                    )
                    
        dim_data['mutual_information'] = mutual_info
        
        self.results['dimensionality'] = dim_data
        
    def analyze_periodicity(self):
        """
        Analyze periodic patterns in the data.
        """
        periodicity_data = {}
        
        for coord in self.coord_cols:
            # Sort data by coordinate
            sorted_data = self.df.sort_values(coord)
            intensity = sorted_data[self.intensity_col].values
            
            # Compute FFT
            fft_result = np.fft.fft(intensity)
            freqs = np.fft.fftfreq(len(intensity))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_result)**2
            dominant_freqs_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
            
            # Autocorrelation
            autocorr = np.correlate(intensity, intensity, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            periodicity_data[coord] = {
                'fft': {'frequencies': freqs, 'power': power_spectrum,
                       'dominant_frequencies': freqs[dominant_freqs_idx],
                       'dominant_powers': power_spectrum[dominant_freqs_idx]},
                'autocorrelation': autocorr
            }
            
        self.results['periodicity'] = periodicity_data
        
    def detect_anomalies(self):
        """
        Detect anomalies using multiple methods.
        """
        anomaly_data = {}
        
        # Z-score method
        z_scores = stats.zscore(self.df[self.intensity_col])
        z_score_anomalies = np.abs(z_scores) > 3
        
        # IQR method
        Q1 = self.df[self.intensity_col].quantile(0.25)
        Q3 = self.df[self.intensity_col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = (self.df[self.intensity_col] < (Q1 - 1.5 * IQR)) | \
                        (self.df[self.intensity_col] > (Q3 + 1.5 * IQR))
        
        # Isolation Forest
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(random_state=42)
        iso_forest_anomalies = iso_forest.fit_predict(self.df[self.intensity_col].values.reshape(-1, 1))
        
        anomaly_data['z_score'] = {
            'anomalies': z_score_anomalies,
            'threshold': 3
        }
        
        anomaly_data['iqr'] = {
            'anomalies': iqr_anomalies,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
        
        anomaly_data['isolation_forest'] = {
            'anomalies': iso_forest_anomalies == -1
        }
        
        self.results['anomalies'] = anomaly_data
        
    def create_visualization_suite(self):
        """
        Create a comprehensive suite of visualizations.
        """
        for coord in self.coord_cols:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 20))
            gs = plt.GridSpec(4, 3)
            
            # 1. Main scatter plot with peaks and valleys
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_main_scatter(ax1, coord)
            
            # 2. Histogram and KDE
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_distribution(ax2, coord)
            
            # 3. Cluster analysis
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_clusters(ax3, coord)
            
            # 4. Anomaly detection
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_anomalies(ax4, coord)
            
            # 5. FFT analysis
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_fft(ax5, coord)
            
            # 6. Autocorrelation
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_autocorrelation(ax6, coord)
            
            # 7. Topology analysis
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_topology(ax7, coord)
            
            # 8. Statistics summary
            ax8 = fig.add_subplot(gs[3, :])
            self._plot_statistics_summary(ax8, coord)
            
            plt.tight_layout()
            plt.show()
            
    # Helper methods
    def _estimate_fractal_dimension(self, points, eps=None):
        if eps is None:
            eps = np.logspace(-10, 10, num=20)
        
        N = []
        for epsilon in eps:
            boxes = np.ceil((points - points.min(0))/epsilon)
            N.append(len(np.unique(boxes, axis=0)))
        
        coeffs = np.polyfit(np.log(eps), np.log(N), 1)
        return -coeffs[0]
    
    def _estimate_curvature(self, points):
        # Fit a smooth curve
        tck = interpolate.splrep(points[:,0], points[:,1], s=0)
        
        # Calculate derivatives
        x = np.linspace(points[:,0].min(), points[:,0].max(), 1000)
        y = interpolate.splev(x, tck, der=0)
        dy = interpolate.splev(x, tck, der=1)
        d2y = interpolate.splev(x, tck, der=2)
        
        # Calculate curvature
        curvature = np.abs(d2y) / (1 + dy**2)**(3/2)
        
        return {'x': x, 'curvature': curvature}
    
    def _find_elbow_point(self, x, y):
        coords = np.vstack((x, y)).T
        line_vec = coords[-1] - coords[0]
        vec_from_first = coords - coords[0]
        vec_to_line = vec_from_first - np.outer(
            np.dot(vec_from_first, line_vec) / np.dot(line_vec, line_vec),
            line_vec
        )
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        return x[np.argmax(dist_to_line)]
    
    def _mutual_information(self, x, y, bins=20):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if c_xy[i,j] != 0:
                    p_xy = c_xy[i,j] / np.sum(c_xy)
                    p_x = np.sum(c_xy[i,:]) / np.sum(c_xy)
                    p_y = np.sum(c_xy[:,j]) / np.sum(c_xy)
                    mi += p_xy * np.log(p_xy / (p_x * p_y))
        return mi

    def _plot_main_scatter(self, ax, coord):
        """Enhanced main scatter plot with additional features."""
        scatter = ax.scatter(self.df[coord], self.df[self.intensity_col],
                           c=self.df[self.intensity_col], cmap='viridis',
                           alpha=0.5, s=50)
        
        # Add peaks and valleys
        peak_data = self.results['peak_analysis'][coord]
        ax.scatter(peak_data['peaks']['coordinates'],
                  peak_data['peaks']['intensities'],
                  color='red', s=100, label='Peaks')
        ax.scatter(peak_data['valleys']['coordinates'],
                  peak_data['valleys']['intensities'],
                  color='blue', s=100, label='Valleys')
        
        # Add trend line
        z = np.polyfit(self.df[coord], self.df[self.intensity_col], 3)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df[coord].min(), self.df[coord].max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', label='Trend')
        
        ax.set_title(f'Main Analysis: {self.intensity_col} vs {coord}')
        ax.set_xlabel(coord)
        ax.set_ylabel(self.intensity_col)
        ax.legend()
        plt.colorbar(scatter, ax=ax)

    def analyze_local_patterns(self, window_size=20):
        """Analyze local patterns and variations in the data."""
        pattern_data = {}
        
        for coord in self.coord_cols:
            # Sort data by coordinate
            sorted_data = self.df.sort_values(coord)
            
            # Calculate rolling statistics
            rolling_stats = pd.DataFrame({
                'mean': sorted_data[self.intensity_col].rolling(window_size).mean(),
                'std': sorted_data[self.intensity_col].rolling(window_size).std(),
                'skew': sorted_data[self.intensity_col].rolling(window_size).skew(),
                'kurt': sorted_data[self.intensity_col].rolling(window_size).kurt()
            })
            
            # Detect local trend changes
            rolling_diff = sorted_data[self.intensity_col].diff().rolling(window_size).mean()
            trend_changes = np.where(np.diff(np.signbit(rolling_diff)))[0]
            
            pattern_data[coord] = {
                'rolling_stats': rolling_stats,
                'trend_changes': trend_changes,
                'coordinates': sorted_data[coord].values
            }
            
        self.results['local_patterns'] = pattern_data

    def analyze_symmetry(self):
        """Analyze symmetry properties of the data."""
        symmetry_data = {}
        
        for coord in self.coord_cols:
            # Sort data by coordinate
            sorted_data = self.df.sort_values(coord)
            intensity = sorted_data[self.intensity_col].values
            coords = sorted_data[coord].values
            
            # Calculate reflection symmetry score
            mid_point = (coords.max() + coords.min()) / 2
            left_side = intensity[coords <= mid_point]
            right_side = intensity[coords > mid_point][::-1]
            min_len = min(len(left_side), len(right_side))
            reflection_score = np.corrcoef(left_side[:min_len], right_side[:min_len])[0,1]
            
            # Calculate rotational symmetry score
            center_intensity = np.interp(mid_point, coords, intensity)
            rotational_score = np.mean(np.abs(intensity - center_intensity))
            
            symmetry_data[coord] = {
                'reflection_symmetry': reflection_score,
                'rotational_symmetry': rotational_score,
                'center_point': mid_point
            }
            
        self.results['symmetry'] = symmetry_data

    def analyze_scale_dependence(self):
        """Analyze how patterns change with scale."""
        scale_data = {}
        
        for coord in self.coord_cols:
            scales = np.logspace(-1, 1, 20)
            variations = []
            
            for scale in scales:
                # Smooth data at different scales
                smoothed = gaussian_filter(self.df[self.intensity_col].values, scale)
                
                # Calculate various metrics at this scale
                variations.append({
                    'scale': scale,
                    'variance': np.var(smoothed),
                    'peak_count': len(signal.find_peaks(smoothed)[0]),
                    'complexity': len(np.unique(np.round(smoothed, decimals=3)))
                })
            
            scale_data[coord] = pd.DataFrame(variations)
            
        self.results['scale_dependence'] = scale_data

    def analyze_persistence(self):
        """Analyze persistence of features across scales."""
        persistence_data = {}
        
        for coord in self.coord_cols:
            sorted_data = self.df.sort_values(coord)
            intensity = sorted_data[self.intensity_col].values
            
            # Calculate persistence diagram
            persistence = []
            scales = np.logspace(-1, 1, 20)
            
            for i, scale in enumerate(scales):
                smoothed = gaussian_filter(intensity, scale)
                peaks, properties = signal.find_peaks(smoothed)
                
                for peak, prop in zip(peaks, properties['prominences']):
                    persistence.append({
                        'scale': scale,
                        'location': sorted_data[coord].iloc[peak],
                        'prominence': prop
                    })
            
            persistence_data[coord] = pd.DataFrame(persistence)
            
        self.results['persistence'] = persistence_data

    def create_feature_space(self):
        """Create and analyze feature space representation."""
        feature_space = {}
        
        for coord in self.coord_cols:
            # Extract features
            sorted_data = self.df.sort_values(coord)
            intensity = sorted_data[self.intensity_col].values
            
            # Calculate various features
            features = pd.DataFrame({
                'original': intensity,
                'gradient': np.gradient(intensity),
                'curvature': np.gradient(np.gradient(intensity)),
                'smoothed': gaussian_filter(intensity, 2),
                'local_std': pd.Series(intensity).rolling(10).std().fillna(0)
            })
            
            # Perform PCA on feature space
            pca = PCA()
            pca_result = pca.fit_transform(features)
            
            feature_space[coord] = {
                'features': features,
                'pca_result': pca_result,
                'explained_variance': pca.explained_variance_ratio_
            }
            
        self.results['feature_space'] = feature_space

    def analyze_critical_points(self):
        """Analyze critical points and their stability."""
        critical_points = {}
        
        for coord in self.coord_cols:
            sorted_data = self.df.sort_values(coord)
            intensity = sorted_data[self.intensity_col].values
            coords = sorted_data[coord].values
            
            # Find critical points
            grad = np.gradient(intensity)
            critical_indices = np.where(np.abs(grad) < np.std(grad)/10)[0]
            
            # Analyze stability
            stability = []
            for idx in critical_indices:
                # Calculate local Hessian
                if idx > 0 and idx < len(intensity)-1:
                    hessian = (intensity[idx+1] - 2*intensity[idx] + intensity[idx-1])
                    stability.append({
                        'coordinate': coords[idx],
                        'intensity': intensity[idx],
                        'stability': np.abs(hessian),
                        'type': 'maximum' if hessian < 0 else 'minimum'
                    })
            
            critical_points[coord] = pd.DataFrame(stability)
            
        self.results['critical_points'] = critical_points

    def plot_advanced_analysis(self):
        """Create advanced analysis plots."""
        for coord in self.coord_cols:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 20))
            gs = plt.GridSpec(4, 3)
            
            # 1. Scale dependence plot
            ax1 = fig.add_subplot(gs[0, 0])
            scale_data = self.results['scale_dependence'][coord]
            ax1.loglog(scale_data['scale'], scale_data['variance'], 'o-')
            ax1.set_title('Scale Dependence')
            ax1.set_xlabel('Scale')
            ax1.set_ylabel('Variance')
            
            # 2. Persistence diagram
            ax2 = fig.add_subplot(gs[0, 1])
            persistence = self.results['persistence'][coord]
            ax2.scatter(persistence['scale'], persistence['prominence'])
            ax2.set_title('Persistence Diagram')
            ax2.set_xscale('log')
            
            # 3. Feature space visualization
            ax3 = fig.add_subplot(gs[0, 2])
            feature_space = self.results['feature_space'][coord]
            ax3.scatter(feature_space['pca_result'][:,0], 
                       feature_space['pca_result'][:,1])
            ax3.set_title('Feature Space (PCA)')
            
            # 4. Critical points
            ax4 = fig.add_subplot(gs[1, :])
            critical = self.results['critical_points'][coord]
            ax4.scatter(critical['coordinate'], critical['intensity'],
                       c=critical['stability'], cmap='viridis')
            ax4.set_title('Critical Points')
            
            # 5. Local patterns
            ax5 = fig.add_subplot(gs[2, :])
            patterns = self.results['local_patterns'][coord]
            ax5.plot(patterns['coordinates'], 
                    patterns['rolling_stats']['mean'],
                    label='Local Mean')
            ax5.fill_between(patterns['coordinates'],
                           patterns['rolling_stats']['mean'] - patterns['rolling_stats']['std'],
                           patterns['rolling_stats']['mean'] + patterns['rolling_stats']['std'],
                           alpha=0.3)
            ax5.set_title('Local Patterns')
            
            # 6. Symmetry analysis
            ax6 = fig.add_subplot(gs[3, :])
            symmetry = self.results['symmetry'][coord]
            sorted_data = self.df.sort_values(coord)
            ax6.plot(sorted_data[coord], sorted_data[self.intensity_col])
            ax6.axvline(symmetry['center_point'], color='r', linestyle='--')
            ax6.set_title(f'Symmetry Analysis (Reflection Score: {symmetry["reflection_symmetry"]:.2f})')
            
            plt.tight_layout()
            plt.show()

# Example usage:
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    
    # Create more complex sample data
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    
    # Create a complex intensity function with multiple features
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(np.sqrt(X**2 + Y**2)) + 
         np.exp(-(X**2 + Y**2)/10) + 
         0.5*np.sin(2*X) + 
         0.3*np.cos(3*Y) + 
         0.2*np.random.normal(0, 1, X.shape))
    
    data = {
        'x': X.flatten(),
        'y': Y.flatten(),
        'intensity': Z.flatten()
    }
    df = pd.DataFrame(data)
    
    # Create analyzer and run all analyses
    analyzer = DataAnalyzer(df)
    results = analyzer.run_all_analyses()
    
    # Create all visualizations
    analyzer.plot_advanced_analysis()