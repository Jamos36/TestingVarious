kmeans_results = kmeans_minima(data, n_clusters=10)
quantile_results = quantile_minima(data, n_quantiles=10)
gradient_results = gradient_minima(data)

distance_results = distance_based_minima(data, delta_y=1.0)
percentile_results = percentile_filtering_minima(data, percentile_threshold=10, n_samples=10)
dp_results = dynamic_programming_minima(data, n_segments=10, smoothing_window=5)



#Density interpolation.py:

# Analyze data density in log space
log_analysis = logarithmic_density_analyzer(data, min_points_per_decade=20)
# Get adaptive interpolation recommendations
adaptive_analysis = adaptive_interpolation_analyzer(data, target_resolution=0.01)
# Analyze nearest neighbor density
density_analysis = nearest_neighbor_density_analyzer(data, radius_factor=0.1)