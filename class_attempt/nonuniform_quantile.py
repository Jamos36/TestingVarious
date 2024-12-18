def create_nonuniform_quantiles(n_quantiles: int, edge_density: float = 0.3) -> np.ndarray:
    """
    Create non-uniform quantile spacing with higher density at edges.
    
    Parameters:
    n_quantiles: Number of quantiles to create
    edge_density: Proportion of points to concentrate at edges (0-1)
    
    Returns:
    Array of quantile values between 0 and 1
    """
    # Create three sets of points:
    # 1. Dense points at the start
    # 2. Regular spacing in the middle
    # 3. Dense points at the end
    
    n_edge = int(n_quantiles * edge_density)  # points for each edge
    n_middle = n_quantiles - (2 * n_edge)     # points for the middle
    
    # Create edge points with exponential spacing
    start_points = np.exp(np.linspace(np.log(1e-6), np.log(0.1), n_edge))
    end_points = 1 - np.exp(np.linspace(np.log(0.1), np.log(1e-6), n_edge))
    
    # Create middle points with linear spacing
    middle_points = np.linspace(0.1, 0.9, n_middle)
    
    # Combine all points and ensure boundaries are included
    quantiles = np.concatenate([[0], start_points, middle_points, end_points, [1]])
    
    # Sort and remove any duplicates
    return np.unique(quantiles)

def quantile_minima(df: pd.DataFrame, x_col: str, z_col: str, 
                   n_quantiles: int = 70, 
                   edge_density: float = 0.3,
                   min_points: int = 3) -> pd.DataFrame:
    """
    Calculate quantile minima with increased density at the extremes.
    
    Parameters:
    df: DataFrame containing the data
    x_col: Column name for x-axis
    z_col: Column name for z-axis (values to minimize)
    n_quantiles: Number of quantiles to create
    edge_density: Proportion of points to concentrate at edges (0-1)
    min_points: Minimum number of points required in a quantile
    
    Returns:
    DataFrame with minimal z points from each quantile
    """
    # Create non-uniform quantile boundaries
    quantiles = create_nonuniform_quantiles(n_quantiles, edge_density)
    x_bounds = df[x_col].quantile(quantiles)
    
    minima = []
    for i in range(len(x_bounds)-1):
        mask = (df[x_col] >= x_bounds[i]) & (df[x_col] < x_bounds[i+1])
        if mask.sum() >= min_points:  # Only process if enough points in quantile
            quantile_data = df[mask]
            min_point = quantile_data.loc[quantile_data[z_col].idxmin()]
            minima.append({
                x_col: min_point[x_col],
                z_col: min_point[z_col],
                'n_points': mask.sum()  # Track number of points in quantile
            })
    
    result_df = pd.DataFrame(minima)
    
    # Add global minimum if not already included
    global_min = df.loc[df[z_col].idxmin()]
    min_point_dict = {
        x_col: global_min[x_col],
        z_col: global_min[z_col],
        'n_points': 1
    }
    
    if len(result_df) == 0 or not any((result_df[x_col] == global_min[x_col]) & 
                                     (result_df[z_col] == global_min[z_col])):
        result_df = pd.concat([result_df, pd.DataFrame([min_point_dict])], 
                            ignore_index=True)
    
    # Sort by x_col
    result_df = result_df.sort_values(x_col)
    
    return result_df

def create_minima_plots(df: pd.DataFrame, str_cols: List[str], trouble_col: str, 
                       output_dir: str = 'minima_results'):
    """
    Create minima plots with enhanced quantile analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for col in str_cols:
        # Calculate minima with enhanced quantiles
        minima_df = quantile_minima(df, col, trouble_col, n_quantiles=70, edge_density=0.3)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot original data
        plt.scatter(df[col], df[trouble_col], alpha=0.3, label='Original Data')
        
        # Plot minima curve
        plt.plot(minima_df[col], minima_df[trouble_col], 'r-', 
                label='Quantile Minima', linewidth=2)
        
        # Plot minima points with size proportional to number of points in quantile
        sizes = np.clip(minima_df['n_points'] * 5, 50, 200)  # Scale sizes
        plt.scatter(minima_df[col], minima_df[trouble_col], 
                   s=sizes, color='red', alpha=0.5,
                   label='Minima Points')
        
        plt.xlabel(col)
        plt.ylabel(trouble_col)
        plt.title(f'{col} vs {trouble_col} with Enhanced Quantile Minima')
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot and data
        plt.savefig(os.path.join(output_dir, f'{col}_minima.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        minima_df.to_csv(os.path.join(output_dir, f'{col}_minima.csv'),
                        index=False)