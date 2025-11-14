#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
snapfood_lower_bound_final.py
-----------------------------
Applies your percentile-bin + spike-removal method
to all 8 datasets.

* Uses only real data points
* Always includes edge points
* Removes upward spikes
* Second-pass rolling-window pruning
* Saves CSV + PDF (high quality)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
BASE_DIR = r"C:\Users\User\Desktop\snapfood-app\outputs"
OUTPUT_DIR = BASE_DIR + r"\min_curves"  # Save CSVs and PDFs in same folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    ("dataset1_flat_gradual.csv", 80, 3),
    ("dataset2_sinusoidal.csv", 100, 2),
    ("dataset3_exponential.csv", 80, 3),
    ("dataset4_polynomial.csv", 100, 2),
    ("dataset5_high_frequency.csv", 120, 2),
    ("dataset6_logarithmic.csv", 100, 2),
    ("dataset7_curved_funnel.csv", 90, 2),
    ("dataset8_spike_gradient.csv", 100, 2),
]

# ----------------------------------------------------------------------
# 2. CORE FUNCTIONS (no self, standalone)
# ----------------------------------------------------------------------
def find_lower_envelope(
    df,
    x_col,
    y_col,
    num_bins=100,
    percentile=2,
    density_factor=0.15,
    max_points_per_bin=25,
):
    """
    Find lower envelope using percentile per bin, keeping more points
    in denser regions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    x_col, y_col : str
        Column names for x and y.
    num_bins : int
        Number of x-bins.
    percentile : float
        Percentile used to select low points in each bin.
    density_factor : float
        Fraction of points per bin to keep (scaled by bin size).
        More points in denser bins:
            n_keep = max(1, int(len(bin_data) * density_factor))
    max_points_per_bin : int
        Upper cap on how many points to keep per bin.
    """
    df_sorted = df.sort_values(x_col).reset_index(drop=True)
    x_min, x_max = df_sorted[x_col].min(), df_sorted[x_col].max()
    num_bins = len(df)//2
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)
    

    lower_indices = []

    for i in range(num_bins):
        mask = (df_sorted[x_col] >= bin_edges[i]) & (df_sorted[x_col] < bin_edges[i + 1])
        bin_data = df_sorted[mask]
        if len(bin_data) == 0:
            continue

        threshold = np.percentile(bin_data[y_col], percentile)
        candidates = bin_data[bin_data[y_col] <= threshold * 1.1]
        if len(candidates) == 0:
            continue

        # Sort by y (lowest first)
        candidates = candidates.sort_values(y_col)

        # Keep more points in denser bins
        n_keep = max(1, int(len(bin_data) * density_factor))
        n_keep = min(max_points_per_bin, n_keep)

        lower_indices.extend(candidates.index[:n_keep].tolist())

    # Last bin (x == x_max)
    mask = df_sorted[x_col] >= bin_edges[-1]
    bin_data = df_sorted[mask]
    if len(bin_data) > 0:
        threshold = np.percentile(bin_data[y_col], percentile)
        candidates = bin_data[bin_data[y_col] <= threshold * 1.1]
        if len(candidates) > 0:
            candidates = candidates.sort_values(y_col)

            n_keep = max(1, int(len(bin_data) * density_factor))
            n_keep = min(max_points_per_bin, n_keep)

            lower_indices.extend(candidates.index[:n_keep].tolist())

    if not lower_indices:
        return pd.DataFrame(columns=[x_col, y_col])

    # Unique, sorted by x
    result_df = df_sorted.loc[sorted(set(lower_indices))].copy()
    result_df = result_df.drop_duplicates().sort_values(x_col).reset_index(drop=True)
    return result_df


def smooth_lower_curve(curve_df, x_col, y_col, remove_spikes=True, slope_threshold=0.5):
    """
    Remove upward spikes (up then down) along the curve.

    Parameters
    ----------
    curve_df : pd.DataFrame
        Lower-envelope points.
    x_col, y_col : str
        Column names.
    remove_spikes : bool
        If False, return unchanged.
    slope_threshold : float
        Minimum absolute slope to treat a segment as a spike.
        Smaller -> more aggressive (removes more points).
        Larger  -> less aggressive (keeps more points).
    """
    if len(curve_df) <= 2 or not remove_spikes:
        return curve_df.copy()

    smoothed = [curve_df.iloc[0]]

    for i in range(1, len(curve_df) - 1):
        prev = smoothed[-1]
        curr = curve_df.iloc[i]
        next_p = curve_df.iloc[i + 1]

        dx1 = curr[x_col] - prev[x_col]
        dy1 = curr[y_col] - prev[y_col]
        dx2 = next_p[x_col] - curr[x_col]
        dy2 = next_p[y_col] - curr[y_col]

        if dx1 > 0 and dx2 > 0:
            slope1 = dy1 / dx1
            slope2 = dy2 / dx2
            # Up then sharply down → spike (threshold now tunable)
            if slope1 > slope_threshold and slope2 < -slope_threshold:
                # Skip this spike point
                continue

        smoothed.append(curr)

    smoothed.append(curve_df.iloc[-1])
    return pd.DataFrame(smoothed).reset_index(drop=True)


def force_edge_points(curve_df, full_df, x_col, y_col):
    """
    Ensure first and last points of full dataset are in the curve.
    """
    if curve_df.empty:
        return full_df.iloc[[0, -1]].copy()

    x0, y0 = full_df[x_col].iloc[0], full_df[y_col].iloc[0]
    xn, yn = full_df[x_col].iloc[-1], full_df[y_col].iloc[-1]

    # Add if missing
    if curve_df[x_col].iloc[0] != x0:
        curve_df = pd.concat([pd.DataFrame([{x_col: x0, y_col: y0}]), curve_df], ignore_index=True)
    if curve_df[x_col].iloc[-1] != xn:
        curve_df = pd.concat([curve_df, pd.DataFrame([{x_col: xn, y_col: yn}])], ignore_index=True)

    curve_df = curve_df.drop_duplicates(subset=[x_col]).sort_values(x_col).reset_index(drop=True)
    return curve_df


def prune_rolling_window(
    curve_df,
    x_col,
    y_col,
    window_size=7,
    rel_threshold=0.3,
    two_sided=False,
):
    """
    Second-pass pruning using a rolling window.

    For each point i (excluding the first/last):
        - Take a window of indices [i - half_window, ..., i + half_window]
        - Exclude i from that window
        - Compute local baseline = median of neighbors' y
        - Compute ratio = y_i / baseline
        - If ratio > 1 + rel_threshold  -> remove as upward spike
        - If two_sided and ratio < 1 - rel_threshold -> remove as downward outlier

    Parameters
    ----------
    curve_df : pd.DataFrame
        Curve to clean (sorted by x).
    x_col, y_col : str
        Column names.
    window_size : int
        Number of points in rolling window (should be odd; if even, it is reduced by 1).
    rel_threshold : float
        Fractional deviation allowed from local median.
        E.g. 0.3 -> 30% higher than neighbors triggers removal.
    two_sided : bool
        If True, also remove points that are too low vs neighbors.

    Returns
    -------
    kept_df : pd.DataFrame
        Curve after pruning.
    removed_df : pd.DataFrame
        Points removed as local outliers (for plotting).
    """
    n = len(curve_df)
    if n <= 2:
        return curve_df.copy(), curve_df.iloc[0:0].copy()

    # Ensure sorted by x, keep original index for reference
    curve_df = curve_df.sort_values(x_col).reset_index(drop=False)  # old index saved as 'index'
    # Adjust window size
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size -= 1
    half = window_size // 2

    kept_rows = []
    removed_rows = []

    for i in range(n):
        row = curve_df.iloc[i]

        # Always keep first and last
        if i == 0 or i == n - 1:
            kept_rows.append(row)
            continue

        # Determine window bounds in index space
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = curve_df.iloc[start:end]

        # Exclude the current point from the baseline calculation
        window_neighbors = window[window.index != row.name]

        if len(window_neighbors) < 2:
            # Not enough neighbors to make a judgment -> keep
            kept_rows.append(row)
            continue

        neighbor_median = window_neighbors[y_col].median()

        # Avoid division issues if local baseline is ~0
        if np.isclose(neighbor_median, 0.0):
            kept_rows.append(row)
            continue

        ratio = row[y_col] / neighbor_median

        remove = False
        if ratio > 1.0 + rel_threshold:
            remove = True
        if two_sided and ratio < 1.0 - rel_threshold:
            remove = True

        if remove:
            removed_rows.append(row)
        else:
            kept_rows.append(row)

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    removed_df = pd.DataFrame(removed_rows).reset_index(drop=True)

    # Drop the temporary 'index' column if present
    if 'index' in kept_df.columns:
        kept_df = kept_df.drop(columns=['index'])
    if 'index' in removed_df.columns:
        removed_df = removed_df.drop(columns=['index'])

    return kept_df, removed_df


# ----------------------------------------------------------------------
# 3. MAIN PROCESSING LOOP
# ----------------------------------------------------------------------
def main():
    print("Finding lower bounding curves that hug the data...\n")

    for dataset_file, num_bins, percentile in DATASETS:
        print(f"Processing {dataset_file}...")

        filepath = os.path.join(BASE_DIR, dataset_file)
        if not os.path.exists(filepath):
            print(f"   [MISSING] {dataset_file}\n")
            continue

        df = pd.read_csv(filepath)

        # 1. Find initial envelope (density-aware)
        curve = find_lower_envelope(
            df,
            'x',
            'y',
            num_bins=num_bins,
            percentile=percentile,
            density_factor=0.15,
            max_points_per_bin=25,
        )

        # 2. Smooth spikes, less aggressive with higher slope_threshold
        curve = smooth_lower_curve(
            curve,
            'x',
            'y',
            remove_spikes=True,
            slope_threshold=0.45,
        )

        # 3. Force edge points so we always include start/end
        curve = force_edge_points(curve, df, 'x', 'y')

        # 4. Second-pass rolling-window pruning
        #    window_size=7 means +/- 3 neighbors, rel_threshold=0.3 means 30% above local median -> removed
        curve, removed_points = prune_rolling_window(
            curve,
            'x',
            'y',
            window_size=2,     # tweak: 5, 7, 9 ...
            rel_threshold=0.3, # tweak: 0.2 (stricter), 0.4 (looser)
            two_sided=False    # True if you ever want to remove "too low" points too
        )

        # 5. Save CSV of final curve (kept points only)
        csv_out = dataset_file.replace('.csv', '_lower_bound.csv')
        curve.to_csv(os.path.join(OUTPUT_DIR, csv_out), index=False)

        # 6. Plot and save PDF
        fig, ax = plt.subplots(figsize=(14, 8))

        # Original data
        ax.scatter(
            df['x'], df['y'],
            alpha=0.5, s=32, c='steelblue',
            edgecolors='none', label='Data points', zorder=1
        )

        # Final bounding curve (kept)
        ax.plot(
            curve['x'], curve['y'],
            'r-', linewidth=3,
            label='Lower bounding curve', zorder=5
        )
        ax.scatter(
            curve['x'], curve['y'],
            c='red', s=60,
            zorder=6, edgecolors='darkred', linewidths=2
        )

        # Removed points as big X's
        if not removed_points.empty:
            ax.scatter(
                removed_points['x'], removed_points['y'],
                marker='x', s=80, linewidths=2,
                c='black', zorder=7, label='Removed (rolling-window outliers)'
            )

        ax.set_xlabel('x', fontsize=13, fontweight='bold')
        ax.set_ylabel('y', fontsize=13, fontweight='bold')
        ax.set_title(
            f'Lower Bounding Curve - {dataset_file.replace('.csv', '').replace('_', ' ')}',
            fontsize=15, fontweight='bold'
        )
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linewidth=0.8)

        pdf_out = dataset_file.replace('.csv', '_with_bound.pdf')
        plt.savefig(os.path.join(OUTPUT_DIR, pdf_out), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   - Original points: {len(df)}")
        print(f"   - Bounding curve points (kept): {len(curve)}")
        print(f"   - Removed rolling-window outliers: {len(removed_points)}")
        print(f"   - Saved: {csv_out}")
        print(f"   - Plot: {pdf_out}\n")

    print("All lower bounding curves computed and saved!")
    print(f"\nOutput folder:\n   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
