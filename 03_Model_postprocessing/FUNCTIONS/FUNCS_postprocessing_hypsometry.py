#%% 
"""Delft3D-4 Flow NetCDF Analysis: Morphological Estuary Analysis: compute hypsometric curves.
Last edit: August 2025
Author: Marloes Bonenkamp
"""

#%% IMPORTS AND SETUP
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import os
import sys

#%%
def calculate_hypsometric_curve(bed_level, x, y, x_min, x_max, y_min, y_max, 
                               bed_threshold=6, n_bins=50):
    """
    Calculate hypsometric curve for a given bed level array within specified bounds.
    
    Parameters:
    - bed_level: 2D array of bed levels (ny, nx)
    - x, y: 2D coordinate arrays
    - x_min, x_max, y_min, y_max: Domain bounds
    - bed_threshold: Threshold to exclude land areas
    - n_bins: Number of elevation bins
    
    Returns:
    - elevations: Array of elevation bin centers
    - cumulative_area: Array of cumulative areas (km²)
    """
    
    # Extract estuary region indices
    x0 = x[:, 0]  # x-coordinates along the estuary
    y0 = y[0, :]  # y-coordinates across the estuary
    
    x_indices = np.where((x0 >= x_min) & (x0 <= x_max))[0]
    y_indices = np.where((y0 >= y_min) & (y0 <= y_max))[0]
    
    # Extract bed level data for estuary region
    bed_estuary = bed_level[np.ix_(x_indices, y_indices)]
    
    # Extract coordinate arrays for estuary region
    x_estuary = x[np.ix_(x_indices, y_indices)]
    y_estuary = y[np.ix_(y_indices, x_indices)].T  # Note: transposed to match dimensions
    
    # Mask out land areas (bed level >= threshold)
    water_mask = bed_estuary < bed_threshold
    
    # Get valid bed levels and corresponding coordinates
    valid_bed_levels = bed_estuary[water_mask]
    valid_x = x_estuary[water_mask]
    valid_y = y_estuary[water_mask]
    
    if len(valid_bed_levels) == 0:
        print("Warning: No valid water points found in estuary region")
        return np.array([]), np.array([])
    
    # Calculate cell areas (accounting for variable grid spacing)
    # For each cell, calculate area based on local grid spacing
    cell_areas = []
    
    for i in range(len(valid_x)):
        # Find the grid cell this point belongs to
        x_coord = valid_x[i]
        y_coord = valid_y[i]
        
        # Find nearest grid indices
        x_idx = np.argmin(np.abs(x0 - x_coord))
        y_idx = np.argmin(np.abs(y0 - y_coord))
        
        # Calculate local grid spacing
        if x_idx < len(x0) - 1:
            dx = x0[x_idx + 1] - x0[x_idx]
        else:
            dx = x0[x_idx] - x0[x_idx - 1]
            
        if y_idx < len(y0) - 1:
            dy = y0[y_idx + 1] - y0[y_idx]
        else:
            dy = y0[y_idx] - y0[y_idx - 1]
        
        cell_area = dx * dy  # in m²
        cell_areas.append(cell_area)
    
    cell_areas = np.array(cell_areas)
    
    # Create elevation bins
    min_elevation = np.min(valid_bed_levels)
    max_elevation = np.max(valid_bed_levels)
    
    bin_edges = np.linspace(min_elevation, max_elevation, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate cumulative area for each elevation
    cumulative_areas = []
    
    for elevation in bin_centers:
        # Find all areas at or below this elevation
        below_mask = valid_bed_levels <= elevation
        total_area = np.sum(cell_areas[below_mask])  # in m²
        total_area_km2 = total_area / 1e6  # convert to km²
        cumulative_areas.append(total_area_km2)
    
    return bin_centers, np.array(cumulative_areas)

def plot_hypsometric_curves(bedlev, x, y, x_min, x_max, y_min, y_max, 
                           bed_threshold=6, timesteps=None, reference_timestep=0,
                           scenario='', save_dir='', save_figure=True):
    """
    Plot hypsometric curves for multiple timesteps.
    
    Parameters:
    - bedlev: 3D array of bed levels (time, ny, nx)
    - x, y: 2D coordinate arrays
    - x_min, x_max, y_min, y_max: Estuary bounds
    - bed_threshold: Threshold to exclude land
    - timesteps: Array of timesteps to plot (default: first 10 with step 2)
    - reference_timestep: Timestep to plot as grey reference line
    - scenario: Scenario name for plot title
    - save_dir: Directory to save figure
    - save_figure: Whether to save the figure
    """
    
    if timesteps is None:
        timesteps = np.arange(1, min(10, bedlev.shape[0]), 2)
    
    plt.figure(figsize=(12, 8))
    
    # Plot reference timestep (t=0) as grey line
    print(f"Calculating hypsometric curve for reference timestep {reference_timestep}...")
    elevations_ref, areas_ref = calculate_hypsometric_curve(
        bedlev[reference_timestep], x, y, x_min, x_max, y_min, y_max, bed_threshold
    )
    
    if len(elevations_ref) > 0:
        plt.plot(areas_ref, elevations_ref, color='grey', linewidth=2.5, 
                label=f't = {reference_timestep} (reference)', alpha=0.8)
    
    # Plot other timesteps with Blues colormap
    if len(timesteps) > 0:
        colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(timesteps)))
        
        for i, timestep in enumerate(timesteps):
            if timestep >= bedlev.shape[0]:
                print(f"Warning: Timestep {timestep} exceeds available data range (max: {bedlev.shape[0]-1})")
                continue
                
            print(f"Calculating hypsometric curve for timestep {timestep}...")
            elevations, areas = calculate_hypsometric_curve(
                bedlev[timestep], x, y, x_min, x_max, y_min, y_max, bed_threshold
            )
            
            if len(elevations) > 0:
                plt.plot(areas, elevations, color=colors[i], linewidth=2, 
                        label=f't = {timestep}', alpha=0.9)
    
    # Formatting
    plt.xlabel('Cumulative Area [km²]', fontsize=12)
    plt.ylabel('Elevation [m]', fontsize=12)
    plt.title(f'Hypsometric Curves - Estuary Evolution\n{scenario}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add horizontal line at bed threshold
    plt.axhline(y=bed_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'Land threshold ({bed_threshold} m)')
    
    plt.tight_layout()
    
    if save_figure and save_dir:
        filename = f'hypsometric_curves_{scenario}_t{reference_timestep}_{timesteps[0]}to{timesteps[-1]}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Hypsometric curves saved to {save_dir}")
    
    plt.show()
    
    return elevations_ref, areas_ref
