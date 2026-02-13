# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:18:11 2025

Spiritwood Warwick model version with area outside of WW in layer 1 inactive

@author: shjordan
"""

import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))

import re
import pyemu
import flopy
import numpy as np
from flopy.utils import Raster
from pyproj import CRS
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pykrige.uk import UniversalKriging
from shapely.geometry import LineString
from matplotlib.backends.backend_pdf import PdfPages
import shutil
from datetime import date
from scipy.interpolate import interp1d
import calendar

# Function to process water levels
from spirit_war01_water_level_process import wl_process_main



# ----------------------------------------------------------------
# Function to find the mf6 executable.
# ----------------------------------------------------------------
def find_mf6_exe(bin_folder=os.path.join("..", "..", "bin")):
    """
    Detect the current OS and return the path to the mf6 executable.
    """
    import platform

    osys = platform.system().lower()
    if "windows" in osys:
        exe_path = os.path.join(bin_folder, "win", "mf6.exe")
    elif "linux" in osys:
        exe_path = os.path.join(bin_folder, "linux", "mf6")
    elif "darwin" in osys:
        exe_path = os.path.join(bin_folder, "mac", "mf6")
    else:
        raise OSError(f"Unsupported platform: {osys}")
    if not os.path.isfile(exe_path):
        raise OSError(f"Expected mf6 binary not found: {exe_path}")
    return exe_path


# ------------------------------------------------
# Determine extent of model based on raster inputs
# Or, use shapefile designating extent
# ------------------------------------------------
def calc_model_extent():
    extent = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','spiritwood_leapfrog_model_extent.shp'))
    bounds = extent.bounds.values[0]
    crs = extent.crs
    return bounds,crs    


# ----------------
# Build model grid
# ----------------
def construct_modelgrid(bounds,crs):
    print("....Building model grid")
    # Get model domain width (Lx) and height (Ly) from raster extent
    Lx = bounds[2] - bounds[0]  # X-axis length (ft)
    Ly = bounds[3] - bounds[1]  # Y-axis length (ft)

    # Compute grid spacing
    delr = int(Lx / ncol)  # Cell width (ft)
    delc = int(Ly / nrow)  # Cell height (ft)

    print(f"Model Domain: {Lx:.2f} ft × {Ly:.2f} ft")
    print(f"Grid Size: {nrow} rows × {ncol} columns")
    print(f"Cell Size: {delr:.2f} ft × {delc:.2f} ft")

    # Define model origin (from raster extent)
    xmin = bounds[0]
    ymin = bounds[1]
    
    modelgrid = flopy.discretization.StructuredGrid(
                delr=np.full(ncol, delr),
                delc=np.full(nrow, delc),
                # Lower left corner
                xoff=xmin,
                yoff=ymin,
                crs=crs
            )
    
    # Save as a shapefile
    savePath = os.path.join('..','..','gis','output_shps','sw_ww')
    print(f"Saving model grid to {savePath}")
    modelgrid.write_shapefile(filename=os.path.join(savePath,'sw_ww_modelgrid.shp'))
    
    return modelgrid, delr, delc


# ----------------------------------------------------------------------
# Sample processed rasters onto the modelgrid to create layer elevations
# ----------------------------------------------------------------------
def sample_rasters(model_grid,
                   force_resample=False,
                   surface_smoothing=True):
    
    # Load grid-scale rasters if they exist, if not, create them
    if os.path.exists(os.path.join('data','analyzed','top_4.npy')) and not force_resample:
        print("loading top, botm, idomain from .npy files")
        top_grid = np.load(os.path.join('data','analyzed','top_4.npy'))
        botm = np.load(os.path.join('data','analyzed','botm_4.npy'))
        idomain = np.load(os.path.join('data','analyzed','idomain_4.npy'))
    
    else:
        print("....Sampling rasters to model grid")
        # --- Load processed rasters
        # Bottom of model
        model_bottom = Raster.load(os.path.join('..','..','gis','output_ras','sw_ww', 'model_bottom.tif'))
        # Spiritwood aquifer
        spiritwood_top = Raster.load(os.path.join('..','..','gis','output_ras','sw_ww','top_of_SW.tif'))
        # Confining Unit
        confining_top = Raster.load(os.path.join('..','..','gis','output_ras','sw_ww','confining_unit.tif'))
        # Top of model
        model_top_frog = Raster.load(os.path.join('..','..','gis','output_ras','sw_ww','model_top.tif'))
        
        model_top = Raster.load(os.path.join('..','..','gis','input_ras','sw_ww','combined_usgs_1_3_arc_sec.tif'))
        
        # --- Sample raster values to model grid
        def raster_2_grid(layer,method='median'):
            # Sample raster to model_grid
            grid_raster = layer.resample_to_grid(
                model_grid,
                band=1,
                method=method,
                extrapolate_edges=False
                ) 
            
            # Set nodata to NaN
            nodata = 1.70141e+38  # --> Raster nodata value 
            grid_raster[(grid_raster==nodata)|(grid_raster==-99999)] = np.nan
            
            return grid_raster
        
        # Top of model
        print("Resampling Top")
        top = raster_2_grid(model_top)
        top = top * 3.2808399  # Meters to ft!
        top_frog = raster_2_grid(model_top_frog)
        
        if surface_smoothing:
            sigma = 1
            print(f"Smoothing top elevation using sigma = {sigma}")
            top = gaussian_filter(top,sigma=sigma)
        
        # Bottom of layer 1 (aka top of confining unit)
        print("Resampling Botm1")
        botm1 = raster_2_grid(confining_top)
        
        # Bottom of layer 2 (aka top of spiritwood)
        print("Resampling Botm2")
        botm2 = raster_2_grid(spiritwood_top)    
        
        # Bottom of main layers, top of bedrock
        print("Resampling Model Bottom")
        botm3 = raster_2_grid(model_bottom)
        
        # Save copies for analysis
        top_grid = top.copy()
        botm1_grid = botm1.copy()
        botm2_grid = botm2.copy()
        botm3_grid = botm3.copy()
        
        # --- Create ibound arrays: 1 where data exists, -1 or 0 where data is NaN
        idom1 = np.where((top_frog==botm1_grid) | (top_frog==botm2_grid) | (top_frog==botm3_grid),-1,1)
        idom1 = np.where(np.isnan(top_frog),-1,idom1)
        idom2 = np.where(np.isnan(botm1_grid),-1,1)
        idom3 = np.where(np.isnan(botm2_grid),-1,1)
        # Bedrock layer --> Always active
        idom4 = np.full((nrow,ncol),1)
        
        # Fix NaNs in layer 1 and set min thickness of 5-feet
        botm1_grid = np.where(top_grid-botm1_grid < 5, top_grid-5, botm1_grid)
        botm1_grid = np.where(np.isnan(botm1_grid), top_grid - 0.1, botm1_grid)
        
        # Fix true thin cells
        botm2_grid = np.where(botm1_grid-botm2_grid < 5, botm1_grid - 5, botm2_grid)
        
        # Fix NaNs in layer 2 and deactivate
        botm2_grid = np.where(np.isnan(botm2_grid), botm3_grid, botm2_grid)
        botm2_grid = np.where(np.isnan(botm2_grid), botm1_grid, botm2_grid)
        botm2_grid = np.where(botm1_grid-botm2_grid < 5, botm1_grid - 0.1, botm2_grid)
        idom2 = np.where(botm1_grid-botm2_grid < 1, -1, idom2)
        
    
        # Ensure at least 5-ft thickness in layer 3
        botm3_grid = np.where(botm2_grid-botm3_grid < 5 ,botm2_grid - 5, botm3_grid)
        botm3_grid = np.where(idom3==-1,botm2_grid-0.1,botm3_grid)
        
        # --- Fixing a few cells manually, removing layer 1 and setting top of layer 2 to these locations
        # --- This is a fix for some cells within Stump lake that end up isolated in layer 1 (surrounded by idom=-1)
        cells = [[48,138],[48,137],[49,138],[49,139],[50,139],[52,142]]
        for cell in cells:
            idom1[cell[0],cell[1]] = -1
            # Set bottom to top elevation
            botm1_grid[cell[0],cell[1]] = top_grid[cell[0],cell[1]]
            # Force 5-ft thickness
            top_grid[cell[0],cell[1]] = top_grid[cell[0],cell[1]] + 5
        
        # Create a 100-ft thick bedrock layer
        bot_bedrock = botm3_grid - 100
        
        # --- Stack idomain and bottom arrays for MODFLOW input
        idomain = np.stack([idom1, idom2, idom3, idom4])
        botm = np.stack([botm1_grid, botm2_grid, botm3_grid, bot_bedrock])
    
        # --- Save as .npy so we don't have to do this again
        print("...Saving structure arrays")
        np.save(os.path.join('data','analyzed','top_4.npy'),top)
        np.save(os.path.join('data','analyzed','botm_4.npy'),botm)
        np.save(os.path.join('data','analyzed','idomain_4.npy'),idomain)
        
    return top_grid,botm,idomain


# ----------------------------
# Build the DRN package inputs
# ----------------------------
def build_DRN(top,bot,idomain,K=1):
    """
    Build the DRN inputs

    Parameters
    ----------
    top : array
        Top elevation of model.
    bot : array
        Bottom elevation of layers

    Returns
    -------
    drn_cells : TYPE
        DESCRIPTION.
    drn_geo : TYPE
        DESCRIPTION.

    """
    print("....Building DRN inputs")
    
    # --- Load grid shapefile
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww','sw_ww_modelgrid.shp'))
    grid = grid.set_crs(2265)
    grid['cell_area'] = grid.geometry.area
    grid['DRN'] = 0  # Initialize flag

    # --- Load DRN lines (e.g., canals)
    drn_lines = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','DRN_lines.shp')).to_crs(grid.crs)

    # Intersect line features with grid and compute segment lengths
    line_intersections = gpd.overlay(grid[['node', 'geometry']], drn_lines[['geometry']], how='intersection',
                                     keep_geom_type=False)
    line_intersections['segment_length'] = line_intersections.geometry.length
    segment_lengths = line_intersections.groupby('node').agg({'segment_length': 'sum'}).reset_index()
    
    # Mark DRN flag and merge segment lengths
    grid.loc[grid['node'].isin(segment_lengths['node']), 'DRN'] = 1
    grid = pd.merge(grid, segment_lengths, on='node', how='left')
    grid['segment_length'] = grid['segment_length'].fillna(0)

    # --- Subset to DRN cells
    drn_cells = grid[grid['DRN'] == 1].copy()
    
    # --- Drop small segment lengths
    drn_cells = drn_cells.loc[drn_cells['segment_length'] > 100]
    
    # --- Elevation of DRNs
    elevations = []
    drop_cells = []
    for i, j in zip(drn_cells['row'], drn_cells['column']):
            elevations.append(top[i-1, j-1])
            if idomain[0,i-1,j-1] != 1:
                drop_cells.append(False)
            else:
                drop_cells.append(True)
            
    drn_cells['elevation'] = elevations
    
    # Drop DRNs not in layer 1
    drn_cells = drn_cells.loc[drop_cells]
    
    # Constant 20-ft width for smaller drains
    drn_cells['width'] = 20
    
    # Larger width for Tolna Coulee
    # for _,row in riv_elevations.iterrows():
    #     r = row['row']
    #     c = row['column']
    #     drn_cells.loc[(drn_cells['row'] == r) &
    #                   (drn_cells['column'] == c),'width'] = 60
    
    
    # Stage is set to the top of the cell, basically a surface discharge
    drn_cells['stage'] = drn_cells['elevation']

    # --- Conductance calculation
    # Assume vertical K = 1 ft/day and vertical thickness = 2 ft
    thickness = 2
    drn_cells['cond'] = (drn_cells['segment_length'] * drn_cells['width']) * K / thickness

    # --- Plot a histogram of conductance values
    avg_cond = drn_cells['cond'].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_title('Histogram of Initial Estimate\n DRN Conductances')
    ax.hist(drn_cells['cond'], bins=50, color='blue', alpha=0.7,edgecolor='black')
    ax.set_xlabel('Conductance (ft/d)')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.axvline(avg_cond, color='black', linestyle='dashed', linewidth=2)
    props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1)
    text_x = avg_cond + avg_cond * 0.35  # Adjust text position
    text_y = 50  # Adjust vertical position
    ax.text(text_x, text_y, f'Average = {avg_cond:,.0f}', color='black', fontsize=10, verticalalignment='center', bbox=props)
    ax.annotate('', xy=(text_x, text_y), xytext=(avg_cond, 50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    
    # --- Cut to only necessary columns
    drn_geo = drn_cells.copy()
    drn_cells = drn_cells[['row','column','stage','cond']]
    
    # --- Add DRN's along Sheyenne River Canyon
    # METHOD 1 -> drains along the entire canyon
    sheyenne_drns = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','DRN_river_valley_V2.shp')).to_crs(grid.crs)
    # sheyenne_drns['tolna'] = False
    # tolna_drns = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','DRN_tolna_valley.shp')).to_crs(grid.crs)
    # tolna_drns['tolna'] = True
    
    # riv_drns = pd.concat([sheyenne_drns,tolna_drns])
    riv_drns = sheyenne_drns
    # METHOD 2 -> Drains at specific locations where it seems more seepage occurs
    # drn_polys = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','DRN_polys.shp')).to_crs(grid.crs)
    # drn_polys = drn_polys[['geometry']]
    # riv_drns = gpd.sjoin(grid,drn_polys)
    
    stages = []
    drop_rows = []
    for _,row in riv_drns.iterrows():
        i = row['row'] - 1
        j = row['column'] - 1
        if idomain[0,i,j] != 1:
            drop_rows.append(False)
        else:
            drop_rows.append(True)
            
        # Set stage to middle of cell --> for method 1
        # stage = bot[0,i,j] + (top[i,j] - bot[0,i,j]) / 2
        
        # Set as bottom of cell --> for method 2
        # Likely needs some refinement... 
        stage = bot[0,i,j] + 0.1
        
        # if row['tolna']:
        #     stage = bot[0,i,j] + (top[i,j] - bot[0,i,j]) / 2
        # else:
        # For Sheyenne DRNs, trying to make them a bit more realistic
        if stage < top[i+1,j]:
            print("STAGE ONE CELL DOWN-GRADIENT")
            stage = top[i+1,j]
        
        stages.append(stage)
        
    riv_drns['stage'] = stages
    riv_drns = riv_drns.loc[drop_rows]
    riv_drns['cond'] = avg_cond / 2
    riv_drns = riv_drns[['row','column','stage','cond']]
    
    drn_cells = pd.concat([drn_cells,riv_drns]).reset_index()    

    # drn_cells = riv_drns.reset_index()
    
    return drn_cells,drn_geo

# ---------------------------
# Creat RIV Inputs for rivers
# ---------------------------
def build_RIV(top,botm,idom,K=1):
    print("....Building RIV inputs")

    # --- Load model grid
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww', 'sw_ww_modelgrid.shp'))
    grid = grid.set_crs(2265)
    grid['cell_area'] = grid.geometry.area

    # --- Load RIV line features (e.g., rivers/canals)
    riv_lines = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww', 'RIV_lines.shp')).to_crs(grid.crs)
    
    # --- Only build the RIV for the Sheyenne - going to model Coulee as a DRN
    riv_lines = riv_lines.loc[~riv_lines['resolution'].isna()]
    
    # Load the RIV elevation reference cells
    # Steps done in QGiS to make elevations:
    #   1. Intersect grid with RIV line
    #   2. Generate points (pixel centroids) inside polygons
    #   3. Sample elevation raster at points
    #   4. Join attriobutes by location (summary)
    #      - Use q1, mean, and q3
    riv_elevations = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww', 'RIV_elevations_reference.shp')).to_crs(grid.crs)
    
    # Using Q1 elevation as RIV elevation to capture low stream channel
    riv_elevations['q1_elevation_ft'] = riv_elevations['SAMPLE_v_1'] * 3.28
    
    def force_2d(geom):
        if geom.has_z:
            return LineString([(x, y) for x, y, *_ in geom.coords])
        return geom
    riv_lines['geometry'] = riv_lines['geometry'].apply(force_2d)

    # --- Intersect RIV lines with model grid
    riv_intersections = gpd.overlay(grid[['node', 'row', 'column', 'geometry']], riv_lines[['geometry']], how='intersection',
                                    keep_geom_type=False)
    
    riv_intersections['segment_length'] = riv_intersections.geometry.length

    # --- Drop small fragments (less than 100 ft)
    # riv_intersections = riv_intersections[riv_intersections['segment_length'] > 100]
    
    # --- Deal with segments within inactive zones of layers 1, 2, or 3
    riv_intersections['k'] = np.zeros(len(riv_intersections))
    drop_indices = []
    for i, row in riv_intersections.iterrows():
        r = int(row['row']) - 1
        c = int(row['column']) - 1
        idom1 = idom[0, r, c]
        idom2 = idom[1, r, c]
        idom3 = idom[2, r, c]
        idom4 = idom[3, r, c]
        
        # Try to put riv in each layer, moving down through the model
        if idom1 == 1:
            riv_intersections.at[i, 'k'] = 0  # layer 1 
        elif idom2 == 1:
            riv_intersections.at[i, 'k'] = 1  # layer 2
        elif idom3 == 1:
            riv_intersections.at[i, 'k'] = 2  # layer 3
        elif idom4 == 1:
            riv_intersections.at[i, 'k'] = 3  # layer 4
        # Drop the riv cell if everything is inactive
        else:
            drop_indices.append(i)
            
    # Drop inactive riv cells
    riv_intersections = riv_intersections.drop(index=drop_indices).reset_index(drop=True)
    
    # Merge elevations with the intersection of the riv and the grid
    riv_intersections = pd.merge(riv_intersections,riv_elevations,on=['row','column'])
    
    # --- Assign elevation from top of model --> Based on the layer the RIV package is in
    elevations = []
    for i, row in riv_intersections.iterrows():
        r = int(row['row']) - 1
        c = int(row['column']) - 1
        k = int(row['k'])
        if k == 0:
            # elev = top[r, c]
            elev = botm[k, r, c]
        else:
            elev = botm[k, r, c]  # bottom of previous layer is the top of current layer
        elevations.append(elev)
    
    # Set RIV elevations based on updated QGiS workflow
    # If that elevation is below cell bottom, default to old workflow
    riv_intersections['elevation'] = [elev if elev>=botm else botm for elev,botm in zip(riv_intersections['q1_elevation_ft'],elevations)]
    
    # Set constant width and a foot of head for stage
    riv_intersections['width'] = 65.0
    # Want RIV cell right at the elevation specified
    # Should be ending with a conductance btwen ~1000 and 20000
    riv_intersections['rbot'] = riv_intersections['elevation'] + 0.1
    # Average gage height on Sheyanne is ~2.33 ft --> This is replaced later with actual time variable gage measurements
    riv_intersections['stage'] = riv_intersections['rbot'] + 2.33

    # --- Compute conductance
    thickness = 2 # --> Constant streambed thickness
    riv_intersections['cond'] = (riv_intersections['segment_length'] * 
                                 riv_intersections['width'] * 
                                 (K / thickness)
                                 )

    # --- Clean output
    riv_intersections['node'] = riv_intersections['node_x']
    riv_intersections['geometry'] = riv_intersections['geometry_x']
    riv_df = riv_intersections[['node', 'row', 'column', 'stage', 'cond', 'rbot', 'segment_length', 'k', 'geometry']].copy()
    riv_df = gpd.GeoDataFrame(riv_df, geometry='geometry', crs=grid.crs)

    # --- Plot histogram of conductance
    avg_cond = riv_df['cond'].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_title('Histogram of Initial Estimate\nRiver Conductances')
    ax.hist(riv_df['cond'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Conductance (ft²/day)')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    ax.axvline(avg_cond, color='black', linestyle='dashed', linewidth=2)
    ax.text(avg_cond, ax.get_ylim()[1]*0.9, f'Avg: {avg_cond:.0f}', ha='right', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    return riv_df


# --------------------------
# Creat RIV Inputs for lakes 
# --------------------------
def build_RIV_lakes(top, botm, idom):
    print("....Building lake-based RIV inputs")

    # --- Load model grid
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww', 'sw_ww_modelgrid.shp'))
    grid = grid.set_crs(2265)
    grid['cell_area'] = grid.geometry.area

    # --- Load lake polygons
    lakes = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','lake_polygons.shp')).to_crs(grid.crs)

    # --- Intersect lakes with grid to compute coverage
    intersection = gpd.overlay(grid[['node', 'row', 'column', 'geometry', 'cell_area']], lakes[['geometry','main_lakes','devil_lake','stump_lake']], how='intersection')
    intersection['intersection_area'] = intersection.geometry.area
    intersection['percent_covered'] = intersection['intersection_area'] / intersection['cell_area']

    # --- Keep cells with >50% coverage
    covered = intersection[intersection['percent_covered'] > 0.5]
    covered = covered.drop_duplicates(subset='node')  # in case of multi-polygon overlap

    
    # --- Deal with segments within inactive zones of layers 1, 2, and 3
    covered['k'] = np.zeros(len(covered))
    drop_indices = []
    for i, row in covered.iterrows():
        r = int(row['row']) - 1
        c = int(row['column']) - 1
        
        # All lakes in layer 2
        idom2 = idom[1, r, c]
        covered.at[i, 'k'] = 1
        
        if idom2 != 1:
            drop_indices.append(i)
            
    covered = covered.drop(index=drop_indices).reset_index(drop=True)
                
    # --- Assign elevation from top of model --> Based on the layer the RIV package is in
    elevations = []
    for i, row in covered.iterrows():
        r = int(row['row']) - 1
        c = int(row['column']) - 1
        k = int(row['k'])
        if k == 0:
            elev = top[r, c]
        else:
            elev = botm[k - 1, r, c]  # bottom of previous layer is the top of current layer
        elevations.append(elev)
    covered['elevation'] = elevations

    # --- Set width from square root of cell area (to mimic full cell interaction)
    covered['width'] = np.sqrt(covered['cell_area'])
    covered['length'] = covered['width']  # assume full cell base

    # --- Set stage and rbot
    covered['rbot'] = covered['elevation'] - 4.5
    covered['stage'] = covered['elevation'] + 40 # Placeholder, this is updated with actual time-variable gage data later
    
    # --- Constant lakebed thickness of 5 ft
    thickness = 5

    # --- Compute conductance using full cell area as interface
    K = 0.1  # --> Low K since its a lake, likely clogged with fines
    covered['cond'] = covered['cell_area'] * (K / thickness) 
    
    # --- Final formatting
    riv_lake_df = covered[['node', 'row', 'column', 'stage', 'rbot', 'cond', 'main_lakes','devil_lake','stump_lake','elevation','geometry']].copy()
    riv_lake_df = gpd.GeoDataFrame(riv_lake_df, geometry='geometry', crs=grid.crs)

    # --- Plot a histogram of conductance values
    avg_cond = riv_lake_df['cond'].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_title('Histogram of Initial Estimate\nLake Conductances')
    ax.hist(riv_lake_df['cond'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Conductance (ft²/day)')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    ax.axvline(avg_cond, color='black', linestyle='dashed', linewidth=2)
    ax.text(avg_cond, ax.get_ylim()[1]*0.9, f'Avg: {avg_cond:.0f}', ha='right', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    return riv_lake_df


# --------------------------------------------------------------
# --- Krig water levels to establish GHBs (and for plotting WLs)
# --------------------------------------------------------------
def krig_wl(gdf,idom,layer,tag=None):
    xmin,ymin,xmax,ymax = bounds[0],bounds[1],bounds[2],bounds[3]
    # Define grid to interpolate over
    grid_x = np.linspace(xmin, xmax, ncol)
    grid_y = np.linspace(ymin, ymax, nrow)
    
    # Use universal kriging to generate the surface
    UK = UniversalKriging(
                gdf.geometry.x, 
                gdf.geometry.y,
                gdf['Water_Level(NAVD88)'],
                variogram_model="spherical",   # --> Spherical model
                verbose=False, 
                enable_plotting=False, # Set True to plot variogram
                # drift_terms=['regional_linear'],
            )
    
    # Compute surface
    initial_heads, _ = UK.execute("grid", grid_x, grid_y)
    initial_heads = np.flipud(initial_heads)
    if idom.shape == (3,nrow,ncol):
        initial_heads = np.where(idom[layer-1,:,:]!=1,np.nan,initial_heads)
    else:
        initial_heads = np.where(idom!=1,np.nan,initial_heads)
    
    if layer == 1:
        t = np.full((nrow,ncol),0)  
        # t = create_WW_K(t,1)
        initial_heads = np.where(t==0,initial_heads,np.nan)
    
    # Load some other plot elements
    lakes = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww', 'devils_stump_outline.shp')).to_crs(2265)
    riv = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww', 'RIV_lines.shp')).to_crs(2265)
    ww = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww', 'Aquifers.shp')).to_crs(2265)
    ww = ww.loc[ww['name']=='Warwick Aquifer']
    
    # Plot
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww','sw_ww_modelgrid.shp'))
    grid = grid.set_crs(2265)
    grid['heads'] = initial_heads.flatten()
    if layer == 3:
        fig,ax = plt.subplots(figsize=[8,8])
    else:
        fig,ax = plt.subplots(figsize=[8,5])
    
    if layer == 1:
        grid = grid.clip(ww)
    grid.plot(column='heads',legend=True,ax=ax)
    gdf.plot(ax=ax,color='k',markersize=2)
    # Draw contour lines
    sw_ic_2d = np.flipud(initial_heads).reshape((nrow, ncol))
    X, Y = np.meshgrid(grid_x, grid_y)
    cs = ax.contour(X, Y, sw_ic_2d, colors='black', linewidths=0.5,levels=12)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    
    # Plot lakes
    lakes.boundary.plot(ax=ax,color='k')
    riv.plot(ax=ax,color='blue')
    ww.boundary.plot(color='red',ax=ax)
    
    # For Warwick, zoom in on the aquifer
    # if layer == 1:
        # ax.set_ylim(290000,345000)
        # ax.set_xlim(right=2.465e6)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    if tag:
        ax.set_title(f"SW WL {tag}")
    else:
        ax.set_title(f"Interpolated Initial Water Levels in layer {layer}")
        
    return initial_heads,fig


# ---------------------------------------------------------
# --- Interp Wl data to get initial and boundary conditions
# ---------------------------------------------------------
def make_SW_IC(idom):
    # --- Load water level and site data
    water_levels = pd.read_csv(os.path.join("data","analyzed","water_levels","processed_water_levels.csv"),index_col=0,parse_dates=True)
    sites = gpd.read_file(os.path.join("data","analyzed","water_levels","processed_sites.shp"))
    
    # --- SW initial water levels
    sw_wl = water_levels.loc[water_levels['Site_Index'].isin(sites.loc[sites['Aquifer']=='Spiritwood','Site_Index'].values)]
    sw_sub = sw_wl.loc[sw_wl.index<pd.Timestamp('1974-01-01')]
    sw_sub = sw_sub[['Site_Index','Water_Level(NAVD88)']].groupby('Site_Index').mean()
    
    # Grab only sites with ~1970 data and merge with average water level value
    sw_sites = pd.merge(sites,sw_sub,left_on='Site_Index',right_on='Site_Index')
    sw_sites = sw_sites.loc[~sw_sites['Site_Index'].isin([3433,])]
    # sw_sites.to_file(os.path.join("data","analyzed","water_levels","sw_ic.shp"))
    
    # Create initial water levels and create BCs based off of that
    sw_ic,_ = krig_wl(sw_sites,idom,layer=3)
    
    return sw_ic

# --- This is not in use, just using model top as initial water levels for WW
def make_WW_IC(idom):
    # --- Load water level and site data
    water_levels = pd.read_csv(os.path.join("data","analyzed","water_levels","processed_water_levels.csv"),index_col=0,parse_dates=True)
    sites = gpd.read_file(os.path.join("data","analyzed","water_levels","processed_sites.shp"))
    
    # --- SW initial water levels
    ww_wl = water_levels.loc[water_levels['Site_Index'].isin(sites.loc[sites['Aquifer']=='Warwick','Site_Index'].values)]
    ww_sub = ww_wl.loc[ww_wl.index<pd.Timestamp('1974-01-01')]
    ww_sub = ww_sub[['Site_Index','Water_Level(NAVD88)']].groupby('Site_Index').mean()
    
    # Grab only sites with ~1970 data and merge with average water level value
    ww_sites = pd.merge(sites,ww_sub,left_on='Site_Index',right_on='Site_Index')
    ww_sites = ww_sites.loc[~ww_sites['Site_Index'].isin([3433,])]
    # sw_sites.to_file(os.path.join("data","analyzed","water_levels","sw_ic.shp"))
    
    # Create initial water levels and create BCs based off of that
    idom = np.full((nrow,ncol),1)
    ww_ic,_ = krig_wl(ww_sites,idom,layer=1)
    
    return ww_ic


# ----------------------------------------------------------------------------
# --- Plot interpolated heads in the Spiritwood aquifer, moving 5-year average
# ----------------------------------------------------------------------------
def plot_interp_WLs(idom,aquifer='Spiritwood',layer=3):
    # --- Load water level and site data
    water_levels = pd.read_csv(os.path.join("data","analyzed","water_levels","processed_water_levels.csv"),index_col=0,parse_dates=True)
    sites = gpd.read_file(os.path.join("data","analyzed","water_levels","processed_sites.shp"))
    y1 = 1970
    y2 = 1975
    with PdfPages(os.path.join("figures",f"{aquifer}_WLs.pdf")) as pdf:
        for i in range(10):
            # --- SW initial water levels
            wl = water_levels.loc[water_levels['Site_Index'].isin(sites.loc[sites['Aquifer']==aquifer,'Site_Index'].values)]
            # if i == 0:
            #     devils_ele = pd.read_csv(os.path.join("data","analyzed","devils_elevation_processed.csv"),
            #                              index_col=0,parse_dates=True)
            #     sw_avg = sw_wl[['Water_Level(NAVD88)']].resample("YE").mean().mean(axis=1)
            #     fig,ax = plt.subplots(figsize=(9,4))
            #     sw_avg.plot(ax=ax,color='blue',label='SW Avg. WL')
            #     ax2=ax.twinx()
            #     devils_ele.plot(ax=ax2,lw=1,color='k',ls='--',label='Devils Lake Elevation')
            #     ax.set_ylabel("Average SW Water Level")
            #     ax.set_xlabel("Year")
            #     ax.grid()
            #     ax.set_xlim(left=pd.Timestamp("1970-01-01"))
            #     ax.legend()
                
            wl_sub = wl.loc[(wl.index >= pd.Timestamp(f'{y1}-01-01')) & (wl.index <= pd.Timestamp(f'{y2}-01-01'))]
            wl_sub = wl_sub[['Site_Index','Water_Level(NAVD88)']].groupby('Site_Index').mean()
            
            # Grab only sites with ~1970 data and merge with average water level value
            sw_sites = pd.merge(sites,wl_sub,left_on='Site_Index',right_on='Site_Index')
            sw_sites = sw_sites.loc[~sw_sites['Site_Index'].isin([3433,5365,9307])]
            
            # m = sw_sites.explore()
            # m.save('gw_sites_temp.html')
            # sw_sites.to_file(os.path.join("data","analyzed","water_levels","sw_ic.shp"))
            
            # Create initial water levels and create BCs based off of that
            sw_ic,fig = krig_wl(sw_sites,idom,layer=layer,tag=f"{y1} - {y2}")
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)
            y1 += 5
            y2 += 5
   
        
# -------------------------------------
# --- Helper function to plot GHB heads
# -------------------------------------
def plot_GHBs(ghb_data,layer):
    """
    Helper function to plot GHB heads.

    Parameters
    ----------
    ghb_data : Dict
        Dictionary of ghb stress_period_info organized for flopy
    layer : int
        Model layer associated with the ghb inputs
    
    Returns
    -------
    None.
    """
    # Create empty array for plotting
    ghb_plot = np.full((nrow, ncol), np.nan)
    
    # Fill GHB locations
    for entry in ghb_data:
        (lay, row, col), head, cond = entry
        if lay == 2:  # Only plot if it's really Layer 3 (redundant here but safe)
            ghb_plot[row, col] = head  # or cond if you prefer
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(ghb_plot, cmap='coolwarm', origin='upper')
    plt.colorbar(label='Boundary Head (ft)')  # or Conductance if you plotted that
    plt.title(f'GHB Heads, layer {layer}')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.grid(False)
    plt.savefig(os.path.join("figures",f"layer_{layer}_ghb.png"),dpi=200,bbox_inches='tight')
    #plt.show()


def process_SS_head_targs(idom):
    """
    Build the steady-state inputs for the hobs package.

    Parameters
    ----------
    idom : Numpy array
        Model idomain array.
    
    Returns
    -------
    WL_SS : DataFrame
        Steady-state head targets organized in a DataFrame
    """
    # Check if water level data exists, if it doesnt, call the script to generate processed data
    WL_path = os.path.join("data","analyzed","water_levels","processed_water_levels.csv")
    
    # Load WLs
    water_levels = pd.read_csv(WL_path,index_col=0,parse_dates=True)
    
    # Load sites (already has r/c index)
    sites = gpd.read_file(os.path.join("data","analyzed","water_levels","processed_sites.shp"))
    WL = water_levels.loc[water_levels['Site_Index'].isin(sites.loc[sites['Aquifer'].isin(['Warwick','Spiritwood']),'Site_Index'].values)]
    WL_SS = WL.loc[WL.index<pd.Timestamp('1974-01-01')]
    WL_SS = WL_SS[['Site_Index','Water_Level(NAVD88)']].groupby('Site_Index').mean()
    # Merge SS average WL's into the sites gdf
    WL_SS = pd.merge(sites,WL_SS,left_on='Site_Index',right_on='Site_Index')
    WL_SS.rename(columns={'column':'col'},inplace=True)

    skip_well = 30250  # --> This well is on the other side of the divide next to 9348, cannot hit both targets
    WL_SS = WL_SS.loc[WL_SS['Site_Index'] != skip_well]

    # Create columns to match Ryan's workflow
    WL_SS['k'] = np.zeros(len(WL_SS)).astype(int)
    WL_SS.loc[WL_SS['Aquifer']=='Spiritwood','k'] = int(2)
    WL_SS['i'] = WL_SS['row'] - 1
    WL_SS['j'] = WL_SS['col'] - 1
    WL_SS['obsprefix'] = [f"ssh_id:{WL_SS.loc[x,'Site_Index']}_k:{WL_SS.loc[x,'k']}_i:{WL_SS.loc[x,'i']}_j:{WL_SS.loc[x,'j']}" for x in range(len(WL_SS))]
    WL_SS['obstype'] = ['ss_head' for x in range(len(WL_SS))]
    WL_SS['layer'] = [x+1 for x in WL_SS['k']]
    WL_SS['idom'] = [1 for x in range(len(WL_SS))]
    WL_SS['aquifer'] = ['ww' for x in range(len(WL_SS))]
    WL_SS.loc[WL_SS['k']==2,'aquifer'] = 'sw'
    WL_SS['gwe_ft'] = WL_SS['Water_Level(NAVD88)']
    WL_SS['group'] = [f"sshead_{WL_SS.loc[x,'aquifer']}" for x in range(len(WL_SS))]
    WL_SS['sp'] = np.ones(len(WL_SS))
    WL_SS['start_dt'] = ['12/31/1969' for x in range(len(WL_SS))]
    WL_SS['end_dt'] = ['1/1/1970' for x in range(len(WL_SS))]
    WL_SS['x_2265'] = WL_SS.geometry.x.values
    WL_SS['y_2265'] = WL_SS.geometry.y.values
    WL_SS['well_depth'] = WL_SS['Total_Dept']
    WL_SS['id'] = [x+1 for x in range(len(WL_SS))]
    # Ensure wells are within active grid domain
    WL_SS['valid'] = WL_SS.apply(lambda row: idom[int(row['k']), int(row['i']), int(row['j'])] == 1, axis=1)
    WL_SS = WL_SS[WL_SS['valid']].drop(columns='valid')
    
    # Drop one that's outside of the WW zone
    # WL_SS = WL_SS.loc[WL_SS['Site_Index']!=9711]
    # WL_SS = WL_SS.reset_index(drop=True)
    
    # Order columns same as Wahp
    WL_SS = WL_SS[["id","obsprefix","obstype","group","layer","row","col","k","i","j","idom","aquifer","well_depth","gwe_ft","sp","start_dt","end_dt","x_2265","y_2265","geometry"]]
    # Save to csv
    WL_SS.to_csv(os.path.join("data","analyzed","SS_target_heads.csv"),index=False)

    return WL_SS


def process_transient_head_targs(idom):
    """
    Build the transient inputs for the hobs package.

    Parameters
    ----------
    idom : Numpy array
        Model idomain array.
    
    Returns
    -------
    trans_out : DataFrame
        Transient head targets organized in a DataFrame
    """
    # Load targets
    transient_targs = pd.read_csv(os.path.join('data','analyzed','transient_well_targets.csv'))
    
    # Load well lookup
    lookup_df = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    
    # Melt to long format: one row per (date, Site_Index)
    trans_long = transient_targs.melt(id_vars='start_datetime', var_name='Site_Index', value_name='gwe_ft')

    # Drop NaNs
    trans_long.dropna(subset=['gwe_ft'], inplace=True)
    trans_long['Site_Index'] = trans_long['Site_Index'].astype(float)
    
    # Skip a particular well
    # --> This well is on the other side of the divide next to 9348, cannot hit both targets
    skip_well = 30250.0
    trans_long = trans_long.loc[trans_long['Site_Index'] != skip_well]
    
    # Merge i, j, k info
    trans_long = trans_long.merge(lookup_df, on='Site_Index', how='left')
    trans_long = trans_long.dropna(subset='i')
    # Convert date and add MODFLOW stress period (0-based)
    trans_long['start_datetime'] = pd.to_datetime(trans_long['start_datetime'])
    trans_long['sp'] = int()
    
    # Stress periods for Annual SPs
    trans_long.loc[trans_long['start_datetime']<pd.Timestamp('2000-01-01'), 'sp'] = (trans_long['start_datetime'].dt.year - 1970) + 1
    
    # Stress periods for monthly SPs -- > help
    monthly_mask = trans_long['start_datetime'] >= pd.Timestamp('2000-01-01')
    monthly_dt = trans_long.loc[monthly_mask, 'start_datetime']
    trans_long.loc[monthly_mask, 'sp'] = (
                (monthly_dt.dt.year - 2000) * 12 + monthly_dt.dt.month + 30
            ).astype(int)
    
    # Calculate modflow-friendly indices
    trans_long['i'] = trans_long['i'].astype(int)
    trans_long['j'] = trans_long['j'].astype(int)
    trans_long['k'] = trans_long['k'].astype(int)
    trans_long['row'] = trans_long['i'] + 1
    trans_long['col'] = trans_long['j'] + 1
    trans_long['layer'] = trans_long['k'] + 1
    
    # Two manual fixes --> Wells are screened in confining layer
    trans_long.loc[trans_long['Site_Index']==30232,'k'] = 1
    
    # Build obsprefix
    trans_long['obsprefix'] = trans_long.apply(lambda row: f"trh_id:{int(row.Site_Index)}_k:{row.k}_i:{row.i}_j:{row.j}", axis=1)

    # Required fields only
    trans_out = trans_long[['obsprefix', 'gwe_ft', 'sp', 'k', 'i', 'j', 'start_datetime']].copy()
    
    # Drop anything outside of active model domain
    trans_out['valid'] = trans_out.apply(lambda row: idom[int(row['k']), int(row['i']), int(row['j'])] == 1, axis=1)
    trans_out = trans_out[trans_out['valid']].drop(columns='valid')
    
    # Save csv
    trans_out.to_csv(os.path.join("data", "analyzed", "transient_head_targets.csv"), index=False)

    return trans_out


def create_WW_K(K, ww_k=1):
    """
    Create layer 1 K values based on extent of Warwick aquifer.

    Parameters
    ----------
    K : float
        Array of shape nrow, ncol filled with HK value for the area in layer 1 
        outside of the Warwick aquifer shapefile.
    ww_K : float
        HK value for the area in layer 1 WITHIN Warwick aquifer shapefile.

    Returns
    -------
    K : Numpy array
        Array of HK values in layer 1.
    """
    # Load shapes and clip grid to ww shape
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww', 'sw_ww_modelgrid.shp')).set_crs(2265)
    ww_shape = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww', 'warwick_larger.shp'))
    
    # Ensure crs    
    if ww_shape.crs != grid.crs:
        ww_shape = ww_shape.to_crs(grid.crs)
    ww_grid = grid.clip(ww_shape)
    
    # Assign K in WW based on clipped shape
    for _,row in ww_grid.iterrows():
        i = row['row'] - 1
        j = row['column'] - 1
        K[i,j] = ww_k
   
    return K


def K_confining_layer(k_low=1e-6,
                      recharge_window=True,
                      rch_window_K=150):
    """
    Define HK in the confining unit (layer 2) based on the recharge window shapefile. 
    """

    # Initialize k grid
    k = np.full((nrow,ncol),k_low)
    
    # --- Include high K recharge window
    if recharge_window:
        # Load recharge_window shapefile
        rch_window = gpd.read_file(os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww','sw_recharge_window_large.shp')).to_crs(2265)
        rch_window['window'] = 1
        grid = gpd.read_file(os.path.join("..", "..", "gis", "output_shps", "sw_ww", "sw_ww_modelgrid.shp")).set_crs(2265)
        grid = grid.clip(rch_window)
        for _,row in grid.iterrows():
            i = row['row'] - 1
            j = row['column'] - 1
            k[i,j] = rch_window_K
    
    plt.imshow(k)
    
    return k


def ww_K_flat_lakes(K,K_33):
    """
    Function that will set the K values at the smaller lakes to 10000
    NOT IN USE. Decided against this method because it seems we want to represent these smaller
    lakes with increased recharge, rather than K discontinuities
    """
    # --- Load model grid
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww', 'sw_ww_modelgrid.shp'))
    grid = grid.set_crs(2265)
    grid['cell_area'] = grid.geometry.area
    
    # --- Load lake polygons
    lakes = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','lake_polygons.shp')).to_crs(grid.crs)
    lakes = lakes.loc[lakes['main_lakes']!=1]
    grid = grid.clip(lakes)
    
    # --- Make HK and VK 10,000 anywhere that a smaller lake exists
    for _,row in grid.iterrows():
        i = row['row'] - 1
        j = row['column'] - 1
        # if K[i,j] == np.max(K):
        K[i,j] = 10000
        K_33[0,i,j] = 10000
            
    return K, K_33


def increase_RCH_lakes():
    """
    Helper function to increase the RCH applied in areas with the smaller lakes. Smaller lakes are, generally, 
    named lakes in the model domain exluding Devils and Stump lake. This is called when building the RCH package.

    NOT in use since we swapped to using SWB for recharge

    Parameters
    ----------
    None.

    Returns
    -------
    rch_increase_arr : Numpy array
        Array of 1's (small lake is present) and 0's (no lake) covering the model domain.

    """
    # --- Load model grid
    grid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww', 'sw_ww_modelgrid.shp'))
    grid = grid.set_crs(2265)
    grid['cell_area'] = grid.geometry.area
    
    # --- Load lake polygons
    lakes = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','lake_polygons.shp')).to_crs(grid.crs)
    lakes = lakes.loc[lakes['main_lakes']!=1]
    # prairie_potholes = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','DRN_polys.shp'))
    grid = grid.clip(lakes)
    
    # --- Create array to increase RCH values where the smaller lakes are present
    rch_increase_arr = np.full((nrow,ncol),0)
    for _,row in grid.iterrows():
        i = row['row'] - 1
        j = row['column'] - 1
        rch_increase_arr[i,j] = 1
            
    return rch_increase_arr


def process_lake_elevations():
    """
    Read in USGS gage height data for Devils and Stump lake and process into stage elevation
    for the RIV package. 
    
    This is only used to prescribe GHBs in the Northern area of Spiritwood, since
    we are no longer modeling the lakes.

    Parameters
    ----------
    None.

    Returns
    -------
    df_devils : DataFrame
        Devils lake elevation data resampled to model stress periods
    df_stump : DataFrame
        Stump lake elvation data resampled to model stress periods
    
    """
    # --- Devils lake
    df = pd.read_csv(os.path.join("data","analyzed","devils_elevation_processed.csv"))
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df = df[df['dateTime'] >= '1970-01-01'].set_index('dateTime')
    
    # Split into two periods
    df_1970_1999 = df.loc['1970-01-01':'1999-12-31']
    df_2000_on = df.loc['2000-01-01':]
    
    # Resample
    df_yr = df_1970_1999.resample('YS').mean()  # 'Y' = calendar year-end
    df_mon = df_2000_on.resample('MS').mean()   # 'M' = calendar month-end
    
    # Combine and reset index if needed
    df_combined = pd.concat([df_yr, df_mon])
    df_combined.index.name = 'dateTime'
    df_combined = df_combined.reset_index()
    first_row = df_combined.iloc[[0]].copy()
    
    # Add in an extra for the SS period
    first_row['dateTime'] = pd.to_datetime('1970-01-01')  # Explicitly set to SS start
    df_combined = pd.concat([first_row, df_combined], ignore_index=True)
    
    # Assign SP indices
    df_combined['stress_period'] = range(len(df_combined))
    df_devils = df_combined.copy()
    # Interpolate NaNs
    df_devils = df_devils.interpolate()
    
    # --- Stump lake
    df = pd.read_csv(os.path.join("data","analyzed","stump_elevation_processed.csv"))
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df = df[df['dateTime'] >= '1970-01-01'].set_index('dateTime')
    
    # Split into two periods
    df_1970_1999 = df.loc['1970-01-01':'1999-12-31']
    df_2000_on = df.loc['2000-01-01':]
    
    # Resample
    df_yr = df_1970_1999.resample('YS').mean()  # 'Y' = calendar year-end
    first_row = df_yr.iloc[[0]].copy()
    first_row['dateTime'] = pd.to_datetime('1970-01-01')  # Explicitly set to SS start
    first_row = first_row.set_index('dateTime',drop=True)
    df_yr = pd.concat([first_row, df_yr], ignore_index=False)
    df_yr = df_yr.resample('YS').ffill()
    df_mon = df_2000_on.resample('MS').mean()   # 'M' = calendar month-end
    
    # Combine and reset index if needed
    df_combined = pd.concat([df_yr, df_mon])
    df_combined.index.name = 'dateTime'
    df_combined = df_combined.reset_index()

    # Assign SP indices
    df_combined['stress_period'] = range(len(df_combined))
    df_stump = df_combined.copy()
    # Interpolate NaNs
    df_stump = df_stump.interpolate()
    
    return df_devils,df_stump


def process_RIV_flows():
    """
    Process the USGS gage data for streamflow at the Sheyenne river and convert into stage
    input for the RIV package. This function takes reccorded flow values, fits a spline curve
    to the USGS provided rating curve data, and uses this to solve for river stage.

    Parameters
    ----------
    None.

    Returns
    -------
    df_combined : DataFrame
        Dataframe of processed Sheyenne river flow data resampled to monthly and yearly 
        periods that correspond to the model timeframe. 
    """
    # Load the USGS rating curve
    file_path = os.path.join("data","raw","sheyenne_rating_curve.txt")
    rating_curve  = pd.read_csv(file_path, delim_whitespace=True, usecols=["stage", "flow_cfs"])
    rating_curve  = rating_curve.dropna(subset=["stage", "flow_cfs"])  # drop any rows missing data
    
    # Flow to Stage intepolator
    def create_flow_to_stage_interpolator(df):
        """Create an interpolating function from flow (cfs) to stage (ft)"""
        return interp1d(df['flow_cfs'], df['stage'], kind='linear', fill_value='extrapolate')
    flow_to_stage = create_flow_to_stage_interpolator(rating_curve)
    
    # --- Same process as for lakes
    # Load Sheyenne flow data
    flow_data = pd.read_csv(os.path.join("data","analyzed","sheyenne_cfs_ts.csv"))
    flow_data = flow_data.loc[flow_data.flow_cfs>-999]
    flow_data['dateTime'] = pd.to_datetime(flow_data['dateTime'])
    flow_data.set_index('dateTime', inplace=True)
    
    # Create stage data
    stage_data = flow_to_stage(flow_data['flow_cfs'].values)
    flow_data['stage'] = stage_data
    
    # Split into two periods
    df_1970_1999 = flow_data.loc['1970-01-01':'1999-12-31']
    df_2000_on = flow_data.loc['2000-01-01':]

    # Resample
    df_yr = df_1970_1999['stage'].resample('YS').mean().to_frame()
    df_mon = df_2000_on['stage'].resample('MS').mean().to_frame()

    # Insert synthetic SS row
    first_row = df_yr.iloc[[0]].copy()
    first_row.index = [pd.to_datetime('1970-01-01')]
    df_yr = pd.concat([first_row, df_yr])

    # Combine and assign SP
    df_combined = pd.concat([df_yr, df_mon])
    df_combined = df_combined.reset_index()
    df_combined.rename(columns={'index': 'dateTime'}, inplace=True)
    df_combined['stress_period'] = range(len(df_combined))
    
    # Quick interp to get rid of NaN values
    df_combined = df_combined.interpolate()
    
    # Create plot
    fig,ax = plt.subplots(figsize=(7,4))
    flow_data['stage'].plot(ax=ax,color='royalblue')
    ax.set_ylabel("Sheyenne River Stage (ft)")
    ax.set_xlabel("Year")
    ax.grid()
    ax.set_title("Yearly Average Sheyenne River Stage")
    ax.set_xlim(left=pd.Timestamp('01-01-1970'),
                right=pd.Timestamp('01-01-2000'))
    #lt.show()
    
    return df_combined


def make_WEL(botm,top,idom,ss_pumping=True):
    """
    Build the inputs to the WEL package (lay, row, col, Q) based on processed water use data

    Parameters
    ----------
    botm : Numpy array
        Model bottom elevation array (3D)
    top : Numpy array
        Model top elevation array (2D)
    idom : Numpy array
        Model idomain array
    ss_pumping : Boolean
        Flag to specify the inclusion of pumping in the steady state period. This is estimated to 
        exist at 'Estiamted_well_1' based on the issue year of the water use permit and the observed
        groundwater heads in that area. 
    
    Returns
    -------
    pump_gdf : GeoDataFrame
        GeoDataFrame holding all necessary inputs for the MODFLOW WEL package.
    
    """
    # Load pumping data and grid
    # !!!
    pump_df = pd.read_csv(os.path.join("data", "analyzed", "use_by_well_monthly_bellCurve.csv"))
    pump_df = pump_df.loc[pump_df['cfd']>0]
    grid = gpd.read_file(os.path.join("..", "..", "gis", "output_shps", "sw_ww", "sw_ww_modelgrid.shp")).set_crs(2265)
        
    # Convert pump data to GeoDataFrame
    pump_geo = gpd.points_from_xy(pump_df['x_2265'].values, pump_df['y_2265'].values, crs=2265)
    pump_gdf = gpd.GeoDataFrame(geometry=pump_geo, data=pump_df)
    pump_gdf = pump_gdf.sjoin(grid)
    
    # Add in suspected pumping in years prior to 1976 at Estimated_Well_1
    # This will also include this pumping in the Steady-State period
    if ss_pumping:
        extra_pump = pump_gdf.loc[(pump_gdf['Year']==1976) & 
                                  (pump_gdf['Well']=='Estimated_Well_1')]
        for year in [1969,1970,1971,1972,1973,1974,1975]:
            extra_pump['Year'] = year
            pump_gdf = pd.concat([pump_gdf,extra_pump])
        
    # Calculate pumping layer (should be either SW or WW, so 1 or 3)
    pump_gdf['k'] = np.zeros(len(pump_gdf), dtype=int)
    for well in pump_gdf['Well'].unique():
        depth = pump_gdf.loc[pump_gdf['Well'] == well, 'bottom_scr'].values[0]
        i = int(pump_gdf.loc[pump_gdf['Well'] == well, 'row'].values[0]) - 1
        j = int(pump_gdf.loc[pump_gdf['Well'] == well, 'column'].values[0]) - 1
        screen_elev = top[i, j] - depth  # elevation of bottom of screen
        # Determine layer: screen elevation > bottom => in this layer
        for ly in range(3):
            if screen_elev > botm[ly, i, j]:
                k = ly
                break
        else:
            # If it didn’t break, assign to bottom layer
            k = 2
        # Redirect any pumping in confining layer (Layer 2) to nearest aquifer
        if k == 1:
            d0 = abs(screen_elev - botm[0, i, j])
            d2 = abs(screen_elev - botm[2, i, j])
            k = 0 if d0 < d2 else 2
        
        # All the estimated wells are in Warwick
        if 'Estimated' in well:
            k = 0
        
        # Last check - make sure the layer is active, if its not, move the well
        if (k == 2) & (idom[2,i,j] != 1):
            print(f"\nInactive in SW, moving {well} to WW")
            print(f"Depth for well = {screen_elev}")
            print(f"Bottom of lay 0 = {botm[0, i, j]}\n")
            k = 0
        # Ugly... But works ?
        if (k == 0) & (idom[0,i,j] != 1):
            if idom[2,i,j] != 1:
                print(f"\n!!!! SW and WW Inactive for {well}")
            else:
                print(f"\nInactive in WW, moving {well} to SW")
                k = 2
        else:
            pass
        
        pump_gdf.loc[pump_gdf['Well'] == well, 'k'] = k
    
    # --- Map years to stress period numbers
    pump_gdf['sp'] = [np.nan for x in range(len(pump_gdf))]
    for i in range(30):
        pump_gdf.loc[pump_gdf['Year']==1970+i,'sp'] = int(i + 1)
    
    # --- Grab post-2000 data
    post_2k = pump_gdf.loc[pump_gdf['Year']>=2000,:].copy()
    
    # Steady state period --> Use all pre-1970 data and then average later
    pump_gdf.loc[pump_gdf['Year']<1970,'sp'] = 0
    
    # -- Average rates per stress period
    pump_gdf = pump_gdf[['Well','cfd','row','column','k','sp']]
    
    # --- Agregate
    pump_gdf = pump_gdf.groupby(['Well','sp']).agg({
            'cfd' : 'mean',
            'column' : 'first',
            'row' : 'first',
            'k' : 'first'}
            ).reset_index().dropna()
    
    # --- Analyze post 2000 data (need to map sp to year and month)
    sp_0 = 30
    for year in range(26):
        for month in range(12):
            sp = sp_0 + (year * 12) + (month + 1)
            post_2k.loc[(post_2k['Year'] == 2000 + year) & 
                        (post_2k['Month'] == month + 1), 'sp'] = sp
            
            
    post_2k = post_2k[['Well','cfd','row','column','k','sp']]
    
    # --- Aggregate
    post_2k = post_2k.groupby(['Well','sp']).agg({
            'cfd' : 'mean',
            'column' : 'first',
            'row' : 'first',
            'k' : 'first'}
            ).reset_index().dropna()
    
    pump_gdf = pd.concat([pump_gdf,post_2k],axis=0)
    
    return pump_gdf


# -------------------------------------------
# --- Write the stress period reference table
# -------------------------------------------
def stress_period_df_gen(mws,strt_yr,annual_only=False):
    strt_yr = 1970
    out_tbl_dir = os.path.join('tables')
    if not os.path.exists(out_tbl_dir):
        os.makedirs(out_tbl_dir)
    sim = flopy.mf6.MFSimulation.load(sim_ws=mws, exe_name='mf6',load_only=['dis'])
    start_date = sim.tdis.start_date_time.data
    period_data = sim.tdis.perioddata.array
    nper = sim.tdis.nper.data    
    
    if annual_only:
        spd = pd.DataFrame(period_data)
        # from spd perlen array, get cumulative days:
        spd.loc[0,'perlen'] = 0
        spd['cum_days'] = spd.perlen.cumsum()
        spd.loc[0,'perlen'] = 1
        spd.loc[0,'cum_days'] = 1
        spd['year'] = np.arange(strt_yr-1,strt_yr-1+nper,1)
        spd['start_datetime'] = pd.to_datetime(spd['year'], format='%Y')
        spd.loc[0,'start_datetime'] = pd.to_datetime(start_date)
        spd['end_datetime'] = pd.to_datetime(spd['cum_days'],unit='D',origin=pd.Timestamp(start_date))
        
        spd['steady_state'] = False
        spd.loc[0,'steady_state'] = True
        spd = spd.reset_index()
        spd = spd.rename(columns={'index':'stress_period'})
        spd['stress_period'] = spd['stress_period'] + 1 # make it 1 indexed
        spd.loc[0,'cum_days'] = 0 # set to zero cuase ss period time doesn't matter
        
        spd.to_csv(os.path.join(out_tbl_dir,'annual_stress_period_info.csv'),index=False)
    else:
        spd = pd.DataFrame(period_data)
        # Initialize cumulative days column
        spd.loc[0, 'perlen'] = 0
        spd['cum_days'] = spd.perlen.cumsum()
        spd.loc[0, 'perlen'] = 1
        spd.loc[0, 'cum_days'] = 0

        start_datetime = []
        end_datetime = []
        
        # get years by taking start date and calucating based on cum_days:
        years = []
        spd['cum_days'] = spd['cum_days'].astype(int) 
        origin = pd.to_datetime(start_date)
        for i in range(nper):
            if i == 0:
                st_dt = origin - pd.Timedelta(days=1) + pd.to_timedelta(spd.loc[i, 'cum_days'], unit='D')
                years.append(st_dt.year)
            else:
                # get the start date for each period
                st_dt = origin + pd.to_timedelta(spd.loc[i, 'cum_days'], unit='D')
                years.append(st_dt.year)
        spd['year'] = years
        
        # start of each stress period in datetime format:
        for i in range(nper):
            if i == 0:
                start_datetime.append(pd.to_datetime(start_date) - pd.Timedelta(days=1))
                end_datetime.append(pd.to_datetime(start_date))
            else:
                start_datetime.append(end_datetime[i-1])
                end_datetime.append(start_datetime[i] + pd.DateOffset(days=spd.loc[i, 'perlen']))
        spd['start_datetime'] = start_datetime
        spd['end_datetime'] = end_datetime
        
        # Steady-state flag (set to True for the first period)
        spd['steady_state'] = False
        spd.loc[0, 'steady_state'] = True

        # Reset index and rename the 'index' column to 'stress_period'
        spd = spd.reset_index()
        spd = spd.rename(columns={'index': 'stress_period'})
        spd['stress_period'] = spd['stress_period'] + 1  # Make it 1-indexed

        # Set steady-state period time to zero
        spd.loc[0, 'cum_days'] = 0

        # Save the dataframe to CSV
        spd.to_csv(os.path.join(out_tbl_dir, 'monthly_stress_period_info.csv'), index=False)

    return spd


def load_precip_for_rch():
    """
    Load the PRISM precipitation data to estimate RCH

    Returns
    -------
    precip_agg : DataFrame
        Precipitation data aggregated (summed) for yearly and monthly model periods

    """
    precip = pd.read_csv(os.path.join('data','raw','PRISM_precip','PRISM_ppt_provisional_4km_197001_202501_47.8746_-98.5477.csv'),
                         skiprows=10,
                         index_col=0,
                         parse_dates=True)
    
    # annual totals, timestamped at Jan 1 of each year, up through 1999
    annual = precip.resample("AS-JAN").sum().loc[: "1999-01-01"]

    # monthly totals, timestamped at the first of the month, from Jan 2000 onward
    monthly = (precip
               .resample("MS")
               .sum()
               .loc["2000-01-01" : "2025-12-31"]
               )

    # concatenate and sort
    precip_agg = pd.concat([annual, monthly]).sort_index()
    
    return precip_agg


def process_swb_for_rch(make_plots=True,
                        idom=None,
                        increase_low_rch=True):
    
    # ---- Load in the SWBs pre-processors
    from process_SWB_to_RCH import main as process_SWB
    
    pre_2000, post_2000_monthly = process_SWB()
    
    # ---- Set some control variables based on the input data
    # Calc Q90 of non-zero recharge values
    all_vals = np.array([])
    for key in post_2000_monthly:
        all_vals = np.concatenate((all_vals,post_2000_monthly[key][0,:,:].flatten()))
    all_vals = all_vals[all_vals > 0]
    q_ = np.quantile(all_vals,0.70)
    
    # Maximum allowable RCH value in units of in/day (Using Q90 as threshold)
    rch_thresh = q_
    for key in post_2000_monthly:
        post_2000_monthly[key] = np.where(post_2000_monthly[key]>rch_thresh, rch_thresh, post_2000_monthly[key])
        
    # ---- Format the pre_2000 data
    rch = []
    years = list(pre_2000.keys())
    
    # TODO - quick fix, rch is zero for SP 7
    pre_2000[1976] = pre_2000[1988]
    
    # Create a SS period average
    avg_pre_2000 = np.zeros(pre_2000[1970].shape)
    for key in pre_2000.keys():
        rch_vals = np.where(np.isnan(pre_2000[key]),0,pre_2000[key])
        
        # Ensure positive!
        rch_vals = np.where(rch_vals<0,0,rch_vals)
        
        # Add to the total for average calc
        avg_pre_2000 += rch_vals
        
        # Append for plotting --> account for active/inactive
        plot_rch = np.where(idom[0,:,:]!=1,np.nan,rch_vals[:,:-1])
        rch.append(np.nansum(plot_rch))
    
    # Calculate average
    avg_pre_2000 /= len(pre_2000.keys())
    
    # Reset keys to stress periods and then add SS
    pre_2000 = {i+1: v for i, (k, v) in enumerate(sorted(pre_2000.items()))}
    pre_2000[0] = avg_pre_2000
    
    # Increase RCH during low years for stability
    if increase_low_rch:
        for key in post_2000_monthly:
            if key[0:4] in ['2002','2005','2006','2018']:
                post_2000_monthly[key] *= 10
    
    # ---- Format the monthly stress periods
    years_2 = list(post_2000_monthly.keys())
    post_2000_monthly = {i+31: v for i, (k, v) in enumerate(sorted(post_2000_monthly.items()))}
    
    # Grab the data for the plot
    rch_2 = []
    for key in post_2000_monthly:
        rch_key = np.where(np.isnan(post_2000_monthly[key]),0,post_2000_monthly[key])
        plot_rch = np.where(idom[0,:,:]!=1,np.nan,rch_key[0,:,:-1])
        rch_2.append(np.nansum(plot_rch))
    
    
    # ---- Plot up the yearly recharge as a bar chart
    if make_plots:
        # Load precip used for linear regression
        precip_data = load_precip_for_rch()
        precip_data = precip_data.resample('YS').sum()
        
        
        # dates for the monthly periods
        years_2 = [pd.Timestamp(x) for x in years_2]
        
        # Create DataFrames
        df1 = pd.DataFrame({'date':[pd.Timestamp(f'01-01-{x}') for x in years],
                            'values':rch})
        df2 = pd.DataFrame({'date':years_2,
                            'values':rch_2})
        
        # Concat
        df = pd.concat([df1,df2],axis=0)
        
        df = df.set_index('date')
        
        # Bfill the rate to daily, resample and sum for total yearly RCH
        df = df.resample('D').ffill().resample('YS').sum()
        
        # ----------------
        # --- Plot
        # ----------------
        fix,ax = plt.subplots(figsize=(14,5))
        
        # Add a vspan for regression period
        start_date = pd.Timestamp("1969-01-01")
        end_date = pd.Timestamp("1999-06-15")
        ax.axvspan(start_date,end_date, color="#1b9e77", alpha=0.1,
                   label='Regression Estimated Period')
        ax.axvline(end_date,
                   color='#1b9e77')
        
        # df.loc[df.index>=pd.to_datetime('01-01-2020'),'values'] *= 1.2
        
        # Plot the RCH
        ax.bar(df.index,
               df['values'] * 3,
               width=240,
               color='#7570b3',
               label='SWB RCH (in)',
               edgecolor='k')
        
        # Plot precip below
        ax.bar(precip_data.index[:-2],
               precip_data['ppt (inches)'][:-2] / 12 * -1,
               label='Yearly Precipitation (feet)',
               color='#d95f02',
               width=240,
               alpha=0.8,
               edgecolor='k')
        
        ax.axhline(0,color='k')
        t = idom[0,:,:].flatten()
        t = t[t==1]
        # Plot 10% of yearly precipitation as a comparison to SWB/regression results
        ax.step(precip_data.index[:-2],
                precip_data['ppt (inches)'][:-2] * 0.10 * len(t),
                label='10% of Precipitation',
                color='red',
                markeredgecolor='k',
                marker='o',
                markersize=5
                )
        
        # Formatting
        ax.grid(axis='y')
        ax.set_xlim(start_date,pd.Timestamp('01-01-2024'))
        ax.legend(loc=2,
                  framealpha=1,
                  bbox_to_anchor=[0.1,-0.13],
                  ncols=4
                  )
        ax.set_yticklabels([label.get_text().replace('−', '') for label in ax.get_yticklabels()])
        ax.set_xlabel("Year")
        ax.set_title("RCH Inputs from SWB")
        #plt.show()
        
    return {"yearly_1970_1999": pre_2000,
            "monthly_2000_2023": post_2000_monthly}


def params_from_jcb(idomain,par='k'):
    obs_jcb_path = os.path.join('master_flow_07',"swww.2.obs.jcb")
    obs_en = pyemu.ObservationEnsemble.from_binary(None,
                                                   obs_jcb_path)
    obs_df = pd.DataFrame(obs_en,
                          columns=obs_en.keys(),
                          index=obs_en.index)
    obs_base = obs_df.loc['base']
        
    # Pull out parameter values
    pat = re.compile(r'oname:{}_k:(\d+)_otype:arr_i:(\d+)_j:(\d+)'.format(par))
    records = []
    for name, val in obs_base.items():
        m = pat.match(name)
        if m:
            k, i, j = map(int, m.groups())
            records.append((k, i, j, val))
    
    df_parsed = pd.DataFrame(records, columns=["k", "i", "j", "val"])
    arrs = {k: np.full((nrow, ncol), np.nan) for k in range(nlay)}

    # fill arrays
    for _, row in df_parsed.iterrows():
        k = int(row["k"])
        i = int(row["i"])
        j = int(row["j"])
        if idomain[k,i,j] != 1:
            arrs[k][i, j] = np.nan
        else:
            arrs[k][i, j] = float(row["val"])
    
    # Plot layer HK
    for k in range(3):
        plt.figure()
        plt.imshow(arrs[k], cmap="viridis")
        plt.colorbar(label=f"k:00{k} values")
        plt.title(f"Layer {k} values from obs_base")
        #plt.show()
    
    return arrs
    
        
def construct_model(top,
                    botm,
                    idom,
                    model_grid,
                    drn_df,
                    riv_df,
                    lake_df,
                    heads_layer3,
                    modname='swww',
                    make_plots=True,
                    export_zones=True,
                    swb2_rch=True,
                    use_jcb_npf=False,
                    outside_ww_active=True,
                    three_layer=True,
                    build_lakes=False,
                    ):
    """
    Function that takes all processed inputs and builds the MF6 model.

    Parameters
    ----------
    top : Numpy array
        Model top elevations.
    botm : Numpy array
        Model bottom elevations.
    idom : Numpy array
        Model idomain (active/inactive cells).
    model_grid : GeoDataFrame
        GeoDataFrame of the active modelgrid.
    drn_df : DataFrame
        Holds all inputs necessary for the DRN package.
    riv_df : DataFrame
        Holds all inputs necessary for the RIV package.
    lake_df : DataFrame
        Holds all inputs necessary for the RIV (lakes) package.
    heads_layer3 : Numpy array
        Starting heads for layer 3 based on interpoalted GWL data.
    modname : str, optional
        Name of the simulation. The default is 'swww'.
    make_plots : bool, optional
        Flag to make some qa/qc plots along the way. The default is True.
    export_zones : bool, optional
        Flag to export the various hydrogeologic zones for PEST setup. The default is True.
    create_wl_ts : bool, optional
        Flag to specify if the input hydrographs still need to be processed. IF True,
        the heads will be processed after calling this function. The reason it is 
        imbedded this way is because the hydrograph script needs the model dis info 
        in order to algin well depths with model layers.The default is False.

    Returns
    -------
    sim : Flopy mfsim object
        Fully constructed mf6 model.
    """
    
    
    print("....Creating MF6 model")
    mf6_exe = find_mf6_exe()
    model_name = modname
    workspace = f"./model_ws/{modname}"
    
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    
    # ---- Create flow model
    sim = flopy.mf6.MFSimulation(sim_name=model_name, 
                                 version="mf6", 
                                 exe_name=mf6_exe, 
                                 sim_ws=workspace,
                                 continue_=True)
    
    gwf = flopy.mf6.ModflowGwf(sim, 
                               modelname=model_name, 
                               save_flows=True,
                               # Using Newton solver to prevent dry cells during calibration
                               newtonoptions=['under_relaxation'],
                               )
    
    # ---- TDIS
    print("Writing TDIS")
    start_year = 1970
    n_yearly_years = 30
    perioddata = [(1.0, 1, 1)]  # SS period
    # Set up yearly period from 1970 --> 2000
    for year in range(start_year, start_year + n_yearly_years):
        days_in_year = 366 if calendar.isleap(year) else 365
        perioddata.append((days_in_year, 1 if year == 1970 else 1, 1.2))
    
    # Set up monthly periods from Jan 2000 to Dec 2024
    start = date(2000, 1, 1)
    end   = date(2023, 12, 31)
    year, month = start.year, start.month
    
    while (year, month) <= (end.year, end.month):
        ndays = calendar.monthrange(year, month)[1]
        # Single timestep per monthly stress period
        perioddata.append((ndays, 1, 1.2))
        
        # increment month
        month += 1
        if month == 13:
            month = 1
            year += 1
    
    nper = len(perioddata)
    
    _ = flopy.mf6.ModflowTdis(
        sim,
        time_units='days',
        start_date_time='1970-01-01',
        nper=nper,
        perioddata=perioddata
    )

    # ---- DIS
    print("Writing DIS")
    _ = flopy.mf6.ModflowGwfdis(gwf,
                                nlay=nlay, 
                                nrow=nrow, 
                                ncol=ncol,
                                delr=delr, 
                                delc=delc, 
                                top=top, 
                                botm=botm,
                                idomain=idom.tolist(),
                                )
    
    # ---- IMS
    print("Writing Solver")
    ims = flopy.mf6.ModflowIms(
                sim,
                pname="ims",
                print_option="SUMMARY",
                complexity="SIMPLE",
                outer_dvclose=0.01,
                outer_maximum=100,
                under_relaxation="NONE",
                inner_maximum=100,
                inner_dvclose=0.01,
                rcloserecord=100,
                linear_acceleration="bicgstab ",
                scaling_method="NONE",
                reordering_method="NONE",
                relaxation_factor=0.97,
            )
    sim.register_ims_package(ims, [gwf.name])    
    
    # sim.write_simulation()
    
    # return
    
    # ---- GHBs
    # --- Layer 3 (Spiritwood)
    print("Writing GHB")
    ghb_sp_dat = {}
    
    # Create a GHB time-series from groundwater head data in Spiritwood-Devils Lake
    # aquifer, along with the lake elevations
    devils_stage,stump_stage = process_lake_elevations()
    
    # Define GHB head based on a ~regression fit~ between observed WL's in SW-Devil's Lake and lake elevations
    devils_stage['ghb_hd'] = 1428 + (devils_stage['gage_height_ft'] * 0.7) - devils_stage['gage_height_ft'][0]
    conductance_array = np.full((nrow, ncol), 50000.0)  
    for sp in range(nper):
        ghb_spd_layer3 = []
        # Grab ghb head for that time-step
        ghb_hd = int(devils_stage.loc[devils_stage['stress_period']==sp,'ghb_hd'].values[0])
        # Top side
        for col in range(ncol):
            if idom[2,:,:][0, col] == 1:
                ghb_spd_layer3.append([
                            (2, 0, col),  # (layer, row, column)  (zero-indexed)
                             ghb_hd,  # Boundary head
                             conductance_array[0,col]  # Conductance
                             ]
                    )
        # Left side
        for row in range(nrow):
            if idom[2,:,:][row, 0] == 1:
                ghb_spd_layer3.append([
                            (2, row, 0),
                             ghb_hd,
                             conductance_array[row,0]
                             ]
                    )
            
        # Check with plot
        if (sp == 0) and make_plots:
            plot_GHBs(ghb_spd_layer3,layer=3)
    
        ghb_sp_dat[sp] = ghb_spd_layer3
    
    
    # Write layer 3 inputs
    _ = flopy.mf6.ModflowGwfghb(
                        gwf,
                        stress_period_data=ghb_sp_dat,
                        pname='GHB3',
                        )
      
                
    # ---- IMS
    print("Writing Solver")
    ims = flopy.mf6.ModflowIms(
                sim,
                pname="ims",
                print_option="SUMMARY",
                complexity="SIMPLE",
                outer_dvclose=0.05,
                outer_maximum=100,
                under_relaxation="NONE",
                inner_maximum=100,
                inner_dvclose=0.05,
                rcloserecord=100,
                linear_acceleration="bicgstab ",
                scaling_method="NONE",
                reordering_method="NONE",
                relaxation_factor=0.97,
            )
    sim.register_ims_package(ims, [gwf.name])

    # --- Setup regional GHBs for the confining unit
    

    
    # ---- DRN
    print("Writing DRN")
    # Precompute spatially constant parts
    drn_lrc_cond = []  # list of ((lay, row, col), cond)
    drn_grid = np.zeros((nrow, ncol))
    
    drop_rows = []
    for idx, row in drn_df.iterrows():
        r = int(row['row'] - 1)
        c = int(row['column'] - 1)
        cond = row['cond']
        # Only allowing DRNs in layer 1
        ly = 0
        if np.isin(idom[0, r, c], [0, -1]):
            drop_rows.append(False)
            continue
        else:
            drop_rows.append(True)
    
        drn_lrc_cond.append(((ly, r, c), cond))
        drn_grid[r, c] = 1  # mark as DRN cell (just once)
    
    # Drop rows where we dropped DRNs
    drn_df = drn_df.loc[drop_rows].reset_index(drop=True)
    
    # Build transient dict with varying stage - reusing the spatial info 'cause 
    # its constant through time
    drn_dict = {}
    for i in range(nper):
        per_spd = []
        for (idx, ((ly, r, c), cond)) in zip(drn_df.index, drn_lrc_cond):
            stage = drn_df.loc[idx, 'stage']  # Replace this if stage varies with time
            per_spd.append(((ly, r, c), stage, cond))
        drn_dict[i] = per_spd

    if make_plots:
        plt.Figure()
        plt.imshow(drn_grid)
        plt.title("Active DRN Cells")
        #plt.show()
    
    _ = flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data=drn_dict,
        pname="drn",
        save_flows=True,
        )
    
    # ---- RIV - Rivers
    print("Writing RIV - Rivers")
    riv_stage = process_RIV_flows()
    # Precompute spatial constants
    riv_lrc_cond_rbot = []  # store ((lay, row, col), cond, rbot)
    riv_grid = np.zeros((nrow, ncol))
    riv_cells = []
    
    for idx, row in riv_df.iterrows():
        r = int(row['row'] - 1)
        c = int(row['column'] - 1)
        cond = row['cond']
        rbot = row['rbot']
        
        # All RIV cells in layer 2 (confining unit)
        ly = 1
        riv_lrc_cond_rbot.append(((ly, r, c), cond, rbot))
        riv_grid[r, c] = 1
        riv_cells.append((ly, r, c))
    
    # Preprocess stage by stress period to speed lookup
    riv_stage_dict = dict(zip(riv_stage['stress_period'], riv_stage['stage'].round(0)))
    
    # Build transient dict
    riv_dict = {}
    for i in range(nper):
        stage_base = riv_stage_dict.get(i, 0)  # default to 0 if missing
        per_spd = []
        for ((ly, r, c), cond, rbot) in riv_lrc_cond_rbot:
            stage = rbot + stage_base
            per_spd.append(((ly, r, c), stage, cond, rbot))
        riv_dict[i] = per_spd

    if make_plots:
        plt.Figure()
        plt.imshow(riv_grid)
        plt.title("Active RIV Cells - River")
        #plt.show()
    
    # Only makes sense if zones outside of WW is active...
    if outside_ww_active:
        _ = flopy.mf6.ModflowGwfriv(
                gwf,
               stress_period_data=riv_dict,
               pname="RIV_rivers",
               save_flows=True,
               filename=f'{modname}-riv.riv'
               )
    
    
    # ---- RIV - Lakes
    if build_lakes:
        print("Writing RIV - Lakes")
        # devils_stage,stump_stage = process_lake_elevations()
        lake_dict = {}
        
        # Load lake elevation time series
        devils_stage_dict = dict(zip(devils_stage['stress_period'], devils_stage['gage_height_ft'].round(0)))
        stump_stage_dict = dict(zip(stump_stage['stress_period'], stump_stage['gage_height_ft'].round(0)))
        
        # Precompute spatial constants for cells to include
        lake_lrc_info = []  # Store ((lay, r, c), cond, rbot, elev, lake_type)
        
        lake_grid = np.zeros((nrow, ncol))
        
        for idx, row in lake_df.iterrows():
            if row['main_lakes'] != 1:
                continue
        
            if not (row['devil_lake'] or row['stump_lake']):
                continue  # Skip ambiguous or undefined lakes
        
            r = int(row['row'] - 1)
            c = int(row['column'] - 1)
            cond = row['cond']
            rbot = row['rbot']
            elev = row['elevation']
        
            # Lake only in layer 1
            ly = 1
        
            lake_type = 'devils' if row['devil_lake'] == 1 else 'stump'
            lake_lrc_info.append(((ly, r, c), cond, rbot, elev, lake_type))
            lake_grid[r, c] = 1
        
        # Build lake_dict by stress period
        lake_dict = {}
        for i in range(nper):
            per_spd = []
            devils_stage = devils_stage_dict.get(i, 0)
            stump_stage = stump_stage_dict.get(i, 0)
    
            for (lyr_rc, cond, rbot, elev, lake_type) in lake_lrc_info:
                if lake_type == 'devils':
                    stage = elev + devils_stage
                else:
                    stage = elev + stump_stage
                per_spd.append((lyr_rc, stage, cond, rbot))
        
            lake_dict[i] = per_spd
    
        if make_plots:
            plt.Figure()
            plt.imshow(lake_grid)
            plt.title("Active RIV Cells - Lakes")
            #plt.show()
        
        _ = flopy.mf6.ModflowGwfriv(
                gwf,
                stress_period_data=lake_dict,
                pname="RIV_lakes",
                save_flows=True,
                filename=f'{modname}-lake.riv'
                )
    
    
    # ---- IC
    print("Writing IC")
    strt_1 = np.where(idom[0,:,:]==0,0,top)
    strt_2 = np.where(idom[1,:,:]==0,0,top)
    strt_3 = np.where(np.isnan(heads_layer3),0,heads_layer3)
    strt = np.zeros((nlay, nrow, ncol))
    strt[0,:,:] = strt_1
    strt[1,:,:] = strt_2
    strt[2,:,:] = strt_3
    # strt[2,:,:] = botm[3,:,:]
    _ = flopy.mf6.ModflowGwfic(gwf, 
                               strt=strt)
    
    # ---- NPF
    print("Writing NPF")
    # --- Creater layer 1 K based on WW extent
    hk_ww = 140 # --> K within ww domain
    K_lay_1 = np.full((nrow,ncol),hk_ww)  
    # Add a bit of heterogeneity
    K_lay_1 = create_WW_K(K_lay_1, hk_ww)
    
    # --- K in layer 2 
    k_conf_low = 5.6e-3
    
    # --- Recharge Window between Spiritwood and Warwick
    window_K = 10
    
    # Add recharge window to confining unit
    K_lay_2 = K_confining_layer(k_low=k_conf_low,
                                recharge_window=True,
                                rch_window_K=window_K
                                )
    
    # Adjust the K value in the confining unit along the RIV cells
    for cell in riv_cells:
        K_lay_2[cell[1],cell[2]] = 1
    
    # --- K in Spiritwood
    # hk_sw_sheyenne = 100
    hk_sw = 120
    hk_sw_high = 180
    k_sw = np.full((nrow,ncol),hk_sw,dtype=float)
    
    # --- Load the high-K zone shapefile (hand drawn using AEM data)
    high_k_zone = gpd.read_file(os.path.join("..","..","gis","input_shps","sw_ww","sw_high_k_v2.shp"))
    high_k_zone = high_k_zone.to_crs(2265)
    for _,row in high_k_zone.iterrows():
        i = row['row'] - 1
        j = row['column'] - 1
        k_sw[i,j] = hk_sw_high
    
    # --- Low K barrier in SW
    barrier_k = 1
    low_k_zone = gpd.read_file(os.path.join("..","..","gis","input_shps","sw_ww","HFB_V7.shp"))
    for _,row in low_k_zone.iterrows():
        i = row['row'] - 1
        j = row['column'] - 1
        k_sw[i,j] = barrier_k
        
        
    if make_plots:
        # Plot Warwick
        plt.figure()
        im = plt.imshow(np.where(idom[0,:,:]!=1,np.nan,K_lay_1))
        plt.colorbar(im)
        plt.title("Layer 1 (WW) HK")
        #plt.show()
        
        # Plot SW
        plt.figure()
        im = plt.imshow(np.where(idom[2,:,:]!=1,np.nan,k_sw))
        plt.colorbar(im)
        plt.title("Layer 3 (SW) HK")
        #plt.show()
        
    # --- Vertical K
    k33 = np.full((nlay, nrow, ncol),0.1)
    # k33[0,:,:] = np.full((nrow,ncol),0.2)
    # HK == VK in confining  unit
    # k33[1,:,:] = np.full((nrow,ncol),0.2)
    # k33[2,:,:] = np.full((nrow,ncol),1)

    #  Horizontal K
    k11 = np.zeros((nlay, nrow, ncol))
    k11[0, :, :] = K_lay_1   # Layer 1: WW
    k11[1, :, :] = K_lay_2   # Layer 2: low-K confining unit
    k11[2, :, :] = k_sw      # Layer 3: SW
    if not three_layer:
        k11[3, :, :] = np.full((nrow,ncol),1e-6)  # Layer 4: Bedrock, really low K
    
    # --- Layer weting
    icell = np.zeros(nlay)
    icell[0] = 1 # WW set as convertible layer
    
    # Just to make sure...
    for i in range(nlay):
        k11[i,:,:] = np.where(np.isnan(k11[i,:,:]),0,k11[i,:,:])
        k33[i,:,:] = np.where(np.isnan(k33[i,:,:]),0,k33[i,:,:])
    
    # --- Write NPF package
    _ = flopy.mf6.ModflowGwfnpf(gwf,
                                pname="npf",
                                save_flows=True,
                                icelltype=icell,
                                k=k11,
                                k33=k33,
                                k33overk=True,
                                )
    
    # ---- Export NPF zones for PEST setup
    if export_zones:
        zone_array = np.full((nlay,nrow,ncol), 0)
        # --- Assign zones for layer 1 --> WW
        # Zone 1 for WW
        zone_array[0,:,:] = np.ones(shape=(nrow,ncol))
        
        # --- Assign zones for layer 2 --> Confining Unit
        # Zone 2 for rch windows, zone 3 for under RIV, zone 4 everywhere else
        zone_array[1,:,:] = np.where(K_lay_2 == np.max(K_lay_2), 2, 4)       
        for cell in riv_cells:
            zone_array[1,cell[1],cell[2]] = 3
        
        # --- Assign zones for layer 2 --> Spiritwood/Spiritwood-Sheyenne
        # Zone 5 for Spiritwood, zone 6 for Spiritwood-Sheyenne, zone 7 for flow barrier
        zone_array[2,:,:] = np.full(shape=(nrow,ncol),fill_value=5)
        
        # Zone 6 for SW-Sheyenne
        sw_shey_cells = gpd.read_file('../../gis/input_shps/sw_ww/sw_sheyenne_cells.shp')
        for _,row in sw_shey_cells.iterrows():
            i = row['row'] - 1
            j = row['column'] - 1
            if idom[2,i,j] != 1:
                continue
            else:
                zone_array[2,i,j] = 6
        
        # Zone 7 for low-K barrier
        for _,row in low_k_zone.iterrows():
            i = row['row'] - 1
            j = row['column'] - 1
            zone_array[2,i,j] = 7
            
        # Save as Numpy array
        np.save('./zone_array.npy',zone_array)
        
    # ---- RCH
    # Two options - SWB2 or Simple fractional precipitation
    rch_dict = {}
    
    # METHOD 1: Precip-based recharge
    if not swb2_rch:
        print("Writing Recharge - Fractional Precip Method")
        # Load precipitation data
        precip_dat = load_precip_for_rch()
        
        # Array of 0's and 1's to increase the RCH at location of smaller lakes
        max_rch_vals = []
        # rch_increase_lakes = increase_RCH_lakes()
        
        # 15% Percent of yearly precip to use as RCH
        rch_per = 0.15
        
        # Steady state RCH, in units of in/year
        SS_RCH = 1.7
                
        for i in range(nper):
            if i == 0:
                # X in/year converted to ft/day for SS period
                rch_val = SS_RCH / 12 / 365
            else:
                # Approx yearly rch
                if i <= 30:
                    rch_val = (rch_per * precip_dat.iloc[i-1]) / 12 / 365
                # Approx monthly rch
                else:
                    rch_val = (rch_per * precip_dat.iloc[i-1]) / 12 / (perioddata[i][0])
            
            rch_array = np.full((nrow, ncol), rch_val) 
                        
            # No RCH on lakes, rivers, or DRNs
            # rch_array = np.where(lake_grid == 1, 0, rch_array)
            rch_array = np.where(riv_grid == 1, 0, rch_array)
            rch_array = np.where(drn_grid == 1, 0, rch_array)
            
            # Increase RCH on smaller lakes by 200%
            # rch_array = np.where(rch_increase_lakes==1,rch_array*2,rch_array)
            
            # No RCH outside of ww
            rch_array = np.where(idom[0,:,:]==1,rch_array,0)
            
            # Save rch for current SP
            rch_dict[i] = rch_array
            
            # Track max recharge value
            max_rch_vals.append(np.max(rch_array))
            
        print(f"Maximum RCH value: {np.max(max_rch_vals)}")
    
    # METHOD 2: SWB-based recharge
    else:
        print("Writing Recharge - SWB2 method")
        # Track maximum recharge values
        max_rch_vals = []
        
        # Load the SWB2 recharge values --> Returns a dict with SP keys
        swb_rch = process_swb_for_rch(idom=idom,
                                      increase_low_rch=True
                                      )
                
        # Specify a reduction multiplier for the SWB2 infiltration
        RCH_factor = 3 # factor to apply to prior SWB rch
        
        for i in range(nper):
            if i <= 30:
                # X in/year converted to ft/day for SS period
                rch_array = swb_rch['yearly_1970_1999'][i]
                rch_array = rch_array[:,1:] / 12 # inches to feet
                rch_array = np.where(np.isnan(rch_array),0,rch_array)
            else:
                rch_array = swb_rch['monthly_2000_2023'][i]
                rch_array = rch_array[0,:,1:] / 12 # inches to feet
                rch_array = np.where(np.isnan(rch_array),0,rch_array)
            
            # multiply RCH by specified factor
            rch_array *= RCH_factor
            
            # No RCH on RIV, DRN, or Lake BCs
            if build_lakes:
                rch_array = np.where(lake_grid == 1, 0, rch_array)
            rch_array = np.where(riv_grid == 1, 0, rch_array)
            rch_array = np.where(drn_grid == 1, 0, rch_array)
            
            # Zero RCH outside of warwick (?)
            # rch_array = np.where(idom[0,:,:]==0,0,rch_array)
            
            if make_plots:
                if i == 0:
                    plt.Figure()
                    im = plt.imshow(rch_array)
                    plt.title("Stress Period 0 Recharge")
                    plt.colorbar(im)
                    #plt.show()
            
            # Save rch for current SP
            rch_dict[i] = rch_array
            
            # Track max recharge value
            max_rch_vals.append(np.max(rch_array))
            
        print(f"Maximum RCH value: {np.max(max_rch_vals)}")
    
    _ = flopy.mf6.ModflowGwfrcha(gwf,
                                 recharge=rch_dict,
                                 save_flows=True,
                                 pname="rch",
                                 fixed_cell=False,
                                 )
    
    # ---- WEL
    wel_dat = make_WEL(botm,top,idom,ss_pumping=True)
    
    total_pumping = (
        wel_dat.groupby('sp')['cfd']
        .sum()
        .abs()
    )
        
    # --- Plot ---
    if make_plots:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(total_pumping.index, total_pumping/43560, marker='o')  # convert to acre-ft
        ax.set_xlabel('Stress Period')
        ax.set_ylabel('Total Pumping (acre-ft)')
        ax.set_title('Total Pumping per Stress Period')
        ax.grid(True)
        plt.tight_layout()
        #plt.show()
    
    mxbnd = 0
    well_dict = {}
    for i in range(nper):
        per_spd = []
        pmp_in_sp = wel_dat.loc[wel_dat['sp']==i]
        if len(pmp_in_sp) > 0:
            for idx, vals in pmp_in_sp.iterrows():
                # name = vals['Well']
                q = round(vals['cfd'] * -1.0, 3)
                if q == 0:
                    continue
                ly = int(vals['k'])
                r = int(vals['row']-1)
                c = int(vals['column']-1)
                
                # Double check here
                if idom[ly,r,c] == 0:
                    print(f"!! WARNING: {idx} well not in active zone, skipping !!")
                    continue
                
                per_spd.append(((ly, r, c), q))
        if len(per_spd) > mxbnd:
            mxbnd = len(per_spd)
        well_dict[i] = per_spd
    
    _ = flopy.mf6.ModflowGwfwel(gwf,
                                stress_period_data=well_dict,
                                # pname=val,
                                save_flows=True,
                                auto_flow_reduce=0.1,
                                maxbound=mxbnd,
                                filename=f'{modname}.wel'
                                )
    
    # ---- OC
    print("Writing OC")
    _ = flopy.mf6.ModflowGwfoc(gwf,
                                pname="oc",
                                budget_filerecord=f"{model_name}.cbb",
                                budgetcsv_filerecord='budget.csv',
                                head_filerecord=f"{model_name}.hds",
                                headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
                                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                #printrecord=[("HEAD", "FIRST"), ("HEAD", "LAST"), ("BUDGET", "LAST")],
                                )
    
    # write out temp modelfiles needed for wl processing:
    sim.write_simulation()
    
    # ---- HOBS    
    print("Writing Head OBS")
    # SS targets
    ss_targs = process_SS_head_targs(idom)
    ss_hd_list = [(row.obsprefix, 'HEAD', (int(row.k), int(row.i), int(row.j))) for _, row in ss_targs.iterrows()]
    
    # Transient targets
    trans_targs, hdiff_targs = wl_process_main(mnm='swww',
                                               annual_flag=False,
                                               gen_plots=False
                                               )

    trans_hd_list = []    
    trans_sites = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    cols = trans_targs.columns.tolist()
    tdf = pd.DataFrame(columns=['obsprefix','grpid' ,'k', 'i', 'j'])
    new_cols = []
    for col in cols:
        grp = int(col.split('_')[0].split('.')[1])
        k = int(col.split('.')[-1])
        site = trans_sites.loc[(trans_sites['group_number'] == grp) & (trans_sites['k'] == k)]
        obsprefix = f'transh_grpid:{grp}_k:{k}_i:{site["row"].values[0]-1}_j:{site["column"].values[0]-1}'
        i = site.row.values[0] - 1
        j = site.column.values[0] - 1
        temp = pd.DataFrame({
            'obsprefix': [obsprefix],
            'grpid': [grp],
            'k': [k],
            'i': [i],
            'j': [j]
            })
        tdf = pd.concat([tdf, temp], ignore_index=True)
        new_cols.append(obsprefix)
    trans_targs.columns = new_cols
    trans_targs.to_csv(os.path.join('data','analyzed','transient_well_targets.csv')) 
    tdf.to_csv(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'), index=False)

    wells = tdf.obsprefix.unique()
    for well in wells:
        subset = tdf[tdf.obsprefix==well].iloc[0,:]
        # check if well is in active domain:
        id = idom[subset.k, subset.i, subset.j]
        if id == 1:
            trans_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
        else:
            print(f"Skipping {well} at ({subset.k}, {subset.i}, {subset.j}) as it is not in the active domain.")
    
    hd_obs = { f"{model_name}.ss_head.obs.output.csv": ss_hd_list,
                f"{model_name}.trans_head.obs.output.csv": trans_hd_list,
                }
    
    _ = flopy.mf6.ModflowUtlobs(gwf,
                                pname="head_obs",
                                filename=f"{model_name}.obs",
                                continuous=hd_obs
                                )
    
    # ---- STO
    print("Writing STO")
    
    # SY
    sy = np.full((nlay,nrow,ncol),0.05)    
    sy[0,:,:] = 0.18  # --> X% for Warwick
    
    # SS --> Starting pretty low...
    ss = np.full((nlay,nrow,ncol),1e-6)
    
    # For layer 1, set ss equal to 1/thickness of the layer
    ss[0,:,:] = np.where(top - botm[0,:,:]>0, 1 / (top - botm[0,:,:]), 0)
    
    # Just to make sure....
    for i in range(nlay):
        sy[i,:,:] = np.where(np.isnan(sy[i,:,:]),0,sy[i,:,:])
        ss[i,:,:] = np.where(np.isnan(ss[i,:,:]),0,ss[i,:,:])
    
    iconv = botm.copy() * 0
    iconv[0,:,:] = 1
    iconv = iconv.astype(int)
    iconv_lst = iconv.tolist()
    
    _ = flopy.mf6.ModflowGwfsto(
                    gwf,
                    pname="sto",
                    save_flows=True,
                    ss_confined_only=True,
                    iconvert=iconv_lst,
                    ss=ss,
                    sy=sy,
                    steady_state={0: True},
                    transient={1: True},
                )
    
    # ---- Georef grid
    cord_sys = CRS.from_epsg(2265)
    gwf.modelgrid.set_coord_info(
        xoff=bounds[0], 
        yoff=bounds[1], 
        crs=cord_sys
        )
    
    # ---- Write inputs
    print("Writing Files to External")
    sim.set_all_data_external()
    sim.write_simulation()
    
    return sim, riv_cells
    

def clean_mf6(org_mws,mnm):
    cws = org_mws + "_clean"
    if os.path.exists(cws):
        shutil.rmtree(cws)
    shutil.copytree(org_mws,cws)

    sim = flopy.mf6.MFSimulation.load(sim_ws=cws, exe_name="mf6",load_only=["dis"])
    # nper = sim.tdis.nper.data
    m = sim.get_model(f"{mnm}")
    # nrow,ncol = m.dis.nrow.data,m.dis.ncol.data
    botm = m.dis.botm.array
    
    def fix_wrapped_format(model_ws='.'):
        """undo the redic wrapped format that reminds me of stone tablets

        Args:
            arr_file (str): array file
            nrow (int): nrow
            ncol (int): ncol

        """
        flow_dir = os.path.join(model_ws)
        sim = flopy.mf6.MFSimulation.load(sim_ws=flow_dir, exe_name="mf6")
        m = sim.get_model()
        pkg_lst = m.get_package_list()
        pkg_lst = [p.lower() for p in pkg_lst]
        nrow = m.dis.nrow.data
        ncol = m.dis.ncol.data
        tags = ["npf_k", "npf_k33","sto_ss","sto_sy", "porosity", 
                "recharge_", "ic_strt", "botm","top", "idomain",
                'icelltype','iconvert']
        arr_files = []
        for tag in tags:
            arr_files.append([os.path.join(model_ws, f) for f in os.listdir(model_ws) if tag in f and f.endswith(".txt")])
        arr_files = [arr_file for arr_file in arr_files if len(arr_file) > 0]
        #unpack the list of lists:
        arr_files = [item for sublist in arr_files for item in sublist]
        # remove any duplicates:
        arr_files = list(set(arr_files))
        for arr_file in arr_files:
            print(arr_file)
            vals = []
            with open(os.path.join(arr_file), 'r') as f:
                for line in f:
                    vals.extend([float(v) for v in line.strip().split()])
            vals = np.array(vals).reshape(nrow, ncol)
            # remove old file:
            os.remove(os.path.join(arr_file))
            # arr_file = arr_file.replace(f"{mnm}.", "")
            if "idomain" in arr_file:
                np.savetxt(os.path.join(arr_file), vals, fmt="%2.0f")
            else:
                np.savetxt(os.path.join(arr_file), vals, fmt="%15.6E")
        
        #adjust none array files:
        tags = ['delr','delc']
        afiles =[]
        for tag in tags:
            afiles.append([os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.endswith(f"{tag}.txt")])
        afiles = [af for af in afiles if len(af) > 0]
        #unpack the list of lists:
        afiles = [item for sublist in afiles for item in sublist]
        
        # find riv_stress files:
        riv_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.riv_stress")]
        drn_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.drn_stress")]
        # append to afiles:
        afiles.extend(riv_stress_files)
        afiles.extend(drn_stress_files)
        
        # check if drain or riv below model cell bottom:
        for drn_file in drn_stress_files:
            df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
            df.columns = ['ly','row','col','stage','cond']
            bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
            df['mbot'] = bot
            df['diff'] = df['stage'] - df['mbot']
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
            df = df.drop(columns=['mbot','diff'])
            df.to_csv(drn_file, sep=' ', index=False, header=False)
        for riv_file in riv_stress_files:
            df = pd.read_csv(riv_file, delim_whitespace=True,header=None)
            df.columns = ['ly','row','col','stage','cond','rbot']
            bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
            df['mbot'] = bot
            df['diff'] = df['stage'] - df['mbot']
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 1.0
            df['diff'] = df['rbot'] - df['mbot']
            df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1 # check on this if it was we want
            df = df.drop(columns=['mbot','diff'])
            df.to_csv(riv_file, sep=' ', index=False, header=False)
            
        for file in afiles:
            ogfile = file
            # file = file.replace(f"{mnm}.", "") 
            os.rename(ogfile, file)     
        
        # adjust paths in  pkg controls:
        # for pkg in pkg_lst:
        #     if 'rcha' in pkg:
        #         pkg = 'rcha' # remove numbering from recharge package name
        #     if pkg in ["oc","head_obs"]:
        #         continue
        #     with open(os.path.join(model_ws, f"{mnm}.{pkg}"), 'r') as f:
        #         # remove f'{mnm.}' where it appears:
        #         lines = f.readlines()
        #         nl = []
        #         for l in lines:
        #             if f"{mnm}." in l:
        #                 l = l.replace(f"{mnm}.", "")
        #             nl.append(l)
        #         #write the lines back to the file:
        #         with open(os.path.join(model_ws, f"{mnm}.{pkg}"), 'w') as f:
        #             for l in nl:
        #                 f.write(l)
                
                
    # fix annoying flopy external wrapped format:
    fix_wrapped_format(cws) 
    # pyemu.utils.run("mf6",cwd=cws)
    


# -------------
# Main function
# -------------
def main_build_model():
    
    # --- Option flags
    # - Set True to set outside of WW active, but make it all layer 2
    outside_ww_active = True
    # - Set True for 3-layer model, False to include 4th bedrock layer
    three_layer = True
    # - True to use s2b recharge, False to use x% of yearly precip
    swb2_rch = True
    # - True to include lake package
    build_lakes = False
    # - True to run base model
    run_base = True
    
    
    # --- Maybe add param values here for NPF to pass to construct_model (?)

    
    # --- globals
    global nrow, ncol, nlay, delr, delc, bounds
    
    # --- Set nrow and ncol here, modelgrid will be built using these values
    nlay, nrow, ncol = 4, 175, 155
    
    # --- Calculate model bounds
    bounds,crs = calc_model_extent()
    
    # --- Build grid
    model_grid,delr,delc = construct_modelgrid(bounds,crs)
    
    # --- Calculate top and bottom arrays from raster data
    # - If you need to regenerate the surfaces from the .tif's, use force_resample=True
    top,botm,idomain = sample_rasters(model_grid,force_resample=False)
    
    # Deactive cells in layers 1 outside of WW model domain
    idomain[0,:,:] = create_WW_K(np.zeros((nrow,ncol)), 1)
    
    # ---- Make area outside WW active but set it all to layer 2 (clay/till)
    if outside_ww_active:
        botm[0,:,:] = np.where(idomain[0,:,:]!=1,top-0.1,botm[0,:,:])
        idomain[1,:,:] = np.ones(shape=(nrow,ncol))
    # Or, layer 2 only active under WW
    else:
        idomain[1,:,:] = np.where(idomain[0,:,:] == 1, 1, 0)
    
    # Layer 4 (bedrock) active within WW or within Spiritwood
    idomain[3,:,:] = np.where((idomain[0,:,:] == 1) | (idomain[2,:,:] == 1) | (idomain[1,:,:] == 1), 1, 0)
        
    # --- Build the DRN inputs
    drn_df, drn_geo = build_DRN(top,botm,idomain,K=0.1)
    
    # --- Build the RIV inputs for the Sheyenne river
    riv_df = build_RIV(top,botm,idomain,K=1)
    
    # --- Build lakes if using
    if build_lakes:
        lake_df = build_RIV_lakes(top,botm,idomain)
    else:
        lake_df = pd.DataFrame()
        
    # --- Interpolate WLs in Spiritwood to get initial conditions and GHB values
    heads_layer3 = make_SW_IC(idomain[2,:,:])
    
    plt.close('all')
    
    # --- Model Name
    modname = 'swww'
    
    # --- Cut down to three layers if flagged
    if three_layer:
        nlay = 3
        botm = botm[0:3,:,:]
        idomain = idomain[0:3,:,:]
    
    # --- Create model
    sim, riv_cells = construct_model(top,
                          botm,
                          idomain,
                          model_grid,
                          drn_df,
                          riv_df,
                          lake_df,
                          heads_layer3,
                          modname=modname,
                          make_plots=False,
                          export_zones=True,
                          swb2_rch=swb2_rch,
                          outside_ww_active=outside_ww_active,
                          three_layer=three_layer,
                          build_lakes=build_lakes,
                          )
    
    # success, buff = sim.run_simulation()
    
    # --- Create a clean dir with external input files
    org_mws = f'./model_ws/{modname}'
    clean_mf6(org_mws,modname)
    
    # --- Run the cleaned model
    if run_base:
        sim.set_sim_path(org_mws + "_clean")
        success, buff = sim.run_simulation()
    
    strt_yr = pd.to_datetime(sim.tdis.start_date_time.data).year
    stress_period_df_gen(org_mws + "_clean",strt_yr,annual_only=False)
    
if __name__ == "__main__":
     main_build_model()
