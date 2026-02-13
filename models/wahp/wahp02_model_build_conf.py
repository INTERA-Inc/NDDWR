#!/usr/bin/env python3
"""
Builds a n-layer MODFLOW 6 model, but current foundation model is 4-layers based off 
these layer definitions:
  - Top: DEM (converted from meters to feet)
  - Bottom of layer 1 = Top - (WSS thickness), with a minimum thickness enforced at 5 ft.
  - Bottom of layer 2 = bottom of layer 1 - (WBV thickness), minimum = 5 ft.
  - Bottom of layer 3 = bottom of layer 2 - (DC thickness), minimum = 5 ft.
  - Bottom of layer 4 = bottom of layer 3 - (WR thickness), minimum = 5 ft.
For each thickness raster, any value < 5 ft is reset to 5 ft:
  - Forced cells in layers 1–3 get ibound = -1.
  - Forced cells in layer 4 get ibound = 0.
Active cells get ibound = 1.

Workflow:
  1. Create a rotated & trimmed grid from an input extent shapefile using create_grid_in_feet.
  2. Extract the proper lower-left corner and grid dimensions using extract_grid_attributes.
  3. Define a target grid (extent, shape, and transform) using ll_corner, nrow, ncol, and cell size.
  4. Sample the DEM and thickness rasters at grid cell centroids via elevs_to_grid_layers.
  5. Build the model layers (top, botm, idomain) by sequential subtraction.
  6. Set up a MODFLOW 6 simulation (DIS package only) in workspace "model_4_lay".

The printed "BEGIN options" output from Flopy is simply the simulation options header.
"""

import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import shutil
import math
import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import overlay
import flopy
import pyemu
#from shapely.geometry import box, Point
from shapely import affinity
from shapely.ops import unary_union
from shapely.geometry import box
from pyproj import CRS
import rasterio
from rasterio.warp import reproject, Resampling
#from rasterio.transform import from_origin
import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from pykrige.uk import UniversalKriging
import wahp01_water_level_process as wlprocess

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('data','pumping')))
import make_use_by_well as wuprocess
from wahp03_setup_pst_sspmp import prep_deps

# set fig formats:
import wahp04_process_plot_results as wpp
wpp.set_graph_specifications()
wpp.set_map_specifications()


def plot_basemap(epsg=2265):
    """Plot the basemap of the model area."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("Easting (ft)")
    ax.set_ylabel("Northing (ft)")
    
    #cnty = gpd.read_file(os.path.join("..", "..", "gis",'input_shps','regional',"counties.shp"))
    #cnty = cnty.to_crs(epsg=epsg)
    #cnty.boundary.plot(ax=ax, color="grey", linewidth=0.5, label="County")
    
    # wahp outline:
    grd = gpd.read_file(os.path.join("..", "..", "gis",'output_shps','wahp',"wahp7ly_cell_size_660ft_epsg2265.grid.shp"))
    grd = grd.dissolve()
    grd = grd.explode(index_parts=False)
    grd = grd.to_crs(epsg=epsg)
    grd.boundary.plot(ax=ax, color="blue", linewidth=0.5, label="Model Grid")
    
    # add wahp aq state outline:
    wahp = gpd.read_file(os.path.join("..", "..", "gis",'input_shps','wahp',"wahp_outline_full.shp"))
    wahp = wahp.to_crs(epsg=epsg)
    wahp.boundary.plot(ax=ax, color="black", linewidth=0.5, label="Wahp Outline")
    
    # zoom to the extent of the grid:
    minx, miny, maxx, maxy = grd.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # comma format for large numbers:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,}'.format(int(x))))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,}'.format(int(x))))
    
    return fig, ax
    
    
def build_grid_shp(nrow,ncol,cell_size,ll_corner,angrot,crs,mnm):
    """function to build shapefile of model grid, with key attributes for each cell"
    Args:
        are all defined as global variables at the bottom of the script
    Returns:
        gdf (shp): geodataframe of grid
    """

    mf = flopy.modflow.Modflow(f"cs{cell_size}.dummy")
    dis = flopy.modflow.ModflowDis(
        mf, nlay=1, nrow=nrow, ncol=ncol, delr=cell_size, delc=cell_size
    )
    modelgrid = mf.modelgrid
    proj4_str = CRS.from_user_input(crs).to_proj4()
    modelgrid.set_coord_info(
        xoff=ll_corner[0], yoff=ll_corner[1], angrot=angrot, proj4=proj4_str
    )

    shpfile = os.path.join(
        "..",
        "..",
        "gis",
        "output_shps",
        "wahp",
        f"{mnm}_cell_size_{cell_size}ft_epsg{crs}.grid.shp",
    )
    modelgrid.write_shapefile(shpfile)

    gdf = gpd.read_file(shpfile)
    gdf = gdf.rename(columns={"column": "col"})
    gdf = gdf.set_crs(f"epsg:{crs}")
    gdf["i"] = gdf.row - 1
    gdf["j"] = gdf.col - 1
    gdf.to_file(shpfile)

    return gdf, shpfile

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

# ----------------------------------------------------------------
# Helper: Resample a raster to a target grid.
# ----------------------------------------------------------------
def resample_raster_to_target(src_path, target_shape, target_transform, target_crs):
    """
    Resample the raster at src_path to the target grid defined by target_shape and target_transform.
    Returns a 2D numpy array.
    """
    with rasterio.open(src_path) as src:
        src_array = src.read(1)
        destination = np.empty(target_shape, dtype=src_array.dtype)
        reproject(
            source=src_array,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    return destination

# ----------------------------------------------------------------
# Grid-building functions.
# ----------------------------------------------------------------
def build_unrotated_grid_gdf(minx, miny, maxx, maxy, cell_size, crs):
    """
    Build an axis-aligned grid covering the given bounds.
    Returns a GeoDataFrame with integer 'i' and 'j' indices and cell geometries.
    """
    num_cells_x = math.ceil((maxx - minx) / cell_size)
    num_cells_y = math.ceil((maxy - miny) / cell_size)
    rows = []
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            x_left = minx + i * cell_size
            y_bottom = miny + j * cell_size
            x_right = x_left + cell_size
            y_top = y_bottom + cell_size
            poly = box(x_left, y_bottom, x_right, y_top)
            rows.append({"i": i, "j": j, "geometry": poly})
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf["grid_id"] = gdf.index + 1
    return gdf


def trim_grid_rows_columns(grid_gdf, shapefile_gdf):
    """
    Remove rows/columns from grid_gdf that do not intersect the shapefile.
    """
    shape_union = shapefile_gdf.union_all()  # use union_all() per deprecation warning
    grid_gdf["overlap"] = grid_gdf.geometry.apply(
        lambda cell: shape_union.intersects(cell)
    )
    overlapping = grid_gdf.loc[grid_gdf["overlap"], ["i", "j"]]
    used_i = set(overlapping["i"])
    used_j = set(overlapping["j"])
    trimmed = grid_gdf[
        (grid_gdf["i"].isin(used_i)) & (grid_gdf["j"].isin(used_j))
    ].copy()
    trimmed.drop(columns="overlap", inplace=True, errors="ignore")
    return trimmed


def extract_grid_attributes(grid_shapefile_path, angrot=0.0):
    """
    Extract the lower-left corner, number of rows, and number of columns from a rotated grid shapefile.
    Scans all exterior vertices and returns the vertex with minimum y (if angrot > 0).
    """
    grid_gdf = gpd.read_file(grid_shapefile_path)
    if not {"i", "j", "geometry"}.issubset(grid_gdf.columns):
        raise ValueError(
            "Grid shapefile must contain 'i', 'j', and 'geometry' columns."
        )
    all_coords = []
    for geom in grid_gdf.geometry:
        all_coords.extend(list(geom.exterior.coords))
    coords_df = pd.DataFrame(all_coords, columns=["x", "y"])
    if angrot > 0:
        min_y_row = coords_df.loc[coords_df["y"].idxmin()]
        ll_corner = (min_y_row["x"], min_y_row["y"])
    elif angrot < 0:
        min_x_row = coords_df.loc[coords_df["x"].idxmin()]
        ll_corner = (min_x_row["x"], min_x_row["y"])
    else:
        ll_corner = (coords_df["x"].min(), coords_df["y"].min())
    nrow = len(grid_gdf["j"].unique())
    ncol = len(grid_gdf["i"].unique())
    return ll_corner, nrow, ncol


def create_grid_in_feet(
    shapefile_path,
    grid_size_feet,
    output_shapefile_path,
    target_crs_epsg,
    rotation_degrees=0.0,
    max_iterations=20,
    expansion_mode="partial",
    columns_to_add=10):
    """
    Create a rotated and trimmed grid covering the input shapefile.
    Returns the trimmed GeoDataFrame, lower-left corner, rotation angle, and grid dimensions.
    """
    input_gdf = gpd.read_file(shapefile_path)
    shapefile_gdf = input_gdf.to_crs(epsg=target_crs_epsg)
    shape_union = shapefile_gdf.union_all()
    minx, miny, maxx, maxy = shape_union.bounds
    centroid_pt = shape_union.centroid
    pivot_x, pivot_y = centroid_pt.x, centroid_pt.y

    grid_minx, grid_miny, grid_maxx, grid_maxy = minx, miny, maxx, maxy
    coverage_ok = False
    for iteration in range(1, max_iterations + 1):
        grid_gdf = build_unrotated_grid_gdf(
            grid_minx,
            grid_miny,
            grid_maxx,
            grid_maxy,
            grid_size_feet,
            crs=f"epsg:{target_crs_epsg}",
        )
        if rotation_degrees != 0.0:
            grid_gdf["geometry"] = grid_gdf["geometry"].apply(
                lambda geom: affinity.rotate(
                    geom, rotation_degrees, origin=(pivot_x, pivot_y)
                )
            )
        union_grid = unary_union(grid_gdf.geometry)
        diff = shape_union.difference(union_grid)
        if diff.is_empty:
            coverage_ok = True
            break
        dmX, dmY, dMX, dMY = diff.bounds
        expand_amount = columns_to_add * grid_size_feet
        if expansion_mode == "all":
            grid_minx -= expand_amount
            grid_maxx += expand_amount
            grid_miny -= expand_amount
            grid_maxy += expand_amount
        else:
            if dmX < grid_minx:
                grid_minx -= expand_amount
            if dMX > grid_maxx:
                grid_maxx += expand_amount
            if dmY < grid_miny:
                grid_miny -= expand_amount
            if dMY > grid_maxy:
                grid_maxy += expand_amount
    if not coverage_ok:
        print(f"Reached max_iterations={max_iterations}, coverage may be incomplete.")
    trimmed_gdf = trim_grid_rows_columns(grid_gdf, shapefile_gdf)
    trimmed_gdf.to_file(output_shapefile_path)
    print(f"Trimmed grid shapefile saved to: {output_shapefile_path}")
    ll_corner, nrow, ncol = extract_grid_attributes(
        output_shapefile_path, rotation_degrees
    )
    return trimmed_gdf, ll_corner, rotation_degrees, nrow, ncol


# -----------------------------------------------------------
# Helper: Fill NaNs with median of 8-neighbor cells.
# -----------------------------------------------------------
def fill_nan_with_neighbors_median(arr):
    """
    Replace any NaN value in a 2D array with the median of its surrounding cells
    (neighbors in an 8-cell window). If no non-NaN neighbors exist, the NaN value
    remains unchanged.
    """
    filled = arr.copy()
    nrow, ncol = filled.shape
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(filled[i, j]):
                neighbors = []
                # Loop through the 3x3 window
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # Skip the cell itself
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nrow and 0 <= nj < ncol:
                            if not np.isnan(filled[ni, nj]):
                                neighbors.append(filled[ni, nj])
                if neighbors:
                    filled[i, j] = np.median(neighbors)
    return filled


# -----------------------------------------------------------
# New function: Sample elevations at grid cell centroids.
# -----------------------------------------------------------
def elevs_to_grid_layers(grid_shp_path,div_ly1=[]):
    """
    Reads the grid shapefile, computes cell centroids, and samples the DEM and thickness rasters
    (WSS, WBV, DC, WR) at each centroid.

    Returns:
      top: 2D numpy array (nrow x ncol) of model top elevations (in feet)
      botm: 3D numpy array (4 x nrow x ncol) of bottom elevations
      idomain: 3D numpy array (4 x nrow x ncol) of ibound values
    """
    grid = gpd.read_file(grid_shp_path)
    grid["centroid"] = grid.geometry.centroid
    grid["centroid_coords"] = grid["centroid"].apply(lambda p: (p.x, p.y))

    def sample_raster(raster_path, gdf):
        with rasterio.open(raster_path) as src:
            vals = [val[0] for val in src.sample(gdf["centroid_coords"])]
        return np.array(vals)

    # Define raster paths (adjust as needed)
    dem_path = os.path.join(
        "..",
        "..",
        "gis",
        "input_ras",
        "wahp",
        "dem",
        "wahp_full_res_dem_100_resampled.tif",
    )
    wss_path = os.path.join(
        "..", "..", "gis", "output_ras", "interpolation_ras", "wss_kriging.tif"
    )
    wbv_path = os.path.join(
        "..", "..", "gis", "output_ras", "interpolation_ras", "wbv_kriging.tif"
    )
    dc_path = os.path.join(
        "..", "..", "gis", "output_ras", "interpolation_ras", "dc_kriging.tif"
    )
    wr_path = os.path.join(
        "..", "..", "gis", "output_ras", "interpolation_ras", "wr_kriging.tif"
    )

    dem_vals = sample_raster(dem_path, grid) * 3.28084 # to feet
    
    top = dem_vals.reshape([nrow,ncol])
    # Fill any NaN values in the DEM with the median of their 8-neighbors.
    top = fill_nan_with_neighbors_median(top)

    wss_vals = sample_raster(wss_path, grid)
    wbv_vals = sample_raster(wbv_path, grid)
    dc_vals = sample_raster(dc_path, grid)
    wr_vals = sample_raster(wr_path, grid)

    # reshape:
    wss_vals = wss_vals.reshape((nrow, ncol))
    wbv_vals = wbv_vals.reshape((nrow, ncol))
    dc_vals = dc_vals.reshape((nrow, ncol))
    wr_vals = wr_vals.reshape((nrow, ncol))

    # Create boolean arrays from the original sampled thickness values.
    forced_wss_bool = (wss_vals < 5).reshape((nrow, ncol))
    forced_wbv_bool = (wbv_vals < 5).reshape((nrow, ncol))
    forced_dc_bool  = (dc_vals  < 5).reshape((nrow, ncol))
    forced_wr_bool  = (wr_vals  < 5).reshape((nrow, ncol))

    # Force thickness values to 5 ft where necessary.
    wss_arr = np.where(forced_wss_bool, 5, wss_vals).reshape([nrow,ncol])
    wbv_arr = np.where(forced_wbv_bool, 5, wbv_vals).reshape([nrow,ncol])
    dc_arr = np.where(forced_dc_bool, 5, dc_vals).reshape([nrow,ncol])
    wr_arr = np.where(forced_wr_bool, 5, wr_vals).reshape([nrow,ncol])

    # Compute bottom elevations sequentially.
    bottom_layer1 = top - wss_arr
    bottom_layer2 = bottom_layer1 - wbv_arr
    bottom_layer3 = bottom_layer2 - dc_arr
    bottom_layer4 = bottom_layer3 - wr_arr
    
    def divide_single_layer(top, wss_arr, div_ly1):
        """
        Divide a single layer into sublayers based on div_ly1 proportions.

        Parameters:
            top : ndarray (nrow, ncol)
                Top surface elevation of the layer.
            wss_arr : ndarray (nrow, ncol)
                Total thickness of the single layer.
            div_ly1 : list of float
                Fractions for each sublayer. Must sum to ~1.0.

        Returns:
            botm : ndarray (n_sublayers, nrow, ncol)
                Bottom elevation of each sublayer.
        """
        # Sanity check
        cum_frac = np.cumsum(div_ly1)
        assert np.isclose(cum_frac[-1], 1.0, atol=1e-6), "div_ly1 must sum to 1.0"

        # Compute bottom of each sublayer
        botm = np.stack([top - wss_arr * f for f in cum_frac], axis=0)
        return botm

    if len(div_ly1) > 0 :
        botm = divide_single_layer(top, wss_arr, div_ly1)  
        botm = np.concatenate([botm, # shape: (n_sublayers, nrow, ncol)
                                bottom_layer2[None, :, :],       
                                bottom_layer3[None, :, :],   
                                bottom_layer4[None, :, :]    
                            ], axis=0)

    else:
        botm = np.stack(
            [bottom_layer1, bottom_layer2, bottom_layer3, bottom_layer4], axis=0
        )
    
    # thickness calcs on layer 1 sub layers:
    nlays = len(div_ly1)
    thickness_arrs = np.empty((nlays, *top.shape))
    forced_bools = np.zeros((nlays, *top.shape), dtype=bool)

    # loop through sublayers and calc thicknesses
    for i in range(nlays):
        top_i = top if i == 0 else botm[i-1]
        bot_i = botm[i]
        thickness = top_i - bot_i

        # enforce minimum thickness of 5
        forced = thickness < 5
        thickness = np.where(forced, 5, thickness)

        thickness_arrs[i] = thickness
        forced_bools[i] = forced
    
    # # plot imshow of layer thickness:
    # for i in range(nlays):
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.set_title(f"Layer {i+1} Thickness (ft)")
    #     cax = ax.imshow(thickness_arrs[i], cmap='viridis', origin='lower')
    #     cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    #     cbar.set_label('Thickness (ft)')


    idomain = []
    if len(div_ly1) > 0:
        for i in range(len(div_ly1)):
            idom = np.where(forced_bools[i], -1, 1)
            idomain.append(idom)
    else:
        idomain.append(np.where(forced_wss_bool, -1, 1))

    # Add WBV, DC, WR layers
    idomain.append(np.where(forced_wbv_bool, -1, 1))
    idomain.append(np.where(forced_dc_bool,  -1, 1))
    idomain.append(np.where(forced_wr_bool,   0, 1))

    idomain = np.stack(idomain, axis=0) 

    # # loop through idomain and plot:
    # for i in range(idomain.shape[0]):
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.set_title(f"Layer {i+1} Ibound")
    #     cax = ax.imshow(idomain[i], cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    #     cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    
    return top, botm, idomain


def reinterp_wbv_bot_in_southern_area(grd, cell_size):
    
    org_grd = grd.copy()
    
    # laod area that needs reinterpolation:
    area_shp = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','southern_area_wbv_thick_reinterp.shp'))
    # load lines to guuide reinterpolation:
    lcenter = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','center_line_reinterp.shp'))
    wline = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','western_edg_reinterp.shp'))
    eline = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','eastern_edg_reinterp.shp'))
    wbvpts = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','wbv_inform_points.shp'))
    
    wline['botm_4'] = wline['botm_4'] - 16.0
    eline['botm_4'] = eline['botm_4'] - 16.0
    
    wbvpts.loc[(wbvpts.i ==136) & (wbvpts.j==45),'botm_4'] = 660.0
    wbvpts.loc[(wbvpts.i ==137) & (wbvpts.j==45),'botm_4'] = 661.0
    wbvpts.loc[(wbvpts.i ==136) & (wbvpts.j==46),'botm_4'] = 665.0
    wbvpts.loc[(wbvpts.i ==137) & (wbvpts.j==46),'botm_4'] = 666.0
    wbvpts.loc[(wbvpts.i ==136) & (wbvpts.j==47),'botm_4'] = 672.0
    

    lcenter['botm_4'] = 657.0
    #wline['botm_4']   = 697.0
    #eline['botm_4']   = 721.0

    grd_in = area_shp.copy()
    
    grd_nogeo = grd_in.drop(columns=['geometry'])
    grd_temp = grd_nogeo.merge(grd[['i','j','geometry']],on=['i','j'],how='left')
    
    
    # remove cells that we will reinterpolate:
    grd = grd.merge(area_shp[['i', 'j']], on=['i', 'j'], how='left', indicator=True)
    grd = grd.loc[grd['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # build training points from the line layers (ONLY rows with a value)
    train = pd.concat(
        [lcenter[['geometry','botm_4']], wline[['geometry','botm_4']], eline[['geometry','botm_4']],
         wbvpts[['geometry','botm_4']]],
        ignore_index=True
    )

    train_pts = train.geometry
    tx = train_pts.centroid.x.values
    ty = train_pts.centroid.y.values
    tz = train['botm_4'].values

    # 5) Fit kriging model
    UK = UniversalKriging(
        tx, ty, tz,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
        # variogram_parameters={"range": ..., "sill": ..., "nugget": ...},  # optional: speeds things up
        # drift_terms=['regional_linear'],  # only if you need a trend
    )

    cent = grd_in.geometry.centroid
    zx, _ = UK.execute("points", cent.x.values, cent.y.values)  # returns values in same order
    grd_in['botm_4_krig'] = np.asarray(zx)
    
    ashp = grd_temp.merge(grd_in[['i','j','botm_4_krig']],on=['i','j'],how='left')

    ashp['botm_4'] = ashp['botm_4_krig']
    ashp = ashp.drop(columns=['botm_4_krig'], errors='ignore')
    
    grd = pd.concat([grd,ashp],ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(8,8))
    grd.plot(column='botm_4',legend=True,ax=ax)
    
    grd["botm_4_masked"] = grd["botm_4"].where(grd["idom_4"]>0, np.nan)
    
    # manual pt adjs:
    grd.loc[(grd.i ==135) & (grd.j==46),'botm_4_masked'] = grd.loc[(grd.i ==135) & (grd.j==46),'botm_4_masked'] -15.0
    grd.loc[(grd.i ==135) & (grd.j==45),'botm_4_masked'] = grd.loc[(grd.i ==135) & (grd.j==45),'botm_4_masked'] -10.0
    grd.loc[(grd.i ==135) & (grd.j==43),'botm_4_masked'] = grd.loc[(grd.i ==135) & (grd.j==43),'botm_4_masked'] -10.0
    
    fig, ax = plt.subplots(figsize=(8,8))
    grd.plot(column="botm_4_masked", legend=True, ax=ax)
    
    # replace botm_4 in grd with masked version where no nan:
    out = grd.copy()
    out['botm_4'] = np.where(~out['botm_4_masked'].isna(),out['botm_4_masked'],out['botm_4'])
    out = out.drop(columns=['botm_4_masked'], errors='ignore')
    
    # sort out to mathch orginal grd order:
    out = out.merge(org_grd[['i','j']],on=['i','j'],how='left',indicator=True)
    out = out.sort_values(by=['i','j']).drop(columns=['_merge'])

    return out


# -----------------------------------------------------------
# River and Drain defintions:
# -----------------------------------------------------------
def riv_definition(qa_figs=True):
    print('creating first pass at riv definition...')

    rivoutdir = os.path.join('..','..','gis','output_shps','wahp','rivs')
    if not os.path.exists(rivoutdir):
        os.makedirs(rivoutdir)
    
    grd = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','wahp7ly_cell_size_660ft_epsg2265.grid.shp'))

    out_figs = os.path.join('prelim_figs_tables','figs','riv')
    if not os.path.exists(out_figs):
        os.makedirs(out_figs)

    
    # load twdb major riv shapefile:
    riv = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','rivs.shp'))
    riv.columns = riv.columns.str.lower()
    riv = riv.to_crs(grd.crs)

    raster_path = os.path.join('..','..','gis','output_ras','dem_no_buildings.tif') # this in meters ah
    
    with rasterio.open(raster_path) as src:
        # Sample the elevation for each point
        ras_crs = src.crs

        from shapely.ops import linemerge
        from shapely.geometry import Point, MultiPoint

        multi_line = riv.unary_union
        
        seg_merg = linemerge(multi_line)
        distance_interval = 250  # Adjust as needed
        points = [seg_merg.interpolate(distance) for distance in range(0, int(seg_merg.length), distance_interval)]
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=grd.crs)
        points_gdf = points_gdf.reset_index()
        points_gdf = points_gdf.rename(columns={'index':'pid'})
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)
        coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
        points_gdf['elevation'] = [val[0] for val in src.sample(coords)]
        points_gdf['elev_ft'] = points_gdf['elevation'] * 3.28084 # convert to feet
        
        # points_gdf = points_gdf.sort_values('pid').reset_index(drop=True)

        # # Ensure monotonic decreasing elevations with a cap on iterations
        # max_iterations = 15 # Prevent endless loop by capping iterations
        # iterations = 0
        # while iterations < max_iterations:
        #     is_monotonic = True
        #     for i in range(1, len(points_gdf)):
        #         if points_gdf.loc[i, 'elevation'] >= points_gdf.loc[i - 1, 'elevation']:
        #             # Non-decreasing elevation found; apply small decrement
        #             points_gdf.loc[i, 'elevation'] = points_gdf.loc[i - 1, 'elevation'] - 1e-3
        #             is_monotonic = False
        #     if is_monotonic:
        #         break
        #     iterations += 1

        #points_gdf.to_file(os.path.join(rivoutdir, f'points_testz_monotonic.shp'))
        # merge with grid:
        grd['geom_cpy'] = grd['geometry']
        mrg = points_gdf.sjoin(grd, how='left',  predicate='within')
        mrg = mrg[['node','row','col','i','j','elevation','geom_cpy']]
        mrg['geometry'] = mrg['geom_cpy']
        riv_df = mrg.drop(columns=['geom_cpy'])
        
    # for each row col drop duplicates keep lowest elevation:
    riv_df = riv_df.sort_values('elevation').drop_duplicates(subset=['row','col'], keep='first')
    riv_df = gpd.GeoDataFrame(riv_df, geometry='geometry', crs=grd.crs)    

    segments = overlay(riv[['fcode','geometry']], riv_df[['node','geometry']], how='intersection')
    segments['segment_length'] = segments.geometry.length
    segments = segments.drop(columns=['geometry'])
    riv_df = riv_df.merge(segments, on='node', how='left')
    riv_df = riv_df.drop(columns=['fcode'])

    riv_df['elevation'] = riv_df['elevation'] * 3.28084 # convert to feet
    
    # assume widths of 150 feet:
    riv_df['width'] = 150.0
    # assume river depths of 5 feet:
    riv_df['rbot'] = riv_df['elevation'] - 3.0
    # assume a foot of stage on top of elevation:
    riv_df['stage'] = riv_df['elevation'] + 1.5 
    riv_df['cond'] = riv_df['segment_length'] * riv_df['width'] * 1.0 / (riv_df['stage'] - riv_df['rbot']) # assuming river bed K = 1.0 ft/d
    
    riv_df['cond'] = riv_df['cond']*.5 # trying half conducatnce run 
    
    # drop riv cells with less than 125 feet of segment length:
    riv_df = riv_df[riv_df['segment_length'] > 100]
    
    if qa_figs:    
        # histogram of cond values:
        avg_cond = riv_df['cond'].mean()
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.set_title('Histogram of Initial Estimate\n River Conductances')
        ax.hist(riv_df['cond'], bins=50, color='blue', alpha=0.7,edgecolor='black')
        ax.set_xlabel('Conductance (ft/d)')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.set_axisbelow(True)

        # Add a vertical line at the average value
        ax.axvline(avg_cond, color='black', linestyle='dashed', linewidth=2)
        
        props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1)

        # Add the text box
        text_x = avg_cond + avg_cond * 0.35  # Adjust text position
        text_y = 50  # Adjust vertical position
        ax.text(text_x, text_y, f'Average = {avg_cond:,.0f}', color='black', fontsize=10, verticalalignment='center', bbox=props)
        ax.annotate('', xy=(text_x, text_y), xytext=(avg_cond, 50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        plt.savefig(os.path.join(out_figs, 'riv_cond_hist.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_figs, 'riv_cond_hist.pdf'), format='pdf', dpi=300)
        plt.close(fig)

        fig, ax = plot_basemap(epsg=2265)
        ax.set_title("River's Initial Conductivity Estimate")
        riv_df.plot(column='cond', cmap='viridis', linewidth=0.5, ax=ax, edgecolor='none')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=riv_df['cond'].min(), vmax=riv_df['cond'].max()))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7) 
        cbar.set_label('Conductance\n(ft/d)')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        #riv.plot(ax=ax, color='lightblue', linewidth=0.25, label='Major Rivers')
        ax.legend()
        
        plt.savefig(os.path.join(out_figs, 'riv_cond_estimate.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_figs, 'riv_cond_estimate.pdf'), format='pdf', dpi=300)
        plt.close(fig)

        fig, ax = plot_basemap(epsg=2265)
        ax.set_title("River's Initial Stage")
        riv_df.plot(column='stage', cmap='viridis', linewidth=0.5, ax=ax, edgecolor='none')

        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=riv_df['stage'].min(), vmax=riv_df['stage'].max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('Stage (ft)')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        #riv.plot(ax=ax, color='lightblue', linewidth=0.25, label='Major Rivers')
        ax.legend()

        plt.savefig(os.path.join(out_figs, 'riv_stage.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_figs, 'riv_stage.pdf'), format='pdf', dpi=300)
        plt.close(fig)

    # get otter tail riv cells and label:
    otter = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','rivs','otter_tail_lines.shp'))
    otter = otter[['gnis_name','geometry']]
    otter['gnis_name'] = 'Otter Tail River'
    otter = otter.rename(columns={'gnis_name':'name'})
    riv_df = riv_df.sjoin(otter, how='left', predicate='intersects')
    
    nm = f'riv_with_init_props.shp'
    riv_df.to_file(os.path.join(rivoutdir,nm))


def drain_definition(qa_figs=False):
    print('creating first pass at drn definition...')

    drnoutdir = os.path.join('..','..','gis','output_shps','wahp','drns')
    if not os.path.exists(drnoutdir):
        os.makedirs(drnoutdir)
    
    grd = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','wahp7ly_cell_size_660ft_epsg2265.grid.shp'))

    out_figs = os.path.join('prelim_figs_tables','figs','drn')
    if not os.path.exists(out_figs):
        os.makedirs(out_figs)

    
    # load twdb major riv shapefile:
    drn = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','flow_lines_all_clipped.shp'))
    drn.columns = drn.columns.str.lower()
    drn = drn.to_crs(grd.crs)

    raster_path = os.path.join('..','..','gis','output_ras','dem_no_buildings.tif') # this in meters ah
    
    ag_drains = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','Drains.shp'))
    
    with rasterio.open(raster_path) as src:
        # Sample the elevation for each point
        ras_crs = src.crs

        from shapely.ops import linemerge
        from shapely.geometry import Point, MultiPoint

        multi_line = drn.unary_union
        
        seg_merg = linemerge(multi_line)
        distance_interval = 250  # Adjust as needed
        points = [seg_merg.interpolate(distance) for distance in range(0, int(seg_merg.length), distance_interval)]
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=grd.crs)
        points_gdf = points_gdf.reset_index()
        points_gdf = points_gdf.rename(columns={'index':'pid'})
        coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
        points_gdf['elevation'] = [val[0] for val in src.sample(coords)]
        points_gdf['elev_ft'] = points_gdf['elevation'] * 3.28084 # convert to feet
        
        ag_drains['centroid'] = ag_drains.geometry.centroid
        ag_drn_cords = [(pt.x, pt.y) for pt in ag_drains.centroid]
        ag_drains['elevation'] = [val[0] for val in src.sample(ag_drn_cords)]
        ag_drains['elev_ft'] = ag_drains['elevation'] * 3.28084 # convert to feet
        
        #points_gdf = points_gdf.sort_values('pid').reset_index(drop=True)
        # # Ensure monotonic decreasing elevations with a cap on iterations
        # max_iterations = 15 # Prevent endless loop by capping iterations
        # iterations = 0
        # while iterations < max_iterations:
        #     is_monotonic = True
        #     for i in range(1, len(points_gdf)):
        #         if points_gdf.loc[i, 'elevation'] >= points_gdf.loc[i - 1, 'elevation']:
        #             # Non-decreasing elevation found; apply small decrement
        #             points_gdf.loc[i, 'elevation'] = points_gdf.loc[i - 1, 'elevation'] - 1e-3
        #             is_monotonic = False
        #     if is_monotonic:
        #         break
        #     iterations += 1

        #points_gdf.to_file(os.path.join('gis', 'output_shps', f'{grp}_points_testz_monotonic.shp'))
        # merge with grid:
        grd['geom_cpy'] = grd['geometry']
        mrg = points_gdf.sjoin(grd, how='left',  predicate='within')
        mrg = mrg[['node','row','col','i','j','elevation','geom_cpy']]
        mrg['geometry'] = mrg['geom_cpy']
        drn_df = mrg.drop(columns=['geom_cpy'])
        
        mrg = ag_drains.sjoin(grd, how='left',  predicate='within')
        mrg = mrg[['node','row','col','i','j','elevation','geom_cpy']]
        mrg['geometry'] = mrg['geom_cpy']
        ag_drn_df = mrg.drop(columns=['geom_cpy'])
        # drop nan rows:
        ag_drn_df = ag_drn_df.dropna(subset=['node'])
        
    # for each row col drop duplicates keep lowest elevation:
    drn_df = drn_df.sort_values('elevation').drop_duplicates(subset=['row','col'], keep='first')
    drn_df = gpd.GeoDataFrame(drn_df, geometry='geometry', crs=grd.crs)    

    ag_drn_df = ag_drn_df.sort_values('elevation').drop_duplicates(subset=['row','col'], keep='first')
    ag_drn_df = gpd.GeoDataFrame(ag_drn_df, geometry='geometry', crs=grd.crs)
    
    segments = overlay(drn[['fcode','geometry']], drn_df[['node','geometry']], how='intersection')
    segments['segment_length'] = segments.geometry.length
    segments = segments.drop(columns=['geometry'])
    drn_df = drn_df.merge(segments, on='node', how='left')
    drn_df = drn_df.drop(columns=['fcode'])

    # load in the riv shapefile and remove any drn segments that are also riv segments:
    # make sure riv shapefile exists:
    assert os.path.exists(os.path.join('..','..','gis','output_shps','wahp','rivs','riv_with_init_props.shp')), 'Riv shapefile not found, run riv_defintion fx first!'
    riv = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','rivs','riv_with_init_props.shp'))
    
    drn_df = drn_df[~drn_df['node'].isin(riv['node'])]
    
    # make sure ag drain segments do not overlap with riv segments and drn_df segments:
    ag_drn_df = ag_drn_df[~ag_drn_df['node'].isin(riv['node'])]
    ag_drn_df = ag_drn_df[~ag_drn_df['node'].isin(drn_df['node'])]
    
    # fig, ax = plot_basemap(epsg=2265)
    # ax.set_title("Agricultural Drain Locations")
    # drn_df.plot(ax=ax, color='red', markersize=5, label='Flow Lines Drains')
    # ag_drn_df.plot(ax=ax, color='cyan', markersize=5, label='Ag Drains')
    # riv.plot(ax=ax, color='lightblue', linewidth=0.25, label='Major Rivers')
    # drn.plot(ax=ax, color='lightblue', linewidth=0.25, label='Stream, river, and canal netwrok')

    drn_df['elevation'] = drn_df['elevation'] * 3.28084 # convert to feet
    ag_drn_df['elevation'] = ag_drn_df['elevation'] * 3.28084 # convert to feet
    
    # assume widths of 50 feet:
    drn_df['width'] = 50.0
    # assume river depths of 5 feet:
    drn_df['rbot'] = drn_df['elevation'] - 5.0
    # assume a foot of stage on top of elevation:
    drn_df['stage'] = drn_df['elevation'] + 1.0 
    drn_df['cond'] = drn_df['segment_length'] * drn_df['width'] * 1.0 / (drn_df['stage'] - drn_df['rbot']) # assuming drn bed K = 1.0 ft/d
    
    drn_df['cond'] = drn_df['cond']*.5 # trying half conducatnce run
    
    # drop riv cells with less than 125 feet of segment length:
    drn_df = drn_df[drn_df['segment_length'] > 200]
    
    ag_drn_df['stage'] = ag_drn_df['elevation'] - 1.0
    ag_drn_df['cond'] = 2000.0 # assign constant conductance of 2000 ft/d for ag drains
    
    drn_df = pd.concat([drn_df, ag_drn_df], ignore_index=True)
    drn_df['row'] = drn_df['row'].astype(int)
    drn_df['col'] = drn_df['col'].astype(int)
    drn_df['i'] = drn_df['i'].astype(int)
    drn_df['j'] = drn_df['j'].astype(int)
    
    # histogram of cond values:
    avg_cond = drn_df['cond'].mean()
    
    if qa_figs:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.set_title('Histogram of Initial Estimate\n Drain Conductances')
        ax.hist(drn_df['cond'], bins=50, color='blue', alpha=0.7,edgecolor='black')
        ax.set_xlabel('Conductance (ft/d)')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.set_axisbelow(True)

        # Add a vertical line at the average value
        ax.axvline(avg_cond, color='black', linestyle='dashed', linewidth=2)
        
        props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1)

        # Add the text box
        text_x = avg_cond + avg_cond * 0.35  # Adjust text position
        text_y = 50  # Adjust vertical position
        ax.text(text_x, text_y, f'Average = {avg_cond:,.0f}', color='black', fontsize=10, verticalalignment='center', bbox=props)
        ax.annotate('', xy=(text_x, text_y), xytext=(avg_cond, 50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

        fig, ax = plot_basemap(epsg=2265)
        ax.set_title("Drain's Initial Conductance Estimate")
        drn_df.plot(column='cond', cmap='viridis', linewidth=0.5, ax=ax, edgecolor='none')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=drn_df['cond'].min(), vmax=drn_df['cond'].max()))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7) 
        cbar.set_label('Conductance\n(ft/d)')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        drn.plot(ax=ax, color='lightblue', linewidth=0.25, label='Stream, river, and canal netwrok')
        ax.legend()
        
        plt.savefig(os.path.join(out_figs, 'drn_cond_estimate.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_figs, 'drn_cond_estimate.pdf'), format='pdf', dpi=300)

    nm = f'drn_with_init_props.shp'
    drn_df.to_file(os.path.join(drnoutdir,nm))    

# -----------------------------------------------------------
# MODFLOW 6 model setup functions.
# -----------------------------------------------------------
def build_model_with_ll_pivot(
    ll_corner, nrow, ncol, delr, delc, rotation_degrees, sim_ws, model_name, exe_name):
    """
    Build a dummy FloPy MF6 model using the provided lower-left corner and grid dimensions.
    The dummy DIS package is later overwritten.
    """
    sim = flopy.mf6.MFSimulation(sim_name=model_name, exe_name=exe_name, sim_ws=sim_ws,continue_=True)
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, save_flows=True,newtonoptions=['under_relaxation'])
    ims = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", outer_hclose=1e-2, inner_hclose=1e-2
    )
    sim.register_ims_package(ims, [gwf.name])
    dis = flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=nrow, ncol=ncol, delr=delr, delc=delc
    )
    return sim, gwf


def setup_mf6(mws, crs, top, botm, idomain, ll_corner, nrow, ncol, cell_size, angrot,
              strt_yr, end_of_hm, pred_period_len,annual_only,grd,mnm,add_early_time_pmp=True):
    """
    Set up a MODFLOW 6 simulation using the DIS package and adds an IC package
    that uses the top elevations for initial heads.
    """
    mf6_exe = find_mf6_exe()
    # copy over mf6 exe:
    shutil.copy(mf6_exe, mws)
    sim, gwf = build_model_with_ll_pivot(
        ll_corner=ll_corner,
        nrow=nrow,
        ncol=ncol,
        delr=cell_size,
        delc=cell_size,
        rotation_degrees=angrot,
        sim_ws=mws,
        model_name=mnm,
        exe_name=mf6_exe,
    )
    
    # create tdis package
    strt_yr = strt_yr - 1
    ss_start_dt = f"{strt_yr}-12-31"
    
    if annual_only == True:
        # create tdis_rc:
        tdis_rc = []
        nper = 0
        for i in range(strt_yr, end_of_hm):
            if nper == 0:
                tdis_rc.append((1.0, 1, 1.2)) # ss period can be any length, setting to 1 for clairty in output
                nper += 1
            else:
                # check if year is a leap year:
                if i % 4 == 0:
                    tdis_rc.append((366, 1.0, 1.2))
                else:
                    tdis_rc.append((365, 1.0, 1.2))
                nper += 1
            
        for i in range(end_of_hm, end_of_hm + pred_period_len):
            if i % 4 == 0:
                tdis_rc.append((366, 1.0, 1.2))
            else:
                tdis_rc.append((365, 1.0, 1.2))
            nper += 1

    else:
        # Create tdis_rc
        tdis_rc = []
        nper = 0
        start_datetime = []  # To store start datetime of each period
        end_datetime = []    # To store end datetime of each period

        # SS period:
        start_datetime.append(pd.Timestamp(ss_start_dt))
        end_datetime.append(pd.Timestamp(ss_start_dt)+ pd.DateOffset(days=1))
        
        
        # Annual stress periods for 1970 through 1999
        for i in range(strt_yr, 2000):
            if i % 4 == 0:
                tdis_rc.append((366, 1.0, 1.2))  # Leap year
            else:
                tdis_rc.append((365, 1.0, 1.2))  # Non-leap year
            nper += 1
            start_datetime.append(pd.Timestamp(f"{i}-01-01"))
            end_datetime.append(pd.Timestamp(f"{i}-12-31"))

        # Monthly stress periods for 2000 through 2024
        for i in range(2000, 2024):
            for month in range(1, 13):  # 12 months for each year
                # Get the number of days in the current month and year
                days_in_month = calendar.monthrange(i, month)[1]
                tdis_rc.append((days_in_month, 1, 1.2))  # Add period based on actual days in month
                nper += 1
                
                # Set the start and end datetime for the current month
                start_date = pd.Timestamp(f"{i}-{month:02d}-01")
                end_date = pd.Timestamp(f"{i}-{month:02d}-{days_in_month:02d}")
                
                start_datetime.append(start_date)
                end_datetime.append(end_date)

        # Annual stress periods for the prediction period (2024 through 2044)
        for i in range(2024, 2024 + pred_period_len):
            if i % 4 == 0:
                tdis_rc.append((366, 1.0, 1.2))  # Leap year
            else:
                tdis_rc.append((365, 1.0, 1.2))  # Non-leap year
            nper += 1
            start_datetime.append(pd.Timestamp(f"{i}-01-01"))
            end_datetime.append(pd.Timestamp(f"{i}-12-31"))

        # Convert start and end datetimes to a DataFrame for better visibility
        dates_df = pd.DataFrame({
            'Stress Period': range(0, nper + 1),
            'Start Date': start_datetime,
            'End Date': end_datetime
        })

    tdis = flopy.mf6.ModflowTdis(sim, 
                                 pname="tdis", 
                                 time_units="days", 
                                 nper=nper, 
                                 perioddata=tdis_rc,
                                 start_date_time = ss_start_dt
                                 )
                                 

    # set ims settings:
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
                            rcloserecord=1000.0,
                            linear_acceleration="BICGSTAB",
                            scaling_method="NONE",
                            reordering_method="NONE",
                            relaxation_factor=0.97,
                            )
    sim.register_ims_package(ims, [gwf.name])
    
    # Convert arrays to NumPy arrays.
    top_arr = np.array(top, dtype=float)
    botm_arr = np.array(botm, dtype=float)
    idomain_arr = np.array(idomain, dtype=int)

    # Convert flipped arrays to lists for DIS package.
    top_list = top_arr.tolist()
    botm_list = botm_arr.tolist()
    idomain_list = idomain.tolist()

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=botm_arr.shape[0],
        nrow=nrow,
        ncol=ncol,
        delr=cell_size,
        delc=cell_size,
        top=top_list,
        botm=botm_list,
        idomain=idomain_list,
        xorigin=ll_corner[0],
        yorigin=ll_corner[1],
        angrot=angrot,
        length_units="FEET",
    )
    
    nlay = dis.nlay.get_data()
    
    # Use the flipped top array for the IC package.
    strt_array = np.tile(top_arr, (nlay, 1, 1))
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt_array)
    
    # init npf package:
    k0_cly = 0.5 # starting at low val from pump test result
    k0_snd = 100.0
    k0_soils = 245.0
    k = []
    k33 = []
    for layer in range(0, dis.nlay.get_data()):
        if lydict[layer] == "clay":
            k.append(k0_cly)
            k33.append(0.0001) # note these are aniso ratios K33/K 
        elif lydict[layer] == "sand":
            k.append(k0_snd)
            k33.append(0.1)
        elif lydict[layer] == "soils":
            k.append(k0_soils)
            k33.append(0.1)
        else:
            raise ValueError(f"Unknown layer type: {lydict[layer]}")
    
    icell = np.zeros(nlay)
    icell[0] = 1 # convertible layer
    
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        icelltype=icell,
        k=k,
        k33=k33,
        k33overk=True,
    )
    
    # intit storage package:
    sy = flopy.mf6.ModflowGwfsto.sy.empty(gwf, layered=True)
    for layer in range(0, nlay):
        sy[layer]["data"] = 0.2

    ss = flopy.mf6.ModflowGwfsto.ss.empty(gwf, layered=True, default_value=0.0007)
    
    iconv = botm_arr.copy()*0
    if len(div_ly1) > 0:
        for ly in range(0, nlay):
            print(ly)
            if ly <= 2:
                iconv[ly,:,:] = 1 # make these convertible
    else:
        iconv[0,:,:] = 1 # make layer 1 convertible
        
    iconv = iconv.astype(int)
    iconv_lst = iconv.tolist()
    
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        save_flows=True,
        iconvert=iconv_lst,
        ss=ss,
        sy=sy,
        steady_state={0: True},
        transient={1: True},
    )
    

    # ---------------- #
    # init riv package:
    # ---------------- #
    
    riv_df = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','rivs','riv_with_init_props.shp'))

    # add riv package for non Otter Tail River segments:
    riv_dict = {}
    for i in range(nper):
        per_spd = []
        for idx, row in riv_df.iterrows():
            if row['name'] == 'Otter Tail River':
                continue # skip otter tail river for now
            ly = 0
            r = row['i']
            c = row['j']
            cond = row['cond']
            rbot = row['rbot']
            stage = row['stage']
            per_spd.append(((ly, r, c), stage,cond,rbot))
        riv_dict[i] = per_spd 
           
    riv = flopy.mf6.ModflowGwfriv(
        gwf,
        stress_period_data=riv_dict,
        pname="riv",
        save_flows=True,
        filename=f'{mnm}.riv')
    
    # add riv package for Otter Tail River segments:
    otter_dict = {}
    for i in range(nper):
        per_spd = []
        for idx, row in riv_df.iterrows():
            if row['name'] != 'Otter Tail River':
                continue # skip non Otter Tail River segments
            ly = 0
            r = row['i']
            c = row['j']
            cond = row['cond']
            rbot = row['rbot']
            stage = row['stage']
            per_spd.append(((ly, r, c), stage,cond,rbot))
        otter_dict[i] = per_spd
    
    otter_riv = flopy.mf6.ModflowGwfriv(
            gwf,
            stress_period_data=otter_dict,
            pname="otriv",
            save_flows=True,
            filename=f'{mnm}.otriv')
            
    
    # ---------------- #
    # init drn package:
    # ---------------- #
    
    drn_df = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','drns','drn_with_init_props.shp'))
    
    drn_dict = {}
    for i in range(nper):
        per_spd = []
        for idx, row in drn_df.iterrows():
            ly = 0
            r = row['i']
            c = row['j']
            stage = row['stage']
            cond = row['cond']
            per_spd.append(((ly, r, c), stage,cond))
        drn_dict[i] = per_spd
        
    drn = flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data=drn_dict,
        pname="drn",
        save_flows=True,
        filename=f'{mnm}.drn')
    
    # init recharge package:
    # create a dummy recharge package to get the model to run:
    # this will be replaced with a real recharge package later.
    # get drain and river locs and set recharge to 0 in those cells:
    riv_cells = riv_df[['i','j']].values.tolist()
    drn_cells = drn_df[['i','j']].values.tolist()
    
    rch_dict = {}
    for i in range(nper):
        rch_temp = np.ones((nrow, ncol), dtype=float)*0.25/(12*365) # 0.25 in/yr to ft/d ~1% of precipitation
        # set riv and drn cells to 0:
        for r, c in riv_cells:
            rch_temp[r, c] = 0.0
        for r, c in drn_cells:
            rch_temp[r, c] = 0.0
        rch_dict[i] = rch_temp

    rch_dict = get_rcha_tseries_from_swb2_nc(gwf,perioddata=tdis_rc,
                                  rch_dict=rch_dict,
                                  start_date_time = ss_start_dt,
                                  annual_only=annual_only,
                                  ftag = 'net_infiltration')
         
    rch = flopy.mf6.ModflowGwfrcha(gwf,
                                   recharge=rch_dict,
                                   save_flows=True,
                                   pname="rch",
                                   filename=f"{mnm}.rch",)     
    # ---------------- #
    # init ghb package:
    # ---------------- #
    # as a start putting a ghb in every model edge cell and setting stage to the top of the model
    ghb_arr = top.copy()
    zero_out_inside = top.copy() * 0.0
    # set edge cells to 1, so 0 and -1 col/row
    zero_out_inside[0,:] = 1
    zero_out_inside[-1,:] = 1
    zero_out_inside[:,0] = 1
    zero_out_inside[:,-1] = 1
    ghb_arr = ghb_arr * zero_out_inside # zero out inside cells
    
    ghb_all = pd.DataFrame()
    for ly in [0,1,2]: # update layers based on Mel email 07/14
        # 2D array of ghb values, to dataframe with k,i,j,stage,cond:
        # reshape ghb:
        ghb_2d = ghb_arr.reshape((nrow*ncol))
        ghb_df = pd.DataFrame({
            'k': ly,
            'i': np.repeat(np.arange(nrow), ncol),
            'j': np.tile(np.arange(ncol), nrow),
            'stage': ghb_2d,
            'cond': (ghb_2d * 0.0) + 1000  # set conductance to 1,000ft^2/d SRM reduced by factor of 10
            })
        # check if in an inactive cell:
        ghb_df['active'] = idomain_arr[ly,:,:].flatten()
        # drop rows with zero stage:
        ghb_df = ghb_df[ghb_df['stage'] > 0]
        ghb_df = ghb_df[ghb_df['active'] > 0] # only keep active cells
        ghb_df = ghb_df.drop(columns=['active'])
        ghb_all = pd.concat([ghb_all, ghb_df], ignore_index=True)

    ghb_dict = {}
    for i in range(nper):
        per_spd = []
        for idx, row in ghb_all.iterrows():
            ly = int(row['k'])
            r = int(row['i'])
            c = int(row['j'])
            stage = row['stage']
            cond = row['cond']
            per_spd.append(((ly, r, c), stage,cond))
        ghb_dict[i] = per_spd
        
    ghb = flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=ghb_dict,
        pname="ghb",
        save_flows=True,
        filename=f'{mnm}.ghb')


    # ---------------- #
    # init wbv ghb package:
    # ---------------- #
    # Create GHB just of the upgradient side of the WBV
    # find perimeter cells
    idomL4 = idomain[4].copy()

    def find_boundary_cells(idomain):
        """
        Find cells where idomain == 1 and at least one neighbor has idomain == -1.
        Returns a list of (row, col) tuples.
        """
        nrow, ncol = idomain.shape
        boundary_cells = []
        boundary_arr = np.zeros_like(idomain, dtype=bool)
        for i in range(nrow):
            for j in range(ncol):
                if idomain[i, j] != 1:
                    continue
                # Check 4-connected neighbors (up, down, left, right)
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nrow and 0 <= nj < ncol:
                        neighbors.append(idomain[ni, nj])
                if any(n == -1 for n in neighbors):
                    boundary_cells.append((i, j))
                    boundary_arr[i, j] = True
        return boundary_cells, boundary_arr
    # find boundary cells:
    boundary_cells, boundary_arr = find_boundary_cells(idomL4)
    # only want souther end, row > 140
    boundary_cells = [cell for cell in boundary_cells if cell[0] > 140]
    boundary_arr[:140,:] = False  # zero out the top rows

    ghb_wbv = pd.DataFrame(columns=['k', 'i', 'j', 'stage', 'cond'])
    ghb_wbv.loc[:,'i'] = [cell[0] for cell in boundary_cells]
    ghb_wbv.loc[:, 'j'] = [cell[1] for cell in boundary_cells]
    ghb_wbv.loc[:, 'k'] = 4
    ghb_wbv.loc[:,'stage'] = 965  # set for now based on pre-development SS heads
    ghb_wbv.loc[:,'cond'] = 300 #100*(660*660)/5280  # Kh * A / L (100 ft/d * (660 * 660 window)/ 5280 ft Length

    ghb_wbv_dict = {}
    for i in range(nper):
        per_spd = []
        for idx, row in ghb_wbv.iterrows():
            ly = int(row['k'])
            r = int(row['i'])
            c = int(row['j'])
            stage = row['stage']
            cond = row['cond']
            per_spd.append(((ly, r, c), stage, cond))
        ghb_wbv_dict[i] = per_spd

    ghb_2 = flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=ghb_wbv_dict,
        pname="ghb_wbv",
        save_flows=True,
        filename=f'{mnm}_wbv.ghb')

    # --------------------------- #        
    # init output control package:
    # --------------------------- #
    
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{mnm}.cbb",
        budgetcsv_filerecord='budget.csv',
        head_filerecord=f"{mnm}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("BUDGET", "LAST")],
    )
    
    # write out temp files:
    sim.write_simulation() # write out temp model files needed for the following package builds...
    
    # ---------------- #
    # init wel package:
    # ---------------- #

    curdir = os.getcwd()
    os.chdir(os.path.join("data","pumping"))
    if annual_only == True:
       pmp = wuprocess.build_use_by_well(monthly=False)
    else:
        pmp = wuprocess.build_use_by_well(monthly=True)
    
    # link model layer:
    pmp_shp = gpd.read_file('WahpetonBV_Wells_with5721.shp')
    pmp_shp = pmp_shp[['well_index', 'site_locat', 'well_no', 'county',
       'aquifer', 'purpose', 'site_owner', 'creation_d', 'total_dept',
       'top_screen', 'bottom_scr']]
    
    # store install date for earliest wells installed by permittee:
    pmp_shp = pmp_shp.rename(columns={'creation_d':'install_date'})
    pmp_shp['install_date'] = pd.to_datetime(pmp_shp['install_date'], errors='coerce')
    install_dates = pmp_shp.groupby('site_owner')['install_date'].min().reset_index()
    # map to permit holder name:
    pnms = {'City of Wahpeton':'wahpeton, city of', 'Froedtert Malting':'froedtert malt corp','Minn-Dak':'minn-dak farmers cooperative', 'ProGold':'cargill incorporated',
       'city of Breckenridge':'city of breckenridge', 'city of Wahpeton':'wahpeton, city of'}
    install_dates['permit_holder_name'] = install_dates['site_owner'].map(pnms)

    pmp = pmp.merge(pmp_shp, left_on='Well', right_on='site_locat', how='left')

    os.chdir(curdir)
    
    spd = stress_period_df_gen(mws,strt_yr+1,annual_only=annual_only) # plus one b/c -1 was applied earlier when building tdis, need to undo that here
    spd_stor = spd.copy()

    pmp.columns = pmp.columns.str.lower()   
    pmp['permit_holder_name'] = pmp['permit_holder_name'].str.lower()
    
    if annual_only == True:
        pmp = pmp.merge(spd[['stress_period','year']], on='year', how='left')
    else:
        spd['month'] = spd['start_datetime'].dt.month
        pmp = pmp.merge(spd[['stress_period','year','month']], on=['year','month'], how='left')
    
    # sort by stress period:
    pmp = pmp.sort_values(['stress_period'])

    # to geodataframe:
    pmp = gpd.GeoDataFrame(pmp, geometry=gpd.points_from_xy(pmp['x_2265'], pmp['y_2265']), crs=grd.crs)
    if annual_only == True:
        grid = grd[['node', 'row', 'col', 'i', 'j', 'geometry', 'idom_0', 'idom_1',
                    'idom_2', 'idom_3', 'idom_4', 'idom_5','idom_6']]
    else:
        grid = grd[['node', 'row', 'col', 'i', 'j', 'geometry', 'idom_0', 'idom_1',
                    'idom_2', 'idom_3', ]]
        
    pmp = pmp.sjoin(grid, how='left', predicate='within')   

    # filling in missing data for permit 5721:
    top_at_5721 = top_arr[pmp.loc[pmp.well=='permit_5721','i'].values[0],pmp.loc[pmp.well=='permit_5721','j'].values[0]]
    top_o_scr = top_at_5721 - 890
    bot_o_scr = top_at_5721 - 900
    pmp.loc[pmp.well=='permit_5721','top_screen'] = top_o_scr
    pmp.loc[pmp.well=='permit_5721','bottom_scr'] = bot_o_scr
    
    pmp['scr_midpt'] = (pmp['top_screen'] + pmp['bottom_scr']) / 2.0
    
    def find_layer(w, bot, top):
        i = w['i']
        j = w['j']
        bot = bot[:, i, j]
        top = top[i, j]
        mid_elev = top - w.scr_midpt

        # Create layer boundaries from top to bottom
        ly_cake = np.concatenate([[top], bot])

        # Check that layers are in descending order
        if not np.all(np.diff(ly_cake) <= 0):
            raise ValueError("Layer boundaries are not in decreasing elevation order.")

        # Find the layer index where mid_elev falls between ly_cake[k] and ly_cake[k+1]
        for k in range(len(ly_cake) - 1):
            upper = ly_cake[k]
            lower = ly_cake[k + 1]
            if upper >= mid_elev >= lower:
                return k, ly_cake
        return np.nan, np.nan # If not within model layers

    well_tags = {'cargill incorporated':'car',
                'froedtert malt corp':'malt',
                'wahpeton, city of':'cow',
                'minn-dak farmers cooperative':'minn',
                'city of breckenridge':'cob'}
    

    # get layer for each well:
    pmp['k'] = pmp.apply(lambda w: find_layer(w, botm_arr, top_arr)[0], axis=1)
    pmp['ly_cake'] = pmp.apply(lambda w: find_layer(w, botm_arr, top_arr)[1], axis=1)
    pmp['model_top_elev'] = pmp['ly_cake'].apply(lambda x: x[0])
    
    if add_early_time_pmp:
        def hindcast_with_trend(
            wshrt: pd.DataFrame,
            start_date: str | pd.Timestamp,
            *,
            annual: bool | None = None,
            trend_kind: str = "linear",            # "linear" or "log"
            weight_half_life_years: float = 5.0,   # emphasize earlier obs for trend fit (used after t0/missing)
            early_window_years: float = 8.0,       # monthly-only: seasonality baseline window
            apply_monthly_seasonality: bool = True,
            year_len: float = 365.2425,
            # NEW: anchor controls
            start_anchor_quantile: float | None = 0.20,
            start_anchor_value: float | None = None,
            anchor_mode: str = "interpolate"
        ) -> pd.DataFrame:
            """
            Hindcast to start_date using a trend model, with an optional anchored start value.
            Adds a boolean column 'observed' (True = observed cfd, False = hindcast).
            """

            if wshrt.empty:
                raise ValueError("wshrt is empty")

            df = wshrt.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["cfd"] = pd.to_numeric(df["cfd"], errors="coerce")
            df = df.sort_values("datetime")
            start_date = pd.to_datetime(start_date)

            # Auto-detect frequency
            if annual is None:
                dts = df["datetime"]
                annual = (dts.dt.day.eq(1).all() and dts.dt.month.eq(1).all()) or (dts.dt.month.nunique() == 1)
            freq = "AS-JAN" if annual else "MS"

            end_date = df["datetime"].max()
            full_idx = pd.date_range(start=start_date, end=end_date, freq=freq)

            # Aggregate to target freq
            s_obs = (
                df.set_index("datetime")["cfd"]
                .groupby(pd.Grouper(freq=freq))
                .sum(min_count=1)
            )
            s_obs_valid = s_obs.dropna().astype(float)
            if s_obs_valid.empty:
                raise ValueError("No valid observed cfd values (after numeric coercion).")

            t0 = s_obs_valid.index.min()
            y0_obs = float(s_obs_valid.loc[t0])

            # Build a full series on the target index
            s_full = s_obs.reindex(full_idx)

            # === Trend fit (for filling missing at/after t0) ===
            def years_from_t0(idx: pd.DatetimeIndex) -> np.ndarray:
                return (idx - t0).days.astype(float) / year_len

            t_obs = years_from_t0(s_obs_valid.index).astype(float)
            y_obs = s_obs_valid.to_numpy(dtype=float)

            tk = trend_kind.lower()
            if tk == "log":
                eps = max(1e-6, 1e-9 * (np.nanmax(y_obs) if len(y_obs) else 1.0))
                y_fit = np.log(np.clip(y_obs, eps, None))
                transform = np.exp
            elif tk == "linear":
                y_fit = y_obs
                transform = lambda x: x
            else:
                raise ValueError("trend_kind must be 'linear' or 'log'.")

            if len(t_obs) == 1:
                a, b = 0.0, float(y_fit[0])
            else:
                t_min = float(np.nanmin(t_obs))
                hl = max(1e-6, float(weight_half_life_years))
                w = (0.5 ** ((t_obs - t_min) / hl)).astype(float)
                W = np.sqrt(w)
                A = np.vstack([W * t_obs, W * np.ones_like(t_obs, dtype=float)]).T.astype(float)
                b_vec = (W * y_fit).astype(float)
                (a, b), *_ = np.linalg.lstsq(A, b_vec, rcond=None)
                a = float(a); b = float(b)

            t_full = years_from_t0(s_full.index).astype(float)
            y_pred_space = a * t_full + b
            y_pred = transform(y_pred_space)

            # Monthly seasonality
            if not annual and apply_monthly_seasonality:
                early_cut = t0 + pd.Timedelta(days=int(early_window_years * year_len))
                early_obs = s_obs.loc[(s_obs.index >= t0) & (s_obs.index <= early_cut)].dropna().astype(float)
                if not early_obs.empty:
                    t_early = years_from_t0(early_obs.index).astype(float)
                    base = transform(a * t_early + b)
                    base = np.where(base <= 0, np.nan, base)
                    ratio = early_obs.to_numpy(dtype=float) / base
                    seas = pd.Series(ratio, index=early_obs.index).groupby(early_obs.index.month).median()
                    med = float(seas.median()) if np.isfinite(seas.median()) else np.nan
                    if np.isfinite(med) and med != 0:
                        seas = seas / med
                    month_idx = pd.Index(s_full.index.month, name="month")
                    seas_vec = month_idx.map(seas).to_numpy()
                    seas_vec = np.where(np.isfinite(seas_vec), seas_vec, 1.0)
                    y_pred = y_pred * seas_vec

            # === Anchor at start_date ===
            if start_anchor_value is not None:
                y_anchor = float(start_anchor_value)
            elif start_anchor_quantile is not None:
                y_anchor = float(pd.to_numeric(df["cfd"], errors="coerce").quantile(start_anchor_quantile))
                if not np.isfinite(y_anchor):
                    y_anchor = float(s_obs_valid.quantile(start_anchor_quantile))
            else:
                y_anchor = float(y_pred[0])
            y_anchor = max(0.0, y_anchor)

            # Build pre-history
            pre_idx = s_full.index[(s_full.index >= start_date) & (s_full.index < t0)]
            if len(pre_idx) > 0:
                if anchor_mode == "flat":
                    y_pre = np.full(len(pre_idx), y_anchor, dtype=float)
                else:
                    if tk == "log":
                        eps = 1e-9
                        y1 = np.log(max(y_anchor, eps))
                        y2 = np.log(max(y0_obs, eps))
                        w = np.linspace(0.0, 1.0, len(pre_idx), endpoint=False)
                        y_pre = np.exp((1 - w) * y1 + w * y2)
                    else:
                        w = np.linspace(0.0, 1.0, len(pre_idx), endpoint=False)
                        y_pre = (1 - w) * y_anchor + w * y0_obs
                s_full.loc[pre_idx] = y_pre

            # Fill post-t0 gaps with prediction
            pred_series = pd.Series(y_pred, index=s_full.index).clip(lower=0.0)
            mask_post = s_full.index >= t0
            s_full.loc[mask_post] = s_full.loc[mask_post].fillna(pred_series.loc[mask_post])
            s_full = s_full.clip(lower=0.0).astype(float)

            # Build observed flag: True where original s_obs had values
            observed_mask = s_full.index.isin(s_obs_valid.index)
            out = s_full.rename_axis("datetime").reset_index(name="cfd")
            out["observed"] = observed_mask.astype(bool)

            return out

        # map well_tags:
        pmp['well_tag'] = pmp['permit_holder_name'].map(well_tags)
        groups = pmp.groupby('well_tag')

        for nm, group in groups:
            #if nm == 'minn':
            #    continue # skip minn dak for now, no early data
            permit = group.copy()

            # datetime per group
            if annual_only:
                permit['datetime'] = pd.to_datetime(
                    permit['year'].astype(str) + '0101', format='%Y%m%d'
                )
            else:
                permit['datetime'] = pd.to_datetime(
                    permit['year'].astype(str) + permit['month'].astype(str).str.zfill(2) + '01',
                    format='%Y%m%d'
                )

            unq_wells = permit['well'].unique()
            for well in unq_wells:
                well_df = permit[permit['well'] == well]
                wshrt = well_df[['datetime','cfd']].copy()

                # choose start_date
                if nm in ('minn'):
                    start_date = pd.to_datetime('1985-01-01')
                elif nm == 'cow':
                    if well == 'permit_5721':
                        continue
                    start_date = pd.to_datetime('1970-01-01')
                else:
                    continue

                # hindcast
                hind = hindcast_with_trend(
                    wshrt,
                    start_date,
                    annual=annual_only,
                    trend_kind="linear",
                    start_anchor_quantile=0.4,
                    start_anchor_value=None,
                    anchor_mode="interpolate"
                )

                # (optional) plot
                # fig, ax = plt.subplots(figsize=(10, 4))
                # ax.plot(wshrt['datetime'], wshrt['cfd'], 'o', label='Observed', markersize=4, zorder=3)
                # ax.plot(hind['datetime'], hind['cfd'], '-x', label='Hindcast', linewidth=1)
                # ax.set_title(f'Pumping Hindcast for {nm} well {well}')
                # ax.set_ylabel('Pumping Rate (cfd)')
                # ax.legend()

                # prepare hind rows → year/month
                fillvals = hind.loc[hind['observed'] == False].copy()
                fillvals['year']  = fillvals['datetime'].dt.year
                fillvals['month'] = 1 if annual_only else fillvals['datetime'].dt.month
                fillvals['day'] = 1
                if annual_only:
                    fillvals = fillvals.merge(spd[['stress_period','year']], on='year', how='left')
                else:
                    fillvals = fillvals.merge(spd[['stress_period','year','month']], on=['year','month'], how='left')
                fillvals['well'] = well
                fillvals['permit_holder_name'] = well_df['permit_holder_name'].values[0]
                fillvals['aquifer'] = well_df['aquifer'].values[0]
                fillvals['top_screen'] = well_df['top_screen'].values[0]
                fillvals['bottom_scr'] = well_df['bottom_scr'].values[0]
                fillvals['scr_midpt'] = well_df['scr_midpt'].values[0]
                fillvals['total_dept'] = well_df['total_dept'].values[0]
                fillvals['x_2265'] = well_df['x_2265'].values[0]
                fillvals['y_2265'] = well_df['y_2265'].values[0]
                fillvals['geometry'] = well_df['geometry'].values[0]
                fillvals['i'] = well_df['i'].values[0]
                fillvals['j'] = well_df['j'].values[0]
                fillvals['model_top_elev'] = well_df['model_top_elev'].values[0]
                fillvals['k'] = well_df['k'].values[0]
                cake = well_df['ly_cake'].values[0]
                fillvals['ly_cake'] = [cake] * len(fillvals)

                pmp = pd.concat([pmp, fillvals], ignore_index=True, axis=0)
                

    if nlay == 7:
        pmp.loc[pmp['k'] > 4, 'k'] = 4 # there is no pumping deeper then WBV so BRUTE FORCE this for now
        # if pmp['aquifer'] == 'Wahpeton Shallow Sand' and layer == 4 the confining unit move k up one layer
        pmp.loc[(pmp['aquifer'] == 'Wahpeton Shallow Sand') & (pmp['k'] == 3), 'k'] = 2


    # plot pumping timesseries and save shp files of pumping:
    out_figs = os.path.join('prelim_figs_tables','figs','pumping')
    if not os.path.exists(out_figs):
        os.makedirs(out_figs)
    pdf_path = os.path.join(out_figs, f'pumping_plots_nly{nlay}.pdf')
    
    for key, val in well_tags.items():
        permit = pmp.loc[pmp['permit_holder_name'] == key]   
        permit = permit.sort_values(['stress_period'])
        if annual_only == True:
            permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + '0101', format='%Y%m%d')
        else:
            permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + permit['month'].astype(str).str.zfill(2) + '01', format='%Y%m%d')
        unq_wells = permit['well'].unique()
        
        with PdfPages(pdf_path) as pdf:
            nrows, ncols = 3, 2
            plots_per_page = nrows * ncols

            for key, val in well_tags.items():
                permit = pmp[pmp['permit_holder_name'] == key].sort_values(['stress_period'])

                if annual_only:
                    permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + '0101', format='%Y%m%d')
                else:
                    permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + permit['month'].astype(str).str.zfill(2) + '01', format='%Y%m%d')

                wells = permit['well'].unique()
                num_wells = len(wells)

                for i in range(0, num_wells, plots_per_page):
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10.5))
                    axes = axes.flatten()

                    for j, well in enumerate(wells[i:i + plots_per_page]):
                        ax = axes[j]
                        well_data = permit[permit['well'] == well]
                        ax.plot(well_data['datetime'], well_data['cfd'], '-o')
                        ly_cake_str = ', '.join([f"{x:.1f}" for x in well_data.ly_cake.values[-1]])
                        ax.set_title((
                            f'{key.title()}\n'
                            f'Well ID: {well}\n'
                            f'Aq. Assignment: {well_data.aquifer.values[0]}\n'
                            f'Model Layer: {well_data.k.values[0] + 1}\n'
                            f'Midpoint Elevation: {well_data.model_top_elev.values[0] - well_data.scr_midpt.values[0]:.2f} ft\n'
                            f'Top of Screen: {well_data.model_top_elev.values[0] - well_data.top_screen.values[0]:.2f} ft\n'
                            f'Bottom of Screen: {well_data.model_top_elev.values[0] - well_data.bottom_scr.values[0]:.2f} ft\n'
                            f'Layer Elevs: [{ly_cake_str}]'),
                            fontsize=10)

                        ax.set_ylabel('Pumping Rate (cfd)', fontsize=8)
                        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
                        ax.tick_params(axis='y', labelsize=7)
                        
                        start_date = spd.loc[0, 'start_datetime']
                        end_date = spd.loc[len(spd) - 1, 'end_datetime']
                        ax.set_xlim(start_date, end_date)
                        import matplotlib.dates as mdates

                        # Major ticks: every 10 years
                        ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

                        # Minor ticks: every 1 year
                        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
                        
                    # Turn off any unused subplots
                    for j in range(num_wells - i, plots_per_page):
                        fig.delaxes(axes[j])
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"PDF of plots saved to {pdf_path}")
        
    # build well package for each permit holder:
    model_well_dicts = {}
    for key, val in well_tags.items():
        permit = pmp.loc[pmp['permit_holder_name'] == key]
        well_dict = {}
        mxbnd = 0
        for i in range(nper):
            per_spd = []
            pmp_in_sp = permit[permit['stress_period'] == i]
            if len(pmp_in_sp) > 0:
                for idx, vals in pmp_in_sp.iterrows():
                    ly = int(vals['k'])
                    r = int(vals['i'])
                    c = int(vals['j'])
                    q = np.round(vals['cfd'] * -1.0, 3)
                    per_spd.append(((ly, r, c), q))
            if len(per_spd) > mxbnd:
                mxbnd = len(per_spd)
            well_dict[i] = per_spd

        model_well_dicts[val] = {sp: lst.copy() for sp, lst in well_dict.items()}

        well = flopy.mf6.ModflowGwfwel(gwf,
                                    stress_period_data=well_dict,
                                    pname=val,
                                    save_flows=True,
                                    auto_flow_reduce=0.1,
                                    maxbound=mxbnd,
                                    filename=f'{mnm}.{val}')
    

    def plot_input_vs_model_pumping_pdf(
        pmp, spd_stor, well_tags, model_well_dicts, out_figs, nlay,
        annual_only=True, nrows=3, ncols=2, agg="mean"
    ):
        """
        Creates PDF 'pumping_plots_input_vs_model_nly{nlay}.pdf' with paired bars per year:
        - Left bar: Reported (input) cfd from `pmp`
        - Right bar: Model (applied) cfd reconstructed from `model_well_dicts`

        Parameters
        ----------
        agg : {'mean','sum'}
            Aggregation to compute annual values when multiple entries per year exist.
            'mean' is appropriate for flow rates (cfd); use 'sum' if you truly want totals.
        """

        assert agg in ("mean", "sum"), "agg must be 'mean' or 'sum'"

        pdf_path2 = os.path.join(out_figs, f'pumping_plots_input_vs_model_nly{nlay}.pdf')

        kij_map = (
            pmp.sort_values('stress_period')
            .groupby('well', as_index=False)
            .first()[['well', 'k', 'i', 'j', 'aquifer', 'model_top_elev',
                    'scr_midpt', 'top_screen', 'bottom_scr', 'ly_cake']]
        ).set_index('well')

        # Stress period timing/index from model
        sp_times = pd.to_datetime(spd_stor['start_datetime'].values)
        sp_index = spd_stor['stress_period'].values
        sp_years = pd.DatetimeIndex(sp_times).year

        plots_per_page = nrows * ncols

        # helper to aggregate a pandas Series by year
        def _annualize(df_year_val):
            if df_year_val.empty:
                return pd.Series(dtype=float)
            if agg == "mean":
                return df_year_val.groupby("year")["val"].mean()
            else:
                return df_year_val.groupby("year")["val"].sum()

        with PdfPages(pdf_path2) as pdf:
            for key, val in well_tags.items():
                permit = pmp[pmp['permit_holder_name'] == key].sort_values(['stress_period']).copy()

                # Add datetime & year for reported
                if annual_only:
                    permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + '0101', format='%Y%m%d')
                else:
                    permit['datetime'] = pd.to_datetime(
                        permit['year'].astype(str) + permit['month'].astype(str).str.zfill(2) + '01',
                        format='%Y%m%d'
                    )
                permit['year'] = permit['datetime'].dt.year

                wells = permit['well'].unique()
                num_wells = len(wells)

                # model_well_dict for this permit tag (val)
                mwd = model_well_dicts.get(val, {})

                for i0 in range(0, num_wells, plots_per_page):
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10.5))
                    axes = axes.flatten()

                    for j, well in enumerate(wells[i0:i0 + plots_per_page]):
                        ax = axes[j]

                        # -------------------
                        # Reported series (input)
                        # -------------------
                        wdat = permit[permit['well'] == well].copy()
                        rep_annual = _annualize(
                            wdat.rename(columns={"cfd": "val"})[["year", "val"]]
                        )

                        # -------------------
                        # Model series (applied)
                        # -------------------
                        if well in kij_map.index:
                            k = int(kij_map.loc[well, 'k'])
                            r = int(kij_map.loc[well, 'i'])
                            c = int(kij_map.loc[well, 'j'])

                            # Build model cfd timeseries over SPs
                            model_cfd = np.full(len(sp_index), np.nan, dtype=float)
                            for idx_sp, sp in enumerate(sp_index):
                                entries = mwd.get(sp, [])
                                # entries: [ ((kk, rr, cc), q), ... ]
                                for (kk, rr, cc), q in entries:
                                    if (kk == k) and (rr == r) and (cc == c):
                                        # q is negative extraction; convert to +cfd
                                        model_cfd[idx_sp] = -float(q)
                                        break

                            # Convert to annual by aggregating per year
                            mod_df = pd.DataFrame({"year": sp_years, "val": model_cfd})
                            mod_df = mod_df.dropna(subset=["val"])
                            mod_annual = _annualize(mod_df)
                        else:
                            mod_annual = pd.Series(dtype=float)

                        # -------------------
                        # Align years & make paired bar plot
                        # -------------------
                        all_years = sorted(set(rep_annual.index.tolist()) | set(mod_annual.index.tolist()))
                        x = np.arange(len(all_years), dtype=float)
                        width = 0.4

                        rep_vals = np.array([rep_annual.get(y, np.nan) for y in all_years], dtype=float)
                        mod_vals = np.array([mod_annual.get(y, np.nan) for y in all_years], dtype=float)

                        # Bars: Reported (left), Model (right)
                        ax.bar(x - width/2, rep_vals, width, label='Reported (input)')
                        ax.bar(x + width/2, mod_vals, width, label='Model (applied)')

                        # Title block (same metadata as before)
                        if well in kij_map.index:
                            ly_cake = kij_map.loc[well, 'ly_cake']
                            ly_cake_str = ', '.join([f"{x:.1f}" for x in ly_cake]) if isinstance(
                                ly_cake, (list, np.ndarray, pd.Series)
                            ) else str(ly_cake)
                            ax.set_title((
                                f'{key.title()}\n'
                                f'Well ID: {well}\n'
                                f'Aq. Assignment: {kij_map.loc[well, "aquifer"]}\n'
                                f'Model Layer: {int(kij_map.loc[well, "k"]) + 1}\n'
                                f'Midpoint Elevation: {kij_map.loc[well, "model_top_elev"] - kij_map.loc[well, "scr_midpt"]:.2f} ft\n'
                                f'Top of Screen: {kij_map.loc[well, "model_top_elev"] - kij_map.loc[well, "top_screen"]:.2f} ft\n'
                                f'Bottom of Screen: {kij_map.loc[well, "model_top_elev"] - kij_map.loc[well, "bottom_scr"]:.2f} ft\n'
                                f'Layer Elevs: [{ly_cake_str}]'
                            ), fontsize=10)
                        else:
                            ax.set_title(f'{key.title()}\nWell ID: {well}\n(No k,i,j mapping found)', fontsize=10)

                        # Ax cosmetics
                        ax.set_ylabel(f'Annual {agg} cfd', fontsize=8)
                        ax.set_xticks(x)
                        ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha='right', fontsize=7)
                        ax.tick_params(axis='y', labelsize=7)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '{:,.0f}'.format(v)))
                        ax.legend(fontsize=7, loc='upper left', frameon=False)

                    # turn off unused axes
                    used = min(plots_per_page, num_wells - i0)
                    for j_off in range(used, plots_per_page):
                        fig.delaxes(axes[j_off])

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        print(f"PDF of input vs model paired-bar plots saved to {pdf_path2}")

    plot_input_vs_model_pumping_pdf(
        pmp, spd_stor, well_tags, model_well_dicts, out_figs, nlay,
        annual_only=annual_only, nrows=nrows, ncols=ncols, agg="mean"
    )

    # obs package set up:
    ss_targs, trans_targs, hdiff_targs = wlprocess.main(False,div_ly1,annual_flag=annual_only,mnm=mnm)
    
    # ss_targs['layer'].unique()
    # # correct for the new layer add WBV obs shift a lyer down
    # ss_targs.loc[ss_targs['layer'] >= 5, 'layer'] = ss_targs.loc[ss_targs['layer'] >=5, 'layer'] + 1
    # ss_targs.loc[ss_targs['k'] >= 4, 'k'] = 5
    # tarns_targs.loc[tarns_targs['k'] >= 4, 'layer'] = 5

    wells = ss_targs.obsprefix.unique()
    ss_hd_list = []
    for well in wells:
        subset = ss_targs[ss_targs.obsprefix==well].iloc[0,:]
        id = idomain_arr[subset.k, subset.i, subset.j]
        if id == 1:
            ss_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
        else:
            print(f"Skipping {well} at ({subset.k}, {subset.i}, {subset.j}) as it is not in the active domain.")
    
    trans_hd_list = []    
    # load transient targets look up:
    trans_sites = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    trans_obs = []
    cols = trans_targs.columns.tolist()
    tdf = pd.DataFrame(columns=['obsprefix','grpid' ,'k', 'i', 'j'])
    new_cols = []
    for col in cols:
        grp = int(col.split('_')[0].split('.')[1])
        k = int(col.split('.')[-1])
        site = trans_sites.loc[(trans_sites['group number'] == grp) & (trans_sites['k'] == k)]
        obsprefix = f'transh_grpid:{grp}_k:{k}_i:{site.row.values[0]-1}_j:{site.col.values[0]-1}'
        i = site.row.values[0] - 1
        j = site.col.values[0] - 1
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
        id = idomain_arr[subset.k, subset.i, subset.j]
        if id == 1:
            trans_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
        else:
            print(f"Skipping {well} at ({subset.k}, {subset.i}, {subset.j}) as it is not in the active domain.")
    
    hd_obs = {f'{mnm}.ss_head.obs.output': ss_hd_list,
             f'{mnm}.trans_head.obs.output': trans_hd_list
             }

    obs_package = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs",
        filename=f"{mnm}.obs",
        continuous=hd_obs
    )
    
    cord_sys = CRS.from_epsg(crs)
    gwf.modelgrid.set_coord_info(
        xoff=ll_corner[0], yoff=ll_corner[1], angrot=angrot, crs=cord_sys
    )
    sim.set_all_data_external()
    sim.write_simulation()
    print(f"MF6 input files written in {mws}")
    return sim, gwf


def get_rcha_tseries_from_swb2_nc(gwf, perioddata =[], rch_dict = {},start_date_time = '1969-12-31',
                                  annual_only=True,ftag="net_infiltration"):
    '''Function pulls infiltration estimates from SWB to create RCH dict

    process_SWB_to_RCH_srm.py script (created by: S Jordan) estimates:
    - annual avg rch prior to 2000 using a infil vs precip linear regression
    - caluclates monthly averages from daily estimates from 2000-2023

    - Depending on "annual_only" flag, SWB estimates are processed to create rch_dict for MF6
        - if annual_only = True, all SPs annual
        - if annual_only = False, monthly SPs for 2000 - 2023
    - RCH values are capped at the 90th percentile of non-zero values to avoid extreme values
    - RCH values are converted from in/day to ft/day for MF6 input
    - Predictive period (2024-2050) RCH values are set to annual average
    - SWB Grid and MF grid do not line up!!
        - RCH values are resampled from SWB grid to the model grid using nearest neighbor

    '''
    import xarray
    from flopy.utils import Raster
    # ---- Load in the SWBs pre-processors
    from process_SWB_to_RCH import main as process_SWB

    pre_2000, post_2000_monthly = process_SWB()

    tdis_df = pd.DataFrame(perioddata, columns=['days', 'time_step', 'period_length'])
    tdis_df['cum_days'] = tdis_df['days'].cumsum()

    start_date = pd.Timestamp(start_date_time)+ pd.DateOffset(days=1)  # start from the next day
    end_date = pd.Timestamp(start_date_time)+ pd.to_timedelta(tdis_df['cum_days'].iloc[-1]-1,unit='d')  # end at the last day of the last period

    # ftag="net_infiltration"
    # make sure modelgrid is rotated and offset
    mg = gwf.modelgrid
    assert mg.angrot != 0, "modelgrid must be rotated"

    # ---- Set some control variables based on the input data
    # Calc Q 90 of non-zero recharge values
    all_vals = np.array([])
    for key in post_2000_monthly:
        all_vals = np.concatenate((all_vals, post_2000_monthly[key][0, :, :].flatten()))
    all_vals = all_vals[all_vals > 0]
    q_90 = np.quantile(all_vals, 0.70)

    # Maximum allowable RCH value in units of in/day (Using Q90 as threshold)
    rch_thresh = q_90
    for key in post_2000_monthly:
        post_2000_monthly[key] = np.where(post_2000_monthly[key] > rch_thresh, rch_thresh, post_2000_monthly[key])

    # ---- Format the pre_2000 data
    rch = []
    years = list(pre_2000.keys())

    # Create a SS period average
    avg_pre_2000 = np.zeros(pre_2000[1970].shape)
    for key in pre_2000.keys():
        rch_vals = np.where(np.isnan(pre_2000[key]), 0, pre_2000[key])

        # Ensure positive!
        rch_vals = np.where(rch_vals < 0, 0, rch_vals)

        # Add to the total for average calc
        avg_pre_2000 += rch_vals

        # Append for plotting
        rch.append(np.mean(rch_vals))

    # Calculate average
    avg_pre_2000 /= len(pre_2000.keys())

    # Reset keys to stress periods and then add SS
    pre_2000 = {i + 1: v for i, (k, v) in enumerate(sorted(pre_2000.items()))}
    pre_2000[0] = avg_pre_2000

    # ---- Format the monthly stress periods
    years_2 = list(post_2000_monthly.keys())
    post_2000_monthly = {i + 31: v for i, (k, v) in enumerate(sorted(post_2000_monthly.items()))}

    # Grab the data for the plot
    rch_2 = []
    for key in post_2000_monthly:
        rch_key = np.where(np.isnan(post_2000_monthly[key]), 0, post_2000_monthly[key])
        rch_2.append(np.mean(rch_key))


    # Get the SWB grid information
    f_swb_ctl = os.path.join('data', 'swb', f'swb_control_file_wahp.ctl')
    with open(f_swb_ctl, 'r') as file:
        for line in file:
            if line.startswith('GRID '):
                grid_info = line.split(' ')
                break
    dxdy = float(grid_info[-1].strip('\n'))
    sxo = float(grid_info[3])
    syo = float(grid_info[4])

    mg_swb = flopy.discretization.StructuredGrid(delr=np.array([dxdy]*int(grid_info[1])),
                                                delc=np.array([dxdy]*int(grid_info[2])),
                                                xoff=sxo, yoff=syo,
                                                crs='EPSG:2265')

    # Extract list of dates from ds.time within the specified range
    date_range = pd.date_range(start=start_date, end=end_date)
    # dates = [pd.Timestamp(date).date() for date in ds.time.values if date in date_range]
    years = np.unique(date_range.year)

    # process each timestep
    if annual_only:
        # If annual only, calculate annual average from monthly for 2000 - 2023
        tsa_dict = rch_dict.copy()

        # Assign SS rch
        rio = Raster.raster_from_array(pre_2000[0], mg_swb)
        arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
        tsa_dict[0] = arr / 12  # convert to ft/d

        pred_avg = []
        for i, year in enumerate(years):
            if year < 2000: # pre 2000 annual
                rio = Raster.raster_from_array(pre_2000[i+1], mg_swb)
                arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
                tsa_dict[i+1] = arr / 12 # convert to ft/d
            elif (year >= 2000) & (year < 2024): # 2000 - 2023 monthly --> estimate annual average
                idx =(np.array(list(post_2000_monthly.keys()))-30)/12
                idx = np.where((idx <= year-1999))[0][-12:]

                month_arr = [post_2000_monthly[id+31] for id in idx]
                month_arr = np.stack(month_arr)
                annual_arr = np.nanmean(month_arr[:,0,:,:], axis=0) # sum over all months and divide by avg day/year
                pred_avg.append(annual_arr)

                rio = Raster.raster_from_array(annual_arr, mg_swb)
                arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
                tsa_dict[i+1] = arr / 12  # convert to ft/d

            elif year >= 2024: # post 2023 use SS avg
                annual_avg = np.nanmean(np.stack(pred_avg), axis=0)
                rio = Raster.raster_from_array(pre_2000[0], mg_swb)
                arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
                tsa_dict[i+1] = arr / 12 # convert to ft/d

    else:
        # Annual and monthly
        # Annual from 1970 - 1999
        # Monthly from 2000 - 2023
        # Annual 2024 -  End of Predictive period
        tsa_dict = rch_dict.copy()
        pre_2000_sps = np.array(list(pre_2000.keys()))
        post_2000_sps = np.array(list(post_2000_monthly.keys()))
        pred_avg = []
        for sp in range(0, len(perioddata)):
            if sp in pre_2000_sps:
                rio = Raster.raster_from_array(pre_2000[sp], mg_swb)
                arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
                tsa_dict[sp] = arr / 12 # convert to ft/d
            elif sp in post_2000_sps:
                pred_avg.append(post_2000_monthly[sp])
                rio = Raster.raster_from_array(post_2000_monthly[sp], mg_swb)
                arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
                tsa_dict[sp] = arr / 12 # convert to ft/d
            else:
                # post 2023 use SS avg
                annual_avg = np.nanmean(np.stack(pred_avg), axis=0)
                rio = Raster.raster_from_array(annual_avg, mg_swb)
                arr = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")
                tsa_dict[sp] = arr / 12 # convert to ft/d

    return tsa_dict



def clean_mf6(org_mws,mnm):
    cws = org_mws + "_clean"
    if os.path.exists(cws):
        shutil.rmtree(cws)
    shutil.copytree(org_mws,cws)

    sim = flopy.mf6.MFSimulation.load(sim_ws=cws, exe_name="mf6",load_only=["dis"])
    nper = sim.tdis.nper.data
    m = sim.get_model(f"{mnm}")
    nrow,ncol = m.dis.nrow.data,m.dis.ncol.data
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
            arr_file = arr_file.replace(f"{mnm}.", "")
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
        otriv_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.otriv_stress")]
        drn_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.drn_stress")]
        ghb_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.ghb_stress")]
        ghb_wbv_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}_wbv.ghb_stress")]
        well_pkg_cob_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.cob_")]
        well_pkg_cow_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.cow_")]
        well_pkg_malt_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.malt_")]
        well_pkg_min_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.minn_")]
        well_pkg_car_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.car_")]
        
        # append to afiles:
        afiles.extend(riv_stress_files)
        afiles.extend(otriv_stress_files)
        afiles.extend(drn_stress_files)
        afiles.extend(ghb_stress_files)
        afiles.extend(ghb_wbv_stress_files)
        afiles.extend(well_pkg_cob_files)
        afiles.extend(well_pkg_cow_files)
        afiles.extend(well_pkg_malt_files)
        afiles.extend(well_pkg_min_files)
        afiles.extend(well_pkg_car_files)
        
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
            # case where stage is below model bottom:
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
            df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1 
            # case where rbot is below model bottom:
            df['diff'] = df['rbot'] - df['mbot']
            df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1 # check on this if it was we want
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0 
            df = df.drop(columns=['mbot','diff'])
            df.to_csv(riv_file, sep=' ', index=False, header=False)
        for otriv_file in otriv_stress_files:
            df = pd.read_csv(otriv_file, delim_whitespace=True,header=None)
            df.columns = ['ly','row','col','stage','cond','rbot']
            bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
            df['mbot'] = bot
            df['diff'] = df['stage'] - df['mbot']
            # case where stage is below model bottom:
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
            df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1
            # case where rbot is below model bottom:
            df['diff'] = df['rbot'] - df['mbot']
            df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
            df = df.drop(columns=['mbot','diff'])
            df.to_csv(otriv_file, sep=' ', index=False, header=False)
            
        for file in afiles:
            ogfile = file
            file = file.replace(f"{mnm}.", "") 
            os.rename(ogfile, file)     
        
        # adjust paths in  pkg controls:
        for pkg in pkg_lst:
            if 'rcha' in pkg:
                pkg = 'rcha' # remove numbering from recharge package name
            if pkg in ["oc","head_obs"]:
                continue
            if pkg == 'ghb_wbv':
                with open(os.path.join(model_ws, f"{mnm}_wbv.ghb"), 'r') as f:
                    # remove f'{mnm.}' where it appears:
                    lines = f.readlines()
                    nl = []
                    for l in lines:
                        if f"{mnm}." in l:
                            l = l.replace(f"{mnm}_wbv.", "wbv_")
                        nl.append(l)
                    # write the lines back to the file:
                    with open(os.path.join(model_ws, f"{mnm}_wbv.ghb"), 'w') as f:
                        for l in nl:
                            f.write(l)

            else:
                with open(os.path.join(model_ws, f"{mnm}.{pkg}"), 'r') as f:
                    # remove f'{mnm.}' where it appears:
                    lines = f.readlines()
                    nl = []
                    for l in lines:
                        if f"{mnm}." in l:
                            l = l.replace(f"{mnm}.", "")
                        nl.append(l)
                    #write the lines back to the file:
                    with open(os.path.join(model_ws, f"{mnm}.{pkg}"), 'w') as f:
                        for l in nl:
                            f.write(l)

                
    # fix annoying flopy external wrapped format:
    fix_wrapped_format(cws) 
    
    # bc_fnames = [os.path.join(d, f"{mnm}.riv"),os.path.join(d, f"{mnm}_edge.ghb"),
    #              os.path.join(d, f"{mnm}.drn")]
    # org_df_fnames = [os.path.join(d,"riv_000.dat"),os.path.join(d,"ghb_edge_000.dat"),
    #                  os.path.join(d, "drn_000.dat")]

    # for org_df_fname,bc_fname in zip(org_df_fnames,bc_fnames):
    #     org_df = pd.read_csv(org_df_fname,delim_whitespace=True)
    #     try:
    #         org_df.loc[:,"boundname"] = org_df.boundname.apply(lambda x: x.replace(" ","-"))
    #     except AttributeError:
    #         pass
    #     prefix = os.path.split(org_df_fname)[1].split("_000")[0]
    #     fnames = []
    #     for kper in range(nper):
    #         fname = "{0}_{1:03d}.dat".format(prefix,kper)
    #         org_df.to_csv(os.path.join(d,fname),index=False,sep=" ")
    #         fnames.append(fname)

    #     lines = open(bc_fname,'r').readlines()

    #     with open(bc_fname,'w') as f:
    #         for line in lines:
    #             if line.lower().startswith("begin period"):
    #                 break
    #             f.write(line)

    #         for kper,fname in enumerate(fnames):
    #             f.write("begin period {0}\n".format(kper+1))
    #             f.write("  open/close {0}\n".format(fname))
    #             f.write("end period {0}\n\n".format(kper+1))

    # skip = ["hds","cbc","csv","list","grb"]
    # mod_files = [f for f in os.listdir(mws) if f.startswith(f"{mnm}") and "_" not in f and f.split(".")[-1] not in skip]
    # print(mod_files)

    # for fname in mod_files:
    #     lines = open(os.path.join(d,fname),'r').readlines()

    #     with open(os.path.join(d,fname),'w') as f:
    #         for line in lines:
    #             if "open/close" in line.lower():
    #                 line = line.replace("./","").replace("'","")
    #             f.write(line)
    #             if fname.endswith(".oc") and "begin options" in line.lower():
    #                 f.write("BUDGETCSV FILEOUT budget.csv\n")
    #             #if fname.endswith(".npf") and "begin options" in line.lower():
    #             #    f.write("K33OVERK\n")

    pyemu.utils.run("mf6",cwd=cws)

    return cws


def write_zbud_nam_file(cbcPath, grbPath):
    with open(os.path.join('zbud.nam'), 'w') as f:
        f.write('BEGIN ZONEBUDGET\n')
        f.write(f'  BUD {cbcPath}\n')
        f.write('  ZON zones_byly.dat\n')
        f.write(f'  GRB {grbPath}\n')
        f.write('END ZONEBUDGET\n')


def write_zone_file(zones, ncpl):
    with open(os.path.join('zones_byly.dat'), 'w') as f:
        f.write('BEGIN DIMENSIONS\n')
        f.write(f'  NCELLS {ncpl}\n')
        f.write('END DIMENSIONS\n')
        f.write('\n')
        f.write('BEGIN GRIDDATA\n')
        f.write('  IZONE\n')
        f.write('  INTERNAL\n')
        np.savetxt(f, zones, fmt='%i')
        f.write('END GRIDDATA\n')


def run_zb_by_layer(w_d,modnm='wahp7ly'):
    prep_deps(w_d)
    curdir = os.getcwd()
    os.chdir(w_d)
    # plot forward run ZB results
    print(f'\n\n\nRun ZB and plot results for layered Model\n\n\n')
    sim = flopy.mf6.MFSimulation.load(load_only=['dis'])
    mf = sim.get_model(modnm)
    nrow = mf.dis.nrow.data
    ncol = mf.dis.ncol.data
    nlay = mf.dis.nlay.data

    lay_arr = np.zeros([nlay,nrow,ncol])
    for i in range(nlay):
        lay_arr[i] += i+1

    # sets up zone file, zb name file and runs ZB
    # Make zone file
    d = {'node': np.arange(1, nrow * ncol * nlay + 1, 1).tolist(),
         'zone': lay_arr.flatten().tolist()}
    zon_file = pd.DataFrame(data=d)

    # Make zone file
    ncpl = nrow * ncol * nlay
    write_zone_file(zon_file.zone.astype(int).values, ncpl)

    # Make ZB nam file
    org_d = os.getcwd()
    cbb = os.path.join(f'{modnm}.cbb')
    grb = os.path.join(f'{modnm}.dis.grb')
    write_zbud_nam_file(cbb, grb)

    pyemu.utils.run("zbud6")

    # Run ZB
    assert os.path.exists('zbud.csv'), 'zbud6 did not run'

    os.chdir(curdir)


def stress_period_df_gen(mws,strt_yr,annual_only=False):
    out_tbl_dir = os.path.join('tables')
    if not os.path.exists(out_tbl_dir):
        os.makedirs(out_tbl_dir)
    sim = flopy.mf6.MFSimulation.load(sim_ws=mws, exe_name='mf6',load_only=['dis'])
    start_date = sim.tdis.start_date_time.data
    period_data = sim.tdis.perioddata.array
    nper = sim.tdis.nper.data    
    
    if annual_only == True:
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
        spd.loc[0, 'cum_days'] = 1

        start_datetime = []
        end_datetime = []
        
        # get years by taking start date and calucating based on cum_days:
        years = []
        spd['cum_days'] = spd['cum_days'].astype(int) 
        origin = pd.to_datetime(start_date)
        for i in range(nper):
            if i == 0:
                st_dt = origin - pd.Timedelta(days=1) + pd.to_timedelta(spd.loc[i, 'cum_days'], unit='D')
                years.append(origin.year)
            else:
                # get the start date for each period
                st_dt = origin + pd.to_timedelta(spd.loc[i, 'cum_days'], unit='D')
                years.append(st_dt.year)
        spd['year'] = years
        
        # start of each stress period in datetime format:
        for i in range(nper):
            if i == 0:
                start_datetime.append(pd.to_datetime(start_date))
                end_datetime.append(pd.to_datetime(start_date) + pd.DateOffset(days=1))
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


def plot_flow_xsec(workspace):
    from flopy.utils import CellBudgetFile, postprocessing as pp
    workspace=os.path.join("model_ws","wahp_clean")
    sim = flopy.mf6.MFSimulation.load(sim_ws=workspace,load_only=["dis","oc","riv","rch"])
    model = sim.get_model()
    
    cbc = CellBudgetFile(os.path.join(workspace,model.oc.budget_filerecord.array[0][0]), precision="double")
    kstpkper = np.array(cbc.get_kstpkper())
    spdis = cbc.get_data(text="SPDIS", kstpkper=(0, 0))[0]
    qx, qy, qz = pp.get_specific_discharge(spdis, model)
    xcent, ycent, zcent = model.modelgrid.xyzcellcenters
    arrow_kw = {}
    row=145
    fig, ax = plt.subplots(figsize=(15,8), dpi=500)
    xs = flopy.plot.PlotCrossSection(model=model, line={"row": row-1},
                                geographic_coords=True, ax=ax)  
    
    xs.plot_bc(name="riv",color="cyan")
    vkw = {"kstep": 1, "hstep": 1}
    vkw.update(arrow_kw or {})
    arrow_plt=xs.plot_vector(qx, qy, qz, ax=ax,**vkw,scale=2,color="b",width=0.002)
    xs.plot_grid(ax=ax, linewidth=0.1)
    xs.plot_inactive(color_noflow='grey')
    
    ax.set_title(f"Xsec: Row {row}, high Kv+low RCH", size=12)
    ax.set_xlabel("Lateral Coordinate (ft)")
    ax.set_ylabel("Elevation (ft)")
    plt.tight_layout()


    #checking for face flux
    flow_riv=cbc.get_data(full3D=True,kstpkper=(0, 0),text='RIV')
    flow_riv[0][0][flow_riv[0][0]==0]=np.nan
    plt.imshow(flow_riv[0][0],vmin=-30000,vmax=30000,cmap="seismic"); plt.colorbar(); plt.show()
    
    flowja = cbc.get_data(idx=2,full3D=True,kstpkper=(0, 0))[0]
    MF6_face_flows=flopy.mf6.utils.get_structured_faceflows(flowja, grb_file=os.path.join(workspace,f".dis.grb"))
    mf6_vflow_array=MF6_face_flows[2]
    mf6_vflow_array[mf6_vflow_array==0]=np.nan
    
    #modifying the RCH pack
    rch_arr=model.rch.recharge.get_data()
    riv_cell=model.riv.stress_period_data.array[0]['cellid']
    riv_cell=[[x[1],x[2]] for x in riv_cell]
    riv_cell=pd.DataFrame(riv_cell, columns=['i','j'])
    rch_arr[0][riv_cell.i.values,riv_cell.j.values]=0
    model.rch.recharge.set_data(rch_arr)
    
    model.rch.write()

# -----------------------------------------------------------
# Main routine.
# -----------------------------------------------------------
def main():

    # be careful with globals!
    global crs, angrot, cell_size, ll_corner, nrow, ncol, mnm, lydict, end_of_hm, pred_period_len, div_ly1
    crs = 2265  # State plane (feet)
    angrot = 40  # Rotation angle (degrees)
    cell_size = 660  # Cell size in feet
    mnm = "wahp7ly" # model name, model name must be a simple string with no spaces, underscores, or special characters
    nlay = 7 # number of layers in the model
    reinterp_wbv_bot = True
    if nlay == 6:
        lydict = {0: "soils", 1: "sand", 2: "sand", 3: "sand", 4: "clay", 5: "sand"} # hardcoded unit type for each layer used to initilize the npf package
    elif nlay == 4:
        lydict = {0: "clay", 1: "sand", 2: "clay", 3: "sand"} # hardcoded unit type for each layer used to initilize the npf package
    elif nlay == 7:
        lydict = {0: "soils", 1: "sand", 2: "sand", 3: "clay", 4: "sand", 5: "clay", 6: "sand"}
    div_ly1 = [0.1,0.3,0.5,0.1] # divide the top layer into 3 equal parts, 0.3, 0.6, and 1.0, if you do not want layer 1 divided set to empty array []. 
    
    strt_yr = 1970 # start of tranisent period
    end_of_hm = 2024 # end of history matching
    annual_only = True # if true, only annual stress periods, otherwise monthly stress periods are used.
    pred_period_len = 20 # length of prediction period
    
    # Create a rotated & trimmed grid using create_grid_in_feet.
    shp_path = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "updated_wahp_model_extent.shp"
    )
    outpth = os.path.join("..", "..", "gis", "output_shps", "wahp")
    os.makedirs(outpth, exist_ok=True)
    output_shp = os.path.join(
        outpth, f"{mnm}_full_extent_grd_sz_{cell_size}_ft_rot{angrot}_trimmed.shp"
    )
    trimmed_gdf, ll_corner, rotation_used, nrow, ncol = create_grid_in_feet(
        shapefile_path=shp_path,
        grid_size_feet=cell_size,
        output_shapefile_path=output_shp,
        target_crs_epsg=crs,
        rotation_degrees=angrot,
        max_iterations=20,
        expansion_mode="partial",
        columns_to_add=10,
    )
    #print(f"Trimmed grid shapefile saved to: {output_shp}")
    print(f"Grid dimensions: {nrow} rows, {ncol} columns")
    print(f"Lower-left corner (from extract_grid_attributes): {ll_corner}")
  
    # Build RH version of the grid shapefile, with upper right corner set to 0,0
    grd, output_shp = build_grid_shp(nrow,ncol,cell_size,ll_corner,angrot,crs,mnm)
    grd = grd[['node', 'row', 'col', 'i', 'j', 'geometry']]
    # Build model layers by sampling the DEM and thickness rasters at grid cell centroids.
    top, botm, idomain = elevs_to_grid_layers(output_shp,div_ly1)
    
    idomain[-1,:,:] = 0 # inactive WR
    
    # make sure idomain is array of ints
    idomain = idomain.astype(int)
    print("Sampled model surfaces computed.")

    riv_definition()
    drain_definition()
    
    # add idom attributes to grid shapefile:
    for k in range(len(idomain)):
        print(k)
        col = f'idom_{k}'
        grd[col] = idomain[k,grd['i'].values,grd['j'].values]
    # add botms to shapefile:
  
    grd['top'] = top[grd['i'].values,grd['j'].values]
    for k in range(len(botm)):
        grd[f'botm_{k}'] = botm[k,grd['i'].values,grd['j'].values]
    # add thickness to grid shapefile:
    for k in range(len(botm)):
        col = f'thk_{k}'
        if k == 0:
            grd[col] = top[grd['i'].values,grd['j'].values] - botm[k,grd['i'].values,grd['j'].values]
        else:
            grd[col] = botm[k-1,grd['i'].values,grd['j'].values] - botm[k,grd['i'].values,grd['j'].values]
    
    grd.to_file(output_shp)    
    
    if reinterp_wbv_bot:
        out = reinterp_wbv_bot_in_southern_area(grd, cell_size)
        out_botm = out['botm_4'].values.reshape(nrow,ncol)
        botm[4,:,:] = out_botm
    
    # update grd:    
    grd['top'] = top[grd['i'].values,grd['j'].values]
    for k in range(len(botm)):
        grd[f'botm_{k}'] = botm[k,grd['i'].values,grd['j'].values]
    # add thickness to grid shapefile:
    for k in range(len(botm)):
        col = f'thk_{k}'
        if k == 0:
            grd[col] = top[grd['i'].values,grd['j'].values] - botm[k,grd['i'].values,grd['j'].values]
        else:
            grd[col] = botm[k-1,grd['i'].values,grd['j'].values] - botm[k,grd['i'].values,grd['j'].values]
    
    grd.to_file(output_shp) 

        
    # Set up the MODFLOW 6 model in workspace.
    mws = os.path.join("model_ws", mnm)
    os.makedirs(mws, exist_ok=True)
    
    sim, gwf = setup_mf6(mws, crs, top, botm, idomain, ll_corner, 
                         nrow, ncol, cell_size, angrot, 
                         strt_yr, end_of_hm, 
                         pred_period_len,annual_only=annual_only,
                         grd=grd,mnm=mnm
                         )
    
    # sim = flopy.mf6.MFSimulation.load(sim_ws=mws, exe_name='mf6', )
    # m = sim.get_model(mnm)
    # botm = m.dis.botm.array
    # fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    # ax.imshow(botm[4,:,:], cmap='terrain', vmin=botm.min(), vmax=top.max())
    
    cws = clean_mf6(mws,mnm)
    

    # setup zone budget by layer:
    run_zb_by_layer(w_d=cws, modnm=mnm)


    # write aux model info: 
    stress_period_df_gen(mws+'_clean',strt_yr,annual_only=annual_only)
    
    print(f"{mnm} model setup complete.")


if __name__ == "__main__":
    print('building model...')
    main()
