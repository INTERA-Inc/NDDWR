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
from shapely.geometry import box, Point
from shapely import affinity
from shapely.ops import unary_union
from shapely.geometry import box
from pyproj import CRS
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import wahp01_water_level_process as wlprocess

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('data','pumping')))
import make_use_by_well as wuprocess


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
    grd = gpd.read_file(os.path.join("..", "..", "gis",'output_shps','wahp',"cell_size_660ft_epsg2265.grid.shp"))
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
    
    
def build_grid_shp(nrow,ncol,cell_size,ll_corner,angrot,crs):
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
        f"cell_size_{cell_size}ft_epsg{crs}.grid.shp",
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

def _lower_left_from_geoms(geoms, atol=1e-6):
    # robust "lower-left": min y, then min x among those at ~min y
    pts = []
    for g in geoms:
        pts.extend(list(g.exterior.coords))
    ys = [p[1] for p in pts]
    miny = min(ys)
    candidates = [p for p in pts if abs(p[1] - miny) <= atol]
    minx = min(p[0] for p in candidates)
    return float(minx), float(miny)

# ----------------------------------------------------------------
# Grid-building functions.
# ----------------------------------------------------------------
def build_unrotated_grid_gdf(minx, miny, maxx, maxy, cell_size, crs):
    """
    Build an axis-aligned grid covering the bounds.
    MODFLOW-style indexing: i=row (0 at TOP), j=column (0 at LEFT).
    """
    num_cols = math.ceil((maxx - minx) / cell_size)  # j
    num_rows = math.ceil((maxy - miny) / cell_size)  # i

    rows = []
    for i in range(num_rows):          # i = 0 at TOP => march downward
        y_top = maxy - i * cell_size
        y_bottom = y_top - cell_size
        for j in range(num_cols):      # j = 0 at LEFT => march rightward
            x_left = minx + j * cell_size
            x_right = x_left + cell_size
            poly = box(x_left, y_bottom, x_right, y_top)
            rows.append({"i": i, "j": j, "geometry": poly})

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf["grid_id"] = gdf.index + 1
    return gdf


def trim_grid_rows_columns_with_buffer(grid_gdf, shapefile_gdf, pad_rows=0, pad_cols=0):
    """
    Keep only cells intersecting the AOI, plus a buffer of rows/cols.
    Assumes MODFLOW indexing (i=row, j=col) already.
    Returns i/j renormalized to start at 0 (top-left), plus row/col (1-based).
    """
    shape_union = shapefile_gdf.union_all()
    hits = grid_gdf[grid_gdf.geometry.intersects(shape_union)]
    if hits.empty:
        raise ValueError("No grid cells intersect the AOI; check CRS/inputs.")

    i_min, i_max = int(hits["i"].min()), int(hits["i"].max())
    j_min, j_max = int(hits["j"].min()), int(hits["j"].max())

    Imin = max(int(grid_gdf["i"].min()), i_min - pad_rows)
    Imax = min(int(grid_gdf["i"].max()), i_max + pad_rows)
    Jmin = max(int(grid_gdf["j"].min()), j_min - pad_cols)
    Jmax = min(int(grid_gdf["j"].max()), j_max + pad_cols)

    trimmed = grid_gdf[
        grid_gdf["i"].between(Imin, Imax) & grid_gdf["j"].between(Jmin, Jmax)
    ].copy()

    # Renormalize to 0-based top-left; add MODFLOW 1-based helpers
    trimmed["i"] = trimmed["i"] - Imin
    trimmed["j"] = trimmed["j"] - Jmin
    trimmed["row"] = trimmed["i"] + 1
    trimmed["col"] = trimmed["j"] + 1
    trimmed = trimmed.sort_values(["i", "j"]).reset_index(drop=True)
    return trimmed

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
    max_iterations=50,
    expansion_mode="all",     # "all" is robust; "partial" grows only where needed
    columns_to_add=10,
    pad_rows=0,               # buffer in rows (i)
    pad_cols=0                # buffer in cols (j)
):
    """
    Build a rotated MODFLOW-style grid that fully covers the AOI,
    ensuring room for +pad_rows/+pad_cols, then trim and write.
    Returns (trimmed_gdf, ll_corner_xy, rotation_degrees, nrow, ncol).
    """
    # AOI in target CRS
    input_gdf = gpd.read_file(shapefile_path)
    shapefile_gdf = input_gdf.to_crs(epsg=target_crs_epsg)
    shape_union = shapefile_gdf.union_all()
    minx, miny, maxx, maxy = shape_union.bounds

    # pivot for rotation
    centroid = shape_union.centroid
    pivot_xy = (centroid.x, centroid.y)

    # start from AOI bounds
    grid_minx, grid_miny, grid_maxx, grid_maxy = minx, miny, maxx, maxy

    # coverage target includes distance pad to guarantee i/j padding later
    req_pad_dist = max(pad_rows, pad_cols) * grid_size_feet
    target_geom = shape_union.buffer(req_pad_dist) if req_pad_dist > 0 else shape_union

    coverage_ok = False
    for _ in range(1, max_iterations + 1):
        # build grid with MODFLOW indexing
        grid_gdf = build_unrotated_grid_gdf(
            grid_minx, grid_miny, grid_maxx, grid_maxy,
            grid_size_feet, crs=f"epsg:{target_crs_epsg}"
        )

        # rotate geometries around AOI centroid
        if rotation_degrees != 0.0:
            grid_gdf["geometry"] = grid_gdf["geometry"].apply(
                lambda geom: affinity.rotate(geom, rotation_degrees, origin=pivot_xy)
            )

        # coverage test vs padded AOI
        union_grid = unary_union(grid_gdf.geometry)
        uncovered = target_geom.difference(union_grid)
        if uncovered.is_empty:
            coverage_ok = True
            break

        # grow base rectangle in world coords
        expand_amount = columns_to_add * grid_size_feet
        if expansion_mode == "all":
            grid_minx -= expand_amount
            grid_maxx += expand_amount
            grid_miny -= expand_amount
            grid_maxy += expand_amount
        else:
            ug_minx, ug_miny, ug_maxx, ug_maxy = union_grid.bounds
            dmX, dmY, dMX, dMY = uncovered.bounds
            if dmX < ug_minx: grid_minx -= expand_amount
            if dMX > ug_maxx: grid_maxx += expand_amount
            if dmY < ug_miny: grid_miny -= expand_amount
            if dMY > ug_maxy: grid_maxy += expand_amount

    if not coverage_ok:
        print(f"Reached max_iterations={max_iterations}, coverage may be incomplete.")

    # trim to AOI with index-space buffer (keeps i=row top, j=col left)
    trimmed_gdf = trim_grid_rows_columns_with_buffer(
        grid_gdf, shapefile_gdf, pad_rows=pad_rows, pad_cols=pad_cols
    )
    
    # add 1-based MODFLOW-style fields for the shapefile
    trimmed_gdf["row"] = (trimmed_gdf["i"] + 1).astype("int32")  # 1..nrow, top to bottom
    trimmed_gdf["col"] = (trimmed_gdf["j"] + 1).astype("int32")  # 1..ncol, left to right

    # write shapefile
    trimmed_gdf.to_file(output_shapefile_path)
    print(f"Trimmed grid shapefile saved to: {output_shapefile_path}")

    # report dims & lower-left corner (world coords)
    nrow = int(trimmed_gdf["i"].nunique())
    ncol = int(trimmed_gdf["j"].nunique())
    ll_corner = _lower_left_from_geoms(trimmed_gdf.geometry)

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

    # Sample DEM and convert to feet.
    dem_vals = sample_raster(dem_path, grid) * 3.28084 # to feet
    
    top = dem_vals.reshape([nrow,ncol])
    # Fill any NaN values in the DEM with the median of their 8-neighbors.
    top = fill_nan_with_neighbors_median(top)

    # Sample thickness rasters.
    wss_vals = sample_raster(wss_path, grid)
    wbv_vals = sample_raster(wbv_path, grid)
    dc_vals = sample_raster(dc_path, grid)
    wr_vals = sample_raster(wr_path, grid)

    # Create boolean arrays from the original sampled thickness values.
    forced_wss_bool = wss_vals < 5
    forced_wbv_bool = wbv_vals < 5
    forced_dc_bool = dc_vals < 5
    forced_wr_bool = wr_vals < 5

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
    
    # if break layer 1 into sub layers:
    if len(div_ly1) > 0:
        bot_soil = top - (wss_arr * div_ly1[0])
        bot_sub_ly2 = top - (wss_arr * div_ly1[1])
        botm = np.stack(
            [bot_soil, bot_sub_ly2, bottom_layer1, bottom_layer2, bottom_layer3,
             bottom_layer4], axis=0
        )
    else:
        botm = np.stack(
            [bottom_layer1, bottom_layer2, bottom_layer3, bottom_layer4], axis=0
        )
    
    # thickness calcs on layer 1 sub layers:
    if len(div_ly1) > 0:
        ly1_1 = top - bot_soil
        ly1_2 = bot_soil - bot_sub_ly2
        ly1_3 = bot_sub_ly2 - bottom_layer1
        ly1_1_arr = np.where(ly1_1 < 5, 5, ly1_1)
        ly1_2_arr = np.where(ly1_2 < 5, 5, ly1_2)
        ly1_3_arr = np.where(ly1_3 < 5, 5, ly1_3)
        forced_ly1_1_bool = ly1_1_arr < 5
        forced_ly1_2_bool = ly1_2_arr < 5
        forced_ly1_3_bool = ly1_3_arr < 5
    
    
    # Build idomain arrays using the original forced booleans.
    if len(div_ly1) > 0:
        idom_layer1 = np.where(forced_ly1_1_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer2 = np.where(forced_ly1_2_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer3 = np.where(forced_ly1_3_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer4 = np.where(forced_wbv_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer5 = np.where(forced_dc_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer6 = np.where(forced_wr_bool.reshape((nrow, ncol)), 0, 1)
        idomain = np.stack([idom_layer1, idom_layer2, idom_layer3, idom_layer4,
                            idom_layer5, idom_layer6], axis=0)
        
    else:
        idom_layer1 = np.where(forced_wss_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer2 = np.where(forced_wbv_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer3 = np.where(forced_dc_bool.reshape((nrow, ncol)), -1, 1)
        idom_layer4 = np.where(forced_wr_bool.reshape((nrow, ncol)), 0, 1)
        idomain = np.stack([idom_layer1, idom_layer2, idom_layer3, idom_layer4], axis=0)
    
    return top, botm, idomain

# -----------------------------------------------------------
# River and Drain defintions:
# -----------------------------------------------------------
def riv_definition(qa_figs=False):
    print('creating first pass at riv definition...')

    rivoutdir = os.path.join('..','..','gis','output_shps','wahp','rivs')
    if not os.path.exists(rivoutdir):
        os.makedirs(rivoutdir)
    
    grd = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','cell_size_660ft_epsg2265.grid.shp'))

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

        fig, ax = plot_basemap(epsg=2265)
        ax.set_title("River's Initial Conductivity Estimate")
        riv_df.plot(column='cond', cmap='viridis', linewidth=0.5, ax=ax, edgecolor='none')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=riv_df['cond'].min(), vmax=riv_df['cond'].max()))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7) 
        cbar.set_label('Conductance\n(ft/d)')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        riv.plot(ax=ax, color='lightblue', linewidth=0.25, label='Major Rivers')
        ax.legend()
        
        plt.savefig(os.path.join(out_figs, 'riv_cond_estimate.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_figs, 'riv_cond_estimate.pdf'), format='pdf', dpi=300)

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
    
    grd = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','cell_size_660ft_epsg2265.grid.shp'))

    out_figs = os.path.join('prelim_figs_tables','figs','drn')
    if not os.path.exists(out_figs):
        os.makedirs(out_figs)

    
    # load twdb major riv shapefile:
    drn = gpd.read_file(os.path.join('..','..','gis','input_shps','wahp','flow_lines_all_clipped.shp'))
    drn.columns = drn.columns.str.lower()
    drn = drn.to_crs(grd.crs)

    raster_path = os.path.join('..','..','gis','output_ras','dem_no_buildings.tif') # this in meters ah
    
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
        
    # for each row col drop duplicates keep lowest elevation:
    drn_df = drn_df.sort_values('elevation').drop_duplicates(subset=['row','col'], keep='first')
    drn_df = gpd.GeoDataFrame(drn_df, geometry='geometry', crs=grd.crs)    

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

    drn_df['elevation'] = drn_df['elevation'] * 3.28084 # convert to feet
    
    # assume widths of 50 feet:
    drn_df['width'] = 50.0
    # assume river depths of 5 feet:
    drn_df['rbot'] = drn_df['elevation'] - 5.0
    # assume a foot of stage on top of elevation:
    drn_df['stage'] = drn_df['elevation'] + 1.0 
    drn_df['cond'] = drn_df['segment_length'] * drn_df['width'] * 1.0 / (drn_df['stage'] - drn_df['rbot']) # assuming drn bed K = 1.0 ft/d
    
    # drop riv cells with less than 125 feet of segment length:
    drn_df = drn_df[drn_df['segment_length'] > 200]
    
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
    sim = flopy.mf6.MFSimulation(sim_name=model_name, exe_name=exe_name, sim_ws=sim_ws)
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
              strt_yr, end_of_hm, pred_period_len,annual_only,grd):
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
        for i in range(2000, 2025):
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

        # Annual stress periods for the prediction period (2025 through 2044)
        for i in range(2025, 2025 + pred_period_len):
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
    idomain_list = idomain_arr.tolist()

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
    k0_snd = 245.0
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
        filename='wahp.riv')
    
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
            filename='wahp.otriv')
            
    
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
        filename='wahp.drn')
    
    # init recharge package:
    # create a dummy recharge package to get the model to run:
    # this will be replaced with a real recharge package later.
    # get drain and river locs and set recharge to 0 in those cells:
    riv_cells = riv_df[['i','j']].values.tolist()
    drn_cells = drn_df[['i','j']].values.tolist()
    
    rch_dict = {}
    for i in range(nper):
        rch_temp = np.ones((nrow, ncol), dtype=float)*2/12/365 # 2 inches per year   
        # set riv and drn cells to 0:
        for r, c in riv_cells:
            rch_temp[r, c] = 0.0
        for r, c in drn_cells:
            rch_temp[r, c] = 0.0
        rch_dict[i] = rch_temp
         
    rch = flopy.mf6.ModflowGwfrcha(gwf,
                                   recharge=rch_dict,
                                   save_flows=True,
                                   pname="rch",
                                   filename="wahp.rch",)     
    # ---------------- #
    # init ghb package:
    # ---------------- #
    # as a start putting a ghb in every model edge cell and setting stage to the top of the model
    ghb = top.copy() 
    zero_out_inside = top.copy() * 0.0
    # set edge cells to 1, so 0 and -1 col/row
    zero_out_inside[0,:] = 1
    zero_out_inside[-1,:] = 1
    zero_out_inside[:,0] = 1
    zero_out_inside[:,-1] = 1
    ghb = ghb * zero_out_inside # zero out inside cells
    
    ghb_all = pd.DataFrame()
    for ly in range(0, nlay):
        # 2D array of ghb values, to dataframe with k,i,j,stage,cond:
        # reshape ghb:
        ghb_2d = ghb.reshape((nrow*ncol))
        ghb_df = pd.DataFrame({
            'k': ly,
            'i': np.repeat(np.arange(nrow), ncol),
            'j': np.tile(np.arange(ncol), nrow),
            'stage': ghb_2d,
            'cond': (ghb_2d * 0.0) + 10000  # set conductance to 10,000ft^2/d
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
        filename='wahp.ghb')
    
    
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
    pmp = pmp.merge(pmp_shp, left_on='Well', right_on='site_locat', how='left')
    os.chdir(curdir)
    
    spd = stress_period_df_gen(mws,strt_yr+1,annual_only=annual_only) # plus one is cause global var got changed above...sloopy on my part
    
    pmp.columns = pmp.columns.str.lower()   
    pmp['permit_holder_name'] = pmp['permit_holder_name'].str.lower()
    
    if annual_only == True:
        pmp = pmp.merge(spd[['stress_period','year']], on='year', how='left')
    else:
        spd['month'] = spd['start_datetime'].dt.month
        pmp = pmp.merge(spd[['stress_period','year','month']], on=['year','month'], how='left')
    
    # to geodataframe:
    pmp = gpd.GeoDataFrame(pmp, geometry=gpd.points_from_xy(pmp['x_2265'], pmp['y_2265']), crs=grd.crs)
    if annual_only == True:
        grid = grd[['node', 'row', 'col', 'i', 'j', 'geometry', 'idom_0', 'idom_1',
                    'idom_2', 'idom_3', 'idom_4', 'idom_5']]
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
    
    # !!!!! DROPPING 5221 FOR NOW !!!!!
    pmp = pmp.loc[pmp['well'] != 'permit_5721']
    
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

    # get layer for each well:
    pmp['k'] = pmp.apply(lambda w: find_layer(w, botm_arr, top_arr)[0], axis=1)
    pmp['ly_cake'] = pmp.apply(lambda w: find_layer(w, botm_arr, top_arr)[1], axis=1)
    pmp['model_top_elev'] = pmp['ly_cake'].apply(lambda x: x[0])
    
    aq2lay = {'Wahpeton Buried Valley':3.0, 'Wahpeton Shallow Sand':2.0}
    # map aquifer to model layer:
    pmp['mod_ly'] = pmp['aquifer'].map(aq2lay)
    # if k is not equal to mod_ly, then set k to mod_ly:
    pmp['k'] = np.where(pmp['k'] != pmp['mod_ly'], pmp['mod_ly'], pmp['k'])
    
    well_tags = {'cargill incorporated':'car',
                 'froedtert malt corp':'malt',
                 'wahpeton, city of':'cow',
                 'minn-dak farmers cooperative':'minn',
                 'city of breckenridge':'cob'}
    
    # plot pumping timesseries and save shp files of pumping:
    out_figs = os.path.join('prelim_figs_tables','figs','pumping')
    if not os.path.exists(out_figs):
        os.makedirs(out_figs)
    pdf_path = os.path.join(out_figs, 'pumping_plots.pdf')
    
    for key, val in well_tags.items():
        permit = pmp.loc[pmp['permit_holder_name'] == key]   
        permit = permit.sort_values(['stress_period'])
        if annual_only == True:
            permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + '0101', format='%Y%m%d')
        else:
            permit['datetime'] = pd.to_datetime(permit['year'].astype(str) + permit['month'].astype(str).str.zfill(2) + '01', format='%Y%m%d')
        unq_wells = permit['well'].unique()
        
        pdf_path = os.path.join(out_figs, 'pumping_plots.pdf')
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
                        ly_cake_str = ', '.join([f"{x:.1f}" for x in well_data.ly_cake.values[0]])
                        ax.set_title((
                            f'{key.title()}\n'
                            f'Well ID: {well}\n'
                            f'Aq. Assignment: {well_data.aquifer.values[0]}\n'
                            f'Model Layer: {well_data.k.values[0] + 1}\n'
                            f'Midpoint Elevation: {well_data.model_top_elev.values[0] - well_data.scr_midpt.values[0]:.2f} ft\n'
                            f'Top of Screen: {well_data.top_screen.values[0]:.2f} ft\n'
                            f'Bottom of Screen: {well_data.bottom_scr.values[0]:.2f} ft\n'
                            f'Layer Elevs: [{ly_cake_str}]'),
                            fontsize=10)

                        ax.set_ylabel('Pumping Rate (cfd)', fontsize=8)
                        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
                        ax.tick_params(axis='y', labelsize=7)
                        
                        start_date = spd.loc[0, 'start_datetime']
                        end_date = spd.loc[len(spd) - 1, 'end_datetime']
                        ax.set_xlim(start_date, end_date)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
                        
                    # Turn off any unused subplots
                    for j in range(num_wells - i, plots_per_page):
                        fig.delaxes(axes[j])

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        print(f"PDF of plots saved to {pdf_path}")
        
    # build well package for each permit holder:
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
        
        well = flopy.mf6.ModflowGwfwel(gwf,
                                    stress_period_data=well_dict,
                                    pname=val,
                                    save_flows=True,
                                    auto_flow_reduce=0.1,
                                    maxbound=mxbnd,
                                    filename=f'wahp.{val}')
    
    # obs package set up:
    ss_targs, tarns_targs = wlprocess.main(False,div_ly1,annual_flag=annual_only)
    wells = ss_targs.obsprefix.unique()
    ss_hd_list = []
    for well in wells:
        subset = ss_targs[ss_targs.obsprefix==well].iloc[0,:]
        if subset.k >=4:
            # only add if layer is greater than 3
            continue
        ss_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
    
    
    trans_hd_list = []    
    wells = tarns_targs.obsprefix.unique()
    for well in wells:
        subset = tarns_targs[tarns_targs.obsprefix==well].iloc[0,:]
        if subset.k >=4:
            # only add if layer is greater than 3
            continue
        trans_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
    
    hd_obs = {"wahp.ss_head.obs.output": ss_hd_list,
             "wahp.trans_head.obs.output": trans_hd_list
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
    MF6_face_flows=flopy.mf6.utils.get_structured_faceflows(flowja, grb_file=os.path.join(workspace,"wahp.dis.grb"))
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
    global crs, angrot, cell_size, ll_corner, nrow, ncol, mnm, lydict, strt_yr, end_of_hm, pred_period_len, div_ly1
    crs = 2265  # State plane (feet)
    angrot = 40  # Rotation angle (degrees)
    cell_size = 660  # Cell size in feet
    mnm = "wahp" # model name, model name must be a simple string with no spaces, underscores, or special characters
    nlay = 6 # number of layers in the model
    if nlay == 6:
        lydict = {0: "soils", 1: "sand", 2: "sand", 3: "sand", 4: "clay", 5: "sand"} # hardcoded unit type for each layer used to initilize the npf package
    elif nlay == 4:
        lydict = {0: "clay", 1: "sand", 2: "clay", 3: "sand"} # hardcoded unit type for each layer used to initilize the npf package
    div_ly1 = [0.1,0.3,0.6] # divide the top layer into 3 equal parts, 0.3, 0.6, and 1.0, if you do not want layer 1 divided set to empty array []. 
    
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
        outpth, f"wahp_full_extent_grd_sz_{cell_size}_ft_rot{angrot}_trimmed.shp"
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
    grd, output_shp = build_grid_shp(nrow,ncol,cell_size,ll_corner,angrot,crs)
    grd = grd[['node', 'row', 'col', 'i', 'j', 'geometry']]
    # Build model layers by sampling the DEM and thickness rasters at grid cell centroids.
    top, botm, idomain = elevs_to_grid_layers(output_shp,div_ly1)
    print("Sampled model surfaces computed.")

    riv_definition()
    drain_definition()
    
    # add idom attributes to grid shapefile:
    for k in range(len(idomain)):
        col = f'idom_{k}'
        grd[col] = idomain[k,grd['i'].values,grd['j'].values]
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
                         grd=grd
                         )
    clean_mf6(mws,mnm)
    
    # write aux model info: 
    stress_period_df_gen(mws+'_clean',strt_yr,annual_only=annual_only)
    
    print(f"{mnm} model setup complete.")


if __name__ == "__main__":
    main()
