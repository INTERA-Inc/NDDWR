# -*- coding: utf-8 -*-
"""
Elk Valley water-level targets → MF6 observation package

Inputs (relative to the script working directory):
    data/raw/obs_data/elk_valley_sites.csv
    data/raw/obs_data/elk_valley_water_level_data.csv

What this does
--------------
1) Loads site metadata + water-level measurements.
2) Filters to sites inside the *active* model domain (idomain > 0).
3) Assigns each site to a model layer using screen midpoint rules:
     - screen_mid elev = midpoint of [top_screen_elev, bottom_screen_elev]
       where each is computed from model top: top_elev - depth_to_screen
     - if bottom_screen == 0, use total_depth (if >0) for bottom_screen
     - if bottom_screen==0 and total_depth==0 → force to layer 2
4) Steady-state targets (SP 0): uses sites.wl_avg_1965.
5) Transient targets (SP 1..): smooths WLs:
     - pre-2000: yearly median
     - 2000+: 60-day rolling mean (centered)
   Then aligns to model stress-period start dates after SP 0.
6) Creates an MF6 UTL-OBS package with HEAD observations (optional).
7) Writes CSVs (and optional shapefile) so you can inspect targets.

Usage
-----
from water_level_process_elk import build_obs_from_elk

ss_df, tr_df, obs_pkg = build_obs_from_elk(
    gwf,                        # flopy.mf6.ModflowGwf
    sites_csv="data/raw/obs_data/elk_valley_sites.csv",
    wl_xlsx="data/raw/obs_data/elk_valley_water_level_data.csv",
    out_dir="data/analyzed/elk_obs",
    make_shapefile=True,
    add_obs_package=True,
    obs_pname="obs",
    obs_filename=None           # default = f"{gwf.name}.obs"
)

Notes
-----
- CRS assumed to be EPSG:2265 for x_2265 / y_2265 columns in sites CSV.
- This builds HEAD observation *definitions*. MF6 produces modelled
  values in output; observed values are written to CSVs for your QA/PEST.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import pyemu
import flopy
import re
import math
import calendar
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from pyproj import CRS
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import defaultdict

# ----------------------------
# ---- helpers / small utils
# ----------------------------
def _ensure_outdir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def _parse_mf6_start_date(tdis) -> pd.Timestamp:
    """
    Try to parse MF6 TDIS start_date_time into a Timestamp.
    Handles a few formats (plain date string; '{internal}\\n(YYYY-MM-DD)'; etc.)
    """
    s = str(getattr(tdis.start_date_time, "data", getattr(tdis, "start_date_time", "")))
    s = s.strip()
    # common Flopy internal print like: "{internal}\n(1964-12-31)"
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", s)
    if m:
        return pd.to_datetime(m.group(1))
    # plain date:
    try:
        return pd.to_datetime(s)
    except Exception:
        pass
    # last resort: if tdis perioddata provides totim=0 as a date?
    raise ValueError(f"Unrecognized TDIS start_date_time format: {s!r}")

def _sp_dates_from_tdis(tdis) -> List[pd.Timestamp]:
    """
    Returns list of stress period START datetimes, length == nper.
    """
    start = _parse_mf6_start_date(tdis)
    pddata = list(tdis.perioddata.get_data())  # (perlen, nstp, tsmult)
    dates = [start]
    cur = start
    for k, (perlen, _nstp, _tsmult) in enumerate(pddata[:-1]):  # we already have SP0 start
        # perlen is in 'DAYS'
        cur = cur + pd.to_timedelta(float(perlen), unit="D")
        dates.append(cur)
    return dates

def _coerce_bool(v):
    """Robust bool cast for 0/1, y/n, t/f, true/false, strings, NaN."""
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(int(v))
    s = str(v).strip().lower()
    return s in {"1","y","yes","t","true"}

def _in_active_domain(idomain: np.ndarray, k: int, i: int, j: int) -> bool:
    try:
        return idomain[k, i, j] > 0
    except Exception:
        return False

def _yearly_median(series: pd.Series) -> pd.Series:
    """Return one value per year: the median of all observations that year."""
    if series.empty:
        return series
    return series.groupby(series.index.year).median().sort_index()

def _pick_layer_from_midpoint(mid_elev: float, top2d: np.ndarray, botm3d: np.ndarray, i: int, j: int) -> int:
    """
    Given midpoint elevation, the cell's vertical layer cake, return layer k (0-based).
    If outside, clamp to 0 or last.
    """
    # construct vertical boundaries at (i,j)
    nlay = botm3d.shape[0]
    ztops = [float(top2d[i, j])]
    for k in range(nlay):
        ztops.append(float(botm3d[k, i, j]))
    # ztops[0]=top; ztops[1]=bot layer 0; ...
    # ensure monotonically decreasing (top >= ... >= bottom)
    if not np.all(np.diff(ztops) <= 1e-8):
        # minor numeric issues are fine; but if grossly wrong, still proceed
        pass
    # find bracket
    for k in range(nlay):
        z_up = ztops[k]
        z_lo = ztops[k+1]
        if z_up >= mid_elev >= z_lo:
            return k
    # clamp
    if mid_elev > ztops[0]:
        return 0
    return nlay - 1

def rc_from_xy(modelgrid, x, y):
    try:
        _, i, j = modelgrid.intersect(float(x), float(y))
        return (int(i), int(j)) if (i is not None and j is not None) else (None, None)
    except Exception:
        return (None, None)

def _xy_to_rc(modelgrid, x: float, y: float):
    """
    Return (i, j) from map coords using modelgrid.intersect(x, y).
    If point is outside grid, return None.
    """
    try:
        k, i, j = modelgrid.intersect(x, y)  # k may be None; we only need i, j
        if i is None or j is None:
            return None
        return int(i), int(j)
    except Exception:
        return None

def _round_trip_shapefile_guard(gdf: gpd.GeoDataFrame, path: str):
    """
    Shapefile field names are limited; we only keep safe columns.
    """
    keep = []
    for c in gdf.columns:
        if c in ("geometry",):
            keep.append(c)
        elif len(c) <= 10:  # DBF 10-char limit
            keep.append(c)
    gdf[keep].to_file(path)

def _mfscalar_to_float(v, default=0.0):
    try:
        # FloPy MF6 objects
        return float(v.data)
    except Exception:
        try:
            # some builds expose .get_data()
            return float(v.get_data())
        except Exception:
            try:
                # sometimes it's already a number or numpy scalar
                return float(v)
            except Exception:
                return float(default)
            
def _find_col_case_insensitive(df: pd.DataFrame, candidates):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"None of the expected columns {candidates} found in grid file: {list(df.columns)}")

def _cells_from_grid_shp(
    sites_df: pd.DataFrame,
    *,
    grid_shp: str,
    row_field: str | None = None,
    col_field: str | None = None,
    sites_crs: str = "EPSG:2265",
) -> pd.DataFrame:
    """Add i,j to sites_df by spatial-joining to the grid shapefile."""
    import geopandas as gpd
    from shapely.geometry import Point

    grid = gpd.read_file(grid_shp)
    # find row/col columns (case-insensitive); common names: row/ROW, col/COL
    if row_field is None:
        row_field = _find_col_case_insensitive(grid, ["row", "r", "ROW"])
    if col_field is None:
        col_field = _find_col_case_insensitive(grid, ["col", "c", "COL"])

    grid = grid[[row_field, col_field, "geometry"]].copy()

    # Build points GeoDataFrame from x_2265,y_2265
    pts = gpd.GeoDataFrame(
        sites_df.copy(),
        geometry=gpd.points_from_xy(sites_df["x_2265"], sites_df["y_2265"]),
        crs=sites_crs,
    )

    # Ensure grid is in same CRS (assume sites CRS is the truth)
    if grid.crs is None:
        grid.set_crs(sites_crs, inplace=True)
    elif str(grid.crs) != str(pts.crs):
        grid = grid.to_crs(pts.crs)

    # spatial join (point-in-polygon)
    j = gpd.sjoin(pts, grid, how="inner", predicate="intersects")

    # If multiple matches (on boundaries), keep first per original row
    if "index_left" in j.columns:
        j = j.sort_values(["index_left"]).drop_duplicates("index_left")
        j = j.sort_index()  # return to original order if desired

    # Map grid row/col to zero-based i,j
    rvals = j[row_field].astype(int).to_numpy()
    cvals = j[col_field].astype(int).to_numpy()

    # Decide 0- vs 1-based indexing by looking at mins
    zero_based = (rvals.min() == 0) or (cvals.min() == 0)
    i = rvals if zero_based else (rvals - 1)
    jcol = cvals if zero_based else (cvals - 1)

    out = j.drop(columns=["geometry"]).copy()
    out["i"] = i
    out["j"] = jcol
    return out

# ----------------------------
# ---- wahp like wl processing
# ----------------------------

def load_input_data(mnm='elk_2lay'):
    # --- Load Water Level site reference
    sites = pd.read_csv(os.path.join('data','raw','obs_data','elk_valley_sites.csv'))
    
    # Only want wells in spiritwood or ww aquifers
    # sites = sites.loc[sites['Aquifer'].isin(['Spiritwood','warwick','Warwick'])]
    # Make into gdf
    points = [Point(xy) for xy in zip(sites['x_2265'], sites['y_2265'])]
    sites = gpd.GeoDataFrame(data=sites,
                             geometry=points)
    sites = sites.set_crs(2265)#.to_crs(2265)
    sites.columns = sites.columns.str.lower()
    sites.columns = sites.columns.str.replace(' ','_')
    
    # if site drop_flag is True, drop the site
    print(f'Dropping {len(sites[sites["drop_flag"] == True])} sites with drop_flag set to True')
    sites = sites.loc[sites['drop_flag'] != True]
    
    # --- Load the modelgrid
    modelgrid = gpd.read_file(os.path.join('..','..','gis','output_shps','elk','elk_cell_size_660ft_epsg2265_rot20.grid.shp'))
    icols = [col for col in modelgrid.columns if 'idom' in col]
    lycnt = len(icols) 

    # --- Load the model to get structure
    model_ws = os.path.join('model_ws',"elk_2lay_monthly")
    model_name = mnm
    sim = flopy.mf6.MFSimulation.load(
        sim_name=f'{model_name}.nam',
        version='mf6',
        exe_name='mf6.exe',
        sim_ws=model_ws,
        load_only=['dis'],
        )
    gwf = sim.get_model(model_name)
    
    top = gwf.dis.top
    bots = gwf.dis.botm.array
    
    # --- Only need data within modelgrid
    sites = sites.clip(modelgrid)
    
    
    # --- Load water level data
    water_levels = pd.read_csv(os.path.join('data','raw','obs_data','elk_valley_water_level_data.csv'))
    water_levels.columns = water_levels.columns.str.lower()

    
    # Filter
    water_levels = water_levels.loc[water_levels['site_index'].isin(sites['site_index'].values)]
    water_levels['date_measured'] = pd.to_datetime(water_levels['date_measured'])
    water_levels = water_levels.set_index('date_measured')
    water_levels.loc[water_levels['water_level(navd88)']>10000,'water_level(navd88)'] = np.nan
    water_levels.loc[water_levels['water_level(navd88)']<-1000,'water_level(navd88)'] = np.nan
    water_levels = water_levels.dropna()
        
    # Filter out sites with less than X measurements
    meas_thresh = 15
    water_levels = water_levels[water_levels['site_index'].map(water_levels['site_index'].value_counts()) > meas_thresh]

    # Drop sites that have less than X measurements
    sites = sites.loc[sites['site_index'].isin(water_levels['site_index'].unique())]

    # --- SORT SITES by 1. aquifer and 2. length of record
    water_levels['datetime'] = water_levels.index
    record_spans = (
            water_levels.groupby('site_index')['datetime']
            .agg(lambda x: (x.max() - x.min()).days)
        )
    sites = sites.merge(record_spans.rename('record_span_days'), left_on='site_index', right_on='site_index',how='left')
    sites['record_span_days'] = sites['record_span_days'].fillna(0).astype(int)
    sites = sites.sort_values(['record_span_days'], ascending=[False])
    
    # --- Get r/c indices for sites
    sites = gpd.sjoin(sites, modelgrid[['geometry', 'row', 'col']], how='left', predicate='intersects')
    
    # add total_depth to sites from water_levels
    wl_td = water_levels[['site_index','total_depth','land_surface_elev_navd88']].drop_duplicates(subset=['site_index'])
    wl_td.reset_index(drop=True,inplace=True)
    sites = sites.merge(wl_td, left_on='site_index', right_on='site_index', how='left')

    # --- Create lithology reference for each site
    layer_data = []
    nlay = gwf.dis.nlay.data
    top_array = top.array
    botm_array = bots

    # --- Do a bit of processing on total depths and then sort by TD
    drop_idxs = []
    for idx, row in sites.iterrows():
        td = row['total_depth']
        bs = row['bot_screen']
        if (td == 0) and (bs != 0):
            sites.at[idx, 'total_depth'] = bs
        if (td != 0) and (bs == 0):
            sites.at[idx, 'bot_screen'] = td
        elif (td == 0) and (bs == 0):
            drop_idxs.append(idx)
    print(f'Dropping {len(drop_idxs)} sites with no total depth or bottom screen depth')
    sites = sites.drop(index=drop_idxs)
    
    # --- Find which layer the site is within
    layer_list = []
    for _, row in sites.iterrows():
        i, j = row['row']-1, row['col']-1
        depth = row['bot_screen']
        surface_elev = row['land_surface_elev_navd88']
        depth_elev = surface_elev - depth
        # find first layer where depth elevation is above bottom
        assigned = False
        for k in range(nlay):
            top_k = top_array[i, j] if k == 0 else botm_array[k - 1, i, j]
            bot_k = botm_array[k, i, j]
            if top_k >= depth_elev >= bot_k:
                layer_list.append(k)
                assigned = True
                break
        if not assigned:
            if depth_elev > top_array[i, j]:
                layer_list.append(0)
            elif depth_elev < botm_array[-1, i, j]:
                layer_list.append(nlay - 1)
            else:
                layer_list.append(np.nan)  # should not happen unless DEM is weird
    # Add result to sites
    sites['model_layer'] = layer_list # zero-based layer index
    # where nan fill with 9999
    sites.loc[sites['model_layer'].isna(), 'model_layer'] = 9999
    sites['model_layer'] = sites['model_layer'].astype(int)+1 # convert to one-based layer index
    sites = sites.sort_values(by='model_layer')
    
    sites['k'] = sites['model_layer'].astype(int)-1
    # where model_layer is not equal to sites['manually_corrected_lay'], k value gets 'manually_corrected_lay' use np.where
    # sites['k'] = np.where(sites['model_layer'] != sites['manually_corrected_lay'],
    #                       sites['manually_corrected_lay'],
    #                       sites['model_layer'])


    return sites,water_levels,modelgrid

def load_perlen(annual_flag):
    if annual_flag:
        perlen = pd.read_csv(os.path.join("tables","annual_stress_period_info.csv"))
    else:
        perlen = pd.read_csv(os.path.join("tables","monthly_stress_period_info.csv"))
    
    return perlen
        
def process_wls(df,subset,annual_flag,smooth_type='median'):

    # smooth_type = 'median' median of year, 'avg_highs' average of yearly highs

    # Load the model perlen data
    perlen = load_perlen(annual_flag) 
    perlen = perlen.tail(len(perlen)-1)
    model_datetime = pd.to_datetime(perlen['start_datetime'])
    
    # --- Interpolate subset of obs WL's to daily
    subset = subset[~subset.index.duplicated(keep='first')]
    interp_ts = subset['water_level(navd88)'].resample('D').interpolate()
    
    # --- Split into pre-2000 and post-2000, cause dealing with different SP discretization 
    split_date = pd.Timestamp('2000-01-01')
    pre2000 = interp_ts[interp_ts.index < split_date]
    post2000 = interp_ts[interp_ts.index >= split_date]
    
    # --- Apply smoothing based on yearly/monthly SP period
    if smooth_type == 'avg_highs':
        # For pre-2000 data, using the yearly high from April, May or June
        pre2000 = pre2000.loc[pre2000.index.month.isin([4,5,6])]
        years = pre2000.index.year
        idx = pre2000.groupby(years).idxmax()
        smooth_pre2000 = pre2000.loc[idx]
        # For post-2000, using a ~2-month rolling average
        smooth_post2000 = post2000.rolling("60D", center=True).mean() 
    
        # --- Recombine to a single time series
        smooth_ts = pd.concat([smooth_pre2000, smooth_post2000]).sort_index()

        # --- Align to model sim times (cum_datetime) to give apples:apples comparison w/ model results
        simtime_ts = smooth_ts.reindex(model_datetime, method="nearest", tolerance=pd.to_timedelta(181, unit="d"))  # --> Tolerance of 181 days
        simtime_ts = simtime_ts.groupby(simtime_ts.index).mean()  # group duplicates to avoid 2 meas on same date
        simtime_ts = simtime_ts.dropna()

    elif smooth_type == 'median':
        years = pre2000.index.year
        smooth_pre2000 = pre2000.rolling("365D", center=True).median()
        smooth_post2000 = post2000.rolling("60D", center=True).median()
        # --- Recombine to a single time series
        smooth_ts = pd.concat([smooth_pre2000, smooth_post2000]).sort_index()
        # --- Align to model sim times (cum_datetime) to give apples:apples comparison w/ model results
        simtime_ts = smooth_ts.reindex(model_datetime, method="nearest", tolerance=pd.to_timedelta(181, unit="d"))  # --> Tolerance of 181 days
        simtime_ts = simtime_ts.groupby(simtime_ts.index).median()  # group duplicates to avoid 2 meas on same date
        simtime_ts = simtime_ts.dropna()
    
    return simtime_ts 

def save_processed_WL_data(sites,water_levels,annual_flag):
    # Process data for each site
    for idx,site_id in enumerate(sites['site_index'].unique()):
        wl_data = water_levels.loc[water_levels['site_index']==site_id,'water_level(navd88)']
        # Calc smoothed WL data
        smoothed_wl = process_wls(water_levels,pd.DataFrame(wl_data),annual_flag)
        t = smoothed_wl.rename(site_id).to_frame()
        t.index = pd.to_datetime(t.index)
        t.columns = [site_id]
        if idx == 0:
            main_df = t
        else:
            main_df = pd.concat([main_df, t], axis=1)
    
    return main_df

def plot_lith_multiwell(sites_grp,master_df,gwf,ax):
    
    top = gwf.dis.top
    bots = gwf.dis.botm.array
    layer_data = []
    nlay = gwf.dis.nlay.data
    top_array = top.array
    botm_array = bots

    # --- Save info when sites have the same row and column
    r_c_dict = {}
    for _, row in sites_grp.iterrows():
        r, c = int(row['row']), int(row['col'])
        aq = row['aquifer']
        r_c_dict[row['location']] = [r,c,aq]
        for k in range(nlay):
            ztop = top_array[r-1, c-1] if k == 0 else botm_array[k-1, r-1, c-1]
            zbot = botm_array[k, r-1, c-1]
            layer_data.append({
                        'loc_id': row['location'],
                        'layer': k,
                        'ztop': ztop,
                        'zbot': zbot
                        }
                )
    layer_df = pd.DataFrame(layer_data)
    
    # --- Create map for sites that share a row/column/aquifer index
    location_to_ids = defaultdict(list)
    for site_id, (r, c, aq) in r_c_dict.items():
        location_to_ids[(r, c, aq)].append(site_id)
    shared_location_dict = {
                site_id: location_to_ids[(r, c, aq)]
                for site_id, (r, c, aq) in r_c_dict.items()
                }
    
    # Colors for the aquifer layers
    colors = {0:"#4682B4", 
              1:"#FFE999", 
              2:"darkblue",
              3:"maroon",
              4:"orange",
              5:"#75D98F",
              6:"#A65628"
              }

    layers = layer_df[layer_df['loc_id'] == site_id]
    
    wellcnt = len(sites_grp)
    def compute_left_positions(n, bar_width=0.4, gap=0.0, x_min=0.0, x_max=1.0):
        span = x_max - x_min
        total = n*bar_width + (n-1)*gap
        if total > span:
            raise ValueError(f"Bars won’t fit: total={total} > span={span}")
        start = x_min + (span - total)/2.0  # center the block
        return start + np.arange(n)*(bar_width + gap)

    # example
    wellcnt = len(sites_grp)
    bar_width = 0.05
    gap = 0.0  # set >0 if you want spacing
    wpos = compute_left_positions(wellcnt, bar_width, gap)  # array of lefts, length = wellcnt
    
    # Plot lithology layers as wide bars

    for _, lyr in layers.iterrows():
        ax.barh(
            y=(lyr['ztop'] + lyr['zbot']) / 2,
            width=0.4,
            height=abs(lyr['ztop'] - lyr['zbot']),
            left=0.3,
            color=colors[lyr['layer']],
            edgecolor='k'
        )
    
    # Overlay well
    cnt = 0
    for idx,well in sites_grp.iterrows():
        mpe = well['land_surface_elev_navd88']
        td = well['total_depth']
        ts = well['top_screen']
        bs = well['bot_screen']

        if td == 0  and ts != 0:
            td = ts
        
        if td == 0 and ts == 0:
            td = mpe - (layers.loc[layers['layer']==well['k'],'ztop'].values+layers.loc[layers['layer']==well['k'],'zbot'].values)/2
        
        if pd.notnull(mpe) and pd.notnull(td):
            well_top = mpe
            well_bot = mpe - td
            ax.barh(
                y=(well_top + well_bot) / 2,
                width=bar_width,
                height=abs(well_top - well_bot),
                left=wpos[cnt],
                color='grey'
            )
            
        # Overlay screen
        if pd.notnull(ts) and pd.notnull(bs):
            top_screen = mpe - ts
            bot_screen = mpe - bs
            ax.barh(
                y=(top_screen + bot_screen) / 2,
                width=0.05,
                height=abs(top_screen - bot_screen),
                left=wpos[cnt],
                color='white',
                hatch='-----',
            )
        

        # id to link to master_df
        grp_num = well['group_number']
        id = f'grp.{grp_num}'+f'_k.{well["k"]}'
        mwl = master_df[[id]].mean(axis=0)
        # --- Plot average WL
        ax.plot(wpos[cnt]+bar_width/2, mwl, 
                marker='x', 
                color='blue', 
                markersize=10)
        
        cnt += 1

def nested_sites(mnm, main_df, sites, water_levels, modelgrid, gen_plots=True):

    model_ws = os.path.join('model_ws',"elk_2lay_monthly")
    mnm = mnm
    sim = flopy.mf6.MFSimulation.load(
        sim_name=f'{mnm}.nam',
        version='mf6',
        exe_name='mf6.exe',
        sim_ws=model_ws,
        load_only=['dis'],
        )
    gwf = sim.get_model(mnm)

    for col in main_df.columns:
        site_info = sites[sites['site_index'] == col]
        new_header = f"sid.{col}_i.{site_info['row'].values[0]-1}_j.{site_info['col'].values[0]-1}_k.{site_info['k'].values[0]}_grp.{site_info['group_number'].values[0]}"
        main_df.rename(columns={col: new_header}, inplace=True)
    
    def parse_column(col):

        """
        Extract (group, layer) as integers from a column like:
        'sid.13304720ABBA3_i.112_j.46_k.1_grp.15'
        """
        m = re.search(r"k\.(\d+)_grp\.(\d+)", col)
        if m:
            layer = int(m.group(1))
            group = int(m.group(2))
            return (group, layer)
        else:
            return (float("inf"), float("inf"))  # fallback if no match
    
    sorted_cols = sorted(main_df.columns, key=parse_column)
    main_df = main_df[sorted_cols]  # reindex with sorted columns   

    # get list of unique group_numbers in columns
    group_nums = main_df.columns.str.extract(r'_grp\.(\d+)')[0].unique()
    master_df = pd.DataFrame(index=main_df.index)
    for grp_num in group_nums:
        # Get columns that match the group_number
        grp_cols = main_df.columns[main_df.columns.str.endswith(f'_grp.{grp_num}')]
      
        if len(grp_cols) > 1:
            # see if we need to combine timeseries if they have the same k value
            k_values = grp_cols.str.extract(r'_k\.(\d+)')[0]
            dup_k_values = []
            # see if there are duplicate k values
            if k_values.duplicated().any():
                # get the duplicated k values
                dup_k_values = k_values[k_values.duplicated()].unique()
                for k_val in dup_k_values:
                    # get column names with this k value
                    dup_cols = [col for col in grp_cols if f'_k.{k_val}' in col]
                    ndf = main_df[dup_cols].median(axis=1,skipna=True)
                    master_df[f'grp.{grp_num}_k.{k_val}'] = ndf
                
                # get the non-duplicated columns
                # these are the columns that do not have the k value in their nam
                # remove dup_k_values from k_values:
                not_duplicated = k_values[~k_values.isin(dup_k_values)].tolist()
                if len(not_duplicated) > 0:
                    for k in not_duplicated:
                        col = [col for col in grp_cols if f'_k.{k}' in col]
                        ndf = main_df[col]  # there should only be one column with this k value
                        # add the non-duplicated columns to master_df
                        master_df[f'grp.{grp_num}_k.{k}'] = main_df[col]
            else:
                for col in grp_cols:
                    k_val = col.split('_k.')[1].split('_')[0]
                    master_df[f'grp.{grp_num}_k.{k_val}'] = main_df[col]
        else:
            for col in grp_cols:
                k_val = col.split('_k.')[1].split('_')[0]
                master_df[f'grp.{grp_num}_k.{k_val}'] = main_df[col]

    def parse_col(col):
        m = re.match(r"grp\.(\d+)_k\.(\d+)", col)
        if m:
            return int(m.group(1)), int(m.group(2))
        else:
            return float("inf"), float("inf")  # fallback if something doesn't match
    sorted_cols = sorted(master_df.columns, key=parse_col)
    master_df = master_df[sorted_cols]  # reindex with sorted columns
    
    # --- Now create dataframe of head differnce obs:
    k_aq = {0:'ClaySilt',1:'EV'}


    # Parse (grp, k) from column names
    def parse_grp_k(col):
        m = re.match(r'^grp\.(\d+)_k\.(\d+)$', col)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    # Build a helper structure: by group -> {k: column_name}
    by_grp = {}
    for col in master_df.columns:
        g, k = parse_grp_k(col)
        if g is None:
            continue
        by_grp.setdefault(g, {})[k] = col
    # filter dictionary down to groups with at least 2 layers
    by_grp_multi = {g: kmap for g, kmap in by_grp.items() if len(kmap) > 1}
    

    for grp, k_to_col in by_grp_multi.items():
        # Map aquifer -> list of (k, col)
        aq_to_layers = {}
        for k, col in k_to_col.items():
            aq = k_aq.get(k)
            if aq is None:
                continue
            aq_to_layers.setdefault(aq, []).append((k, col))


    if gen_plots:
        # Create PDF file
        pdf_path = os.path.join("data","analyzed",f"{mnm}_well_summary_plots_grouped.pdf")
        
        grp_nums = master_df.columns.str.extract(r'grp\.(\d+)')[0].unique()
        with PdfPages(pdf_path) as pdf:

            for grp in grp_nums:
                # get master_df columns that have the group_number
                grp_cols = master_df.columns[master_df.columns.str.startswith(f'grp.{grp}_')]
                
                # --- Load the site to plot
                print(f"\nMaking figure for group: {grp}\n")
               
                fig = plt.figure(figsize=(12, 6))
                gs = GridSpec(
                        nrows=2,
                        ncols=3,
                        height_ratios=[1, 1],
                        width_ratios=[1.01, 0.5, 1.2],  # make map column wider
                        figure=fig
                    )
                shared_colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628",]

                # --- Bottom Row: Hydrograph Time Series
                ax_hydro = fig.add_subplot(gs[1, :])

                # get sites with this group_number
                sites_grp = sites[sites['group_number'] == int(grp)]
                color_cnt = 0
                for site_id in sites_grp['site_index'].unique():
                    wl_data = water_levels.loc[water_levels['site_index']==site_id,'water_level(navd88)']

                    # Daily resample on high-frequency transducer data
                    if len(wl_data) > 1200:
                        wl_data_resamp = wl_data.resample('D').mean()
                        wl_data_resamp = wl_data_resamp.dropna()
                        ax_hydro.plot(wl_data_resamp,
                                    marker='o',
                                    ms=4,
                                    color=shared_colors[color_cnt],
                                    label=site_id,
                                    alpha=0.5)
                        color_cnt += 1
                    else:
                        ax_hydro.plot(wl_data,
                                    marker='o',
                                    ms=4,
                                    color=shared_colors[color_cnt],
                                    label=site_id,
                                    alpha=0.5)
                        color_cnt += 1

                # now plot master_df for this group:
                for col in grp_cols:
                    smoothed_wl = master_df[col]
                    k = int(col.split('_k.')[1].split('_')[0])
                    aq = ''
                    if k == 0:
                        aq = 'Clay/Silt'
                        aq_color = "#774B12"
                        marker = 'x'
                    elif k == 1:
                        aq = 'EV'
                        aq_color = "#061B79"
                        marker = '^'

                    ax_hydro.plot(smoothed_wl,
                                  color=aq_color,
                                  ls='-',
                                  marker=marker,
                                  ms=5,
                                  linewidth=1.5,
                                  markerfacecolor=None,
                                  label=aq)
          
                ax_hydro.set_ylabel("Water Level\n(feet above mean sea level)")
                ax_hydro.set_xlabel("Year")
                ax_hydro.set_xlim(right=pd.Timestamp('2025-05-01'))
                # h,l = ax_hydro.get_legend_handles_labels()
                ax_hydro.grid()
                ax_hydro.legend()
                
                # --- Color the plot by wet and dry periods
                # for i in range(len(wet_dry)-1):
                #     color = wet_dry.iloc[i].color
                #     if color == 'grey':
                #         continue
                #     d1 = wet_dry.iloc[i].name
                #     d2 = wet_dry.iloc[i+1].name - pd.Timedelta(days=10)
                #     ax_hydro.axvspan(d1,d2,color=color,
                #                      alpha=0.2,
                #                      edgecolor=None)
                
                # --- Metadata Text Block (Left)
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_meta.axis("off")
                swn = sites_grp['location'].unique()
                swn_str = '\n'.join(swn)
                ax_meta.text(0.3, 0.5, f"Group Number: {grp}\nState Well Numbers:\n {swn_str}\n", 
                    va="top", fontsize=12,
                    ha='center')

                # --- B: Lithology Stack Bar
                ax_lith = fig.add_subplot(gs[0, 1])
                plot_lith_multiwell(sites_grp,master_df,gwf,ax_lith)
                #plot_lith(site,site_id,layer_df,avg_wl,ax=ax_lith)
                
                ax_lith.set_xlim(0.4, 0.6)
                for spine in ['top', 'right', 'bottom','left']:
                    ax_lith.spines[spine].set_visible(False)
                ax_lith.tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)
                ax_lith.set_ylabel("Elevation (feet ASL)")
                
        
                # --- C: Map View
                ax_map = fig.add_subplot(gs[0, 2])
                
                elk = gpd.read_file(os.path.join("..","..","gis","input_shps","elk","elk_boundary.shp"))
                elk.boundary.plot(ax=ax_map,
                                    color='blue')
                sites.plot(ax=ax_map,
                           color='grey',
                           edgecolor='k',
                           markersize=8)
                
                # Plot selected site
                sites_grp.plot(ax=ax_map,
                                    color='orange',
                                    edgecolor='k',
                                    markersize=35)
                modelgrid.dissolve().boundary.plot(ax=ax_map,
                                                color='k')
                
                # --- Custom map legend
                legend_elements = [
                            Patch(edgecolor='k', facecolor='none', linewidth=1.5, label='Model Boundary'),
                            Patch(edgecolor='blue', facecolor='none', linewidth=1.5, label='Wahp Aquifer\nExtent'),
                            Line2D([0], [0], marker='o', color='grey', markeredgecolor='k',
                                markersize=6, linestyle='None', label=f'Monitoring Wells'),
                            Line2D([0], [0], marker='o', color='orange', markeredgecolor='k',
                                markersize=10, linestyle='None', label='Selected Well'),
                            ]
                # Add legend to map axis
                ax_map.legend(handles=legend_elements,
                            loc='center left', 
                            fontsize=8, 
                            frameon=True,
                            bbox_to_anchor=(0.8, 0.08),
                            framealpha=1,
                            edgecolor='black'
                            )
                ax_map.axis('off')
                
                # --- Custom lith legend
                ax_legend_lith = fig.add_axes([0.58, 0.82, 0.1, 0.1])  # [left, bottom, width, height]
                ax_legend_lith.axis('off')
                
                {0: "soils", 1: "sand", 2: "clay", 3: "clay", 4: "sand", 5: "clay", 6: "sand"}
                colors = {0:"#4682B4", 
                        1:"#FFE999", 
                        2:"darkblue",
                        3:"maroon",
                        4:"orange",
                        5:"#75D98F",
                        }
                # get columns in modelgrid that start with idom_
                icols = modelgrid.columns[modelgrid.columns.str.startswith('idom_')]
                lycnt = len(icols)

                lith_legend_elements = [
                            Patch(facecolor="#4682B4", edgecolor='k', label="Clay/Silt"),
                            Patch(facecolor="#FFE999", edgecolor='k', label="EV Aquifer"),
                            Patch(facecolor='none', edgecolor='k', hatch='-----', label='Screen Interval'),
                            Line2D([0], [0], marker='x', color='blue', linestyle='None', markersize=8, label='Mean Water Level')
                                ]

                         
                    
                # Add the lithology legend
                ax_legend_lith.legend(
                            handles=lith_legend_elements,
                            loc='upper left',
                            fontsize=8,
                            frameon=True,
                            framealpha=1,
                            edgecolor='black'
                        )
                
                fig.tight_layout()
                # plt.show()
                pdf.savefig(fig,bbox_inches='tight' )
                plt.close(fig)
    return master_df

# -------------------------------------------------------
# ---- main worker: build SS+Transient obs & (optionally)
# ---- attach a ModflowUtlobs package to the model
# -------------------------------------------------------
def build_obs_from_elk(
    gwf: flopy.mf6.ModflowGwf,
    *,
    sites_csv: str = "data/raw/obs_data/elk_valley_sites.csv",
    wl_xlsx: str = "data/raw/obs_data/elk_valley_water_level_data.csv",
    out_dir: str = "data/analyzed",
    make_shapefile: bool = True,
    add_obs_package: bool = True,
    obs_pname: str = "obs",
    obs_filename: Optional[str] = None,
    post2000_rolling_days: int = 60,
    pre2000_cutoff_year: int = 2000,
    mode: str = "monthly",        # "monthly" or "annual"
    grid_shp: str | None = None,  # path to grid shapefile
    grid_row_field: str | None = None,
    grid_col_field: str | None = None,
    tdf: pd.DataFrame | None = None,  # optional transient obs nodes (not used here but kept for API compat)
):
    """
    Build steady-state (SP0) and (optionally) transient HEAD targets from Elk Valley
    inputs, dropping any sites with drop_flag=True from BOTH SS and transient.

    NOTE: This version returns only ss_df (to match your current calling code).
    """
    # ------------------------
    # small helper
    # ------------------------
    def _coerce_bool(v):
        """Robust bool cast for 0/1, y/n, t/f, true/false, strings, NaN."""
        if pd.isna(v):
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer, float, np.floating)):
            return bool(int(v))
        s = str(v).strip().lower()
        return s in {"1", "y", "yes", "t", "true"}

    # ------------------------
    # setup
    # ------------------------
    out_dir = _ensure_outdir(out_dir)
    if obs_filename is None:
        obs_filename = f"{gwf.name}.obs"

    # model arrays
    top = np.asarray(gwf.dis.top.array)
    botm = np.asarray(gwf.dis.botm.array)
    idomain = np.asarray(gwf.dis.idomain.array, int)
    nlay, nrow, ncol = idomain.shape

    # stress period starts
    tdis = gwf.simulation.get_package("tdis")
    sp_starts = _sp_dates_from_tdis(tdis)
    nper = len(sp_starts)

    # ------------------------
    # load + normalize sites
    # ------------------------
    sites = pd.read_csv(sites_csv)
    # normalize columns (lowercase + underscores)
    sites.columns = (
        sites.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # common renames (normalized form)
    # (handles typical headers you’ve shown; harmless if already present)
    colmap_norm = {
        "site_index": "site_id",
        "location": "location",
        "group_identifier": "group_identifier",
        "group_number": "group_number",
        "top_screen": "top_screen",
        "bot_screen": "bottom_screen",
        "total_depth": "total_depth",
        "x_2265": "x_2265",
        "y_2265": "y_2265",
        "wl_avg_1965": "wl_avg_1965",
    }
    for old, new in colmap_norm.items():
        if old in sites.columns and new not in sites.columns:
            sites = sites.rename(columns={old: new})

    # drop flagged sites BEFORE anything else (affects SS + transient)
    if "drop_flag" in sites.columns:
        sites["drop_flag"] = sites["drop_flag"].map(_coerce_bool)
        n_drop = int(sites["drop_flag"].sum())
        if n_drop:
            print(f"Dropping {n_drop} sites with drop_flag=True (SS + transient).")
        sites = sites.loc[~sites["drop_flag"]].copy()

    required = ["site_id", "location", "x_2265", "y_2265", "wl_avg_1965"]
    missing = [r for r in required if r not in sites.columns]
    if missing:
        raise ValueError(f"sites CSV is missing required columns after normalization/rename: {missing}")

    # ------------------------
    # assign cells via grid shapefile
    # ------------------------
    if grid_shp is None:
        raise ValueError("Please provide grid_shp=path/to/grid.shp for cell assignment.")
    sites = _cells_from_grid_shp(
        sites,
        grid_shp=grid_shp,
        row_field=grid_row_field,
        col_field=grid_col_field,
        sites_crs="EPSG:2265",
    )
    if sites.empty:
        raise ValueError("No site points joined to grid shapefile (check CRS/coordinates).")

    # keep only sites in *active* domain (any layer active at i,j)
    def _in_bounds(i, j):
        return (0 <= i < nrow) and (0 <= j < ncol)
    def _active_any(i, j) -> bool:
        if not _in_bounds(i, j):
            return False
        return (idomain[:, i, j] > 0).any()

    sites = sites[[ _active_any(i, j) for i, j in zip(sites["i"], sites["j"]) ]].copy()
    if sites.empty:
        raise ValueError("No sites intersect the active model domain.")

    # ------------------------
    # layer (k) selection via screen midpoint rule
    # ------------------------
    # numeric depths
    sites["top_screen"] = pd.to_numeric(sites.get("top_screen", np.nan), errors="coerce")
    sites["bottom_screen"] = pd.to_numeric(sites.get("bottom_screen", np.nan), errors="coerce")

    # if bottom_screen <= 0 or NaN → force layer 2 (k=1), else use midpoint
    force_l2_mask = (sites["bottom_screen"].fillna(0) <= 0)

    # get cell-top at (i,j)
    top_at_cell = np.array([ float(top[i, j]) for i, j in zip(sites["i"], sites["j"]) ], dtype=float)

    # convert depths (ft below surface) → elevations
    top_depth = sites["top_screen"].fillna(0.0).to_numpy(float)
    bot_depth = sites["bottom_screen"].to_numpy(float)
    bot_depth = np.where(np.isnan(bot_depth), top_depth, bot_depth)

    top_screen_elev = top_at_cell - top_depth
    bot_screen_elev = top_at_cell - bot_depth
    mid_elev = (top_screen_elev + bot_screen_elev) / 2.0

    k_list = []
    for midz, i, j, force_l2 in zip(mid_elev, sites["i"], sites["j"], force_l2_mask):
        if force_l2:
            k_list.append(min(1, nlay - 1))  # clamp if single-layer
        else:
            k_list.append(_pick_layer_from_midpoint(float(midz), top, botm, int(i), int(j)))

    def _ensure_active_k(i, j, k):
        if idomain[k, i, j] > 0:
            return k
        for kk in range(k + 1, nlay):
            if idomain[kk, i, j] > 0:
                return kk
        for kk in range(k - 1, -1, -1):
            if idomain[kk, i, j] > 0:
                return kk
        return k

    sites["k"] = [
        _ensure_active_k(int(i), int(j), int(k)) for i, j, k in zip(sites["i"], sites["j"], k_list)
    ]

    # ------------------------
    # Steady-state targets (SP 0)
    # ------------------------
    ss_df = sites[["site_id","location","i","j","k","x_2265","y_2265","wl_avg_1965"]].copy()
    ss_df = ss_df.rename(columns={"wl_avg_1965": "head_target"})
    ss_df["head_target"] = pd.to_numeric(ss_df["head_target"], errors="coerce")
    ss_df = ss_df.dropna(subset=["head_target"]).copy()

    ss_df["sp"] = 0
    ss_df["start_dt"] = sp_starts[0]
    ss_df["end_dt"] = sp_starts[0] + pd.to_timedelta(1, unit="D")
    ss_df["obsprefix"] = (
        "ssh_id:" + ss_df["site_id"].astype(str)
        + "_k:" + ss_df["k"].astype(int).astype(str)
        + "_i:" + ss_df["i"].astype(int).astype(str)
        + "_j:" + ss_df["j"].astype(int).astype(str)
    )

    # ------------------------
    # Transient inputs (filtered by the same dropped sites)
    # ------------------------
    try:
        wlx = pd.read_excel(wl_xlsx, dtype={"Site_Index": str, "Location": str}, engine="openpyxl")
    except Exception:
        # allow CSV fallback if needed
        if os.path.splitext(wl_xlsx)[1].lower() == ".csv":
            wlx = pd.read_csv(wl_xlsx)
        else:
            raise

    # normalize transient headers (lower + underscores)
    wlx.columns = (
        wlx.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    # map common names
    colmap2 = {
        "site_index": "site_id",
        "location": "location",
        "water_level(navd88)": "wl_navd88",
        "water_level_navd88": "wl_navd88",
        "date_measured": "date_meas",
    }
    for old, new in colmap2.items():
        if old in wlx.columns and new not in wlx.columns:
            wlx = wlx.rename(columns={old: new})

    # build site_key on both sides using DROPPED set
    wlx["site_id"] = wlx.get("site_id", wlx.get("location")).astype(str)
    wlx["location"] = wlx.get("location", wlx.get("site_id")).astype(str)
    wlx["site_key"] = wlx["site_id"].fillna(wlx["location"]).astype(str).str.strip()

    sites["site_key"] = sites["site_id"].fillna(sites["location"]).astype(str).str.strip()
    keep_keys = set(sites["site_key"].unique())

    # filter transient to only the kept site keys (so drop_flag sites are OUT)
    wlx = wlx[wlx["site_key"].isin(keep_keys)].copy()
    if not wlx.empty:
        wlx["date_meas"] = pd.to_datetime(wlx["date_meas"], errors="coerce")
        wlx["wl_navd88"] = pd.to_numeric(wlx["wl_navd88"], errors="coerce")
        wlx = wlx.dropna(subset=["date_meas","wl_navd88"])

    # (Your transient smoothing/alignment can proceed here; omitted in return)

    # ------------------------
    # Write outputs (SS)
    # ------------------------
    ss_csv = os.path.join(out_dir, "elk_ss_targets.csv")
    ss_df.to_csv(ss_csv, index=False)

    # if make_shapefile:
    #     gss = gpd.GeoDataFrame(
    #         ss_df.copy(),
    #         geometry=gpd.points_from_xy(ss_df["x_2265"], ss_df["y_2265"]),
    #         crs="EPSG:2265",
    #     )
    #     shp_path = os.path.join(out_dir, "elk_ss_targets.shp")
    #     _round_trip_shapefile_guard(gss, shp_path)

    # (obs package wiring left as-is/commented in your original code)

    # Return steady-state targets (matches how you’re calling this function)
    return ss_df


def main(sim_ws,mnm,annual_flag,gen_plots=True):
    sites,water_levels,modelgrid = load_input_data()
    main_df = save_processed_WL_data(sites,water_levels,annual_flag)
    wls = nested_sites(mnm,main_df,sites,water_levels,modelgrid,gen_plots=gen_plots)   
    
    sites = sites.sort_values(by='group_number')
    sites.to_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    wls.to_csv(os.path.join('data','analyzed','transient_well_targets.csv'))
    
    trans_hd_list = []    
    # load transient targets look up:
    trans_sites = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    trans_obs = []
    cols = wls.columns.tolist()
    tdf = pd.DataFrame(columns=['obsprefix','grpid' ,'k', 'i', 'j'])
    new_cols = []
    for col in cols:
        grp = int(col.split('_')[0].split('.')[1])
        k = int(col.split('.')[-1])
        site = trans_sites.loc[(trans_sites['group_number'] == grp) & (trans_sites['k'] == k)]
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
    wls.columns = new_cols
    wls.to_csv(os.path.join('data','analyzed','transient_well_targets.csv')) 
    tdf.to_csv(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'), index=False)

    sim = flopy.mf6.MFSimulation.load(
        sim_ws=sim_ws,
    )
    gwf = sim.get_model(mnm)
    xo  = _mfscalar_to_float(getattr(gwf.dis, "xorigin", 0.0), 0.0)
    yo  = _mfscalar_to_float(getattr(gwf.dis, "yorigin", 0.0), 0.0)
    ang = _mfscalar_to_float(getattr(gwf.dis, "angrot", 0.0), 0.0)

    gwf.modelgrid.set_coord_info(xoff=xo, yoff=yo, angrot=ang, crs=CRS.from_epsg(2265))
    
    # taking transient obs from RH water level processing and SS from Mel's build_obs_from_elk fx, and changing
    # the obs package build to be built in the model build script because the OBS package was not getting added previously
    ss_df= build_obs_from_elk(
        gwf,
        mode="annual",  # or "monthly"
        grid_shp=os.path.join("..", "..", "gis", "output_shps", "elk", "elk_cell_size_660ft_epsg2265_rot20.grid.shp"),   # <-- your grid shapefile
        # grid_row_field="row", grid_col_field="col",  # only set if your field names differ
        tdf=tdf,
    )
    #gwf.simulation.write_simulation()
    return ss_df, tdf


# --------------------------
# ---- tiny CLI / demo run
# --------------------------
if __name__ == "__main__":
    # Example: load an existing model workspace and build obs.
    # Adjust these if you want to run as a script.
    simws = os.path.join("model_ws", "elk_2lay_monthly")   # your workspace
    mnm = "elk_2lay"
    annual_flag = False
    ss, trans = main(simws,mnm,annual_flag,gen_plots=True)






