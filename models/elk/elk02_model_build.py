# -*- coding: utf-8 -*-
"""
Elk Valley MF6 build: 2-layer model + TDIS (calendar), IMS, uniform recharge (2 in/yr),
and a RIV package built once from topography and reused for all stress periods.

Well package intentionally deferred (to be added later).
"""

import os
import sys
import re
import glob
import platform
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import pyemu
import flopy
from pathlib import Path
import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import overlay
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.errors import RasterioIOError
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
import importlib
import shutil
from pyproj import CRS
import matplotlib.pyplot as plt
import elk01_water_level_processing as elk_obs
import elk03_setup_pst as elk_pst
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from typing import Optional, Tuple, List, Dict, Any, Sequence
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter  # for regression smoothing
from flopy.utils import Raster


# --------------------------
# QA Plotting
# --------------------------
def plot_basemap(epsg=2265):
    """Plot the basemap of the model area."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('Easting (ft)')
    ax.set_ylabel('Northing (ft)')
    
    #cnty = gpd.read_file(os.path.join("..", "..", "gis",'input_shps','regional',"counties.shp"))
    #cnty = cnty.to_crs(epsg=epsg)
    #cnty.boundary.plot(ax=ax, color="grey", linewidth=0.5, label="County")
    
    # wahp outline:
    grd = gpd.read_file(os.path.join('..', '..', 'gis','output_shps','elk','elk_2lay_cell_size_660.0ft_epsg2265.grid.shp'))
    grd = grd.dissolve()
    grd = grd.explode(index_parts=False)
    grd = grd.to_crs(epsg=epsg)
    grd.boundary.plot(ax=ax, color="blue", linewidth=0.5, label="Model Grid")
    
    # add wahp aq state outline:
    elk = gpd.read_file(os.path.join('..', '..', 'gis','input_shps','elk','elk_boundary.shp'))
    elk = elk.to_crs(epsg=epsg)
    elk.boundary.plot(ax=ax, color='black', linewidth=0.5, label='Elk Outline')

    # zoom to the extent of the grid:
    minx, miny, maxx, maxy = grd.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # comma format for large numbers:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,}'.format(int(x))))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,}'.format(int(x))))
    
    return fig, ax


# --------------------------
# UTILS
# --------------------------
def find_mf6_exe(bin_folder=os.path.join("..", "..", "bin")):
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

def _plain_gtiff_profile(width, height, count, dtype, crs, transform, *, nodata=None):
    p = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        p["nodata"] = nodata
    return p

def convert_surfer_grids_to_tifs(
    grd_dir: str,
    out_dir: str,
    dst_epsg: int = 2265,
    source_epsg: int | None = None,
    resampling: Resampling = Resampling.nearest,
    overwrite: bool = False,
) -> list[Path]:
    grd_dir = Path(grd_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for grd_path in sorted(grd_dir.rglob("*.grd")):
        tif_path = out_dir / f"{grd_path.stem}.tif"
        if tif_path.exists():
            if overwrite:
                try:
                    tif_path.unlink()
                except OSError:
                    pass
            else:
                written.append(tif_path)
                continue
        try:
            with rasterio.open(grd_path) as src:
                src_crs = src.crs
                if src_crs is None:
                    if source_epsg is None:
                        raise ValueError(
                            f"{grd_path.name} has no CRS. Provide source_epsg= "
                            "(e.g., 2265 if already in EPSG:2265)."
                        )
                    src_crs = rasterio.crs.CRS.from_epsg(source_epsg)
                dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg)
                nodata = src.nodata if src.nodata is not None else -9999.0
                if src_crs == dst_crs:
                    profile = _plain_gtiff_profile(
                        src.width, src.height, src.count, src.dtypes[0],
                        dst_crs, src.transform, nodata=nodata
                    )
                    with rasterio.open(tif_path, "w", **profile) as dst:
                        for b in range(1, src.count + 1):
                            arr = src.read(b, masked=True).filled(nodata)
                            dst.write(arr, b)
                    written.append(tif_path)
                    continue
                transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds
                )
                profile = _plain_gtiff_profile(
                    width, height, src.count, src.dtypes[0],
                    dst_crs, transform, nodata=nodata
                )
                with rasterio.open(tif_path, "w", **profile) as dst:
                    for b in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, b),
                            destination=rasterio.band(dst, b),
                            src_transform=src.transform,
                            src_crs=src_crs,
                            src_nodata=nodata,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            dst_nodata=nodata,
                            resampling=resampling,
                        )
                written.append(tif_path)
        except (RasterioIOError, ValueError) as e:
            print(f"[skip] {grd_path.name}: {e}")
            try:
                if tif_path.exists() and tif_path.stat().st_size == 0:
                    tif_path.unlink()
            except OSError:
                pass
    return written

def _is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

def _cell_centers_xy(gwf):
    mg = gwf.modelgrid
    return np.asarray(mg.xcellcenters), np.asarray(mg.ycellcenters)

def _sample_tif_at_points(tif_path, xcc, ycc):
    with rasterio.open(tif_path) as src:
        coords = [(float(x), float(y)) for x, y in zip(xcc.ravel(), ycc.ravel())]
        vals = np.fromiter((v[0] for v in src.sample(coords)), dtype=float, count=xcc.size)
        arr = vals.reshape(xcc.shape)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    return arr

def _enforce_min_thickness(
    top: np.ndarray,
    botm1_in: np.ndarray,
    botm2_in: np.ndarray,
    *,
    min_thk: float = 5.0,
    check_l1_top_bottom: bool = True,
    enforce_even_if_no_overlap: bool = True,
    mask: np.ndarray | None = None,
    nodata_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groundwater Vistas-style repair for layer overlap / thin layers for a 2-layer stack.

    Behavior
    --------
    - Only pushes bottoms **downward** (numerically smaller elevations), never moves `top`.
    - If `check_l1_top_bottom` is True, enforce thickness between `top` and `botm1`.
    - If `enforce_even_if_no_overlap` is True, enforce min_thk even if thickness is positive but < min_thk.
      If False, only fix where thickness <= 0 (true overlap).
    - Optional `mask` (bool, shape nrow×ncol) can restrict the cells we modify.
    - Optional `nodata_value` skips cells where any input equals that value.

    Returns
    -------
    top, botm1, botm2 (same shapes as inputs)
    """
    top   = np.asarray(top, dtype=float)
    botm1 = np.asarray(botm1_in, dtype=float).copy()
    botm2 = np.asarray(botm2_in, dtype=float).copy()

    nrow, ncol = top.shape

    valid = np.ones((nrow, ncol), dtype=bool)
    if mask is not None:
        valid &= mask.astype(bool)
    if nodata_value is not None:
        valid &= (top   != nodata_value)
        valid &= (botm1 != nodata_value)
        valid &= (botm2 != nodata_value)

    def _push_down(cur_top: np.ndarray, cur_bot: np.ndarray, apply_check: bool) -> np.ndarray:
        """
        Push `cur_bot` downward just enough to resolve overlap and/or meet min_thk.
        Operates only on `valid` cells. Never raises bottoms.
        """
        thickness = cur_top - cur_bot
        if enforce_even_if_no_overlap:
            need_fix = (thickness < min_thk)
        else:
            need_fix = (thickness <= 0.0)
        need_fix &= valid

        # target bottom = top - min_thk (downward)
        target_bot = cur_top - float(min_thk)
        fixed = cur_bot.copy()
        if apply_check:
            # choose the *lower* bottom so we only push downward
            fixed[need_fix] = np.minimum(cur_bot[need_fix], target_bot[need_fix])
        return fixed

    # 1) Layer 1: compare TOP vs BOTM1 (optional)
    if check_l1_top_bottom:
        botm1 = _push_down(top, botm1, apply_check=True)

    # 2) Layer 2: its "top" is BOTM1 (always check)
    botm2 = _push_down(botm1, botm2, apply_check=True)

    return top, botm1, botm2

def _make_idomain_from_boundary(boundary_shp, xcc, ycc, nlay=2, target_epsg=2265):
    elk = gpd.read_file(boundary_shp)
    if elk.crs is None or int(str(elk.crs).split(":")[-1]) != target_epsg:
        elk = elk.to_crs(epsg=target_epsg)
    poly = elk.union_all()
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xcc.ravel(), ycc.ravel())], crs=f"EPSG:{target_epsg}")
    inside = pts.apply(poly.covers).to_numpy().reshape(xcc.shape)
    idom2d = inside.astype(int)
    idomain = np.repeat(idom2d[np.newaxis, :, :], nlay, axis=0)
    return idomain

def prep_deps(d):
    """copy exes to a directory based on platform
    Args:
        d (str): directory to copy into
    Note:
        currently only mf6 for mac and win and mp7 for win are in the repo.
            need to update this...
    """
    # copy in deps and exes
    if "window" in platform.platform().lower():
        bd = os.path.join("..","..","bin", "win")
        
    elif "linux" in platform.platform().lower():
        bd = os.path.join("..","..","bin", "linux")
        
    else:
        bd = os.path.join("..","..","bin", "mac")
        
    for f in os.listdir(bd):
            shutil.copy2(os.path.join(bd, f), os.path.join(d, f))

    try:
        shutil.rmtree(os.path.join(d,"flopy"))
    except:
        pass

    try:
        shutil.copytree(os.path.join('..','..','dependencies','flopy'), os.path.join(d,"flopy"))
    except:
        pass

    try:
        shutil.rmtree(os.path.join(d,"pyemu"))
    except:
        pass

    try:
        shutil.copytree(os.path.join('..','..','dependencies',"pyemu"), os.path.join(d,"pyemu"))
    except:
        pass

# --------------------------
# GRID (uses your wahp module)
# --------------------------
def make_grid(boundary_shp: str, out_grid_shp: str, grid_size_ft: float, rotation_degrees: float, target_epsg: int):
    import importlib, inspect
    mod_name = "wahp02_model_build"
    wahp_dir = os.path.abspath(os.path.join("..", "wahp"))
    if wahp_dir not in sys.path:
        sys.path.insert(0, wahp_dir)
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    mod = importlib.reload(mod)
    print("Loaded:", mod.__file__)
    print("Signature:", inspect.signature(mod.create_grid_in_feet))

    trimmed_gdf, ll_corner, angrot, nrow, ncol = mod.create_grid_in_feet(
        shapefile_path=boundary_shp,
        grid_size_feet=grid_size_ft,
        output_shapefile_path=out_grid_shp,
        target_crs_epsg=target_epsg,
        rotation_degrees=rotation_degrees,
        max_iterations=100,
        expansion_mode="all",
        columns_to_add=10,
        pad_rows=5,
        pad_cols=5
    )
    print("---- Elk Valley Grid Summary ----")
    print(f"Saved: {out_grid_shp}")
    print(f"angrot (deg): {angrot}")
    print(f"lower-left (rot-aware) corner: {ll_corner}")
    print(f"nrow x ncol: {nrow} x {ncol}")
    print(f"cell size (ft): {grid_size_ft}")
    return ll_corner, nrow, ncol, angrot

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
        "elk",
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

def riv_definition(qa_figs=True, gwf=None, top_clearance_ft: float = 1.0):
    print("creating first pass at riv definition...")

    rivoutdir = os.path.join("..", "..", "gis", "output_shps", "elk", "rivs")
    os.makedirs(rivoutdir, exist_ok=True)

    grd = gpd.read_file(
        os.path.join(
            "..",
            "..",
            "gis",
            "output_shps",
            "elk",
            "elk_2lay_cell_size_660.0ft_epsg2265.grid.shp",
        )
    )

    out_figs = os.path.join("prelim_figs_tables", "figs", "riv")
    os.makedirs(out_figs, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load named river shapefile, normalize fields, classify riv_pac
    # ------------------------------------------------------------------
    riv = gpd.read_file(
        os.path.join("..", "..", "gis", "input_shps", "elk", "nhd_flow_named.shp")
    )
    riv.columns = riv.columns.str.lower()
    riv = riv.to_crs(grd.crs)

    # helper to classify each GNIS name to a river "package"
    def _classify_riv_pac(name: str | None) -> str | None:
        if not isinstance(name, str):
            return None
        n = name.strip().lower()
        # Turtle system: Whisky + Turtle main + branches
        turtle_names = {
            "whisky creek",
            "turtle river",
            "south branch turtle river",
            "north branch turtle river",
        }
        if n in turtle_names:
            return "turtle"
        if n == "little goose river":
            return "goose"
        if n == "hazen brook":
            return "hazen"
        return None

    if "gnis_name" in riv.columns:
        riv["riv_pac"] = riv["gnis_name"].apply(_classify_riv_pac)
    else:
        # fallback: everything unclassified
        riv["riv_pac"] = None

    raster_path = os.path.join("data", "processed", "lf_surfaces", "Topography.tif")

    # ------------------------------------------------------------------
    # 2) Sample elevations along merged river centerlines
    # ------------------------------------------------------------------
    with rasterio.open(raster_path) as src:
        from shapely.ops import linemerge
        from shapely.geometry import Point, MultiPoint
        from geopandas import overlay

        multi_line = riv.unary_union
        seg_merg = linemerge(multi_line)

        distance_interval = 250  # ft along-stream
        points = [
            seg_merg.interpolate(distance)
            for distance in range(0, int(seg_merg.length), distance_interval)
        ]
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=grd.crs)
        points_gdf = points_gdf.reset_index().rename(columns={"index": "pid"})

        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)

        coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
        points_gdf["elevation"] = [val[0] for val in src.sample(coords)]
        points_gdf["elev_ft"] = points_gdf["elevation"]  # topo already in ft

        # merge with grid:
        grd["geom_cpy"] = grd["geometry"]
        mrg = points_gdf.sjoin(grd, how="left", predicate="within")
        mrg = mrg[["node", "row", "col", "i", "j", "elevation", "geom_cpy"]]
        mrg["geometry"] = mrg["geom_cpy"]
        riv_df = mrg.drop(columns=["geom_cpy"])

    # ------------------------------------------------------------------
    # 3) For each row/col keep the lowest elevation point
    # ------------------------------------------------------------------
    riv_df = (
        riv_df.sort_values("elevation")
        .drop_duplicates(subset=["row", "col"], keep="first")
    )
    riv_df = gpd.GeoDataFrame(riv_df, geometry="geometry", crs=grd.crs)

    # ------------------------------------------------------------------
    # 4) Overlay back on original lines to get segment_length + riv_pac
    # ------------------------------------------------------------------
    use_cols = ["fcode", "geometry"]
    if "riv_pac" in riv.columns:
        use_cols.append("riv_pac")
    if "gnis_name" in riv.columns:
        use_cols.append("gnis_name")

    segments = overlay(riv[use_cols], riv_df[["node", "geometry"]], how="intersection")
    segments["segment_length"] = segments.geometry.length
    segments = segments.drop(columns=["geometry"])
    riv_df = riv_df.merge(segments, on="node", how="left")

    # we don't need fcode anymore
    if "fcode" in riv_df.columns:
        riv_df = riv_df.drop(columns=["fcode"])

    # ------------------------------------------------------------------
    # 5) Enforce clearance below top, then assign RIV properties
    # ------------------------------------------------------------------
    riv_df["elevation"] = riv_df["elevation"]  # base elev in ft

    if gwf is not None:
        top = np.asarray(gwf.dis.top.array, float)  # (nrow, ncol)
        clearance = float(top_clearance_ft)
        for idx, row in riv_df.iterrows():
            try:
                i = int(row["i"])
                j = int(row["j"])
            except Exception:
                continue
            if not (0 <= i < top.shape[0] and 0 <= j < top.shape[1]):
                continue
            t_ij = float(top[i, j])
            max_elev = t_ij - clearance
            if np.isfinite(t_ij) and row["elevation"] > max_elev:
                riv_df.at[idx, "elevation"] = max_elev

    # simple geometry assumptions
    riv_df["width"] = 150.0
    riv_df["rbot"] = riv_df["elevation"] - 3.0      # 3 ft thick bed
    riv_df["stage"] = riv_df["elevation"] + 1.5     # 1.5 ft above bed top
    riv_df["cond"] = (
        riv_df["segment_length"] * riv_df["width"] * 0.07
        / (riv_df["stage"] - riv_df["rbot"])
    )  # K = 0.07 ft/d

    # drop very short segments
    riv_df = riv_df[riv_df["segment_length"] > 100]

    # ------------------------------------------------------------------
    # 6) QA plots + shapefile(s)
    # ------------------------------------------------------------------
    if qa_figs and len(riv_df) > 0:
        avg_cond = riv_df["cond"].mean()

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.set_title("Histogram of Initial Estimate\n River Conductances")
        ax.hist(riv_df["cond"], bins=50, color="blue", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Conductance (ft/d)")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.axvline(avg_cond, color="black", linestyle="dashed", linewidth=2)

        props = dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            linewidth=1,
        )
        text_x = avg_cond + avg_cond * 0.35
        text_y = 50
        ax.text(
            text_x,
            text_y,
            f"Average = {avg_cond:,.0f}",
            color="black",
            fontsize=10,
            verticalalignment="center",
            bbox=props,
        )
        ax.annotate(
            "",
            xy=(text_x, text_y),
            xytext=(avg_cond, 50),
            arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->"),
        )
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x))
        )
        plt.savefig(os.path.join(out_figs, "riv_cond_hist.png"), dpi=300, bbox_inches="tight")
        plt.savefig(
            os.path.join(out_figs, "riv_cond_hist.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.close(fig)

        fig, ax = plot_basemap(epsg=2265)
        ax.set_title("River's Initial Conductivity Estimate")
        riv_df.plot(
            column="cond",
            cmap="viridis",
            linewidth=0.5,
            ax=ax,
            edgecolor="none",
        )
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(
                vmin=riv_df["cond"].min(), vmax=riv_df["cond"].max()
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Conductance\n(ft/d)")
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x))
        )
        ax.legend()
        plt.savefig(
            os.path.join(out_figs, "riv_cond_estimate.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(out_figs, "riv_cond_estimate.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.close(fig)

    # main combined shapefile
    nm = "riv_with_init_props.shp"
    riv_df.to_file(os.path.join(rivoutdir, nm))

    # optional: separate QA shapefiles per riv_pac
    for pac in ("turtle", "goose", "hazen"):
        sub = riv_df[riv_df.get("riv_pac") == pac]
        if not sub.empty:
            sub.to_file(os.path.join(rivoutdir, f"riv_{pac}.shp"))

    return riv_df



def drain_definition(qa_figs: bool = False, gwf=None, top_clearance_ft: float = 1.0):
    """
    Build initial DRN definition from nhd_flow_elk.shp and Topography.tif.

    Returns
    -------
    GeoDataFrame
        Columns include at least:
        ['node', 'row', 'col', 'i', 'j',
         'elevation', 'segment_length', 'width', 'rbot', 'stage', 'cond',
         'geometry', 'loc' (if present in source shapefile)]
    """
    print("creating first pass at drn definition...")

    # ------------------------------------------------------------------
    # 0) Paths and output dirs
    # ------------------------------------------------------------------
    drnoutdir = os.path.join("..", "..", "gis", "output_shps", "elk", "drns")
    os.makedirs(drnoutdir, exist_ok=True)

    grd = gpd.read_file(
        os.path.join(
            "..",
            "..",
            "gis",
            "output_shps",
            "elk",
            "elk_2lay_cell_size_660.0ft_epsg2265.grid.shp",
        )
    )

    out_figs = os.path.join("prelim_figs_tables", "figs", "drn")
    os.makedirs(out_figs, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load DRN centerline shapefile & normalize
    # ------------------------------------------------------------------
    drn = gpd.read_file(
        os.path.join("..", "..", "gis", "input_shps", "elk", "nhd_flow_elk.shp")
    )
    drn.columns = drn.columns.str.lower()
    drn = drn.to_crs(grd.crs)

    # ------------------------------------------------------------------
    # 2) Sample elevations along merged drain lines
    # ------------------------------------------------------------------
    raster_path = os.path.join("data", "processed", "lf_surfaces", "Topography.tif")

    with rasterio.open(raster_path) as src:
        from shapely.ops import linemerge

        multi_line = drn.unary_union
        seg_merg = linemerge(multi_line)

        distance_interval = 250.0  # ft along stream/drain
        points = [
            seg_merg.interpolate(distance)
            for distance in range(0, int(seg_merg.length), int(distance_interval))
        ]
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=grd.crs)
        points_gdf = points_gdf.reset_index().rename(columns={"index": "pid"})

        # Make sure sampling CRS matches raster
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)

        coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
        points_gdf["elevation"] = [val[0] for val in src.sample(coords)]
        # Topography.tif is already in ft
        points_gdf["elev_ft"] = points_gdf["elevation"]

        # merge with grid (assign node, i, j, row, col)
        grd["geom_cpy"] = grd["geometry"]
        mrg = points_gdf.sjoin(grd, how="left", predicate="within")
        mrg = mrg[["node", "row", "col", "i", "j", "elevation", "geom_cpy"]]
        mrg["geometry"] = mrg["geom_cpy"]
        drn_df = mrg.drop(columns=["geom_cpy"])

    # ------------------------------------------------------------------
    # 3) One DRN point per cell: keep lowest elevation
    # ------------------------------------------------------------------
    drn_df = (
        drn_df.sort_values("elevation")
        .drop_duplicates(subset=["row", "col"], keep="first")
    )
    drn_df = gpd.GeoDataFrame(drn_df, geometry="geometry", crs=grd.crs)

    # ------------------------------------------------------------------
    # 4) Overlay back on original shapefile to get segment_length + loc
    # ------------------------------------------------------------------
    use_cols = ["fcode", "geometry"]
    if "loc" in drn.columns:
        use_cols.append("loc")
    if "gnis_name" in drn.columns:
        use_cols.append("gnis_name")

    segments = overlay(drn[use_cols], drn_df[["node", "geometry"]], how="intersection")
    segments["segment_length"] = segments.geometry.length
    segments = segments.drop(columns=["geometry"])

    drn_df = drn_df.merge(segments, on="node", how="left")

    # Drop fcode, keep loc if present
    if "fcode" in drn_df.columns:
        drn_df = drn_df.drop(columns=["fcode"])

    # ------------------------------------------------------------------
    # 5) Remove any DRN cells that coincide with RIV cells (by node)
    # ------------------------------------------------------------------
    riv_path = os.path.join(
        "..", "..", "gis", "output_shps", "elk", "rivs", "riv_with_init_props.shp"
    )
    assert os.path.exists(
        riv_path
    ), "Riv shapefile not found, run riv_definition fx first!"
    riv = gpd.read_file(riv_path)

    if "node" in riv.columns:
        drn_df = drn_df[~drn_df["node"].isin(riv["node"])]

    # ------------------------------------------------------------------
    # 6) Enforce clearance below model top, if gwf provided
    # ------------------------------------------------------------------
    drn_df["elevation"] = drn_df["elevation"]  # base elevation (ft)

    if gwf is not None:
        top = np.asarray(gwf.dis.top.array, float)  # (nrow, ncol)
        clearance = float(top_clearance_ft)
        for idx, row in drn_df.iterrows():
            try:
                i = int(row["i"])
                j = int(row["j"])
            except Exception:
                continue
            if not (0 <= i < top.shape[0] and 0 <= j < top.shape[1]):
                continue
            t_ij = float(top[i, j])
            max_elev = t_ij - clearance
            if np.isfinite(t_ij) and row["elevation"] > max_elev:
                drn_df.at[idx, "elevation"] = max_elev

    # ------------------------------------------------------------------
    # 7) Assign DRN properties: width, rbot, stage, cond
    # ------------------------------------------------------------------
    # assume widths of 50 feet:
    drn_df["width"] = 50.0
    # assume drain depths of 5 feet (rbot below elevation):
    drn_df["rbot"] = drn_df["elevation"] - 5.0
    # assume stage 1.0 ft above elevation:
    drn_df["stage"] = drn_df["elevation"] + 1.0
    # K = 0.07 ft/d
    drn_df["cond"] = (
        drn_df["segment_length"] * drn_df["width"] * 0.07
        / (drn_df["stage"] - drn_df["rbot"])
    )

    # drop drain cells with less than 200 ft of segment length
    drn_df = drn_df[drn_df["segment_length"] > 200]

    # ------------------------------------------------------------------
    # 8) QA plots
    # ------------------------------------------------------------------
    if qa_figs and len(drn_df) > 0:
        avg_cond = drn_df["cond"].mean()

        # Histogram of conductance
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.set_title("Histogram of Initial Estimate\n Drain Conductances")
        ax.hist(
            drn_df["cond"],
            bins=50,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Conductance (ft/d)")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        ax.set_axisbelow(True)

        ax.axvline(avg_cond, color="black", linestyle="dashed", linewidth=2)

        props = dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            linewidth=1,
        )
        text_x = avg_cond + avg_cond * 0.35 if avg_cond != 0 else 1.0
        text_y = 50
        ax.text(
            text_x,
            text_y,
            f"Average = {avg_cond:,.0f}",
            color="black",
            fontsize=10,
            verticalalignment="center",
            bbox=props,
        )
        ax.annotate(
            "",
            xy=(text_x, text_y),
            xytext=(avg_cond, 50),
            arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->"),
        )

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x))
        )

        plt.savefig(
            os.path.join(out_figs, "drn_cond_hist.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(out_figs, "drn_cond_hist.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.close(fig)

        # Spatial conductance map
        fig, ax = plot_basemap(epsg=2265)
        ax.set_title("Drain Initial Conductivity Estimate")
        drn_df.plot(
            column="cond",
            cmap="viridis",
            linewidth=0.5,
            ax=ax,
            edgecolor="none",
        )
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(
                vmin=drn_df["cond"].min(), vmax=drn_df["cond"].max()
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Conductance\n(ft/d)")
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x))
        )
        ax.legend()

        plt.savefig(
            os.path.join(out_figs, "drn_cond_estimate.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(out_figs, "drn_cond_estimate.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9) Write shapefiles: full + optional per-loc
    # ------------------------------------------------------------------
    nm = "drn_with_init_props.shp"
    drn_df.to_file(os.path.join(drnoutdir, nm))

    # Optional: write per-loc shapefiles (helps with visual QA, PEST zones)
    if "loc" in drn_df.columns:
        for loc_val, label in [
            ("south", "s"),
            ("midsouth", "ms"),
            ("midnorth", "mn"),
            ("north", "n"),
        ]:
            sub = drn_df[drn_df["loc"].astype(str).str.lower() == loc_val]
            if not sub.empty:
                sub.to_file(
                    os.path.join(drnoutdir, f"drn_{label}.shp")
                )

    return drn_df



def build_ghb_from_shp(
    ghb_shp: str,
    grid_shp: str,
    *,
    gwf=None,                       # <— NEW: pass gwf so we can read DIS top/bottom
    head_offset_ft: float = 0.0,    # add to model top if you want a cushion
    min_clearance_ft: float = 0.1,  # ensure head ≥ cell bottom + clearance
    width_ft: float = 150.0,
    bed_thk_ft: float = 4.5,
    k_bed_ft_per_d: float = 1.0,
    min_seg_length_ft: float = 50.0,
    epsg: int = 2265,
) -> gpd.GeoDataFrame:
    """
    Build GHB cells by intersecting a boundary line with the model grid.

    - head = DIS top[i,j] + head_offset_ft (then lifted if below bottom+clearance)
    - cond = width * segment_length * K_bed / bed_thk
    - returns columns: [i, j, head, cond, segment_length, geometry]
    """
    if not os.path.exists(ghb_shp):
        print(f"[GHB] GHB SHP not found: {ghb_shp}")
        return gpd.GeoDataFrame(columns=["i","j","head","cond","segment_length","geometry"])

    if gwf is None:
        raise ValueError("build_ghb_from_shp now requires gwf to read DIS top/bottom.")

    grd = gpd.read_file(grid_shp).to_crs(epsg=epsg)
    line_src = gpd.read_file(ghb_shp).to_crs(grd.crs)

    # ---- extract linework robustly
    uu = line_src.unary_union
    line_geoms = []
    if uu.geom_type == "LineString":
        line_geoms = [uu]
    elif uu.geom_type == "MultiLineString":
        line_geoms = list(uu.geoms)
    elif uu.geom_type == "GeometryCollection":
        for g in uu.geoms:
            if g.geom_type == "LineString":
                line_geoms.append(g)
            elif g.geom_type == "MultiLineString":
                line_geoms.extend(list(g.geoms))
    if not line_geoms:
        print("[GHB] No linework found in boundary shapefile.")
        return gpd.GeoDataFrame(columns=["i","j","head","cond","segment_length","geometry"], crs=grd.crs)

    # try to merge connected parts; fall back if not possible
    try:
        merged = linemerge(MultiLineString(line_geoms))
        if merged.geom_type == "LineString":
            line_geoms = [merged]
        elif merged.geom_type == "MultiLineString":
            line_geoms = list(merged.geoms)
    except Exception:
        pass

    # ---- per-cell segments
    line_gdf = gpd.GeoDataFrame(geometry=line_geoms, crs=grd.crs)
    inter = gpd.overlay(line_gdf[["geometry"]], grd[["i","j","geometry"]], how="intersection")
    if inter.empty:
        print("[GHB] No intersections between boundary line and grid.")
        return gpd.GeoDataFrame(columns=["i","j","head","cond","segment_length","geometry"], crs=grd.crs)

    inter["segment_length"] = inter.geometry.length
    inter = inter.loc[inter["segment_length"] > float(min_seg_length_ft)].copy()
    if inter.empty:
        print("[GHB] All intersections shorter than min_seg_length_ft.")
        return gpd.GeoDataFrame(columns=["i","j","head","cond","segment_length","geometry"], crs=grd.crs)

    # ---- head from model DIS top (layer 0) + offset, with bottom clearance
    top = np.asarray(gwf.dis.top.array, dtype=float)
    botm0 = np.asarray(gwf.dis.botm.array, dtype=float)[0]  # bottom of layer 0

    inter[["i","j"]] = inter[["i","j"]].astype(int)
    ij = inter[["i","j"]].to_numpy()
    base_head = top[ij[:,0], ij[:,1]] + float(head_offset_ft)
    cellbot   = botm0[ij[:,0], ij[:,1]]

    # enforce head ≥ bottom + clearance
    head = np.maximum(base_head, cellbot + float(min_clearance_ft))

    # ---- conductance as before
    cond = (float(width_ft) * inter["segment_length"].to_numpy() * float(k_bed_ft_per_d)) / max(0.1, float(bed_thk_ft))

    ghb_df = inter.copy()
    ghb_df["head"] = head
    ghb_df["cond"] = cond

    return gpd.GeoDataFrame(ghb_df[["i","j","head","cond","segment_length","geometry"]], crs=grd.crs)

def fix_drn_to_cellbottom(gwf, drn_df, layer=0, clearance_ft=0.1, mode="raise"):
    """
    Ensure DRN elevation >= cell bottom + clearance.
    mode="raise"  -> lift offending elevations just above bottom (keeps rows)
    mode="drop"   -> remove offending rows

    drn_df must have integer i,j and columns: ["i","j","stage","cond"] 
    (or use whatever you named drain "elevation" — adjust below if needed).
    """
    import numpy as np
    import pandas as pd

    if drn_df is None or len(drn_df) == 0:
        return drn_df

    df = drn_df.copy()
    # If your column is named 'stage' for drains, that's the elevation used by DRN.
    elev_col = "stage"   # change to "elev" if you used that name
    df[["i","j"]] = df[["i","j"]].astype(int)

    botm = np.asarray(gwf.dis.botm.array)[layer]  # (nrow, ncol)
    cellbot = botm[df["i"], df["j"]].astype(float)

    ok = df[elev_col].astype(float) >= (cellbot + float(clearance_ft))

    if mode == "drop":
        before = len(df)
        df = df.loc[ok].copy()
        print(f"[DRN] kept {len(df):,} / {before:,} cells; dropped {before - len(df):,} below cell bottom.")
        return df

    # mode == "raise"
    elev_old = df[elev_col].to_numpy(dtype=float)
    elev_new = np.maximum(elev_old, cellbot + float(clearance_ft))
    changed = (elev_new != elev_old).sum()
    df[elev_col] = elev_new
    if changed:
        print(f"[DRN] raised elevation in {int(changed)} cell(s) to clear cell bottoms.")
    return df

def build_drn2_dataframe_from_seep(
    seep_shp: str,
    grid_shp: str,
    dem_tif: str,
    *,
    width_ft: float = 50.0,
    stage_offset_ft: float = 1.0,
    depth_ft: float = 5.0,
    sample_dx_ft: float = 250.0,
    min_seg_len_ft: float = 200.0,
    epsg: int = 2265,
):
    """
    Returns GeoDataFrame with i, j, stage, cond, segment_length, geometry.
    Robust to LineString/MultiLineString/GeometryCollection seepage inputs.
    """

    grd = gpd.read_file(grid_shp).to_crs(epsg=epsg)
    seep = gpd.read_file(seep_shp).to_crs(grd.crs)

    # ---- collect line-like parts safely
    uu = seep.unary_union
    line_geoms = []
    if uu.geom_type == "LineString":
        line_geoms = [uu]
    elif uu.geom_type == "MultiLineString":
        line_geoms = list(uu.geoms)
    elif uu.geom_type == "GeometryCollection":
        line_geoms = [g for g in uu.geoms if g.geom_type in ("LineString", "MultiLineString")]
        # flatten nested multilines
        flat = []
        for g in line_geoms:
            if isinstance(g, MultiLineString):
                flat.extend(list(g.geoms))
            else:
                flat.append(g)
        line_geoms = flat
    else:
        print(f"[DRN2] No linework found in {seep_shp} (got {uu.geom_type}).")
        return gpd.GeoDataFrame(columns=["i","j","stage","cond","segment_length","geometry"], crs=grd.crs)

    if not line_geoms:
        print("[DRN2] Seep line merged geometry is empty.")
        return gpd.GeoDataFrame(columns=["i","j","stage","cond","segment_length","geometry"], crs=grd.crs)

    # Try to merge connected parts; keep unmerged parts too
    try:
        merged = linemerge(MultiLineString(line_geoms))
        if merged.geom_type == "LineString":
            parts = [merged]
        elif merged.geom_type == "MultiLineString":
            parts = list(merged.geoms)
        else:
            parts = line_geoms
    except Exception:
        parts = line_geoms  # fall back

    # ---- densify: points every ~sample_dx_ft along each part
    pts = []
    for ls in parts:
        if not isinstance(ls, LineString) or ls.is_empty or ls.length == 0:
            continue
        dists = np.arange(0.0, float(ls.length), float(sample_dx_ft))
        pts.extend(ls.interpolate(d) for d in dists)
    if not pts:
        print("[DRN2] No points generated from seep linework.")
        return gpd.GeoDataFrame(columns=["i","j","stage","cond","segment_length","geometry"], crs=grd.crs)

    pts_gdf = gpd.GeoDataFrame(geometry=pts, crs=grd.crs).reset_index().rename(columns={"index":"pid"})

    # ---- sample DEM (assumed feet)
    with rasterio.open(dem_tif) as src:
        pts_dem = pts_gdf.to_crs(src.crs) if pts_gdf.crs != src.crs else pts_gdf
        coords = [(p.x, p.y) for p in pts_dem.geometry]
        elev_ft = np.array([v[0] for v in src.sample(coords)], dtype=float)
    pts_gdf["elev_ft"] = elev_ft

    # ---- map to grid cells; keep lowest elevation per (i,j)
    pts_in_grid = gpd.sjoin(pts_gdf.to_crs(grd.crs), grd[["i","j","geometry"]], how="left", predicate="within")
    pts_in_grid = pts_in_grid.dropna(subset=["i","j"])
    cell_min = pts_in_grid.sort_values("elev_ft").drop_duplicates(subset=["i","j"], keep="first")

    # ---- segment length inside each cell (use original seep geometries)
    inter = overlay(seep[["geometry"]], grd[["i","j","geometry"]], how="intersection")
    inter["segment_length"] = inter.geometry.length

    df = cell_min.merge(inter[["i","j","segment_length"]], on=["i","j"], how="left")
    df = df.dropna(subset=["segment_length"])
    df = df.loc[df["segment_length"] > float(min_seg_len_ft)].copy()

    # ---- DRN attributes
    df["stage"] = df["elev_ft"] + float(stage_offset_ft)
    df["cond"]  = (df["segment_length"] * float(width_ft) * 1.0) / max(0.1, float(depth_ft))

    return gpd.GeoDataFrame(df, geometry="geometry", crs=grd.crs)

# --------------------------
#Drain and GHB helpers
# --------------------------

def build_ag_drn_from_points(
    gwf,
    *,
    points_shp: str,
    grid_shp: str,
    layer_k: int = 0,             # 0-based layer index
    stage_offset_ft: float = 2.0,  # stage = top - stage_offset_ft
    bed_thk_ft: float = 5.0,       # thickness of drain "bed" used only for conductance formula
    k_bed_ft_per_d: float = 1.0,   # hydraulic conductivity of drain bed
    width_ft: float = 50.0,        # effective drain width across the cell
    eff_length_ft: float | None = None,  # if None, uses min(delr, delc)
    clearance_ft: float = 0.10,    # ensure stage >= botm + clearance_ft
    drop_if_conflict_cells: set[tuple[int,int]] | None = None,  # {(i,j), ...}
    epsg: int | None = None,       # reproject points to this EPSG before join (optional)
) -> tuple["gpd.GeoDataFrame", dict]:
    """
    Build a static DRN package from point drains. Intersects points with grid cells,
    sets stage = top - stage_offset_ft, and computes conductance as:
        cond = (width * eff_length * k_bed) / bed_thk

    Returns:
        (ag_drn_gdf, drn_dict)  where drn_dict is {kper: [((k,i,j), stage, cond), ...]}
    """
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point

    # --- load inputs
    grid = gpd.read_file(grid_shp)
    pts  = gpd.read_file(points_shp)

    # CRS harmonization
    if epsg is not None:
        target = f"EPSG:{epsg}"
        if grid.crs is None:
            grid.set_crs(target, inplace=True)
        elif str(grid.crs) != target:
            grid = grid.to_crs(target)
        if pts.crs is None:
            pts.set_crs(target, inplace=True)
        elif str(pts.crs) != target:
            pts = pts.to_crs(target)
    else:
        # try to coerce to grid CRS
        if pts.crs is None and grid.crs is not None:
            pts.set_crs(grid.crs, inplace=True)
        elif grid.crs is not None and str(pts.crs) != str(grid.crs):
            pts = pts.to_crs(grid.crs)

    # --- spatial join: assign each point to a single grid cell
    # use 'intersects' so points on boundaries still match a cell
    j = gpd.sjoin(pts, grid[["row", "col", "i", "j", "geometry"]], how="inner", predicate="intersects")

    if j.empty:
        raise ValueError("No point drains intersect the model grid (check CRS/extent).")

    # If multiple matches for same original point (on cell edge), keep first
    if "index_left" in j.columns:
        j = j.sort_values(["index_left"]).drop_duplicates("index_left")

    # --- model arrays & geometry
    top   = np.asarray(gwf.dis.top.array, float)
    botm  = np.asarray(gwf.dis.botm.array, float)
    idom  = np.asarray(gwf.dis.idomain.array, int)
    nlay, nrow, ncol = idom.shape
    if not (0 <= layer_k < nlay):
        raise ValueError(f"layer_k={layer_k} out of range [0..{nlay-1}]")

    # default effective length = min(delr, delc)
    delr = float(np.atleast_1d(gwf.dis.delr.array).min()) if hasattr(gwf.dis, "delr") else float(width_ft)
    delc = float(np.atleast_1d(gwf.dis.delc.array).min()) if hasattr(gwf.dis, "delc") else float(width_ft)
    if eff_length_ft is None:
        eff_len = min(delr, delc)
    else:
        eff_len = float(eff_length_ft)

    # optional conflicts set
    conflicts = drop_if_conflict_cells or set()

    # --- build rows in lock-step with a geometry list
    rows = []
    geoms = []

    # Pre-get centroids of grid cells we matched to keep alignment exact
    # We'll use j.index_right to find the matched grid feature row
    # and store its centroid in the same loop that builds each row.
    grid_geoms = grid.geometry

    # loop matches
    for _, r in j.iterrows():
        ii = int(r["i"]); jj = int(r["j"])
        if not (0 <= ii < nrow and 0 <= jj < ncol):
            continue
        # active?
        if idom[layer_k, ii, jj] != 1:
            continue
        # conflict filter?
        if (ii, jj) in conflicts:
            continue

        # stage & cond
        stg = float(top[ii, jj]) - float(stage_offset_ft)
        min_stage = float(botm[layer_k, ii, jj]) + float(clearance_ft)
        if stg < min_stage:
            stg = min_stage

        cond = (float(width_ft) * float(eff_len) * float(k_bed_ft_per_d)) / float(bed_thk_ft)

        rows.append(dict(
            k=layer_k, i=ii, j=jj,
            stage=stg, cond=float(cond),
        ))

        # geometry aligned with the kept row
        # use the *cell centroid* for a clean map symbol
        # r["index_right"] is the matched grid row
        gidx = r["index_right"]
        geoms.append(grid_geoms.iloc[gidx].centroid)

    if not rows:
        raise ValueError("No valid drain cells were created (all filtered by idomain/conflicts/clearance?).")

    # --- GeoDataFrame with geometry length matching rows
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=grid.crs)

    # --- static per-period dict (no transience)
    nper = int(gwf.simulation.get_package("tdis").nper.get_data())
    per = [((int(v["k"]), int(v["i"]), int(v["j"])), float(v["stage"]), float(v["cond"])) for v in rows]
    drn_dict = {sp: per for sp in range(nper)}

    return gdf, drn_dict

def _segments_from_line(grid_shp: str, line_shp: str, *, min_seg_len_ft: float, epsg: int = 2265) -> gpd.GeoDataFrame:
    """Return per-cell segments: columns ['i','j','segment_length','geometry'] in grid CRS."""
    grd  = gpd.read_file(grid_shp).to_crs(epsg=epsg)
    line = gpd.read_file(line_shp).to_crs(grd.crs)

    uu = line.unary_union
    parts: List[LineString] = []
    if uu.geom_type == "LineString":
        parts = [uu]
    elif uu.geom_type == "MultiLineString":
        parts = list(uu.geoms)
    elif uu.geom_type == "GeometryCollection":
        for g in uu.geoms:
            if g.geom_type == "LineString": parts.append(g)
            elif g.geom_type == "MultiLineString": parts.extend(list(g.geoms))
    if not parts:
        return gpd.GeoDataFrame(columns=["i","j","segment_length","geometry"], crs=grd.crs)

    try:
        merged = linemerge(MultiLineString(parts))
        parts = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    except Exception:
        pass

    line_gdf = gpd.GeoDataFrame(geometry=parts, crs=grd.crs)
    inter = gpd.overlay(line_gdf[["geometry"]], grd[["i","j","geometry"]], how="intersection")
    if inter.empty:
        return gpd.GeoDataFrame(columns=["i","j","segment_length","geometry"], crs=grd.crs)

    inter["segment_length"] = inter.geometry.length
    inter = inter.loc[inter["segment_length"] > float(min_seg_len_ft)].copy()
    inter[["i","j"]] = inter[["i","j"]].astype(int)
    return inter[["i","j","segment_length","geometry"]].copy()

def _idw_on_grid(xc: np.ndarray, yc: np.ndarray,
                 pts_xy: np.ndarray, pts_val: np.ndarray,
                 power: float = 2.0, k: Optional[int] = 12) -> np.ndarray:
    """Simple IDW onto (xc,yc)."""
    nrow, ncol = xc.shape
    G = np.column_stack([xc.ravel(), yc.ravel()])
    P = pts_xy
    V = pts_val
    d2 = (G[:,None,0]-P[None,:,0])**2 + (G[:,None,1]-P[None,:,1])**2
    if k is not None and k < P.shape[0]:
        idx = np.argpartition(d2, k-1, axis=1)[:, :k]
        rows = np.arange(d2.shape[0])[:,None]
        d2 = d2[rows, idx]
        V  = V[idx]
    w   = 1.0 / np.maximum(d2, 1e-12)**(power/2.0)
    num = np.nansum(w*V, axis=1)
    den = np.nansum(w,   axis=1)
    out = (num/den).reshape(nrow, ncol)
    return out

def _sp_dates_from_tdis(tdis) -> List[pd.Timestamp]:
    start = pd.to_datetime(str(tdis.start_date_time.data)).date()
    start = pd.to_datetime(start)
    pddata = list(tdis.perioddata.get_data())
    dates = [start]
    cur = start
    for perlen, *_ in pddata[:-1]:
        cur = cur + pd.to_timedelta(float(perlen), unit="D")
        dates.append(cur)
    return dates

def build_wl_surfaces(
    gwf,
    *,
    sites_csv: str,
    wl_xlsx: str,
    grid_shp: str,  # used only for plotting boundary in debug
    topo_tif: Optional[str] = None,  # OPTIONAL: if omitted, use gwf.dis.top array
    annual: bool = False,
    pre2000_cutoff_year: int = 2000,
    post2000_roll_days: int = 60,
    use_median: bool = True,  # True=median (matches your obs), False=mean
    idw_power: float = 2.0,
    idw_k: Optional[int] = 12,
    min_sites_per_sp: int = 10,  # threshold for computing a period-specific surface
    start_interp_year: int = 1990,  # build IDW from this year...
    end_interp_year: int = 2024,    # ...through this year (inclusive). After this, use mean.
    # ---- DEBUG CONTROLS ----
    debug: bool = False,
    debug_outdir: str = os.path.join("figs", "wl_debug"),
    debug_dump_csv: bool = False,
) -> Tuple[List[pd.Timestamp], Dict[int, np.ndarray]]:
    """
    Builds groundwater-elevation (NAVD88) surfaces by first interpolating Depth-to-Water (DTW, ft bgs)
    from 'elk_valley_water_level_data.csv' (or similarly structured file) and then converting to elevation via:
        GW_elev = TOPO(x,y) - DTW

    Policy:
      - For SP dates in [start_interp_year .. end_interp_year], build an IDW *DTW* surface if there are
        > min_sites_per_sp distinct sites aligned to that SP. Otherwise, fill with the mean of all "good"
        IDW *DTW* surfaces within that window.
      - For SP dates < start_interp_year OR > end_interp_year: always use that same window-mean DTW surface.
      - Finally convert all DTW surfaces to groundwater elevation by subtracting from topography.

    Returns (sp_starts, {sp -> 2D groundwater-elevation surface in NAVD88}).
    """

    # --- stress periods
    tdis = gwf.simulation.get_package("tdis")
    sp_starts = _sp_dates_from_tdis(tdis)
    nper = len(sp_starts)
    start_dt = pd.Timestamp(f"{int(start_interp_year)}-01-01")
    end_dt = pd.Timestamp(f"{int(end_interp_year)}-12-31")

    # --- model grid (structured DIS assumed here)
    mg = gwf.modelgrid
    grid_type = getattr(mg, "grid_type", "").lower()
    if "structured" not in grid_type:
        raise ValueError("This plotting path is for DIS (structured). If you need DISV, I can switch to polygon draw.")

    nrow, ncol = int(mg.nrow), int(mg.ncol)
    xc = np.asarray(mg.xcellcenters)  # (nrow, ncol)
    yc = np.asarray(mg.ycellcenters)  # (nrow, ncol)

    # rotation-aware vertices for pcolormesh
    XE = YE = None
    try:
        XV = np.asarray(mg.xvertices)
        YV = np.asarray(mg.yvertices)
        if XV.shape == (nrow+1, ncol+1) and YV.shape == (nrow+1, ncol+1):
            XE, YE = XV, YV
    except Exception:
        XE = YE = None

    if XE is None or YE is None:
        # fallback (non-rotated)
        def _edges_from_centers_1d(c1d: np.ndarray) -> np.ndarray:
            c = c1d.astype(float).ravel()
            inc = (c[-1] > c[0])
            if not inc:
                c = c[::-1]
            mids = 0.5*(c[1:]+c[:-1])
            first = c[0] - (mids[0]-c[0])
            last = c[-1] + (c[-1]-mids[-1])
            e = np.concatenate([[first], mids, [last]])
            return e if inc else e[::-1]
        xe = _edges_from_centers_1d(xc[0, :])
        ye = _edges_from_centers_1d(yc[:, 0])
        XE, YE = np.meshgrid(xe, ye)  # shape (nrow+1, ncol+1)

    def _ensure_rowcol(arr: np.ndarray) -> np.ndarray:
        if arr.shape == (nrow, ncol): return arr
        if arr.shape == (ncol, nrow): return arr.T
        if arr.T.shape == (nrow, ncol): return arr.T
        raise ValueError(f"Surface shape {arr.shape} ≠ ({nrow},{ncol}).")

    # =========================
    # Load & normalize SITES
    # =========================
    sites = pd.read_csv(sites_csv)
    orig_cols_sites = list(sites.columns)
    sites.columns = (
        sites.columns.astype(str)
        .str.strip().str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True).str.lower()
    )

    if "drop_flag" in sites.columns:
        sites = sites.loc[~sites["drop_flag"].astype(str).str.upper().isin(["TRUE", "T", "YES", "1"])].copy()

    col_map_sites = {
        "site_index": "site_id", "location": "location",
        "x": "x_2265", "y": "y_2265", "x_easting": "x_2265", "y_northing": "y_2265",
        "x2265": "x_2265", "y2265": "y_2265", "x_2265": "x_2265", "y_2265": "y_2265",
    }
    for c in list(sites.columns):
        if c in col_map_sites and col_map_sites[c] not in sites.columns:
            sites.rename(columns={c: col_map_sites[c]}, inplace=True)

    if not {"x_2265", "y_2265"}.issubset(sites.columns):
        raise ValueError(f"sites_csv needs x_2265/y_2265. Original: {orig_cols_sites} → normalized: {list(sites.columns)}")

    sites["x_2265"] = pd.to_numeric(sites["x_2265"], errors="coerce")
    sites["y_2265"] = pd.to_numeric(sites["y_2265"], errors="coerce")
    sites = sites.dropna(subset=["x_2265", "y_2265"])
    if sites.empty:
        raise ValueError("No valid site coordinates after cleaning.")

    # =========================
    # Load & normalize WL (DTW)
    # =========================
    wlx = pd.read_csv(wl_xlsx)
    orig_cols_wlx = list(wlx.columns)
    wlx.columns = (
        wlx.columns.astype(str)
        .str.strip().str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True).str.lower()
    )

    # map possible column names to a standard set
    col_map_wlx = {
        "site_index": "site_id", "location": "location",
        "date_measured": "date_meas", "date": "date_meas",
        # depth to water possibilities
        "dtw_model_top": "dtw"
    }
    for c in list(wlx.columns):
        if c in col_map_wlx and col_map_wlx[c] not in wlx.columns:
            wlx.rename(columns={c: col_map_wlx[c]}, inplace=True)

    key_col = "site_id" if ("site_id" in wlx.columns and "site_id" in sites.columns) else "location"
    if key_col not in wlx.columns or key_col not in sites.columns:
        raise ValueError(f"Need shared key (site_id or location). sites: {list(sites.columns)} wl: {list(wlx.columns)}")

    wlx["site_key"] = wlx[key_col].astype(str)
    sites["site_key"] = sites[key_col].astype(str)
    keep = set(sites["site_key"].unique())
    wlx = wlx[wlx["site_key"].isin(keep)].copy()

    if "date_meas" not in wlx.columns or "dtw" not in wlx.columns:
        raise ValueError(
            f"WL file needs 'date_meas' and a depth-to-water column. "
            f"Original: {orig_cols_wlx} → normalized: {list(wlx.columns)}"
        )

    wlx["date_meas"] = pd.to_datetime(wlx["date_meas"], errors="coerce")
    wlx["dtw"] = pd.to_numeric(wlx["dtw"], errors="coerce")  # ft below ground surface
    wlx = wlx.dropna(subset=["date_meas", "dtw"])
    if wlx.empty:
        raise RuntimeError("No valid depth-to-water measurements after cleaning.")

    xy_map = sites.set_index("site_key")[["x_2265", "y_2265"]].to_dict(orient="index")

    # -------- TOPO sampling setup --------
    topo_from_grid = topo_tif is None
    if topo_from_grid:
        # use the model's current top array as topography
        topo_grid = np.asarray(gwf.dis.top.array, dtype=float)
        if topo_grid.shape != (nrow, ncol):
            topo_grid = _ensure_rowcol(topo_grid)
        # for point sampling (debug), use nearest cell center
        from scipy.spatial import cKDTree as KDTree
        _tree = KDTree(np.c_[xc.ravel(), yc.ravel()])
        def _sample_topo_points(Px: np.ndarray, Py: np.ndarray) -> np.ndarray:
            _, idx = _tree.query(np.c_[Px, Py], k=1)
            return topo_grid.ravel()[idx]
    else:
        # read from raster
        try:
            import rasterio
            from rasterio.transform import rowcol
        except Exception as e:
            raise RuntimeError(f"rasterio is required to sample topography from 'topo_tif': {e}")

        def _sample_tif_grid(path: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            with rasterio.open(path) as src:
                rows, cols = rowcol(src.transform, X.ravel(), Y.ravel(), op=round)
                rows = np.clip(rows, 0, src.height - 1)
                cols = np.clip(cols, 0, src.width - 1)
                data = src.read(1)
                vals = data[rows, cols].reshape(X.shape).astype(float)
                if src.nodata is not None:
                    vals[vals == src.nodata] = np.nan
                return vals

        def _sample_tif_points(path: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            with rasterio.open(path) as src:
                rows, cols = rowcol(src.transform, x, y, op=round)
                rows = np.clip(rows, 0, src.height - 1)
                cols = np.clip(cols, 0, src.width - 1)
                data = src.read(1).astype(float)
                vals = data[rows, cols]
                if src.nodata is not None:
                    vals[vals == src.nodata] = np.nan
                return vals

        topo_grid = _sample_tif_grid(topo_tif, xc, yc)
        def _sample_topo_points(Px: np.ndarray, Py: np.ndarray) -> np.ndarray:
            return _sample_tif_points(topo_tif, Px, Py)

    # -------- Align DTW series to SPs --------
    def _align_series_to_sps(s: pd.Series) -> pd.Series:
        s = s.sort_index()
        if annual:
            agg = s.groupby(s.index.year).median() if use_median else s.groupby(s.index.year).mean()
            out = {}
            for sp, dt in enumerate(sp_starts):
                if dt.year in agg.index:
                    out[sp] = float(agg.loc[dt.year])
            return pd.Series(out, dtype=float)

        # pre-2000: annual; post-2000: rolling smooth
        pre = s[s.index.year < pre2000_cutoff_year]
        post = s[s.index.year >= pre2000_cutoff_year]

        pre_agg = pre.groupby(pre.index.year).median() if use_median else pre.groupby(pre.index.year).mean()
        if not pre_agg.empty:
            pre_agg.index = pd.to_datetime([f"{int(y)}-07-01" for y in pre_agg.index])

        post_sm = (
            post.rolling(f"{post2000_roll_days}D", center=True).median()
            if use_median else
            post.rolling(f"{post2000_roll_days}D", center=True).mean()
        )

        smooth = pd.concat([pre_agg, post_sm]).dropna()
        if smooth.empty:
            return pd.Series(dtype=float)

        if smooth.index.has_duplicates:
            smooth = (smooth.groupby(level=0).median() if use_median else smooth.groupby(level=0).mean())
        smooth = smooth[~smooth.index.isna()].sort_index()

        sim_idx = pd.DatetimeIndex(sp_starts)
        if smooth.index.has_duplicates:
            smooth = smooth[~smooth.index.duplicated(keep="last")]

        aligned = smooth.reindex(sim_idx, method="nearest", tolerance=pd.Timedelta(days=183))
        mask = ~aligned.isna()
        if not mask.any():
            return pd.Series(dtype=float)
        sp_idxs = np.where(mask.to_numpy())[0]
        return pd.Series(aligned[mask].to_numpy(), index=sp_idxs, dtype=float)

    # collect DTW points per SP
    by_sp_pts: Dict[int, List[Tuple[float, float, float, str]]] = {}
    for skey, g in wlx.groupby("site_key"):
        s = g.set_index("date_meas")["dtw"].astype(float)  # DTW time series (ft bgs)
        a = _align_series_to_sps(s)
        if a.empty or skey not in xy_map:
            continue
        xpt = float(xy_map[skey]["x_2265"])
        ypt = float(xy_map[skey]["y_2265"])
        for sp, val in a.items():
            by_sp_pts.setdefault(sp, []).append((xpt, ypt, float(val), skey))  # store DTW value

    # IDW helper over grid (returns array coerced to (nrow,ncol))
    def _make_idw(plist):
        P = np.array([(p[0], p[1]) for p in plist], dtype=float)
        V = np.array([p[2] for p in plist], dtype=float)  # DTW values
        k_use = None if (idw_k is None or len(P) <= 3) else min(idw_k, len(P))
        return _idw_on_grid(xc, yc, P, V, power=idw_power, k=k_use)

    # Build per-SP DTW surfaces inside the window; track "good" IDW within window
    dtw_surfaces: Dict[int, np.ndarray] = {}
    good_dtw_surfs: List[np.ndarray] = []
    surf_meta: Dict[int, Dict[str, Any]] = {}

    for sp, dt in enumerate(sp_starts):
        if not (start_dt <= dt <= end_dt):
            # fill later with window mean
            surf_meta[sp] = {"method": "window_mean_fill", "n_sites": len({p[3] for p in by_sp_pts.get(sp, [])}), "dt": dt}
            continue

        plist = by_sp_pts.get(sp, [])
        unique_sites = len({p[3] for p in plist})
        if unique_sites > min_sites_per_sp:
            surf = _ensure_rowcol(_make_idw(plist))  # DTW surface
            dtw_surfaces[sp] = surf
            good_dtw_surfs.append(surf)
            surf_meta[sp] = {"method": "idw", "n_sites": unique_sites, "dt": dt}
        else:
            surf_meta[sp] = {"method": "window_mean_fill", "n_sites": unique_sites, "dt": dt}

    # Compute window-mean DTW surface from good (1990–2024) IDWs
    if good_dtw_surfs:
        window_mean_dtw_surface = np.nanmean(np.stack(good_dtw_surfs, axis=0), axis=0)
    else:
        # emergency: build a static IDW from all available aligned DTW points inside the window
        pooled = []
        for sp, plist in by_sp_pts.items():
            if start_dt <= sp_starts[sp] <= end_dt:
                pooled.extend([(x, y, v) for (x, y, v, _sid) in plist])
        if not pooled:
            raise RuntimeError("No aligned points in 1990–2024 window to build a fallback mean DTW surface.")
        P = np.array([(x, y) for x, y, _ in pooled], dtype=float)
        V = np.array([v for *_xy, v in pooled], dtype=float)
        window_mean_dtw_surface = _ensure_rowcol(_idw_on_grid(xc, yc, P, V, power=idw_power, k=None))

    # Fill DTW for everything outside the window, and window SPs that didn’t meet the threshold
    for sp, dt in enumerate(sp_starts):
        if sp not in dtw_surfaces:
            dtw_surfaces[sp] = window_mean_dtw_surface

    # Convert DTW → groundwater elevation on grid
    def _dtw_to_elev(dtw_arr: np.ndarray) -> np.ndarray:
        # GW elev = topo - DTW; ensure shapes match
        return topo_grid - _ensure_rowcol(dtw_arr)

    surfaces: Dict[int, np.ndarray] = {sp: _dtw_to_elev(arr) for sp, arr in dtw_surfaces.items()}
    window_mean_surface = _dtw_to_elev(window_mean_dtw_surface)

    # ====================
    # DEBUG PLOTTING (every SP)
    # ====================
    if debug:
        os.makedirs(debug_outdir, exist_ok=True)
        boundary_gdf = None
        if gpd is not None and isinstance(grid_shp, str) and os.path.exists(grid_shp):
            try:
                boundary_gdf = gpd.read_file(grid_shp)
                if len(boundary_gdf) > 0:
                    boundary_gdf = boundary_gdf.to_crs(epsg=2265).dissolve()
            except Exception:
                boundary_gdf = None

        def _plot_surface(sp: int, surface: np.ndarray, points: List[Tuple[float, float, float, str]], meta: Dict[str, Any]):
            """
            points contain DTW; for plotting & residuals we convert those to GW elevation via topo(point) - DTW(point).
            """
            dt = meta.get("dt")
            method = meta.get("method", "?")
            n_sites = int(meta.get("n_sites", 0))

            Z = _ensure_rowcol(surface)  # GW elevation
            vmin = np.nanpercentile(Z, 2)
            vmax = np.nanpercentile(Z, 98)
            vmean = float(np.nanmean(Z))
            vstd = float(np.nanstd(Z))

            fig = plt.figure(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.25, wspace=0.25)

            ax = fig.add_subplot(gs[0, 0])
            ax.set_facecolor('white')
            pcm = ax.pcolormesh(XE, YE, Z, vmin=vmin, vmax=vmax, shading="flat")
            cmap_for_points = pcm.get_cmap(); norm_for_points = pcm.norm

            if boundary_gdf is not None:
                boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="k")

            if method == "idw" and points:
                Px = np.array([p[0] for p in points])
                Py = np.array([p[1] for p in points])
                Pdtw = np.array([p[2] for p in points], float)
                Ptopo = _sample_topo_points(Px, Py)
                Pv_gw = Ptopo - Pdtw  # convert to GW elevation for consistent plotting
                ax.scatter(Px, Py, c=Pv_gw, s=45, edgecolor='k', linewidths=0.6,
                           cmap=cmap_for_points, norm=norm_for_points, zorder=3)
                for (x, y, val_gw) in zip(Px[:12], Py[:12], Pv_gw[:12]):
                    if np.isfinite(val_gw):
                        ax.text(x, y, f"{int(round(val_gw))}", fontsize=7, ha="center", va="bottom")

            cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.set_facecolor('white')
            cbar.set_label("Groundwater elevation (ft NAVD88)")

            title_suffix = "" if method == "idw" else " (avg fill; points not shown)"
            ax.set_title(f"GW Elev Surface — SP {sp} ({dt.date()}), method={method}, sites={n_sites}{title_suffix}")
            ax.set_xlabel("X (ft, EPSG:2265)"); ax.set_ylabel("Y (ft, EPSG:2265)")

            axh = fig.add_subplot(gs[1, 0]); axh.set_facecolor('white')
            axh.hist(Z[np.isfinite(Z)], bins=40)
            axh.set_xlabel("GW Elev (ft NAVD88)"); axh.set_ylabel("Cell count")
            axh.set_title(f"Surface value dist (mean={vmean:,.1f}, std={vstd:,.1f})")

            axt = fig.add_subplot(gs[:, 1]); axt.set_facecolor('white'); axt.axis("off")
            wnd = f"{start_interp_year}–{end_interp_year}"
            if method == "idw" and points:
                Px = np.array([p[0] for p in points])
                Py = np.array([p[1] for p in points])
                Pdtw = np.array([p[2] for p in points], float)
                Ptopo = _sample_topo_points(Px, Py)
                Pv_gw = Ptopo - Pdtw
                ptxt = f"Point GW elev min/median/max: {np.nanmin(Pv_gw):,.1f} / {np.nanmedian(Pv_gw):,.1f} / {np.nanmax(Pv_gw):,.1f}"
            else:
                ptxt = "Points: n/a (avg fill)"

            txt = [
                f"SP: {sp}", f"Date: {dt.date()}", f"Method: {method}", f"Input sites used: {n_sites}",
                f"GW elev min/mean/max: {np.nanmin(Z):,.1f} / {vmean:,.1f} / {np.nanmax(Z):,.1f}",
                f"GW elev p2/p98: {vmin:,.1f} / {vmax:,.1f}",
                ptxt,
                f"Annual mode: {annual}",
                f"DTW pre<{pre2000_cutoff_year} agg: {'median' if use_median else 'mean'}; Post roll={post2000_roll_days}d",
                f"IDW power: {idw_power}, k: {idw_k}",
                f"Min sites per SP: {min_sites_per_sp}",
                f"IDW window: {wnd} (outside = window mean)",
                "Note: interpolated DTW → GW elev via topo - dtw",
            ]
            axt.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=9, family="monospace")

            out_png = os.path.join(debug_outdir, f"gw_elev_surface_SP{sp:04d}_{dt.date()}_{method}.png")
            fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor='white')
            plt.close(fig)

            # residuals vs nearest cell center, using point GW elevation
            if method == "idw" and points:
                try:
                    from scipy.spatial import cKDTree as KDTree
                    tree = KDTree(np.c_[xc.flatten(), yc.flatten()])
                    Pxy = np.c_[Px, Py]
                    _, idx = tree.query(Pxy, k=1)
                    z_hat = Z.flatten()[idx]
                    z_obs = Pv_gw  # observed GW elevation at points
                    resid = z_obs - z_hat
                    pd.DataFrame({
                        "site_key": [p[3] for p in points],
                        "x_2265": Pxy[:, 0], "y_2265": Pxy[:, 1],
                        "gw_obs": z_obs, "gw_surface_nearest": z_hat,
                        "residual_obs_minus_surface": resid
                    }).to_csv(os.path.join(debug_outdir, f"gw_point_residuals_SP{sp:04d}_{dt.date()}.csv"), index=False)
                except Exception as e:
                    print(f"[build_wl_surfaces] Residuals skipped for SP {sp}: {e}")

        os.makedirs(debug_outdir, exist_ok=True)
        boundary_gdf = None
        if gpd is not None and isinstance(grid_shp, str) and os.path.exists(grid_shp):
            try:
                boundary_gdf = gpd.read_file(grid_shp).to_crs(epsg=2265).dissolve()
            except Exception:
                boundary_gdf = None

        for sp in range(nper):
            _plot_surface(sp, surfaces[sp], by_sp_pts.get(sp, []),
                          surf_meta.get(sp, {"dt": sp_starts[sp], "method": "window_mean_fill", "n_sites": 0}))

        # also export the window-mean GW elev surface once (no points)
        _plot_surface(
            sp=-1,
            surface=window_mean_surface,
            points=[],
            meta={"dt": pd.Timestamp(f"{start_interp_year}-01-01"), "method": "window_mean_surface", "n_sites": 0}
        )

    return sp_starts, surfaces

def build_ghb_from_shp_timevarying(
    gwf: flopy.mf6.ModflowGwf,
    *,
    ghb_line_shp: str,
    grid_shp: str,
    wl_surfaces: Dict[int, np.ndarray],
    width_ft: float = 150.0,
    bed_thk_ft: float = 4.5,
    k_bed_ft_per_d: float = 1.0,
    min_seg_len_ft: float = 50.0,
    head_offset_ft: float = 0.0,     # optional cushion above WL
    layer_k: int = 1,                # force layer 2 (0-based)
    clearance_ft: float = 0.1,       # enforce >= bottom + clearance
    epsg: int = 2265,
) -> Dict[int, pd.DataFrame]:
    """
    Returns {sp -> DataFrame[i,j,k,head,cond,segment_length]}.
    Conductance is fixed from geometry; head varies by SP from WL surface.
    """
    segs = _segments_from_line(grid_shp, ghb_line_shp, min_seg_len_ft=min_seg_len_ft, epsg=epsg)
    if segs.empty:
        return {}

    # constant conductance from geometry
    cond = (width_ft * segs["segment_length"].to_numpy() * k_bed_ft_per_d) / max(0.1, bed_thk_ft)
    segs_const = segs.copy()
    segs_const["cond"] = cond
    segs_const["k"] = int(layer_k)

    botm = np.asarray(gwf.dis.botm.array, float)[layer_k]  # bottom of layer 2 (0-based index 1)

    by_sp: Dict[int, pd.DataFrame] = {}
    for sp, surf in wl_surfaces.items():
        if surf is None: continue
        # sample WL at each segment cell
        ii = segs_const["i"].to_numpy()
        jj = segs_const["j"].to_numpy()
        wl = surf[ii, jj]
        h  = wl + float(head_offset_ft)
        # raise to clear cell bottom
        cellbot = botm[ii, jj]
        h = np.maximum(h, cellbot + float(clearance_ft))
        df = segs_const.copy()
        df["head"] = h
        # drop cells with NaN surface (no data)
        df = df.dropna(subset=["head","cond"])
        by_sp[sp] = df[["i","j","k","head","cond","segment_length","geometry"]].copy()

    return by_sp

def build_drn2_from_seep_timevarying(
    gwf: flopy.mf6.ModflowGwf,
    *,
    seep_line_shp: str,
    grid_shp: str,
    wl_surfaces: Dict[int, np.ndarray],
    width_ft: float = 50.0,
    depth_ft: float = 5.0,           # used ONLY for cond calc here (same as your static)
    min_seg_len_ft: float = 200.0,
    stage_offset_ft: float = 0.0,    # add to WL (e.g., +1.0 if desired)
    layer_k: int = 1,
    clearance_ft: float = 0.1,
    epsg: int = 2265,
) -> Dict[int, pd.DataFrame]:
    """
    Returns {sp -> DataFrame[i,j,k,stage,elev(cond is 'cond'),segment_length]} for MF6 DRN.
    Conductance fixed from geometry; stage varies by SP from WL surface.
    """
    segs = _segments_from_line(grid_shp, seep_line_shp, min_seg_len_ft=min_seg_len_ft, epsg=epsg)
    if segs.empty:
        return {}

    # constant conductance (same as your function: cond = seg_len * width * K / depth; K=1)
    cond = (segs["segment_length"].to_numpy() * width_ft * 1.0) / max(0.1, depth_ft)
    segs_const = segs.copy()
    segs_const["cond"] = cond
    segs_const["k"] = int(layer_k)

    botm = np.asarray(gwf.dis.botm.array, float)[layer_k]

    by_sp: Dict[int, pd.DataFrame] = {}
    for sp, surf in wl_surfaces.items():
        if surf is None: continue
        ii = segs_const["i"].to_numpy()
        jj = segs_const["j"].to_numpy()
        stage = surf[ii, jj] + float(stage_offset_ft)
        # ensure drain elevation clears bottom of layer 2
        stage = np.maximum(stage, botm[ii, jj] + float(clearance_ft))
        df = segs_const.copy()
        df["stage"] = stage
        df = df.dropna(subset=["stage","cond"])
        by_sp[sp] = df[["i","j","k","stage","cond","segment_length","geometry"]].copy()

    return by_sp

def attach_timevarying_bc(
    gwf,
    *,
    sp_starts,
    ghb_by_sp=None,
    drn_by_sp=None,
    ghb_pname="ghb_wl",
    drn_pname="drn2_wl",
    ghb_filename=None,
    drn_filename=None,
):
    """Build MF6 GHB/DRN with stress_period_data dicts from per-SP DataFrames."""
    import pandas as pd
    import numpy as np
    import flopy

    # -------- GHB --------
    if ghb_by_sp:
        ghb_spd = {}
        for sp, df in ghb_by_sp.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            # ensure expected columns exist
            missing = [c for c in ["i","j","k","head","cond"] if c not in d.columns]
            if missing:
                # try lowercase/strip fix
                d.columns = [str(c).strip().lower() for c in d.columns]
                missing = [c for c in ["i","j","k","head","cond"] if c not in d.columns]
                if missing:
                    continue  # skip malformed frame

            # type coercions
            d["i"] = pd.to_numeric(d["i"], errors="coerce").astype("Int64")
            d["j"] = pd.to_numeric(d["j"], errors="coerce").astype("Int64")
            d["k"] = pd.to_numeric(d["k"], errors="coerce").astype("Int64")
            d["head"] = pd.to_numeric(d["head"], errors="coerce")
            d["cond"] = pd.to_numeric(d["cond"], errors="coerce")
            d = d.dropna(subset=["i","j","k","head","cond"])
            if d.empty:
                continue

            recs = [((int(k), int(i), int(j)), float(h), float(c))
                    for i, j, k, h, c in d[["i","j","k","head","cond"]].to_numpy()]
            if recs:
                ghb_spd[sp] = recs

        if ghb_spd:
            if gwf.get_package(ghb_pname):
                gwf.remove_package(ghb_pname)
            flopy.mf6.ModflowGwfghb(
                gwf,
                pname=ghb_pname,
                filename=ghb_filename or f"{gwf.name}.{ghb_pname}.ghb",
                stress_period_data=ghb_spd,
            )

    # -------- DRN --------
    if drn_by_sp:
        drn_spd = {}
        for sp, df in drn_by_sp.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            missing = [c for c in ["i","j","k","stage","cond"] if c not in d.columns]
            if missing:
                d.columns = [str(c).strip().lower() for c in d.columns]
                missing = [c for c in ["i","j","k","stage","cond"] if c not in d.columns]
                if missing:
                    continue

            d["i"] = pd.to_numeric(d["i"], errors="coerce").astype("Int64")
            d["j"] = pd.to_numeric(d["j"], errors="coerce").astype("Int64")
            d["k"] = pd.to_numeric(d["k"], errors="coerce").astype("Int64")
            d["stage"] = pd.to_numeric(d["stage"], errors="coerce")
            d["cond"] = pd.to_numeric(d["cond"], errors="coerce")
            d = d.dropna(subset=["i","j","k","stage","cond"])
            if d.empty:
                continue

            recs = [((int(k), int(i), int(j)), float(s), float(c))
                    for i, j, k, s, c in d[["i","j","k","stage","cond"]].to_numpy()]
            if recs:
                drn_spd[sp] = recs

        if drn_spd:
            if gwf.get_package(drn_pname):
                gwf.remove_package(drn_pname)
            flopy.mf6.ModflowGwfdrn(
                gwf,
                pname=drn_pname,
                filename=drn_filename or f"{gwf.name}.{drn_pname}.drn",
                stress_period_data=drn_spd,
            )

# --------------------------
# MF6 BUILDERS
# --------------------------
def build_2layer_model(
    ll_corner, nrow, ncol, delr, delc, rotation_degrees, sim_ws, model_name, exe_name,
    top_elev=1000.0, layer_thicknesses=(80.0, 60.0), icelltype=(1, 0),
    hk=(20.0, 5.0), k33=(2.0, 0.5), idomain_mask=None,
    save_head=True, save_budget=True
):
    nlay = 2
    os.makedirs(sim_ws, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=model_name, exe_name=exe_name, sim_ws=sim_ws,continue_=True)

    # temporary 1-SP TDIS; we replace it later
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)])

    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, save_flows=True,
                               newtonoptions=['UNDER_RELAXATION'])

    ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY", outer_hclose=1e-2, inner_hclose=1e-2)
    sim.register_ims_package(ims, [gwf.name])

    # top/botm
    if np.isscalar(top_elev):
        top = np.full((nrow, ncol), float(top_elev), dtype=float)
    else:
        top = np.asarray(top_elev, dtype=float)

    botm = []
    cur = top.copy()
    for thk in layer_thicknesses:
        lb = cur - float(thk)
        botm.append(lb)
        cur = lb
    botm = np.asarray(botm)

    # idomain
    if idomain_mask is None:
        idomain = np.ones((nlay, nrow, ncol), dtype=int)
    else:
        idomain = np.asarray(idomain_mask, dtype=int)

    # DIS
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delr, delc=delc,
        top=top, botm=botm,
        idomain=idomain,
        xorigin=float(ll_corner[0]),
        yorigin=float(ll_corner[1]),
        angrot=float(rotation_degrees),
        length_units="FEET",
    )

    # NPF
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        icelltype=list(icelltype),
        k=list(hk),
        k33=list(k33),
        k33overk=True,
    )

    # IC (layered start heads ≈ just below tops)
    layer_tops = np.empty((nlay, nrow, ncol), float)
    layer_tops[0] = top
    layer_tops[1] = botm[0]
    strt = np.empty((nlay, nrow, ncol), float)
    for k in range(nlay):
        strt[k] = np.maximum(layer_tops[k] - 0.1, botm[k] + 0.01)
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # OC
    saverec = []
    if save_head:   saverec.append(("HEAD", "ALL"))
    if save_budget: saverec.append(("BUDGET", "ALL"))
    if saverec:
        flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord=f"{model_name}.cbb",
            budgetcsv_filerecord='budget.csv',
            head_filerecord=f"{model_name}.hds",
            saverecord=saverec,
            printrecord=[("BUDGET", "LAST")],
        )
    return sim, gwf

def build_tdis_and_sto(
    sim: flopy.mf6.MFSimulation,
    gwf: flopy.mf6.ModflowGwf,
    mnm: str,
    *,
    start_year: int = 1965,
    end_year: int = 2043,
    annual_only: bool = True,
    monthly_start: int | None = 2000,
    nstp_annual: int = 1,
    nstp_monthly: int = 1,
    tsmult: float = 1.0,
    # --- storage controls ---
    sy_vals: tuple[float, float] = (0.20, 0.20),  # Sy per layer (k=0, k=1)
    ss_default: float = 1e-6,                     # Ss for k>=1 [1/ft]
    min_thk: float = 0.5,                         # min thickness guard for 1/thk
) -> pd.DataFrame:
    import calendar
    import numpy as np
    import pandas as pd

    # ---- build TDIS record ----
    tdis_rc: list[tuple[float, int, float]] = []
    sp_start: list[pd.Timestamp] = []
    sp_end:   list[pd.Timestamp] = []

    ss_anchor_year = start_year - 1
    ss_start_dt = pd.Timestamp(f"{ss_anchor_year}-12-31")
    tdis_rc.append((1.0, 1, tsmult))
    sp_start.append(ss_start_dt)
    sp_end.append(ss_start_dt + pd.Timedelta(days=1))

    def _is_leap(y: int) -> bool:
        return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

    def _add_annual(year: int):
        perlen = 366.0 if _is_leap(year) else 365.0
        tdis_rc.append((perlen, nstp_annual, tsmult))
        sp_start.append(pd.Timestamp(f"{year}-01-01"))
        sp_end.append(pd.Timestamp(f"{year}-12-31"))

    def _add_months(year: int):
        for m in range(1, 13):
            days = calendar.monthrange(year, m)[1]
            tdis_rc.append((float(days), nstp_monthly, tsmult))
            sp_start.append(pd.Timestamp(f"{year}-{m:02d}-01"))
            sp_end.append(pd.Timestamp(f"{year}-{m:02d}-{days:02d}"))

    first_annual = start_year
    if annual_only:
        for y in range(first_annual, end_year+1):
            _add_annual(y)
    else:
        if monthly_start is None:
            raise ValueError("When annual_only=False you must set monthly_start (e.g., 2023).")
        for y in range(first_annual, monthly_start):
            _add_annual(y)
        for y in range(monthly_start, end_year+1):
            _add_months(y)

    nper = len(tdis_rc)

    # ---- (re)create TDIS ----
    td_old = sim.get_package("tdis")
    if td_old is not None:
        sim.remove_package(td_old)

    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=nper,
        perioddata=tdis_rc,
        start_date_time=str(ss_start_dt.date()),
        filename=f"{mnm}.tdis",
    )

    # ---- STORAGE (with Ss = 1/thickness for layer 1 only) ----
    # Make both layers convertible in NPF and STO
    dis = gwf.dis
    nlay = int(dis.nlay.get_data())
    if nlay < 2:
        raise ValueError("build_tdis_and_sto expects a 2-layer model for the Ss trick on layer 1.")

    # NPF: set both layers convertible (1)
    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found on model.")
    npf.icelltype = np.ones(nlay, dtype=int)

    # Remove existing STO
    sto_old = gwf.get_package("sto")
    if sto_old is not None:
        gwf.remove_package(sto_old)

    # Thickness arrays
    top  = np.asarray(dis.top.array, dtype=float)           # (nrow, ncol)
    botm = np.asarray(dis.botm.array, dtype=float)          # (nlay, nrow, ncol)
    thk  = np.empty_like(botm)
    thk[0] = top - botm[0]
    for k in range(1, nlay):
        thk[k] = botm[k-1] - botm[k]
    thk = np.maximum(thk, float(min_thk))

    idomain = np.asarray(dis.idomain.array, dtype=int)

    # iconvert: both layers convertible, masked outside idomain
    iconvert = np.where(idomain == 1, 1, 0).astype(int)

    # Sy layered constants
    sy = np.zeros_like(botm, dtype=float)
    for k in range(nlay):
        sy[k, :, :] = float(sy_vals[k] if k < len(sy_vals) else sy_vals[-1])

    # Ss: layer 1 uses 1/thickness; deeper layers use default
    ss = np.full_like(botm, float(ss_default), dtype=float)
    ss[0, :, :] = 1.0 / thk[0]

    # mask outside idomain
    sy = np.where(idomain == 1, sy, sy)
    ss = np.where(idomain == 1, ss, ss)

    steady_map = {0: True}
    trans_map  = {sp: True for sp in range(1, nper)}

    flopy.mf6.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=iconvert,  # layered array
        ss=ss,
        sy=sy,
        steady_state=steady_map,
        transient=trans_map,
        filename=f"{mnm}.sto",
        ss_confined_only=True,
    )

    # ---- summary table ----
    df = pd.DataFrame(
        {
            "sp": range(nper),
            "perlen_days": [p[0] for p in tdis_rc],
            "nstp": [p[1] for p in tdis_rc],
            "tsmult": [p[2] for p in tdis_rc],
            "start": sp_start,
            "end": sp_end,
            "mode": (["SS"] + ["TR"] * (nper - 1)),
        }
    )
    return df

def stress_period_df_gen(mws, strt_yr, annual_only=False):
    out_tbl_dir = os.path.join("tables")
    if not os.path.exists(out_tbl_dir):
        os.makedirs(out_tbl_dir)

    sim = flopy.mf6.MFSimulation.load(sim_ws=mws, exe_name="mf6", load_only=["dis", "tdis"])
    start_date = pd.to_datetime(sim.tdis.start_date_time.data)
    period_data = sim.tdis.perioddata.array
    nper = int(sim.tdis.nper.data)

    # Wrap perioddata in a DataFrame for convenience
    # period_data has dtype with fields like ('perlen','nstp','tsmult','steady')
    spd = pd.DataFrame(period_data)

    if annual_only:
        # ----- ANNUAL-ONLY BRANCH -----
        # Build cumulative days from perlen as given by TDIS
        perlen_days = spd.iloc[:, 0].astype(float).values  # first column is perlen
        cum_days = np.cumsum(perlen_days)

        spd["perlen"] = perlen_days
        spd["cum_days"] = cum_days

        # Year labels: first SP is steady-state (strt_yr - 1),
        # then transient SPs start at strt_yr, strt_yr+1, ...
        years = np.arange(strt_yr - 1, strt_yr - 1 + nper, 1, dtype=int)
        spd["year"] = years

        # Start/end datetimes
        start_datetime = [start_date]
        end_datetime = [start_date + pd.to_timedelta(perlen_days[0], unit="D")]
        for i in range(1, nper):
            st_i = end_datetime[i - 1]
            en_i = st_i + pd.to_timedelta(perlen_days[i], unit="D")
            start_datetime.append(st_i)
            end_datetime.append(en_i)

        spd["start_datetime"] = start_datetime
        spd["end_datetime"] = end_datetime

        # Steady-state flag
        spd["steady_state"] = False
        spd.loc[0, "steady_state"] = True

        # Reset index → 1-indexed stress period
        spd = spd.reset_index().rename(columns={"index": "stress_period"})
        spd["stress_period"] = spd["stress_period"] + 1

        # SS period time doesn’t really matter → set cum_days to 0 for SP1
        spd.loc[0, "cum_days"] = 0

        spd.to_csv(os.path.join(out_tbl_dir, "annual_stress_period_info.csv"), index=False)

    else:
        # ----- MIXED ANNUAL + MONTHLY BRANCH -----
        perlen_days = spd.iloc[:, 0].astype(float).values  # first column is perlen
        spd["perlen"] = perlen_days

        # Build start/end datetimes by walking forward from start_date
        start_datetime = []
        end_datetime = []
        years = []

        current_start = start_date
        for i in range(nper):
            start_datetime.append(current_start)
            perlen_i = perlen_days[i]
            end_i = current_start + pd.to_timedelta(perlen_i, unit="D")
            end_datetime.append(end_i)
            years.append(current_start.year)
            current_start = end_i

        spd["start_datetime"] = start_datetime
        spd["end_datetime"] = end_datetime
        spd["year"] = years

        # Cumulative days since model start (for convenience)
        cum_days = (spd["end_datetime"] - start_date).dt.days.astype(int)
        spd["cum_days"] = cum_days

        # Steady-state flag: SP1 (index 0) is SS
        spd["steady_state"] = False
        spd.loc[0, "steady_state"] = True

        # 1-indexed stress period
        spd = spd.reset_index().rename(columns={"index": "stress_period"})
        spd["stress_period"] = spd["stress_period"] + 1

        # For SS period, set cum_days to 0 if you want that convention
        spd.loc[0, "cum_days"] = 0

        spd.to_csv(os.path.join(out_tbl_dir, "monthly_stress_period_info.csv"), index=False)

    return spd

def clean_mf6(org_mws,mnm):
    cws = org_mws + "_clean"
    if os.path.exists(cws):
        shutil.rmtree(cws)
    shutil.copytree(org_mws,cws)
    
    # add exe
    elk_pst.prep_deps(cws)

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
            if "idomain" in arr_file or "icell" in arr_file or "iconvert" in arr_file:
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
        riv_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith(f"{mnm}.riv") and "stress" in f
        ]
        
        drn_stress_files = [
            os.path.join(model_ws, f)
            for f in os.listdir(model_ws)
            if f.startswith(f"{mnm}.drn_") and "stress" in f
            and "drn_wl" not in f
            and "drn_ag" not in f
        ]
        
        drn2_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.drn_wl_stress")]
        drn_ag_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.drn_ag_stress")]
        #ghb_wl_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.ghb_wl_stress")]
        well_pkg_wel_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"{mnm}.wel_")]

        
        # append to afiles:
        afiles.extend(riv_stress_files)
        afiles.extend(drn_stress_files)
        afiles.extend(drn2_stress_files)
        afiles.extend(drn_ag_stress_files)
        #afiles.extend(ghb_wl_stress_files)
        afiles.extend(well_pkg_wel_files)

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
        for drn_file in drn2_stress_files:
            df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
            df.columns = ['ly','row','col','stage','cond']
            bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
            df['mbot'] = bot
            df['diff'] = df['stage'] - df['mbot']
            df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
            df = df.drop(columns=['mbot','diff'])
            df.to_csv(drn_file, sep=' ', index=False, header=False)
        for drn_file in drn_ag_stress_files:
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
            
        for file in afiles:
            ogfile = file
            file = file.replace(f"{mnm}.", "") 
            os.rename(ogfile, file)     
        
        # adjust paths in  pkg controls:
        for pkg in pkg_lst:
            print(pkg)
            if 'rch' in pkg:
                pkg = 'rch' # remove numbering from recharge package name
            if pkg in ["oc","head_obs"]:
                continue
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

    pyemu.utils.run("mf6",cwd=cws)

    return cws

def model_packages_to_shp(d='.',modnm='elk_2lay'):
    sim = flopy.mf6.MFSimulation.load(sim_ws=d,)
    mf = sim.get_model(modnm)
    epsg = 2265

    # Create model package shapefile directory
    o_d = os.path.join(d,'output_shapefiles','model_packages')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    # list packages:
    mf.get_package_list()

    # dis
    mf.dis.top.export(os.path.join(o_d, 'top.shp'), epsg=epsg)
    mf.dis.botm.export(os.path.join(o_d, 'bottoms.shp'), epsg=epsg)

    # Drn Streams
    # mf.drn.stress_period_data.export(os.path.join(o_d, 'drn.shp'), epsg=epsg)
    # drn = gpd.read_file(os.path.join(o_d,'drn.shp'))
    # drn['elev'] = drn.elev11
    # drn['cond'] = drn.cond11
    # # Remove cells that don't have drains
    # drn = drn[drn.elev!=0]
    # # Drop unneccessary columns
    # drn = drn.loc[:, ['node','row','column','elev','cond', 'geometry']]
    # drn = drn.set_crs('epsg:2265', allow_override=True)
    # drn.to_file(os.path.join(o_d, 'drn.shp'))

    # riv
    mf.riv.stress_period_data.export(os.path.join(o_d, 'riv.shp'), epsg=epsg)
    riv = gpd.read_file(os.path.join(o_d,'riv.shp'))
    riv['stage'] = riv.stag11 
    riv['cond'] = riv.cond11 
    riv['rbot'] = riv.rbot11
    # Remove cells that don't have drains
    riv = riv[riv.stage!=0]
    # Drop unneccessary columns
    riv = riv.loc[:, ['node','row','column','stage','cond', 'rbot', 'geometry']]
    riv = riv.set_crs('epsg:2265', allow_override=True)
    riv.to_file(os.path.join(o_d, 'riv.shp'))

    # ghb edge cells
    # mf.ghb.stress_period_data.export(os.path.join(o_d, 'ghb_edge.shp'), epsg=epsg)
    # ghb_edge = gpd.read_file(os.path.join(o_d, 'ghb_edge.shp'))
    # # Remove cells and columns  that don't have drains
    # cols_active = [f for f in ghb_edge.columns[:-1] if ghb_edge[f].sum() !=0]

    # recharge
    mf.rch.export(os.path.join(o_d,'rch.shp'),epsg=epsg)
    rch = gpd.read_file(os.path.join(o_d, 'rch.shp'))
    rch = rch.loc[:, [f for f in rch.columns if 'irch' not in f]]
    rch = rch.set_crs('epsg:2265', allow_override=True)
    rch.to_file(os.path.join(o_d, 'rch.shp'))

    # npf
    # get list of npf_icelltype files in dir:
    icell_files = [f for f in os.listdir(d) if f.startswith('npf_icelltype')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(d, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(d, file), icelltype_data, fmt='%i')

    mf.npf.export(os.path.join(o_d,'npf.shp'), epsg=epsg)
    npf = gpd.read_file(os.path.join(o_d, 'npf.shp'))
    #npf = npf.loc[:, [f for f in npf.columns if 'icell' not in f]]
    npf = npf.set_crs('epsg:2265', allow_override=True)
    npf.to_file(os.path.join(o_d, 'npf.shp'))

    # sto
    icell_files = [f for f in os.listdir(d) if f.startswith('sto_iconvert')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(d, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(d, file), icelltype_data, fmt='%i')

    mf.sto.export(os.path.join(o_d,'sto.shp'), epsg=epsg)

    # wels
    nam = 'elk_2lay.wel'
    print(f'Saving {nam.upper()} well package')
    mf.wel.stress_period_data.export(os.path.join(o_d,f'{nam}_wel.shp'), epsg=epsg)
    df = gpd.read_file(os.path.join(o_d,f'{nam}_wel.shp'))
    df = df.set_crs('epsg:2265', allow_override=True)
    df.to_file(os.path.join(o_d, f'{nam}_wel.shp'))

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

def run_zb_by_layer(w_d,modnm='elk_2lay'):
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

# --------------------------
# RIVER BUILDER (from centerlines + DEM + grid)
# --------------------------
def build_riv_dataframe(centerlines_shp: str, grid_shp: str, dem_tif: str,
                        width_ft=150.0, stage_offset_ft=1.5, depth_ft=3.0,
                        sample_dx_ft=250.0, epsg=2265) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with columns: i, j, stage, rbot, cond, segment_length, geometry
    Assumes the DEM is already in FEET. No unit conversion is applied.
    """
    if not os.path.exists(centerlines_shp):
        print(f"[RIV] Centerlines not found: {centerlines_shp} – skipping RIV build.")
        return None

    grd = gpd.read_file(grid_shp).to_crs(epsg=epsg)
    riv = gpd.read_file(centerlines_shp).to_crs(grd.crs)

    # densify centerlines into points every ~sample_dx_ft
    from shapely.ops import linemerge
    merged = linemerge(riv.unary_union)
    if merged.is_empty:
        print("[RIV] Centerlines merged geometry is empty.")
        return None
    total_len = merged.length
    distances = np.arange(0, int(total_len), sample_dx_ft)
    pts = [merged.interpolate(d) for d in distances]
    pts_gdf = gpd.GeoDataFrame(geometry=pts, crs=grd.crs).reset_index().rename(columns={'index': 'pid'})

    # sample DEM at points (assumed FEET)
    with rasterio.open(dem_tif) as src:
        pts_dem = pts_gdf.to_crs(src.crs) if pts_gdf.crs != src.crs else pts_gdf
        coords = [(p.x, p.y) for p in pts_dem.geometry]
        elev_ft = np.array([v[0] for v in src.sample(coords)], dtype=float)

    # map back into grid cells, keep lowest elevation per (i,j)
    pts_gdf["elev_ft"] = elev_ft
    pts_in_grid = gpd.sjoin(pts_gdf.to_crs(grd.crs), grd[["i","j","geometry"]], how="left", predicate="within")
    pts_in_grid = pts_in_grid.dropna(subset=["i","j"])
    cell_min = pts_in_grid.sort_values("elev_ft").drop_duplicates(subset=["i","j"], keep="first")

    # segment length inside each cell
    inter = gpd.overlay(riv[["geometry"]], grd[["i","j","geometry"]], how="intersection")
    inter["segment_length"] = inter.geometry.length

    riv_df = cell_min.merge(inter[["i","j","segment_length"]], on=["i","j"], how="left")
    riv_df = riv_df.dropna(subset=["segment_length"])

    # stage/rbot/cond in FEET
    riv_df["stage"] = riv_df["elev_ft"] + stage_offset_ft
    riv_df["rbot"]  = riv_df["elev_ft"] - depth_ft
    riv_df["cond"]  = (width_ft * riv_df["segment_length"]) / (riv_df["stage"] - riv_df["rbot"])
    riv_df = riv_df.loc[riv_df["segment_length"] > 100.0].copy()

    return gpd.GeoDataFrame(riv_df, geometry="geometry", crs=grd.crs)

def itize_riv_dfsan_for_idomain(gwf, riv_df, layer=0):
    """Keep only rows where (layer,i,j) is inside the active model domain."""
    nrow = gwf.dis.nrow.get_data()
    ncol = gwf.dis.ncol.get_data()
    idom = gwf.dis.idomain.array[layer].astype(int)

    # ensure integer indices
    riv_df = riv_df.copy()
    riv_df["i"] = riv_df["i"].astype(int)
    riv_df["j"] = riv_df["j"].astype(int)

    in_bounds = (
        (riv_df["i"] >= 0) & (riv_df["i"] < nrow) &
        (riv_df["j"] >= 0) & (riv_df["j"] < ncol)
    )

    # mask by idomain
    active_mask = np.zeros(len(riv_df), dtype=bool)
    ib = riv_df.loc[in_bounds, ["i","j"]].to_numpy(dtype=int)
    active_mask[in_bounds] = (idom[ib[:,0], ib[:,1]] == 1)

    before = len(riv_df)
    riv_df = riv_df.loc[in_bounds & active_mask].copy()
    after = len(riv_df)
    print(f"[RIV] kept {after:,} / {before:,} cells (dropped {before-after:,} outside idomain/bounds)")

    return riv_df

def raise_riv_to_cellbottom(gwf, riv_df, layer=0, bed_thk_ft=3.0, clearance_ft=0.1):
    """
    Adjusts each RIV row so that:
        rbot = max(rbot, cellbot + clearance_ft)
        stage = max(stage, rbot + max(0.1, bed_thk_ft*0.2))
    Keeps conductance unchanged.
    """
    if riv_df is None or len(riv_df) == 0:
        return riv_df
    riv_df = riv_df.copy()
    riv_df[["i","j"]] = riv_df[["i","j"]].astype(int)

    botm1 = gwf.dis.botm.array[layer]
    cellbot = botm1[riv_df["i"], riv_df["j"]].astype(float)

    # raise rbot first to be above cell bottom
    riv_df["rbot"] = np.maximum(riv_df["rbot"].astype(float), cellbot + float(clearance_ft))

    # ensure stage is above rbot by a small head (min 0.1 ft or 20% of bed thickness)
    min_head = max(0.1, 0.2 * bed_thk_ft)
    riv_df["stage"] = np.maximum(riv_df["stage"].astype(float), riv_df["rbot"] + min_head)

    # report how many changed
    changed = ((riv_df["rbot"] < cellbot + clearance_ft) | (riv_df["stage"] <= riv_df["rbot"] + min_head)).sum()
    print(f"[RIV] adjusted {int(changed)} cells to clear cell bottoms.")
    return riv_df

def drop_riv_below_cellbottom(gwf, riv_df, layer=0, clearance_ft=0.1):
    """Drop RIV rows where stage/rbot are below the DIS cell bottom."""
    if riv_df is None or len(riv_df) == 0:
        return riv_df
    riv_df = riv_df.copy()
    riv_df[["i","j"]] = riv_df[["i","j"]].astype(int)

    botm1 = gwf.dis.botm.array[layer]  # layer 0 bottom (= layer 1 bottom in 1-based terms)
    cellbot = botm1[riv_df["i"], riv_df["j"]]

    ok = (riv_df["rbot"] >= cellbot + clearance_ft) & (riv_df["stage"] > riv_df["rbot"])
    before = len(riv_df)
    riv_df = riv_df.loc[ok].copy()
    print(f"[RIV] kept {len(riv_df):,} / {before:,} cells; dropped {before - len(riv_df):,} below cell bottom.")
    return riv_df

def fix_ghb_to_cellbottom(gwf, ghb_df, layer=0, clearance_ft=0.1, mode="raise"):
    """
    Ensure GHB head >= cell bottom + clearance.
    mode="raise" -> lift offending heads just above bottom (keeps rows)
    mode="drop"  -> remove offending rows

    Expects ghb_df with integer i,j and columns: ["i","j","head","cond"].
    Conductance is left unchanged.
    """
    import numpy as np
    import pandas as pd

    if ghb_df is None or len(ghb_df) == 0:
        return ghb_df

    df = ghb_df.copy()
    df[["i","j"]] = df[["i","j"]].astype(int)

    # cell bottoms for the target layer
    cellbot = np.asarray(gwf.dis.botm.array)[layer][df["i"], df["j"]].astype(float)

    ok = df["head"].astype(float) >= (cellbot + float(clearance_ft))

    if mode == "drop":
        before = len(df)
        df = df.loc[ok].copy()
        print(f"[GHB] kept {len(df):,} / {before:,} cells; dropped {before - len(df):,} below cell bottom.")
        return df

    # mode == "raise"
    head_old = df["head"].to_numpy(dtype=float)
    head_new = np.maximum(head_old, cellbot + float(clearance_ft))
    changed = (head_new != head_old).sum()
    df["head"] = head_new
    if changed:
        print(f"[GHB] raised head in {int(changed)} cell(s) to clear cell bottoms.")
    return df

# --------------------------
# RECHARGE BUILDER (from SWB)
# --------------------------
def get_rcha_tseries_from_swb2_nc(
    gwf,
    perioddata,
    rch_dict=None,
    annual_only=False,
    rch_mult=1.0,
    smooth_sigma=None,
    smooth_mode="reflect",
    plot_debug=False,
    plot_sps=None,
    pdf_path=None,
):
    """
    Build a recharge time series (dict of SP -> 2D array in ft/day) from SWB outputs.

    Uses process_SWB_to_RCH_elk.main(), which returns:
        pre_2000: dict {year: 2D array in/day} for 1965–1999, built from:
            - PRISM annual precip
            - SWB 2000–2023 regression
            - SWB 2000–2023 spatial pattern
        post_2000_monthly: dict {"YYYY-MM": 2D array in/day} for 2000–2023
            from SWB NetCDF monthly means.

    Behavior (annual_only=False, main use case):
    - For each stress period (SP), use its end date (TDIS 'end'):
        * year < 2000:
            - Use pre_2000[year] (in/day), resampled to MF grid, scaled, optionally
              smoothed, and clipped to ANNUAL thresholds.
        * 2000 ≤ year < 2024:
            - Use post_2000_monthly["YYYY-MM"] (in/day), resampled to MF grid,
              scaled, optionally smoothed, and clipped to MONTHLY thresholds.
        * year ≥ 2024:
            - Use a MONTHLY CLIMATOLOGY from 2003–2023:
                · For each SP, use the mean SWB recharge for that calendar month
                  across 2003–2023 (Jan climatology, Feb climatology, …),
                  then resampled, smoothed, and clipped to MONTHLY thresholds.

    Behavior (annual_only=True):
    - All SPs treated as annual:
        * pre-2000: pre_2000[year]
        * 2000–2023: monthly SWB → annual average per year
        * ≥2024: mean of 2000–2023 annual arrays
      Each is resampled, scaled, optionally smoothed, and clipped to ANNUAL thresholds.

    Units & thresholds:
    - SWB arrays are in in/day. We do:
          arr_in_scaled = arr_in * rch_mult         # in/day
          [optional gaussian_filter on arr_in_scaled]
          arr_smooth_ft = arr_in_smooth / 12.0      # ft/day (smoothed, unclipped)
          arr_ft        = np.clip(arr_smooth_ft, min_thresh, max_thresh)

    Annual thresholds (ft/day) ~ 0.5–10 in/yr:
        annual_min_ft_per_day ≈ 1.14e-04
        annual_max_ft_per_day ≈ 2.28e-03

    Monthly thresholds (ft/day):
        monthly_min_ft_per_day = 1e-5
        monthly_max_ft_per_day = 2.875e-2

    Smoothing & plotting:
    - If smooth_sigma is None or <=0 → no smoothing.
    - If smooth_sigma > 0:
        - Apply gaussian_filter(arr_in_scaled, sigma, mode=smooth_mode).
    - If plot_debug True:
        - For each SP in plot_sps (or all SPs if plot_sps is None),
          produce 3-panel plots in ft/day:
              1) Before smoothing (scaled, unclipped)
              2) Smoothed, unclipped
              3) Smoothed + clipped
        - Inactive cells (idomain/ibound == 0) are masked from plots.
        - Each colorbar has horizontal lines at SP min/max cutoff values.
        - Figure suptitle shows thresholds in ft/day and converted to
          in/yr (for ANNUAL) or in/mo (for MONTHLY/MONTHLY_PRED).
        - If pdf_path is given, plots go to a multipage PDF; otherwise they show.

    Returns
    -------
    tsa_dict : dict[int, np.ndarray]
        {sp_index: 2D array (nrow, ncol) in ft/day}.
    """
    import os
    import numpy as np
    import pandas as pd
    import flopy
    from flopy.utils import Raster

    # --- smoothing + plotting imports ---
    gaussian_filter = None
    if smooth_sigma is not None and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter as _gf
            gaussian_filter = _gf
        except ImportError as e:
            raise ImportError(
                "scipy is required for Gaussian smoothing (smooth_sigma>0). "
                "Install scipy or set smooth_sigma=None."
            ) from e

    if plot_debug:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    else:
        PdfPages = None  # type: ignore

    pdf = None
    if plot_debug and pdf_path is not None:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        pdf = PdfPages(pdf_path)

    if perioddata is None:
        raise ValueError("perioddata (TDIS dataframe) must be provided.")
    if rch_dict is None:
        rch_dict = {}

    from process_SWB_to_RCH_elk import main as process_SWB

    # ---------- thresholds in ft/day ----------
    annual_min_ft_per_day = (0.1 / 12.0) / 365.0
    annual_max_ft_per_day = (5.0 / 12.0) / 365.0

    monthly_min_ft_per_day = 1e-5
    monthly_max_ft_per_day = 0.02875

    # ---- Load pre- and post-2000 SWB outputs ----
    pre_2000, post_2000_monthly = process_SWB()

    # ---- Time info from TDIS/perioddata ----
    tdis_df = perioddata.copy()
    tdis_df["cum_days"] = tdis_df["perlen_days"].cumsum()

    start_date = tdis_df.loc[0, "start"] + pd.Timedelta(days=1)
    end_date = tdis_df.loc[tdis_df.shape[0] - 1, "end"]

    mg = gwf.modelgrid
    nrow, ncol = mg.nrow, mg.ncol

    # ---- idomain / ibound mask (for plotting only) ----
    idomain_2d = None
    try:
        idom = gwf.dis.idomain.get_data()
        idom = np.array(idom)
        if idom.ndim == 3:
            idomain_2d = idom[0, :, :]
        elif idom.ndim == 2:
            idomain_2d = idom
    except Exception:
        idomain_2d = None  # if not available, just plot everything

    # ---- SWB grid geometry from control file ----
    f_swb_ctl = os.path.join("data", "swb", "swb_control_file_elk.ctl")
    with open(f_swb_ctl, "r") as file:
        for line in file:
            if line.startswith("GRID "):
                grid_info = line.split(" ")
                break
    dxdy = float(grid_info[-1].strip("\n"))
    sxo = float(grid_info[3])
    syo = float(grid_info[4])

    mg_swb = flopy.discretization.StructuredGrid(
        delr=np.array([dxdy] * int(grid_info[1])),
        delc=np.array([dxdy] * int(grid_info[2])),
        xoff=sxo,
        yoff=syo,
        crs="EPSG:2265",
    )

    # ---- for annual_only branch ----
    date_range = pd.date_range(start=start_date, end=end_date)
    years = np.unique(date_range.year)

    # For extended debugging: store raw stats by SP
    debug_info = {}

    # ------------ plotting helper (3 panels) ------------
    def _plot_triplet(arr_before_ft, arr_smooth_ft, arr_after_ft,
                      sp_idx, date, kind, min_thr, max_thr):
        if not plot_debug:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        import calendar

        # Mask inactive cells (idomain == 0) for plotting & stats
        if idomain_2d is not None:
            mask = (idomain_2d == 0)
            arr_before_plot = arr_before_ft.copy()
            arr_smooth_plot = arr_smooth_ft.copy()
            arr_after_plot = arr_after_ft.copy()
            arr_before_plot[mask] = np.nan
            arr_smooth_plot[mask] = np.nan
            arr_after_plot[mask] = np.nan
        else:
            arr_before_plot = arr_before_ft
            arr_smooth_plot = arr_smooth_ft
            arr_after_plot = arr_after_ft

        def _nan_stats(a):
            if a is None:
                return np.nan, np.nan
            finite = np.isfinite(a)
            if not finite.any():
                return np.nan, np.nan
            return float(np.nanmin(a)), float(np.nanmax(a))

        bmin, bmax = _nan_stats(arr_before_plot)
        smin, smax = _nan_stats(arr_smooth_plot)
        cmin, cmax = _nan_stats(arr_after_plot)

        # Common vmin/vmax from "before" map
        try:
            vmin = np.nanpercentile(arr_before_plot, 2.0)
            vmax = np.nanpercentile(arr_before_plot, 98.0)
        except Exception:
            vmin = np.nanmin(arr_before_plot)
            vmax = np.nanmax(arr_before_plot)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        # NOTE: origin="upper" => row 0 at top, col 0 at left (MODFLOW-style)

        # Panel 1: before smoothing (scaled, unclipped)
        im0 = axes[0].imshow(arr_before_plot, origin="upper", vmin=vmin, vmax=vmax)
        axes[0].set_title(
            "Before smoothing\n"
            f"min={bmin:.3e}, max={bmax:.3e} ft/day"
        )
        cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.7)
        cb0.ax.axhline(min_thr, color="k", linestyle="--", linewidth=1.0)
        cb0.ax.axhline(max_thr, color="k", linestyle="--", linewidth=1.0)

        # Panel 2: smoothed, before clipping
        im1 = axes[1].imshow(arr_smooth_plot, origin="upper", vmin=vmin, vmax=vmax)
        axes[1].set_title(
            "Smoothed, unclipped\n"
            f"min={smin:.3e}, max={smax:.3e} ft/day"
        )
        cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.7)
        cb1.ax.axhline(min_thr, color="k", linestyle="--", linewidth=1.0)
        cb1.ax.axhline(max_thr, color="k", linestyle="--", linewidth=1.0)

        # Panel 3: smoothed + clipped
        im2 = axes[2].imshow(arr_after_plot, origin="upper", vmin=vmin, vmax=vmax)
        axes[2].set_title(
            "Smoothed + clipped\n"
            f"min={cmin:.3e}, max={cmax:.3e} ft/day"
        )
        cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.7)
        cb2.ax.axhline(min_thr, color="k", linestyle="--", linewidth=1.0)
        cb2.ax.axhline(max_thr, color="k", linestyle="--", linewidth=1.0)

        # --- Suptitle: thresholds in ft/day and in/yr or in/mo ---
        if "ANNUAL" in kind.upper():
            # Annual conversion
            days = 366 if calendar.isleap(date.year) else 365
            min_in = min_thr * 12.0 * days
            max_in = max_thr * 12.0 * days
            sup = (
                f"SP {sp_idx} {date.date()} – {kind} thresholds: "
                f"min={min_thr:.3e} ft/day = {min_in:.2f} in/yr, "
                f"max={max_thr:.3e} ft/day = {max_in:.2f} in/yr"
            )
        else:
            # Monthly conversion
            days = calendar.monthrange(date.year, date.month)[1]
            min_in = min_thr * 12.0 * days
            max_in = max_thr * 12.0 * days
            sup = (
                f"SP {sp_idx} {date.date()} – {kind} thresholds: "
                f"min={min_thr:.3e} ft/day = {min_in:.2f} in/mo, "
                f"max={max_thr:.3e} ft/day = {max_in:.2f} in/mo"
            )

        fig.suptitle(sup)

        if pdf is not None:
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ------------ core processing helper ------------
    def _process_arr(arr_in, *, sp_idx, date, kind, min_thr, max_thr):
        """
        arr_in: MF-grid array in in/day (after resampling).
        Returns arr_ft (ft/day) after scaling, optional smoothing, and clipping.
        Also handles plotting triplet if requested.
        """
        # Scale (in/day) and convert to ft/day BEFORE smoothing
        arr_in_scaled = arr_in * rch_mult                  # in/day
        arr_before_ft = arr_in_scaled / 12.0               # ft/day (un-smoothed, un-clipped)

        # Optional smoothing in in/day space
        if gaussian_filter is not None:
            arr_in_smooth = gaussian_filter(
                arr_in_scaled,
                sigma=smooth_sigma,
                mode=smooth_mode,
            )
        else:
            arr_in_smooth = arr_in_scaled

        # Smoothed, un-clipped ft/day
        arr_smooth_ft = arr_in_smooth / 12.0

        # Clip smoothed ft/day to thresholds
        arr_ft = np.clip(arr_smooth_ft, min_thr, max_thr)

        debug_info[sp_idx] = {
            "type": kind,
            "raw_in_min":    float(np.nanmin(arr_in)),
            "raw_in_max":    float(np.nanmax(arr_in)),
            "ft_before_min": float(np.nanmin(arr_before_ft)),
            "ft_before_max": float(np.nanmax(arr_before_ft)),
            "ft_smooth_min": float(np.nanmin(arr_smooth_ft)),
            "ft_smooth_max": float(np.nanmax(arr_smooth_ft)),
            "thresh_min":    min_thr,
            "thresh_max":    max_thr,
            "ft_final_min":  float(np.nanmin(arr_ft)),
            "ft_final_max":  float(np.nanmax(arr_ft)),
            "smooth_sigma":  smooth_sigma if gaussian_filter is not None else None,
        }

        if plot_debug and (plot_sps is None or sp_idx in plot_sps):
            _plot_triplet(
                arr_before_ft,
                arr_smooth_ft,
                arr_ft,
                sp_idx,
                date,
                kind,
                min_thr,
                max_thr,
            )

        return arr_ft

    # ======================================================================
    # annual_only branch
    # ======================================================================
    if annual_only:
        tsa_dict = rch_dict.copy()
        pred_avg = []  # 2000–2023 annual arrays for ≥2024

        for i, year in enumerate(years):
            date = pd.Timestamp(year=int(year), month=12, day=31)

            if year < 2000:
                if year not in pre_2000:
                    continue
                rio = Raster.raster_from_array(pre_2000[year], mg_swb)
                arr_in = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")  # in/day

                arr_ft = _process_arr(
                    arr_in,
                    sp_idx=i,
                    date=date,
                    kind="ANNUAL",
                    min_thr=annual_min_ft_per_day,
                    max_thr=annual_max_ft_per_day,
                )
                tsa_dict[i] = arr_ft

            elif 2000 <= year < 2024:
                year_keys = [key for key in post_2000_monthly if str(year) in key]
                if not year_keys:
                    continue
                month_arr = [post_2000_monthly[k] for k in year_keys]
                month_arr = np.stack(month_arr)
                annual_arr_in = np.nanmean(month_arr, axis=0)  # in/day
                pred_avg.append(annual_arr_in)

                rio = Raster.raster_from_array(annual_arr_in, mg_swb)
                arr_in = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")

                arr_ft = _process_arr(
                    arr_in,
                    sp_idx=i,
                    date=date,
                    kind="ANNUAL",
                    min_thr=annual_min_ft_per_day,
                    max_thr=annual_max_ft_per_day,
                )
                tsa_dict[i] = arr_ft

            else:
                if not pred_avg:
                    continue
                annual_avg_in = np.nanmean(np.stack(pred_avg), axis=0)  # in/day
                rio = Raster.raster_from_array(annual_avg_in, mg_swb)
                arr_in = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")

                arr_ft = _process_arr(
                    arr_in,
                    sp_idx=i,
                    date=date,
                    kind="ANNUAL_PRED",
                    min_thr=annual_min_ft_per_day,
                    max_thr=annual_max_ft_per_day,
                )
                tsa_dict[i] = arr_ft

    # ======================================================================
    # annual + monthly branch (main use)
    # ======================================================================
    else:
        tsa_dict = rch_dict.copy()

        # --------------------------------------------------------------
        # Build monthly climatology (in/day) for 2003–2023:
        # month_clim[1..12] = mean SWB recharge for that month across
        # all years 2003–2023. If a month has no data in that range,
        # fall back to averaging 2000–2023 for that month, and if that
        # is also missing (very unlikely), average all months.
        # --------------------------------------------------------------
        month_clim = {}
        all_months_by_m = {m: [] for m in range(1, 13)}   # 2000–2023 (fallback)
        core_months_by_m = {m: [] for m in range(1, 13)}  # 2003–2023

        for key, arr in post_2000_monthly.items():
            yr_str, mo_str = key.split("-")
            yr = int(yr_str)
            mo = int(mo_str)
            all_months_by_m[mo].append(arr)
            if 2003 <= yr <= 2023:
                core_months_by_m[mo].append(arr)

        for m in range(1, 13):
            if core_months_by_m[m]:
                month_clim[m] = np.nanmean(np.stack(core_months_by_m[m]), axis=0)
            elif all_months_by_m[m]:
                month_clim[m] = np.nanmean(np.stack(all_months_by_m[m]), axis=0)
            else:
                # Extremely unlikely: no data for this month at all.
                month_clim[m] = np.nanmean(
                    np.stack(list(post_2000_monthly.values())), axis=0
                )

        # --------------------------------------------------------------
        # Loop over SPs and assign annual (pre-2000), monthly (2000–2023),
        # and monthly-climatology (≥2024) recharge arrays.
        # --------------------------------------------------------------
        for sp in range(len(perioddata)):
            date = tdis_df.loc[sp, "end"]
            year = int(date.year)

            if year < 2000:
                # Pre-2000 annual SPs → ANNUAL thresholds
                if year not in pre_2000:
                    continue
                rio = Raster.raster_from_array(pre_2000[year], mg_swb)
                arr_in = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")  # in/day

                arr_ft = _process_arr(
                    arr_in,
                    sp_idx=sp,
                    date=date,
                    kind="ANNUAL",
                    min_thr=annual_min_ft_per_day,
                    max_thr=annual_max_ft_per_day,
                )
                tsa_dict[sp] = arr_ft

            elif 2000 <= year < 2024:
                # 2000–2023 monthly SPs → MONTHLY thresholds
                key = f"{year}-{date.month:02d}"
                if key not in post_2000_monthly:
                    continue

                rio = Raster.raster_from_array(post_2000_monthly[key], mg_swb)
                arr_in = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")  # in/day

                arr_ft = _process_arr(
                    arr_in,
                    sp_idx=sp,
                    date=date,
                    kind="MONTHLY",
                    min_thr=monthly_min_ft_per_day,
                    max_thr=monthly_max_ft_per_day,
                )
                tsa_dict[sp] = arr_ft

            else:
                # ≥2024: use monthly climatology (2003–2023) → MONTHLY thresholds
                m = int(date.month)
                clim_in = month_clim[m]  # in/day for that calendar month
                rio = Raster.raster_from_array(clim_in, mg_swb)
                arr_in = rio.resample_to_grid(mg, band=rio.bands[0], method="nearest")  # in/day

                arr_ft = _process_arr(
                    arr_in,
                    sp_idx=sp,
                    date=date,
                    kind="MONTHLY_PRED",
                    min_thr=monthly_min_ft_per_day,
                    max_thr=monthly_max_ft_per_day,
                )
                tsa_dict[sp] = arr_ft

    # Extended debug: print some key SPs
    debug_sps = [0, 1, 35, 36, 323, 324]
    for sp in debug_sps:
        if sp in tsa_dict and sp in tdis_df.index and sp in debug_info:
            info = debug_info[sp]
            arr_ft = tsa_dict[sp]
            print(
                f"[DEBUG RCH] SP {sp} ({tdis_df.loc[sp, 'end'].date()}): "
                f"type={info.get('type','?')} "
                f"raw_in_min={info.get('raw_in_min', float('nan')):.9e} in/day, "
                f"raw_in_max={info.get('raw_in_max', float('nan')):.9e} in/day, "
                f"ft_before_min={info.get('ft_before_min', float('nan')):.9e} ft/day, "
                f"ft_before_max={info.get('ft_before_max', float('nan')):.9e} ft/day, "
                f"ft_smooth_min={info.get('ft_smooth_min', float('nan')):.9e} ft/day, "
                f"ft_smooth_max={info.get('ft_smooth_max', float('nan')):.9e} ft/day, "
                f"ft_final_min={np.nanmin(arr_ft):.9e}, "
                f"ft_final_max={np.nanmax(arr_ft):.9e} ft/day, "
                f"sigma={info.get('smooth_sigma', None)}"
            )

    if pdf is not None:
        pdf.close()

    return tsa_dict


def add_wells_from_process_script(
    gwf,
    *,
    monthly: bool = True,          # False -> use per_well_allocation_main_monthlyFalse.csv (annual)
    force_rebuild: bool = False,    # True -> re-run process_well_data.py
    pname: str = "wel",
    filename: str | None = None,
    verbose: bool = True,
):
    """
    Build a MODFLOW 6 WEL package from the outputs of process_well_data.py.

    Steps:
      1) Ensure process_well_data.py output exists (re-run if missing or force_rebuild=True).
      2) Load the allocation CSV:
           - monthly=True  -> per_well_allocation_main_monthlyTrue.csv (includes Month)
           - monthly=False -> per_well_allocation_main_monthlyFalse.csv (annual, Month may be NaN)
      3) For each well, compute layer by mid-screen elevation:
           - If bottom_scr == 0 and total_dept > 0, use bottom_scr := total_dept for midpoint.
           - If bottom_scr == 0 and total_dept == 0, force to layer 2 (index 1).
           - mid_elev = cell_top - 0.5*(top_screen + bottom_scr)  [depths below land surface]
           - choose k where botm[k] <= mid_elev <= (top if k==0 else botm[k-1]).
      4) Map (x_2265,y_2265) → (i,j) using modelgrid.intersect().
      5) Drop wells that fall outside idomain.
      6) Map (Year, Month) → stress period via TDIS start_date_time + perioddata:
           - monthly=True, Month not NaN: use mid-month (Year-Month-15)
           - monthly=False or Month NaN: use mid-year (Year-07-01)
      7) For each SP, sum discharges for wells falling in the same (k,i,j);
         convert to negative (extraction) and build WEL stress_period_data.
      8) Create/replace the WEL package.
    """
    import os
    import re
    import runpy
    import numpy as np
    import pandas as pd
    import warnings
    from collections import defaultdict

    sim = gwf.simulation
    mgrid = gwf.modelgrid

    if filename is None:
        filename = f"{gwf.name}.wel"

    # ---------------------------
    # 1) Ensure allocation exists
    # ---------------------------
    proj_root = os.getcwd()  # assume running from your project root/notebook working dir
    proc_script = os.path.join(proj_root, "process_well_data.py")

    out_dir = os.path.join("data", "processed", "water_use")
    main_name = f"per_well_allocation_main_{'monthlyTrue' if monthly else 'monthlyFalse'}.csv"
    alloc_csv = os.path.join(out_dir, main_name)

    need_run = force_rebuild or (not os.path.exists(alloc_csv))
    if need_run:
        if not os.path.exists(proc_script):
            raise FileNotFoundError(f"process_well_data.py not found at {proc_script}")
        if verbose:
            print(f"[WEL] Running {proc_script} to produce allocations...")
        # run in-process so paths remain relative
        runpy.run_path(proc_script, run_name="__main__")
        if not os.path.exists(alloc_csv):
            raise FileNotFoundError(f"Expected allocation CSV not found after run: {alloc_csv}")

    if verbose:
        print(f"[WEL] Using allocation file: {alloc_csv}")

    # ---------------------------
    # 2) Load allocation table
    # ---------------------------
    df = pd.read_csv(alloc_csv, dtype={"Well": str})

    # minimally required columns
    required_cols = ["Well", "x_2265", "y_2265", "Year", "cfd",
                     "top_screen", "bottom_scr", "total_dept"]
    if monthly:
        required_cols.append("Month")  # must exist for monthly mode

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Allocation CSV missing required column: {c}")

    # robust numeric conversion
    num_cols = ["x_2265", "y_2265", "Year", "cfd",
                "top_screen", "bottom_scr", "total_dept"]
    if "Month" in df.columns:
        num_cols.append("Month")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without coords or year;
    # keep rows even if cfd == 0 (they'll just contribute 0)
    df = df.dropna(subset=["x_2265", "y_2265", "Year"])
    if df.empty:
        warnings.warn("Allocation CSV is empty after basic filtering; no WEL package will be created.")
        return None, pd.DataFrame()

    # ---------------------------
    # 3) TDIS → stress period dates & Year/Month map
    # ---------------------------
    tdis = sim.get_package("tdis")
    if tdis is None:
        raise ValueError("TDIS package is required to map Years/Months → Stress Periods.")

    raw = tdis.start_date_time.get_data() if hasattr(tdis.start_date_time, "get_data") else tdis.start_date_time
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(raw))
    if not m:
        raise ValueError(f"Cannot parse start_date_time from TDIS: {raw!r}")
    start_date = pd.to_datetime(m.group(0))

    per = list(tdis.perioddata.get_data())  # (perlen, nstp, tsmult)
    perlen_days = [float(p[0]) for p in per]
    nper = len(perlen_days)

    # Build start/end date for each SP
    sp_start = [start_date]
    for i in range(1, nper):
        sp_start.append(sp_start[-1] + pd.Timedelta(days=perlen_days[i-1]))
    sp_end = [d + pd.Timedelta(days=pl) for d, pl in zip(sp_start, perlen_days)]

    # Helper: map (Year, Month) -> SP index
    # Helper: map (Year, Month) -> SP index
    def _row_to_sp(year, month):
        """
        For monthly=True and valid month: use mid-month date (Year-Month-15).
        For monthly=False or Month NaN: use mid-year (Year-07-01).
        Returns integer SP index or None if outside TDIS range.
        """
        if pd.isna(year):
            return None
        y = int(year)

        if monthly and not pd.isna(month):
            mth = int(month)
            # clamp month to [1,12] just in case
            mth = max(1, min(12, mth))
            day = 15
        else:
            # annual / pre-2000 style: mid-year
            mth = 7
            day = 1

        try:
            target = pd.Timestamp(year=y, month=mth, day=day)
        except Exception:
            return None

        for sp, (s, e) in enumerate(zip(sp_start, sp_end)):
            if s <= target < e:
                return sp
        return None  # outside model time


    # ---------------------------
    # 4) Grid arrays & helpers
    # ---------------------------
    top = np.asarray(gwf.dis.top.array)
    botm = np.asarray(gwf.dis.botm.array)   # shape (nlay, nrow, ncol)
    idomain = np.asarray(gwf.dis.idomain.array) if gwf.dis.idomain.array is not None else np.ones_like(botm, dtype=int)
    nlay, nrow, ncol = botm.shape

    def within_model(i, j):
        return (0 <= i < nrow) and (0 <= j < ncol)

    def cell_k_from_mid_elev(mid_elev, i, j):
        """
        Choose layer index k such that:
          k=0: botm[0,i,j] <= mid_elev <= top[i,j]
          k>0: botm[k,i,j] <= mid_elev <= botm[k-1,i,j]
        Return None if outside all layers.
        """
        # top boundary for each layer (top for k=0; botm[k-1] for k>0)
        upper = [top[i, j]] + [botm[k-1, i, j] for k in range(1, nlay)]
        lower = [botm[k, i, j] for k in range(nlay)]

        for k in range(nlay):
            up = float(upper[k])
            lo = float(lower[k])
            if np.isnan(up) or np.isnan(lo):
                continue
            if up >= mid_elev >= lo:
                return k
        return None

    # ---------------------------
    # 5) Locate (i,j), compute k per mid-screen rules, drop outside idomain
    # ---------------------------
    def to_ij(x, y):
        try:
            res = mgrid.intersect(x, y, local=False)
            if res is None:
                return None, None
            if isinstance(res, tuple):
                if len(res) == 3:
                    _, i, j = res
                elif len(res) == 2:
                    i, j = res
                else:
                    return None, None
                return int(i), int(j)
            return None, None
        except Exception:
            return None, None

    work = df.copy()
    ij = work.apply(lambda r: pd.Series(to_ij(r["x_2265"], r["y_2265"]), index=["i", "j"]), axis=1)
    work[["i", "j"]] = ij
    work = work.dropna(subset=["i", "j"])
    work[["i", "j"]] = work[["i", "j"]].astype(int)

    # mid-screen depth rules → mid_elev
    bs_eff = work["bottom_scr"].fillna(0.0).copy()
    td = work["total_dept"].fillna(0.0)
    bs_eff = np.where((bs_eff <= 0.0) & (td > 0.0), td, bs_eff)

    both_zero_mask = ((work["bottom_scr"].fillna(0.0) <= 0.0) & (td <= 0.0))

    work["_top_depth"] = work["top_screen"].fillna(0.0).clip(lower=0.0)
    work["_bot_depth"] = np.asarray(bs_eff, dtype=float).clip(min=0.0)
    swap_mask = work["_bot_depth"] < work["_top_depth"]
    if swap_mask.any():
        tmp = work.loc[swap_mask, "_bot_depth"].copy()
        work.loc[swap_mask, "_bot_depth"] = work.loc[swap_mask, "_top_depth"].values
        work.loc[swap_mask, "_top_depth"] = tmp.values

    work["_cell_top"] = work.apply(
        lambda r: float(top[int(r.i), int(r.j)]) if within_model(int(r.i), int(r.j)) else np.nan,
        axis=1,
    )
    work = work.dropna(subset=["_cell_top"])

    work["_mid_elev"] = work["_cell_top"] - 0.5 * (work["_top_depth"] + work["_bot_depth"])

    ks = []
    for idx, r in work.iterrows():
        i, j = int(r.i), int(r.j)
        if not within_model(i, j):
            ks.append(np.nan)
            continue

        if both_zero_mask.loc[idx]:
            # both 0 → force to layer 2 (index 1) if exists
            k = 1 if nlay >= 2 else 0
        else:
            k = cell_k_from_mid_elev(float(r["_mid_elev"]), i, j)
            if k is None:
                valid = [kk for kk in range(nlay) if idomain[kk, i, j] == 1]
                k = (valid[-1] if valid else 0)
        ks.append(k)
    work["k"] = np.array(ks, dtype=float)

    work = work.dropna(subset=["k"])
    work["k"] = work["k"].astype(int)

    # drop inactive cells
    inact_mask = work.apply(lambda r: idomain[int(r.k), int(r.i), int(r.j)] != 1, axis=1)
    work = work.loc[~inact_mask].copy()

    if work.empty:
        warnings.warn("No wells fall inside the active model domain; no WEL package will be created.")
        return None, pd.DataFrame()

    # ---------------------------
    # 6) Map each row to SP and sum by (sp, k, i, j)
    # ---------------------------
    # Convert positive cfd to negative extraction
    work["q_cfd"] = -work["cfd"].astype(float).fillna(0.0)

    # Add SP index per row
    if "Month" not in work.columns:
        work["Month"] = np.nan  # for safety
    work["sp"] = work.apply(
        lambda r: _row_to_sp(r["Year"], r["Month"]),
        axis=1,
    )
    work = work.dropna(subset=["sp"])
    work["sp"] = work["sp"].astype(int)

    if work.empty:
        warnings.warn("No rows could be mapped to model stress periods; no WEL package will be created.")
        return None, pd.DataFrame()

    # Sum by (sp, k, i, j)
    grouped = (
        work.groupby(["sp", "k", "i", "j"], dropna=False)["q_cfd"]
            .sum(min_count=1)
            .reset_index()
    )

    # Build wel_spd dict
    wel_spd = {}
    for sp, gSP in grouped.groupby("sp"):
        recs = []
        for _, r in gSP.iterrows():
            cell = (int(r["k"]), int(r["i"]), int(r["j"]))
            q = float(r["q_cfd"])
            recs.append((cell, q))
        if recs:
            wel_spd[int(sp)] = recs

    if verbose:
        filled = sorted(wel_spd.keys())
        print(f"[WEL] SPs with wells: {filled[:10]}{'...' if len(filled) > 10 else ''} (total {len(filled)})")

    # ---------------------------
    # 7) Create/replace WEL package
    # ---------------------------
    existing = gwf.get_package(pname)
    if existing is not None:
        gwf.remove_package(existing)

    wel_pkg = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd,
        pname=pname,
        save_flows=True,
        auto_flow_reduce=0.1,
        filename=filename,
    )

    # Small summary (first few rows)
    summary_rows = []
    for sp, rows in wel_spd.items():
        for cell, q in rows:
            k, i, j = cell
            summary_rows.append({"sp": sp, "k": k, "i": i, "j": j, "q_cfd": q})
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = (
            summary.groupby(["sp", "k", "i", "j"])["q_cfd"]
                   .sum()
                   .reset_index()
                   .sort_values(["sp", "k", "i", "j"])
                   .reset_index(drop=True)
        )

    if verbose:
        total_q = summary["q_cfd"].sum() if not summary.empty else 0.0
        print(
            f"[WEL] Created WEL package '{pname}' with {sum(len(v) for v in wel_spd.values()):,} "
            f"entries across {len(wel_spd)} stress periods. Total Q (cfd) = {total_q:,.2f}"
        )

    return wel_pkg, summary

def apply_monthly_climatology_to_predictive_wel(
    gwf,
    *,
    sp_starts,
    pred_start_year=2024,
    clim_years=(2019, 2020, 2021, 2022, 2023),
    pname="wel",
    fill_missing_as_zero=True,
    drop_zero_records=True,
):
    """
    Overwrite the MF6 WEL package stress_period_data for predictive stress periods (>= pred_start_year)
    using monthly climatology from clim_years.

    The climatology is computed per cellid (k,i,j) and calendar month.
    For each month m:
        q_pred(cellid, m) = mean_{year in clim_years}( q_hist(cellid, year, m) )
    Missing cellid in a given (year, month) can be treated as 0 (default) or ignored.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model object.
    sp_starts : sequence of datetime-like
        Start date for each stress period (len == nper).
    pred_start_year : int
        Predictive starts at this year (inclusive).
    clim_years : iterable[int]
        Years used for monthly climatology.
    pname : str
        Package name to fetch (your script uses "wel").
    fill_missing_as_zero : bool
        If True, missing cellid in a given year-month counts as 0 for averaging.
        If False, average only over years where the cellid exists.
    drop_zero_records : bool
        If True, do not write records with q==0 into predictive SPDs.

    Returns
    -------
    None
        Edits the WEL package in-place.
    """
    import numpy as np

    wel = gwf.get_package(pname)
    if wel is None:
        raise ValueError(f"Could not find a WEL package named '{pname}' on this model.")

    spd = wel.stress_period_data.get_data()  # dict: kper -> recarray/list

    def _iter_cell_q_records(recdata):
        """
        Yield (cellid_tuple, q_float) from flopy MF6 WEL stress_period_data formats.
        Handles:
          - numpy recarray with fields like ('cellid','q') or ('k','i','j','q')
          - list of tuples like ((k,i,j), q) or (k,i,j,q)
        """
        if recdata is None:
            return
        # numpy structured/recarray
        if hasattr(recdata, "dtype") and getattr(recdata.dtype, "names", None):
            names = recdata.dtype.names
            if "cellid" in names and "q" in names:
                for r in recdata:
                    cid = tuple(int(x) for x in r["cellid"])
                    q = float(r["q"])
                    yield cid, q
            elif all(n in names for n in ("k", "i", "j", "q")):
                for r in recdata:
                    cid = (int(r["k"]), int(r["i"]), int(r["j"]))
                    q = float(r["q"])
                    yield cid, q
            else:
                raise ValueError(f"Unrecognized WEL recarray fields: {names}")
        else:
            # list/iterable
            for r in recdata:
                if isinstance(r, (list, tuple)):
                    if len(r) == 2 and isinstance(r[0], (list, tuple)):
                        cid = tuple(int(x) for x in r[0])
                        q = float(r[1])
                        yield cid, q
                    elif len(r) == 4:
                        cid = (int(r[0]), int(r[1]), int(r[2]))
                        q = float(r[3])
                        yield cid, q
                    else:
                        raise ValueError(f"Unrecognized WEL tuple record: {r}")
                else:
                    raise ValueError(f"Unrecognized WEL record type: {type(r)}")

    # ---- Build year-month -> {cellid: q} from clim years ----
    ym_cell_q = {}   # (year, month) -> dict[cellid] = q
    all_cells = set()

    for sp, dt in enumerate(sp_starts):
        if dt.year not in clim_years:
            continue
        recs = spd.get(sp, None)
        cell_q = {}
        for cid, q in _iter_cell_q_records(recs):
            cell_q[cid] = cell_q.get(cid, 0.0) + q  # if duplicates, sum within SP
            all_cells.add(cid)
        ym_cell_q[(dt.year, dt.month)] = cell_q

    if not ym_cell_q:
        raise ValueError(
            f"No WEL stress-period data found for clim_years={tuple(clim_years)}. "
            "Check sp_starts alignment and that historical wells exist in those years."
        )

    years = list(clim_years)

    # ---- Month -> dict[cellid] = climatology mean ----
    month_clim = {m: {} for m in range(1, 13)}

    for m in range(1, 13):
        for cid in all_cells:
            vals = []
            for y in years:
                q = ym_cell_q.get((y, m), {}).get(cid, None)
                if q is None:
                    if fill_missing_as_zero:
                        vals.append(0.0)
                else:
                    vals.append(float(q))
            if not vals:
                continue
            month_clim[m][cid] = float(np.mean(vals))

    # ---- Overwrite predictive SPDs ----
    for sp, dt in enumerate(sp_starts):
        if dt.year >= pred_start_year:
            m = dt.month
            clim_map = month_clim.get(m, {})
            new_recs = []
            for cid, q in clim_map.items():
                if drop_zero_records and (q == 0.0):
                    continue
                new_recs.append((cid, q))
            # Flopy accepts list of ((k,i,j), q)
            spd[sp] = new_recs

    # Push back into package
    wel.stress_period_data.set_data(spd)


def apply_monthly_climatology_to_predictive_rch(
    rch_dict,
    *,
    sp_starts,
    pred_start_year=2024,
    clim_years=(2019, 2020, 2021, 2022, 2023),
):
    """
    Overwrite recharge arrays in rch_dict for predictive stress periods (>= pred_start_year)
    using a monthly climatology computed from clim_years.

    Parameters
    ----------
    rch_dict : dict[int, array_like]
        Mapping kper -> 2D recharge array (nrow, ncol).
    sp_starts : sequence of datetime-like
        Start date for each stress period; len(sp_starts) == nper.
    pred_start_year : int
        Predictive period begins at this calendar year (inclusive).
    clim_years : iterable[int]
        Years used for monthly averaging (e.g., 2019-2023).

    Returns
    -------
    dict[int, np.ndarray]
        Updated rch_dict (same object, modified in-place, also returned).
    """
    import numpy as np

    # month -> list of arrays from clim years
    month_arrays = {m: [] for m in range(1, 13)}
    for sp, dt in enumerate(sp_starts):
        if dt.year in clim_years and sp in rch_dict:
            month_arrays[dt.month].append(np.asarray(rch_dict[sp], float))

    # month -> mean array
    month_mean = {}
    for m, arrs in month_arrays.items():
        if arrs:
            month_mean[m] = np.nanmean(np.stack(arrs, axis=0), axis=0)

    # overwrite predictive SPs
    for sp, dt in enumerate(sp_starts):
        if dt.year >= pred_start_year:
            m = dt.month
            if m in month_mean:
                rch_dict[sp] = month_mean[m].copy()

    return rch_dict



def build_and_plot_group_summary(
    *,
    sim_ws: str,
    model_name: str,
    out_pdf: str,
    annual_flag: bool = True,
    hds_path: str | None = None,
    budget_csv_name: str = "budget.csv",
    index_flag_column: str = "index_well_flag",
) -> str:
    """
    Build smoothed/grouped WL targets (master_df) + cell lookup (tdf) using elk_obs,
    then generate a multi-page PDF:
      p1: 1:1 Obs vs Sim with Overall RMSE  (INDEX wells highlighted)
      p2: Budget IN/OUT time-series from budget.csv (if present)
      p3: Diagnostic: map view of Layer 1 & 2 simulated head contours at SP 60 (2023),
          with BC overlays (RIV green, DRN yellow, GHB blue, WELL red) and flooded/dry masks
      p4: Same diagnostic maps + average residual points (Obs - Sim; red=+, blue=-), jittered to mitigate overlap
      p5+: Per-group pages (hydrograph + lith + map + Group RMSE)

    Now also:
      • On per-group hydrograph page, if an observation's cell (k,i,j) coincides with a pumping well
        in the WEL package, plot the pumping rate on a second y-axis (twinx) as a time series.

    Requires:
      import elk01_water_level_processing as elk_obs
    """
    import os, re, glob
    from typing import Tuple, List, Dict
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap
    import flopy
    try:
        import geopandas as gpd  # optional
    except Exception:
        gpd = None

    # ---------------- helper functions ----------------
    def _parse_mf6_start_date(tdis) -> pd.Timestamp:
        s = str(getattr(tdis.start_date_time, "data", getattr(tdis, "start_date_time", ""))).strip()
        m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", s)
        if m:
            return pd.to_datetime(m.group(1))
        try:
            return pd.to_datetime(s)
        except Exception:
            raise ValueError(f"Unrecognized TDIS start_date_time: {s!r}")

    def _rmse(a, b) -> float:
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.size == 0: return np.nan
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _coerce_bool(v) -> bool:
        if pd.isna(v): return False
        if isinstance(v, (bool, np.bool_)): return bool(v)
        if isinstance(v, (int, np.integer, float, np.floating)): return bool(int(v))
        return str(v).strip().lower() in {"1","y","yes","t","true"}

    def _sim_ts_for_cell(hds_path: str, k: int, i: int, j: int, t0: pd.Timestamp) -> pd.Series:
        from flopy.utils import HeadFile
        hf = HeadFile(hds_path)
        totims = np.asarray(hf.get_times(), float)
        vals = [float(hf.get_data(totim=t)[k, i, j]) for t in totims]
        idx = pd.to_datetime(t0) + pd.to_timedelta(totims, unit="D")
        s = pd.Series(vals, index=idx).sort_index()
        s = s.where(np.abs(s) < 1e20, np.nan).dropna()
        return s

    def _sp_starts_and_totims(tdis) -> Tuple[List[pd.Timestamp], List[float]]:
        """Compute per-SP start datetimes and cumulative end-of-SP totims (days)."""
        t0 = _parse_mf6_start_date(tdis)
        pdta = tdis.perioddata.array
        perlen = [float(row[0]) for row in pdta]  # perlen, nstp, tsmult, ...
        starts = [t0]
        for L in perlen[:-1]:
            starts.append(starts[-1] + pd.to_timedelta(L, unit="D"))
        # cumulative END times in days after t0
        totims = list(np.cumsum(perlen))
        return starts, totims

    def _wel_series_for_cell(gwf, tdis, k: int, i: int, j: int, t0: pd.Timestamp) -> pd.Series:
        """
        Build a pumping time series (cfd) for a given cell (k,i,j) using WEL package(s).

        Assumes:
            - Model arrays and (k,i,j) arguments are 0-based.
            - Flopy MF6 WEL stress-period cellids are already 0-based
            (layer, row, col), so they can be compared directly.
        """
        wel_pkgs = []
        for name in gwf.get_package_list():
            pkg = gwf.get_package(name)
            ptype = getattr(pkg, "package_type", "").lower()
            if ptype == "wel" or ptype.endswith("wel"):
                wel_pkgs.append(pkg)

        if not wel_pkgs:
            return pd.Series(dtype=float)

        pdta = tdis.perioddata.array
        perlen = [float(row[0]) for row in pdta]
        nper = len(perlen)
        q_vals = np.zeros(nper, dtype=float)

        for pkg in wel_pkgs:
            spd = getattr(pkg, "stress_period_data", None)
            if spd is None:
                continue

            for kper in range(nper):
                try:
                    data = spd.get_data(kper)
                except Exception:
                    data = None
                if data is None or len(data) == 0:
                    continue

                val_field = None
                for cand in ("q", "rate", "flux"):
                    if cand in data.dtype.names:
                        val_field = cand
                        break
                if val_field is None:
                    continue

                cids = data["cellid"]
                qs   = np.asarray(data[val_field], float)

                for cid, q in zip(cids, qs):
                    try:
                        # already 0-based from flopy
                        kk = int(cid[0])
                        ii = int(cid[1])
                        jj = int(cid[2])
                    except Exception:
                        continue

                    if kk == k and ii == i and jj == j:
                        q_vals[kper] += q

        # SP midpoints
        idx = []
        cur = pd.to_datetime(t0)
        for L in perlen:
            mid = cur + pd.to_timedelta(L / 2.0, unit="D")
            idx.append(mid)
            cur = cur + pd.to_timedelta(L, unit="D")

        ser = pd.Series(q_vals, index=pd.DatetimeIndex(idx))
        ser = ser.where(ser != 0.0, np.nan).dropna()
        return ser



    def _heads_at_sp(hds_path: str, tdis, kper: int) -> np.ndarray:
        """Get full (k,i,j) head array at end of stress period kper (0-based)."""
        from flopy.utils import HeadFile
        hf = HeadFile(hds_path)
        times = np.asarray(hf.get_times(), float)
        _, sp_totims = _sp_starts_and_totims(tdis)
        tgt = float(sp_totims[kper])
        tsel = float(times[np.argmin(np.abs(times - tgt))])
        H = hf.get_data(totim=tsel)
        return H

    def _vertices_for_pcolor(mg) -> Tuple[np.ndarray, np.ndarray]:
        """Return cell corner grids (nrow+1, ncol+1) honoring rotation."""
        nrow, ncol = int(mg.nrow), int(mg.ncol)
        try:
            XV = np.asarray(mg.xvertices)
            YV = np.asarray(mg.yvertices)
            if XV.shape == (nrow+1, ncol+1) and YV.shape == (nrow+1, ncol+1):
                return XV, YV
        except Exception:
            pass
        # Fallback (non-rotated)
        xc, yc = np.asarray(mg.xcellcenters), np.asarray(mg.ycellcenters)
        def _edges_from_centers_1d(c1d):
            c = np.ravel(c1d).astype(float)
            inc = (c[-1] > c[0])
            if not inc: c = c[::-1]
            mids = 0.5 * (c[1:] + c[:-1])
            first = c[0] - (mids[0] - c[0]); last = c[-1] + (c[-1] - mids[-1])
            e = np.concatenate([[first], mids, [last]])
            return e if inc else e[::-1]
        xe = _edges_from_centers_1d(xc[0,:])
        ye = _edges_from_centers_1d(yc[:,0])
        XE, YE = np.meshgrid(xe, ye)
        return XE, YE

    def _layer_top_bot_arrays(gwf) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return lists of per-layer top and bottom arrays shaped (nrow,ncol)."""
        dis = gwf.dis
        top = np.asarray(dis.top.array)
        botm = np.asarray(dis.botm.array)  # shape (nlay, nrow, ncol)
        nlay = int(dis.nlay.data)
        tops, bots = [], []
        for k in range(nlay):
            lk_top = top if k == 0 else botm[k-1, :, :]
            lk_bot = botm[k, :, :]
            tops.append(lk_top)
            bots.append(lk_bot)
        return tops, bots

    def _collect_bc_masks_for_sp(gwf, kper: int, types=("RIV","DRN","GHB","WEL")) -> Dict[str, np.ndarray]:
        """
        For the given stress period, return dict of {type:(nlay,nrow,ncol) bool mask}.
        Package type is matched by 'package_type' (riv, drn, ghb, wel).
        NOTE: This preserves layer specificity so we can draw BCs only on the layer they are in.
        """
        dis = gwf.dis
        nlay, nrow, ncol = int(dis.nlay.data), int(dis.nrow.data), int(dis.ncol.data)
        out = {t: np.zeros((nlay, nrow, ncol), dtype=bool) for t in [t.upper() for t in types]}
        want = {t.lower() for t in types}
        for name in gwf.get_package_list():
            pkg = gwf.get_package(name)
            ptype = getattr(pkg, "package_type", "").lower()
            if ptype not in want:
                continue
            spd = getattr(pkg, "stress_period_data", None)
            if spd is None:
                continue
            try:
                rec = spd.get_data(kper)
            except Exception:
                rec = None
            if rec is None:
                continue
            for cell in rec:
                cellid = cell["cellid"]
                try:
                    k, r, c = int(cellid[0]), int(cellid[1]), int(cellid[2])
                except Exception:
                    k = int(getattr(cellid, "layer", 0))
                    r = int(getattr(cellid, "row", -1))
                    c = int(getattr(cellid, "column", -1))
                if 0 <= k < nlay and 0 <= r < nrow and 0 <= c < ncol:
                    out[ptype.upper()][k, r, c] = True
        return out

    # ---------------- load model & inputs ----------------
    print("loading simulation...")
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    gwf = sim.get_model(model_name)
    tdis = sim.get_package("tdis")
    t0 = _parse_mf6_start_date(tdis)
    mg = gwf.modelgrid
    nlay = int(gwf.dis.nlay.data)
    nrow, ncol = int(mg.nrow), int(mg.ncol)
    xcc, ycc = np.asarray(mg.xcellcenters), np.asarray(mg.ycellcenters)
    XE, YE = _vertices_for_pcolor(mg)  # rotation-aware

    # resolve headfile
    if hds_path is None:
        candidates = []
        candidates.extend(glob.glob(os.path.join(sim_ws, "*.hds*")))
        candidates.extend(glob.glob(os.path.join(sim_ws, model_name, "*.hds*")))
        if not candidates:
            raise FileNotFoundError("Could not auto-detect headfile (*.hds*). Pass hds_path= explicitly.")
        hds_path = candidates[0]

    # elk_obs-driven inputs
    from elk01_water_level_processing import (
        plot_lith_multiwell as _plot_lith_multiwell,
        load_input_data as _elk_load_input_data,
        save_processed_WL_data as _elk_save_processed_WL_data,
    )
    sites, water_levels, modelgrid = _elk_load_input_data(mnm=model_name)

    # normalize/ensure index flag
    if index_flag_column in sites.columns:
        sites[index_flag_column] = sites[index_flag_column].map(_coerce_bool)
    else:
        sites[index_flag_column] = False

    # smoothed per-site series (columns=site_index)
    main_df = _elk_save_processed_WL_data(sites, water_levels, annual_flag)

    # rename each column to include i,j,k,grp metadata
    for col in main_df.columns:
        site_info = sites[sites["site_index"] == col]
        new_header = (
            f"sid.{col}"
            f"_i.{int(site_info['row'].values[0]-1)}"
            f"_j.{int(site_info['col'].values[0]-1)}"
            f"_k.{int(site_info['k'].values[0])}"
            f"_grp.{int(site_info['group_number'].values[0])}"
        )
        main_df.rename(columns={col: new_header}, inplace=True)

    def _parse_sort_key(c):
        m = re.search(r"k\.(\d+)_grp\.(\d+)", c)
        if m: return (int(m.group(2)), int(m.group(1)))
        return (10**9, 10**9)

    main_df = main_df[sorted(main_df.columns, key=_parse_sort_key)]

    # collapse duplicates per (grp,k) using median across columns with same k in group
    master_df = pd.DataFrame(index=main_df.index)
    groups = main_df.columns.str.extract(r"_grp\.(\d+)$")[0].astype(int)
    by_grp_k: Dict[Tuple[int,int], List[str]] = {}
    for col in main_df.columns:
        g = int(re.search(r"_grp\.(\d+)$", col).group(1))
        k = int(re.search(r"_k\.(\d+)_", col).group(1))
        by_grp_k.setdefault((g, k), []).append(col)

    for (grp, k), cols in sorted(by_grp_k.items()):
        if len(cols) == 1:
            master_df[f"grp.{grp}_k.{k}"] = main_df[cols[0]]
        else:
            master_df[f"grp.{grp}_k.{k}"] = main_df[cols].median(axis=1, skipna=True)

    # build tdf (grp,k → i,j) from sites (first match)
    tdf_rows = []
    for col in master_df.columns:
        grp = int(re.search(r"grp\.(\d+)_k\.\d+", col).group(1))
        k   = int(re.search(r"_k\.(\d+)$", col).group(1))
        sub = sites[(sites["group_number"] == grp) & (sites["k"] == k)]
        if sub.empty:
            continue
        i = int(sub["row"].values[0] - 1)
        j = int(sub["col"].values[0] - 1)
        obsprefix = f"transh_grpid:{grp}_k:{k}_i:{i}_j:{j}"
        tdf_rows.append(dict(obsprefix=obsprefix, grpid=grp, k=k, i=i, j=j))
    tdf = pd.DataFrame(tdf_rows)

    # ---------------- collect pairs for 1:1 scatter & prep residual calc ----------------
    def _pairs_for_col(c, g, k):
        sm = master_df[c].dropna()
        rc = key_to_rc.get((g, k))
        if sm.empty or not rc: return None
        i, j = rc
        ser_sim = _sim_ts_for_cell(hds_path, k, i, j, t0)
        if ser_sim.empty: return None
        ab = sm.astype(float).to_frame("obs").join(ser_sim.astype(float).to_frame("sim"), how="inner").dropna()
        if ab.empty: return None
        return ab

    key_to_rc = { (int(r["grpid"]), int(r["k"])): (int(r["i"]), int(r["j"])) for _, r in tdf.iterrows() }
    parsed_cols = []
    rx = re.compile(r"^grp\.(\d+)_k\.(\d+)$")
    for c in master_df.columns:
        m = rx.match(c)
        if not m: continue
        g = int(m.group(1)); k = int(m.group(2))
        parsed_cols.append((c, g, k))

    # mark index pairs (grp,k)
    index_pairs = set()
    if index_flag_column in sites.columns:
        idx_sites = sites[sites[index_flag_column]]
        for _, r in idx_sites[["group_number","k"]].dropna().iterrows():
            try:
                index_pairs.add((int(r["group_number"]), int(r["k"])))
            except Exception:
                pass

    # For page 1 (overall RMSE)
    all_obs_vals, all_sim_vals = [], []
    idx_obs_vals, idx_sim_vals, oth_obs_vals, oth_sim_vals = [], [], [], []
    for c, g, k in parsed_cols:
        ab = _pairs_for_col(c, g, k)
        if ab is None: continue
        obs_vals = ab["obs"].values.tolist()
        sim_vals = ab["sim"].values.tolist()
        all_obs_vals.extend(obs_vals); all_sim_vals.extend(sim_vals)
        if (g, k) in index_pairs:
            idx_obs_vals.extend(obs_vals); idx_sim_vals.extend(sim_vals)
        else:
            oth_obs_vals.extend(obs_vals); oth_sim_vals.extend(sim_vals)

    # order groups: index groups first
    sites["is_index"] = sites[index_flag_column].map(bool)
    index_groups = sorted(sites.loc[sites["is_index"], "group_number"].astype(int).unique().tolist())
    all_groups   = sorted({g for _, g, _ in parsed_cols})
    ordered_groups = index_groups + [g for g in all_groups if g not in index_groups]

    # ---------------- open PDF and render pages ----------------
    out_pdf = os.path.abspath(out_pdf)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    elk_boundary = None
    try:
        elk_boundary = gpd.read_file(os.path.join("..", "..", "gis", "input_shps", "elk", "elk_boundary.shp"))
    except Exception:
        pass
    aq_color_map = {0: "#774B12", 1: "#061B79"}

    with PdfPages(out_pdf) as pdf:
        # Page 1: 1:1 scatter (INDEX emphasized)
        if len(all_obs_vals):
            all_obs_vals = np.asarray(all_obs_vals, float)
            all_sim_vals = np.asarray(all_sim_vals, float)
            overall_rmse = _rmse(all_obs_vals, all_sim_vals)

            fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
            if len(oth_obs_vals):
                ax.scatter(np.asarray(oth_obs_vals, float), np.asarray(oth_sim_vals, float),
                           s=10, alpha=0.25, edgecolors="none", label="Other wells")
            if len(idx_obs_vals):
                ax.scatter(np.asarray(idx_obs_vals, float), np.asarray(idx_sim_vals, float),
                           s=16, alpha=0.8, edgecolors="none", label="INDEX wells")
            if not len(oth_obs_vals) and not len(idx_obs_vals):
                ax.scatter(all_obs_vals, all_sim_vals, s=10, alpha=0.35, edgecolors="none", label="All wells")
            lo = float(np.nanmin([all_obs_vals.min(), all_sim_vals.min()]))
            hi = float(np.nanmax([all_obs_vals.max(), all_sim_vals.max()]))
            pad = 0.02 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.25)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.grid(True, alpha=0.3)
            ax.set_title("Observed vs Simulated Heads — All Groups")
            ax.set_xlabel("Observed (ft NAVD88)")
            ax.set_ylabel("Simulated (ft NAVD88)")
            ax.text(0.02, 0.98, f"Overall RMSE: {overall_rmse:.2f} ft\nN = {all_obs_vals.size}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
            ax.legend(loc="best", frameon=True)
            pdf.savefig(fig); plt.close(fig)

        # Page 2: Budget (optional)
        budget_csv = os.path.join(sim_ws, budget_csv_name)
        if os.path.exists(budget_csv):
            def _read_budget_csv(budget_csv_path: str, t0: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
                bdf = pd.read_csv(budget_csv_path)
                candidates = [c for c in bdf.columns if c.lower() in ("time", "totim", "datetime", "date")]
                if not candidates:
                    raise ValueError("No time/totim/datetime column found in budget CSV.")
                tcol = candidates[0]
                tser = bdf[tcol]
                if np.issubdtype(tser.dtype, np.number):
                    idx = pd.to_datetime(t0) + pd.to_timedelta(tser.values, unit="D")
                else:
                    try:
                        idx = pd.to_datetime(tser, errors="raise")
                    except Exception:
                        vals = pd.to_numeric(tser, errors="coerce")
                        if np.isfinite(vals).any():
                            idx = pd.to_datetime(t0) + pd.to_timedelta(vals.values, unit="D")
                        else:
                            raise
                in_cols  = [c for c in bdf.columns if c.endswith("_IN")]
                out_cols = [c for c in bdf.columns if c.endswith("_OUT")]
                def _drop_all_zero(df):
                    if df.empty: return df
                    keep = (df.fillna(0).abs().sum(axis=0) > 0)
                    return df.loc[:, keep.values]
                df_in  = _drop_all_zero(bdf[in_cols].copy()  if in_cols  else pd.DataFrame(index=bdf.index))
                df_out = _drop_all_zero(bdf[out_cols].copy() if out_cols else pd.DataFrame(index=bdf.index))
                df_in.index  = idx
                df_out.index = idx
                return df_in, df_out, idx

            try:
                df_in, df_out, _ = _read_budget_csv(budget_csv, t0)
                fig, (ax_in, ax_out) = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), sharex=True, constrained_layout=True)
                if not df_in.empty:
                    for c in df_in.columns: ax_in.plot(df_in.index, df_in[c].values, label=c)
                    ax_in.set_title("Flow Terms — IN"); ax_in.set_ylabel("Flow"); ax_in.grid(True, alpha=0.3)
                    ax_in.legend(fontsize=8, ncol=2, frameon=True, loc="upper left")
                if not df_out.empty:
                    for c in df_out.columns: ax_out.plot(df_out.index, df_out[c].values, label=c)
                    ax_out.set_title("Flow Terms — OUT"); ax_out.set_ylabel("Flow"); ax_out.set_xlabel("Time"); ax_out.grid(True, alpha=0.3)
                    ax_out.legend(fontsize=8, ncol=2, frameon=True, loc="upper left")
                pdf.savefig(fig); plt.close(fig)
            except Exception as e:
                print(f"[budget] Skipping budget page: {e}")

        # ---- Precompute for Page 3 & 4 diagnostic maps ----
        try:
            sp_starts, sp_totims = _sp_starts_and_totims(tdis)
            target_year = 2023
            kper60 = None
            for kper, dt in enumerate(sp_starts):
                if dt.year == target_year:
                    kper60 = kper
                    break
            if kper60 is None:
                kper60 = 59  # fallback if schedule differs

            H = _heads_at_sp(hds_path, tdis, kper60)  # shape (nlay, nrow, ncol)
            tops, bots = _layer_top_bot_arrays(gwf)

            def _flood_dry_masks_for_k(k):
                Z = np.asarray(H[k, :, :], float)
                topk = np.asarray(tops[k], float)
                botk = np.asarray(bots[k], float)
                flooded = (Z > topk)
                dry = (Z < botk)
                bad = ~np.isfinite(Z) | (np.abs(Z) > 1e20)
                flooded[bad] = False
                dry[bad] = False
                return Z, flooded, dry

            bc_masks = _collect_bc_masks_for_sp(gwf, kper60, types=("RIV","DRN","GHB","WEL"))

            color_bc = {"RIV": "#2ca02c", "DRN": "#ffd700", "GHB": "#1f77b4", "WEL": "#d62728"}
            alpha_bc = 0.45
            color_flood = "#4169e1"
            color_dry   = "#87cefa"
            alpha_fd    = 0.28

            def _draw_layer(ax, k: int):
                Z, flooded, dry = _flood_dry_masks_for_k(k)
                finiteZ = Z[np.isfinite(Z) & (np.abs(Z) < 1e20)]
                if finiteZ.size == 0:
                    levels = None
                else:
                    vmin, vmax = np.nanpercentile(finiteZ, [5, 95])
                    if vmin == vmax:
                        levels = np.linspace(vmin-1, vmax+1, 5)
                    else:
                        levels = np.linspace(vmin, vmax, 10)
                if levels is not None:
                    cs = ax.contour(xcc, ycc, Z, levels=levels, colors="k", linewidths=0.8, alpha=0.9)
                    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
                if k == 0:
                    arr_flood = np.where(flooded, 1.0, np.nan)
                    if np.isfinite(arr_flood).any():
                        ax.pcolormesh(XE, YE, arr_flood, cmap=ListedColormap([color_flood]),
                                      vmin=0, vmax=1, alpha=alpha_fd, shading="flat")
                arr_dry   = np.where(dry, 1.0, np.nan)
                if np.isfinite(arr_dry).any():
                    ax.pcolormesh(XE, YE, arr_dry, cmap=ListedColormap([color_dry]),
                                  vmin=0, vmax=1, alpha=alpha_fd, shading="flat")
                for bc_name in ("RIV","DRN","GHB","WEL"):
                    m3d = bc_masks.get(bc_name, None)
                    if m3d is None or k >= m3d.shape[0]:
                        continue
                    mask_k = m3d[k, :, :]
                    if mask_k.any():
                        ax.pcolormesh(XE, YE, np.where(mask_k, 1.0, np.nan),
                                      cmap=ListedColormap([color_bc[bc_name]]),
                                      vmin=0, vmax=1, alpha=alpha_bc, shading="flat")
                try:
                    if gpd is not None and elk_boundary is not None and len(elk_boundary) > 0:
                        elk_boundary.boundary.plot(ax=ax, color="blue", linewidth=1.0)
                except Exception:
                    pass
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("X (ft)"); ax.set_ylabel("Y (ft)")
                ax.set_title(f"Layer {k+1} head contours — SP {kper60+1} ({target_year})")

            # Page 3: diagnostic (no residuals)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5), constrained_layout=True)
            _draw_layer(ax1, 0)
            if nlay >= 2:
                _draw_layer(ax2, 1)
            else:
                ax2.axis("off"); ax2.text(0.5, 0.5, "Model has < 2 layers", ha="center", va="center")
            leg_ax = fig.add_axes([0.82, 0.12, 0.16, 0.25]); leg_ax.axis("off")
            handles = [
                Patch(facecolor=color_flood, edgecolor="none", alpha=alpha_fd, label="Flooded (head > top, L1 only)"),
                Patch(facecolor=color_dry,   edgecolor="none", alpha=alpha_fd, label="Dry (head < bottom)"),
                Patch(facecolor=color_bc["RIV"], edgecolor="none", alpha=alpha_bc, label="RIV"),
                Patch(facecolor=color_bc["DRN"], edgecolor="none", alpha=alpha_bc, label="DRN"),
                Patch(facecolor=color_bc["GHB"], edgecolor="none", alpha=alpha_bc, label="GHB"),
                Patch(facecolor=color_bc["WEL"], edgecolor="none", alpha=alpha_bc, label="WELL"),
                Line2D([0], [0], color="k", lw=0.8, label="Head contours"),
            ]
            leg_ax.legend(handles=handles, loc="center left", frameon=True, fontsize=8)
            pdf.savefig(fig); plt.close(fig)

            # -------- Page 4: residual maps (average obs - sim) --------
            # Compute residuals with SAME pairing as used for RMSE
            res_rows = []
            for c, g, k in parsed_cols:
                rc = key_to_rc.get((g, k))
                if not rc: continue
                i, j = rc
                ab = _pairs_for_col(c, g, k)
                if ab is None or ab.empty: continue
                # average residual (obs - sim) across overlapping times
                mean_res = float((ab["obs"] - ab["sim"]).mean())
                res_rows.append(dict(grp=g, k=k, i=i, j=j, resid=mean_res))
            residuals_df = pd.DataFrame(res_rows)

            # jitter utility to separate overlapping markers in same cell
            try:
                delx = np.nanmedian(np.diff(np.unique(xcc.flatten())))
                dely = np.nanmedian(np.diff(np.unique(ycc.flatten())))
                cell_scale = float(np.nanmin([delx, dely]))
                if not np.isfinite(cell_scale) or cell_scale <= 0:
                    cell_scale = 1.0
            except Exception:
                cell_scale = 1.0

            def _seed_from_tuple(*ints) -> int:
                h = hash(tuple(int(x) for x in ints)) & 0xFFFFFFFF
                if h == 0:
                    h = 12345
                return int(h)

            def _jitter(g:int, k:int, i:int, j:int) -> tuple[float, float]:
                rng = np.random.default_rng(_seed_from_tuple(g, k, i, j))
                ang = rng.uniform(0.0, 2.0*np.pi)
                rad = rng.uniform(0.05, 0.18) * cell_scale
                return rad*np.cos(ang), rad*np.sin(ang)

            # size scaling
            if not residuals_df.empty:
                mag = residuals_df["resid"].abs().values
                p95 = np.nanpercentile(mag, 95) if np.isfinite(mag).any() else 1.0
                scale = 180.0 / (p95 if p95 > 0 else 1.0)  # px per ft at 95th percentile
                base  = 20.0
            else:
                scale, base = 150.0, 15.0

            # draw map with residuals
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5), constrained_layout=True)
            for kk, ax in enumerate([ax1, ax2]):
                _draw_layer(ax, kk)
                if residuals_df.empty: 
                    continue
                res_sub = residuals_df[residuals_df["k"] == kk]
                if res_sub.empty: 
                    continue
                # split by sign
                pos = res_sub[res_sub["resid"] >= 0.0]
                neg = res_sub[res_sub["resid"] <  0.0]

                def _plot_res(df_res, color):
                    if df_res.empty: return
                    xs, ys, ss = [], [], []
                    for _, r in df_res.iterrows():
                        i, j = int(r["i"]), int(r["j"])
                        if not (0 <= i < nrow and 0 <= j < ncol):
                            continue
                        x0, y0 = xcc[i, j], ycc[i, j]
                        dx, dy = _jitter(int(r["grp"]), int(r["k"]), i, j)
                        xs.append(x0 + dx)
                        ys.append(y0 + dy)
                        ss.append(base + scale * abs(float(r["resid"])))
                    if xs:
                        ax.scatter(xs, ys, s=ss, c=color, alpha=0.75, edgecolors="k", linewidths=0.4, zorder=10)

                _plot_res(neg, "#d62728")  # red
                _plot_res(pos, "#1f77b4")  # blue

            # add residual legend
            leg_ax = fig.add_axes([0.82, 0.12, 0.16, 0.25]); leg_ax.axis("off")
            # example sizes
            ex_vals = [0.25, 1.0, 3.0]  # ft
            ex_sizes = [base + scale*v for v in ex_vals]
            handles = [
                Line2D([0], [0], marker="o", color="k", markerfacecolor="#d62728", markersize=np.sqrt(s),
                       linestyle="None", label="+ residual (Obs < Sim)", markeredgewidth=0.6)
                for s in [max(ex_sizes)]  # one color entry
            ] + [
                Line2D([0], [0], marker="o", color="k", markerfacecolor="#1f77b4", markersize=np.sqrt(s),
                       linestyle="None", label="- residual (Obs > Sim)", markeredgewidth=0.6)
                for s in [max(ex_sizes)]  # one color entry
            ]
            leg_ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
            # size scale text
            y0 = 0.05
            leg_ax.text(0.0, y0+0.35, "Marker size ∝ |residual| (ft)", fontsize=8)
            for kx, (v, s) in enumerate(zip(ex_vals, ex_sizes)):
                leg_ax.scatter([0.08 + 0.12*kx], [y0], s=s, c="#666666", edgecolors="k")
                leg_ax.text(0.12 + 0.12*kx, y0, f"{v:g}", va="center", fontsize=8)
            pdf.savefig(fig); plt.close(fig)

        except Exception as e:
            print(f"[diagnostic SP60/residuals] Skipping diagnostic and/or residual maps: {e}")

        # Pages 5+: per-group
        for grp in ordered_groups:
            grp_cols = [c for (c, g, _) in parsed_cols if g == grp]
            if not grp_cols: continue

            # accumulate group RMSE
            grp_obs, grp_sim = [], []
            for c, g, k in [t for t in parsed_cols if t[1] == grp]:
                ab = _pairs_for_col(c, g, k)
                if ab is None: continue
                grp_obs.extend(ab["obs"].values.tolist())
                grp_sim.extend(ab["sim"].values.tolist())
            grp_rmse = _rmse(np.asarray(grp_obs, float), np.asarray(grp_sim, float))
            n_grp = len(grp_obs)

            # layout
            fig = plt.figure(figsize=(11, 7), constrained_layout=True)
            gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.0, 1.6], width_ratios=[1.0, 0.55, 1.15])

            # bottom: hydrographs (no inset overlaps)
            ax_h = fig.add_subplot(gs[1, :])
            ax_q = ax_h.twinx()  # secondary axis for pumping
            had_wel = False      # track whether we actually plot any pumping

            sites_grp = sites[sites["group_number"] == int(grp)]
            shared_colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628"]
            cc = 0
            for site_id, is_idx in zip(sites_grp["site_index"].unique(),
                                       sites_grp.drop_duplicates("site_index")[index_flag_column].tolist()):
                wl = water_levels.loc[water_levels["site_index"] == site_id, "water_level(navd88)"].sort_index()
                if wl.empty: continue
                series = wl if len(wl) <= 1200 else wl.resample("D").mean().dropna()
                ms = 5 if is_idx else 3
                alpha = 0.85 if is_idx else 0.45
                ax_h.plot(series, marker="o", ms=ms, lw=0.0,
                          color=shared_colors[cc % len(shared_colors)], alpha=alpha,
                          label=(f"{site_id} (INDEX)" if is_idx else str(site_id)))
                cc += 1

            added_labels = set()
            for c in grp_cols:
                k_val = int(re.search(r"_k\.(\d+)$", c).group(1))
                sm = master_df[c].dropna()
                if sm.empty:
                    continue

                aq_color = aq_color_map.get(k_val, "#333333")
                obs_lbl = "Obs (Clay/Silt)" if k_val == 0 else "Obs (EV)" if k_val == 1 else f"Obs (k={k_val})"
                if obs_lbl not in added_labels:
                    ax_h.plot(sm.index, sm.values,
                              color='green', ls="-", marker="x", ms=5, lw=1.0,
                              label=obs_lbl)
                    added_labels.add(obs_lbl)
                else:
                    ax_h.plot(sm.index, sm.values,
                              color='green', ls="-", marker="x", ms=5, lw=1.0)

                rc = key_to_rc.get((int(grp), k_val))
                if rc:
                    i, j = rc
                    print(f"[DEBUG] Group {grp}, layer {k_val}: using cell (row={i+1}, col={j+1}) for pumping overlay")

                    # Simulated heads at that cell
                    sim_s = _sim_ts_for_cell(hds_path, k_val, i, j, t0)
                    if not sim_s.empty:
                        sim_lbl = "Sim (Clay/Silt)" if k_val == 0 else "Sim (EV)" if k_val == 1 else f"Sim (k={k_val})"
                        if sim_lbl not in added_labels:
                            ax_h.plot(sim_s.index, sim_s.values,
                                      color='blue', ls="--", lw=2.0,
                                      label=sim_lbl)
                            added_labels.add(sim_lbl)
                        else:
                            ax_h.plot(sim_s.index, sim_s.values,
                                      color='blue', ls="--", lw=2.0)

                    # Pumping at that same cell (if any WEL entries exist)
                    wel_s = _wel_series_for_cell(gwf, tdis, k_val, i, j, t0)
                    if not wel_s.empty:
                        # Plot positive extraction on secondary axis
                        ax_q.step(wel_s.index, -(wel_s.values/192.5), where="mid",
                                  color="orange", lw=1.5, alpha=0.8,
                                  label="Pumping (cfd)" if not had_wel else "_nolegend_")
                        had_wel = True

            ax_h.set_ylabel("Water Level (ft NAVD88)")
            ax_h.set_xlabel("Year")
            ax_h.grid(True, alpha=0.3)
            ax_h.set_xlim(right=pd.Timestamp("2025-05-01"))

            if had_wel:
                ax_q.set_ylabel("Pumping rate (gpm)")
            else:
                # no WEL at any obs cell for this group; hide secondary ticks
                ax_q.set_yticks([])
                ax_q.set_ylabel("")

            # Combined legend (heads + pumping)
            h1, l1 = ax_h.get_legend_handles_labels()
            h2, l2 = ax_q.get_legend_handles_labels()
            h2_labeled = [(h, l) for h, l in zip(h2, l2) if l and l != "_nolegend_"]
            if h2_labeled:
                h2_f, l2_f = zip(*h2_labeled)
                ax_h.legend(list(h1) + list(h2_f), list(l1) + list(l2_f),
                            ncol=2, fontsize=8, frameon=True, loc="best")
            else:
                ax_h.legend(ncol=2, fontsize=8, frameon=True, loc="best")

            # Group RMSE annotation
            ax_h.text(0.01, 0.99, f"Group RMSE: {grp_rmse:.2f} ft\nN = {n_grp}",
                      transform=ax_h.transAxes, va="top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9),
                      fontsize=9)

            # INDEX banner
            if sites_grp[index_flag_column].any():
                ax_h.text(0.5, 1.05, "***INDEX WELL***",
                          transform=ax_h.transAxes,
                          ha="center", va="bottom",
                          fontsize=16, fontweight="bold", color="black")

            # top-left: metadata
            ax_meta = fig.add_subplot(gs[0, 0]); ax_meta.axis("off")
            swn = sites_grp["location"].unique()
            swn_str = "\n".join(swn) if len(swn) else "(no labels)"
            ax_meta.text(0.03, 0.97, f"Group Number: {grp}\nState Well Numbers:\n{swn_str}",
                         va="top", ha="left", fontsize=11)

            # top-middle: lith panel
            ax_lith = fig.add_subplot(gs[0, 1])
            try:
                _plot_lith_multiwell(sites_grp, master_df, gwf, ax_lith)
            except Exception as e:
                ax_lith.text(0.5, 0.5, f"Lith plot error:\n{e}", ha="center", va="center")
            ax_lith.set_xlim(0.4, 0.6)
            for s in ["top", "right", "bottom", "left"]:
                ax_lith.spines[s].set_visible(False)
            ax_lith.tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)
            ax_lith.set_ylabel("Elevation (ft NAVD88)")

            # top-right: map (robust well location plotting)
            ax_map = fig.add_subplot(gs[0, 2])
            try:
                if elk_boundary is not None:
                    elk_boundary.boundary.plot(ax=ax_map, color="blue")
            except Exception:
                pass

            # plot model boundary if available
            try:
                if hasattr(modelgrid, "dissolve"):
                    modelgrid.dissolve().boundary.plot(ax=ax_map, color="k")
            except Exception:
                pass

            # plot wells: prefer geometry if present; else fall back to (row,col)->(x,y)
            try:
                if gpd is not None and "geometry" in sites_grp.columns and sites_grp["geometry"].notna().any():
                    # draw non-index small grey, index gold star
                    gdf_sites = sites_grp.dropna(subset=["geometry"])
                    if not gdf_sites.empty:
                        non_idx = gdf_sites[~gdf_sites[index_flag_column]]
                        idx_    = gdf_sites[gdf_sites[index_flag_column]]
                        if len(non_idx):
                            non_idx.plot(ax=ax_map, color="grey", edgecolor="k", markersize=8)
                        if len(idx_):
                            idx_.plot(ax=ax_map, color="gold", edgecolor="k", markersize=90, marker="*")
                else:
                    # fallback: compute XY from row/col (1-based in sites)
                    pts = []
                    for _, r in sites_grp.dropna(subset=["row","col"]).iterrows():
                        i = int(r["row"]-1); j = int(r["col"]-1)
                        if 0 <= i < nrow and 0 <= j < ncol:
                            pts.append((xcc[i,j], ycc[i,j], bool(r[index_flag_column])))
                    if pts:
                        xs = [p[0] for p in pts if not p[2]]
                        ys = [p[1] for p in pts if not p[2]]
                        if xs:
                            ax_map.scatter(xs, ys, s=18, c="grey", edgecolors="k", linewidths=0.4)
                        xs = [p[0] for p in pts if p[2]]
                        ys = [p[1] for p in pts if p[2]]
                        if xs:
                            ax_map.scatter(xs, ys, s=120, c="gold", marker="*", edgecolors="k", linewidths=0.6)
            except Exception:
                pass

            legend_elements = [
                Patch(edgecolor="k", facecolor="none", linewidth=1.2, label="Model Boundary"),
                Patch(edgecolor="blue", facecolor="none", linewidth=1.2, label="Wahp Aquifer Extent"),
                Line2D([0], [0], marker="o", color="grey", markeredgecolor="k",
                       markersize=5, linestyle="None", label="Monitoring Wells"),
                Line2D([0], [0], marker="*", color="gold", markeredgecolor="k",
                       markersize=12, linestyle="None", label="INDEX Well"),
            ]
            ax_map.legend(handles=legend_elements, loc="lower left", fontsize=8, frameon=True)
            ax_map.axis("off")

            pdf.savefig(fig); plt.close(fig)

    return out_pdf



# --------------------------
# MAIN
# --------------------------
def main():
    # --------------------------
    # CONFIG / PATHS (edit here)
    # --------------------------
    SIM_WS                = os.path.join("..", "..", "models", "elk", "model_ws", "elk_2lay_monthly")
    MODEL_NAME            = "elk_2lay"
    TARGET_EPSG           = 2265
    CELL_SIZE_FT          = 660.0
    GRID_ROT_DEG          = 20.0

    # Input GIS
    ELK_BOUND_SHP         = os.path.join("..", "..", "gis", "input_shps", "elk", "elk_boundary_lf.shp")
    GRID_OUT_DIR          = os.path.join("..", "..", "gis", "output_shps", "elk")
    GRID_SHP              = os.path.join(GRID_OUT_DIR, f"elk_cell_size_{int(CELL_SIZE_FT)}ft_epsg{TARGET_EPSG}_rot{int(GRID_ROT_DEG)}.grid.shp")

    # NEW: time-varying boundary shapefiles
    GHB_SHP               = os.path.join("..", "..", "gis", "input_shps", "elk", "ghb_north.shp")
    SEEP_SHP              = os.path.join("..", "..", "gis", "input_shps", "elk", "turtle_river_seepage_face.shp")

    # Surfer → GeoTIFF setup
    GRD_DIR               = os.path.join("data", "raw", "lf_surfaces")
    TIF_DIR               = os.path.join("data", "processed", "lf_surfaces")
    TOPO_TIF              = os.path.join(TIF_DIR, "Topography.tif")
    TOA_TIF               = os.path.join(TIF_DIR,  "top_of_aquifer.tif")
    BOA_TIF               = os.path.join(TIF_DIR,  "bottom_of_aquifer.tif")
    surface_smoothing     = True

    # River centerlines (for your RIV builder)
    RIV_CENTERLINES_SHP   = os.path.join("..","..","gis","input_shps","elk", "nhd_flow_named.shp")
    AG_DRAINS_SHP = os.path.join("..", "..", "gis", "input_shps", "elk", "Drains.shp")
    
    # Time discretization switches
    mnm         = MODEL_NAME
    strt_yr     = 1965
    end_yr      = 2043
    annual_only = False      # False -> annual through 1999 then monthly starting 2000
    mnth_strt   = 2000

    # Recharge switches
    USE_SWB_RCH            = True   # False = constant value; True = SWB-based (with regression inside)
    ZERO_RCH_AT_RIVER_CELLS = True


    # 1) Convert Surfer GRDs -> GeoTIFFs
    outputs = convert_surfer_grids_to_tifs(
        GRD_DIR, TIF_DIR, dst_epsg=TARGET_EPSG, source_epsg=TARGET_EPSG, overwrite=True
    )
    print(f"Wrote {len(outputs)} GeoTIFF(s) to {TIF_DIR}")

    # 2) Build / load grid and get geometry
    os.makedirs(GRID_OUT_DIR, exist_ok=True)
    ll_corner, nrow, ncol, angrot = make_grid(ELK_BOUND_SHP, GRID_SHP, CELL_SIZE_FT, GRID_ROT_DEG, TARGET_EPSG)

    # 3) Build initial MF6 (we'll replace TDIS later)
    exe = find_mf6_exe()
    sim, gwf = build_2layer_model(
        ll_corner=ll_corner,
        nrow=nrow, ncol=ncol,
        delr=CELL_SIZE_FT, delc=CELL_SIZE_FT,
        rotation_degrees=angrot,
        sim_ws=SIM_WS, model_name=mnm, exe_name=exe,
        top_elev=1000.0,
        layer_thicknesses=(80.0, 60.0),
        icelltype=(1, 0),
        hk=(100.0, 100.0),
        k33=(0.1, 0.1)
    )
    sim.write_simulation()

    # 4) Sample rasters onto the grid & set DIS/idomain/IC with real arrays
    xcc, ycc = _cell_centers_xy(gwf)
    topo = _sample_tif_at_points(TOPO_TIF, xcc, ycc)
    toa  = _sample_tif_at_points(TOA_TIF,  xcc, ycc)
    boa  = _sample_tif_at_points(BOA_TIF,  xcc, ycc)

    idomain = _make_idomain_from_boundary(ELK_BOUND_SHP, xcc, ycc, nlay=2, target_epsg=TARGET_EPSG)

    def _assert_no_nan_inside(a, name):
        bad = np.isnan(a) & (idomain[0] == 1)
        if bad.any():
            raise ValueError(f"{name} has {bad.sum()} NaN values inside Elk boundary; fill or fix inputs.")

    _assert_no_nan_inside(topo, "Topography")
    _assert_no_nan_inside(toa,  "top_of_aquifer")
    _assert_no_nan_inside(boa,  "bottom_of_aquifer")

    top, botm1, botm2 = _enforce_min_thickness(topo, toa, boa, min_thk=5.0)

    if surface_smoothing:
        sigma = 1
        print(f"Smoothing top elevation using sigma = {sigma}")
        top = gaussian_filter(top, sigma=sigma)
        top, botm1, botm2 = _enforce_min_thickness(
            top, botm1, botm2, min_thk=5.0,
            check_l1_top_bottom=True, enforce_even_if_no_overlap=True
        )

    gwf.dis.top.set_data(top)
    gwf.dis.botm.set_data(np.stack([botm1, botm2], axis=0))
    gwf.dis.idomain.set_data(idomain)
    gwf.simulation.write_simulation()

    # grid shapefile for QA
    grd, output_shp = build_grid_shp(nrow, ncol, CELL_SIZE_FT, ll_corner, angrot, TARGET_EPSG, MODEL_NAME)
    grd = grd[['node', 'row', 'col', 'i', 'j', 'geometry']]
    idomain = idomain.astype(int)
    for k in range(len(idomain)):
        grd[f'idom_{k}'] = idomain[k, grd['i'].values, grd['j'].values]
    botm = np.asarray(gwf.dis.botm.array); top_arr = np.asarray(gwf.dis.top.array)
    grd['top'] = top_arr[grd['i'].values,grd['j'].values]
    for k in range(len(botm)):
        grd[f'botm_{k}'] = botm[k,grd['i'].values,grd['j'].values]
    for k in range(len(botm)):
        if k == 0:
            grd[f'thk_{k}'] = top_arr[grd['i'].values,grd['j'].values] - botm[k,grd['i'].values,grd['j'].values]
        else:
            grd[f'thk_{k}'] = botm[k-1,grd['i'].values,grd['j'].values] - botm[k,grd['i'].values,grd['j'].values]
    grd.to_file(output_shp)

    # STRT
    strt = np.empty_like(gwf.dis.botm.data)
    strt[0] = np.maximum(top - 0.1, botm1 + 0.01)
    strt[1] = np.maximum(botm1 - 0.1, botm2 + 0.01)
    for k in range(2):
        strt[k] = np.where(idomain[k] == 1, strt[k], (botm2 - 1.0))
    if hasattr(gwf, "ic"): gwf.ic.strt.set_data(strt)
    else: flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # 5) Replace TDIS with your calendar & add STO
    sp_df = build_tdis_and_sto(
        sim, gwf, mnm,
        start_year=strt_yr, end_year=end_yr,
        annual_only=annual_only, monthly_start=mnth_strt,
        nstp_annual=1, nstp_monthly=1, tsmult=1.0,
        sy_vals=(0.20, 0.20), ss_default=1e-4, min_thk=0.5,
    )
    td = sim.get_package("tdis")
    nper = int(td.nper.get_data())

    # 6) IMS
    ims_old = sim.get_package("ims")
    if ims_old is not None:
        sim.remove_package(ims_old)
    ims = flopy.mf6.ModflowIms(
        sim, pname="ims", print_option="SUMMARY", complexity="COMPLEX",
        outer_dvclose=0.001, outer_maximum=100, under_relaxation="NONE",
        inner_maximum=100, inner_dvclose=0.001, rcloserecord=1000.0,
        linear_acceleration="BICGSTAB", scaling_method="NONE",
        reordering_method="NONE", relaxation_factor=0.97,
        filename=f"{MODEL_NAME}.ims",
    )
    sim.register_ims_package(ims, [gwf.name])

    # ---------------- #
    # RIV: split into turtle, goose, hazen
    # ---------------- #
    riv_df = riv_definition(gwf=gwf, top_clearance_ft=1.0)
    riv_df = riv_df.copy()

    # clean up i,j and enforce ints
    riv_df[["i", "j"]] = (
        riv_df[["i", "j"]].apply(pd.to_numeric, errors="coerce").dropna().astype(np.int64)
    )

    # make sure riv_pac exists so we can group
    if "riv_pac" not in riv_df.columns:
        print("[RIV] riv_pac column missing; assigning all to 'turtle' as fallback.")
        riv_df["riv_pac"] = "turtle"

    # adjust elevations vs cell-bottoms, as before
    riv_df = raise_riv_to_cellbottom(
        gwf, riv_df, layer=0, bed_thk_ft=3.0, clearance_ft=0.1
    )

    # convenience: build stress_period_data dict from subset
    def _build_riv_spd(df_sub: pd.DataFrame) -> dict[int, list[tuple]]:
        spd = {}
        for sp in range(nper):
            per_spd = []
            for _, row in df_sub.iterrows():
                ly = 0
                r, c = int(row["i"]), int(row["j"])
                if idomain[ly, r, c] != 1:
                    continue
                per_spd.append(
                    (
                        (ly, r, c),
                        float(row["stage"]),
                        float(row["cond"]),
                        float(row["rbot"]),
                    )
                )
            spd[sp] = per_spd
        return spd

    # split by package
    groups = {
        "riv_turtle": riv_df[riv_df["riv_pac"] == "turtle"].copy(),
        "riv_goose":  riv_df[riv_df["riv_pac"] == "goose"].copy(),
        "riv_hazen":  riv_df[riv_df["riv_pac"] == "hazen"].copy(),
    }

    for pname, df_sub in groups.items():
        if df_sub.empty:
            print(f"[RIV] {pname}: no cells; skipping package.")
            continue

        print(f"[RIV] Building RIV package '{pname}' with {len(df_sub)} cells.")
        spd = _build_riv_spd(df_sub)

        flopy.mf6.ModflowGwfriv(
            gwf,
            stress_period_data=spd,
            pname=pname,
            save_flows=True,
            filename=f"{mnm}.{pname}",
        )

    # ---------------- #
    # DRN: split into south / mid-south / mid-north / north
    # ---------------- #
    drn_df = drain_definition(gwf=gwf, top_clearance_ft=1.0)
    drn_df = drn_df.copy()

    # clean i,j
    drn_df[["i", "j"]] = (
        drn_df[["i", "j"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .astype(np.int64)
    )

    # fix vs cell bottom
    drn_df = fix_drn_to_cellbottom(
        gwf, drn_df, layer=0, clearance_ft=0.1, mode="raise"
    )

    # normalize loc to lowercase for grouping
    if "loc" in drn_df.columns:
        drn_df["loc"] = drn_df["loc"].astype(str).str.lower()
    else:
        # fallback if loc somehow missing: treat everything as "south"
        drn_df["loc"] = "south"

    # helper to build SPD dict for a subset
    def _build_drn_spd(df_sub: pd.DataFrame) -> dict[int, list[tuple]]:
        spd = {}
        for sp in range(nper):
            per_spd = []
            for _, row in df_sub.iterrows():
                ly = 0
                r, c = int(row["i"]), int(row["j"])
                if idomain[ly, r, c] != 1:
                    continue
                per_spd.append(
                    (
                        (ly, r, c),
                        float(row["stage"]),
                        float(row["cond"]),
                    )
                )
            spd[sp] = per_spd
        return spd

    # split into four geographic packages
    drn_groups = {
        "drn_s":  drn_df[drn_df["loc"] == "south"].copy(),
        "drn_ms": drn_df[drn_df["loc"] == "midsouth"].copy(),
        "drn_mn": drn_df[drn_df["loc"] == "midnorth"].copy(),
        "drn_n":  drn_df[drn_df["loc"] == "north"].copy(),
    }

    for pname, df_sub in drn_groups.items():
        if df_sub.empty:
            print(f"[DRN] {pname}: no cells; skipping package.")
            continue

        print(f"[DRN] Building DRN package '{pname}' with {len(df_sub)} cells.")
        spd = _build_drn_spd(df_sub)

        flopy.mf6.ModflowGwfdrn(
            gwf,
            stress_period_data=spd,
            pname=pname,
            save_flows=True,
            filename=f"{mnm}.{pname}",
        )


    # =======================================================================
    # NEW: Build WL surfaces, then time-varying layer-2 GHB & DRN2 (seepage)
    # =======================================================================
    # Build WL surfaces that match obs
    sp_starts, wl_surfaces = build_wl_surfaces(
        gwf,
        sites_csv=os.path.join("data","raw","obs_data","elk_valley_sites.csv"),
        wl_xlsx=os.path.join("data","raw","obs_data","elk_valley_water_level_data.csv"),
        grid_shp=GRID_SHP,
        annual=annual_only,              # annual-only OR annual→monthly (controlled by annual_only)
        pre2000_cutoff_year=2000,
        post2000_roll_days=60,
        use_median=True,                 # match your obs median default
        idw_power=2.0, idw_k=12
    )

    # This boundary condtion says GHB because that's what it originally was, but it ends up being drn_wl
    ghb_tv = build_ghb_from_shp_timevarying(
        gwf,
        ghb_line_shp=GHB_SHP,
        grid_shp=GRID_SHP,
        wl_surfaces=wl_surfaces,
        width_ft=150.0, bed_thk_ft=4.5, k_bed_ft_per_d=0.01,
        min_seg_len_ft=50.0,
        head_offset_ft=-5.0,
        layer_k=1,                      # layer 2 (0-based)
        clearance_ft=0.1,
        epsg=TARGET_EPSG,
    )

    # drn2_tv = build_drn2_from_seep_timevarying(
    #     gwf,
    #     seep_line_shp=SEEP_SHP,
    #     grid_shp=GRID_SHP,
    #     wl_surfaces=wl_surfaces,
    #     width_ft=50.0, depth_ft=5.0,
    #     min_seg_len_ft=200.0,
    #     stage_offset_ft=5.0,
    #     layer_k=1,                      # layer 2 (0-based)
    #     clearance_ft=0.1,
    #     epsg=TARGET_EPSG,
    # )

    # # Extra guard: filter TV GHB/DRN2 to active idomain in layer 2
    idom = np.asarray(gwf.dis.idomain.array, int)
    if ghb_tv:
        for sp, df in list(ghb_tv.items()):
            if df.empty: continue
            mask = idom[1, df.i.values, df.j.values] == 1
            ghb_tv[sp] = df.loc[mask].copy()
    # if drn2_tv:
    #     for sp, df in list(drn2_tv.items()):
    #         if df.empty: continue
    #         mask = idom[1, df.i.values, df.j.values] == 1
    #         drn2_tv[sp] = df.loc[mask].copy()


    # Convert time-varying GHB-style DataFrames → DRN-style (head → stage)
    drn_tv = {}
    if ghb_tv:
        for sp, df in ghb_tv.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            # `attach_timevarying_bc` expects a "stage" column for drains
            if "stage" not in d.columns and "head" in d.columns:
                d = d.rename(columns={"head": "stage"})
            # ensure layer index column exists (it should already from build_ghb_from_shp_timevarying)
            if "k" not in d.columns:
                d["k"] = 1  # layer 2 (0-based) as a fallback
            drn_tv[sp] = d


    # # Attach as per-period packages (GHB on k=1, DRN on k=1)
    attach_timevarying_bc(
        gwf,
        sp_starts=sp_starts,
        ghb_by_sp=None,          # no GHB
        drn_by_sp=drn_tv,        # use the converted dict
        drn_pname="drn_wl",
        drn_filename=f"{gwf.name}.drn_wl",       
    )

    # # =======================================================================
    # # (optional) avoid duplicating cells that already have DRN/RIV:
    existing_drn_cells = {(int(r), int(c)) for r, c in drn_df[["i","j"]].to_numpy()} if "drn_df" in locals() else None
    existing_riv_cells = {(int(r), int(c)) for r, c in riv_df[["i","j"]].to_numpy()} if "riv_df" in locals() else None
    conflicts = set()
    if existing_drn_cells: conflicts |= existing_drn_cells
    if existing_riv_cells: conflicts |= existing_riv_cells

    ag_drn_gdf, ag_drn_dict = build_ag_drn_from_points(
        gwf,
        points_shp=AG_DRAINS_SHP,
        grid_shp=GRID_SHP,
        layer_k=0,                 # upper layer
        stage_offset_ft=4.0,       # stage = top - 4 ft
        bed_thk_ft=5.0,
        k_bed_ft_per_d=0.1,
        width_ft=50.0,
        eff_length_ft=None,        # defaults to cell-size based length
        clearance_ft=0.10,
        drop_if_conflict_cells=conflicts,
        epsg=TARGET_EPSG,
    )

    print(f"[AG-DRN] candidate cells: {len(ag_drn_gdf)}")
    ag_out_shp = os.path.join(GRID_OUT_DIR, "ag_drains_points.shp")
    ag_drn_gdf.to_file(ag_out_shp)
    print(f"[AG-DRN] wrote QA shapefile: {ag_out_shp}")

    # Attach a separate DRN package for ag drains
    flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data=ag_drn_dict,
        pname="drn_ag",
        save_flows=True,
        filename=f"{mnm}.drn_ag",
    )

    # ---------------------------- #
    # Recharge 
    # ---------------------------- #

    # Recharge switches (only 2 modes now)

    nrow, ncol = gwf.dis.nrow.get_data(), gwf.dis.ncol.get_data()

    def _inb(r, c):
        return 0 <= r < nrow and 0 <= c < ncol

    riv_cells     = {(int(i), int(j)) for i, j in riv_df[["i", "j"]].to_numpy() if _inb(int(i), int(j))}
    drn_cells     = {(int(i), int(j)) for i, j in drn_df[["i", "j"]].to_numpy() if _inb(int(i), int(j))}
    ag_drn_cells  = {(int(i), int(j)) for i, j in ag_drn_gdf[["i", "j"]].to_numpy() if _inb(int(i), int(j))}

    drn2_cells = set()
    if drn_tv:
        for df in drn_tv.values():
            drn2_cells |= {(int(i), int(j)) for i, j in df[["i", "j"]].to_numpy() if _inb(int(i), int(j))}

    # Smoothing controls for SWB path
    RCH_SMOOTH_SIGMA = 2.5
    RCH_USE_SMOOTHING = True

    if not USE_SWB_RCH:
        # ------------------------ #
        # Mode A: constant recharge
        # ------------------------ #
        rch_dict = {}
        base_ft_day = 3.5 / 12.0 / 365.0  # original constant

        for sp in range(nper):
            rch_arr = np.full((nrow, ncol), base_ft_day, dtype=float)

            # Optional smoothing of constant recharge
            if RCH_USE_SMOOTHING and RCH_SMOOTH_SIGMA is not None and RCH_SMOOTH_SIGMA > 0:
                rch_arr = gaussian_filter(rch_arr, sigma=RCH_SMOOTH_SIGMA, mode="reflect")

            # Zero boundaries
            for r, c in riv_cells:
                rch_arr[r, c] = 0.0
            for r, c in drn_cells:
                rch_arr[r, c] = 0.0
            for r, c in drn2_cells:
                rch_arr[r, c] = 0.0
            for r, c in ag_drn_cells:
                rch_arr[r, c] = 0.0

            rch_dict[sp] = rch_arr

    else:
        # ------------------------ #
        # Mode B: SWB-based recharge
        #         (pre-2000 synthetic + 2000–2023 monthly + ≥2024 pred)
        #         with smoothing + PDF before/after plots
        # ------------------------ #
        print("Using SWB-derived recharge (pre-2000 regression + post-2000 SWB)...")

        pdf_path = os.path.join("figs", f"{mnm}_rch_swb_smoothing_debug.pdf")

        rch_dict = get_rcha_tseries_from_swb2_nc(
            gwf,
            perioddata=sp_df,
            rch_dict={},
            annual_only=False,
            rch_mult=1.0,
            smooth_sigma=RCH_SMOOTH_SIGMA if RCH_USE_SMOOTHING else None,
            smooth_mode="reflect",
            plot_debug=False,         # write before/after for each SP
            plot_sps=None,           # None => all SPs
            pdf_path=pdf_path,
        )

    # Final pass: zero at river cells (optionally others)
    if ZERO_RCH_AT_RIVER_CELLS:
        for sp, arr in rch_dict.items():
            rch_arr = arr.copy()
            for r, c in riv_cells:
                rch_arr[r, c] = 0.0
            # If you want the others zeroed here too, uncomment:
            for r, c in drn_cells:    rch_arr[r, c] = 0.0
            for r, c in drn2_cells:   rch_arr[r, c] = 0.0
            for r, c in ag_drn_cells: rch_arr[r, c] = 0.0
            rch_dict[sp] = rch_arr

    rch_dict = apply_monthly_climatology_to_predictive_rch(
        rch_dict,
        sp_starts=sp_starts,
        pred_start_year=2024,
        clim_years=(2019, 2020, 2021, 2022, 2023),
    )

    flopy.mf6.ModflowGwfrcha(
        gwf,
        recharge=rch_dict,
        save_flows=True,
        pname="rch",
        filename=f"{mnm}.rch",
    )
    
    # re-zero again over RIV (and optionally DRN/DRN2 if desired)
    for sp in range(nper):
        rch_arr = rch_dict[sp].copy()
        for r, c in riv_cells:  rch_arr[r, c] = 0.0
        for r, c in drn_cells:    rch_arr[r, c] = 0.0
        for r, c in drn2_cells:   rch_arr[r, c] = 0.0
        for r, c in ag_drn_cells: rch_arr[r, c] = 0.0
        rch_dict[sp] = rch_arr

    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_dict, save_flows=True, pname="rch", filename=f"{mnm}.rch")



    # Write files so far
    sim.write_simulation()
    print(f"MF6 input files written to: {SIM_WS}")

    # Wells
    wel_pkg, wel_summary = add_wells_from_process_script(
        gwf, monthly=True, force_rebuild=True, pname="wel", filename=f"{gwf.name}.wel", verbose=True,
    )
    
    apply_monthly_climatology_to_predictive_wel(
        gwf,
        sp_starts=sp_starts,
        pred_start_year=2024,
        clim_years=(2019, 2020, 2021, 2022, 2023),
        pname="wel",
    )
    
    gwf.simulation.write_simulation()

    # Stress period info CSVs
    stress_period_df_gen(SIM_WS, strt_yr, annual_only=annual_only)

    # CRS on grid (for any downstream plotting)
    gwf.modelgrid.set_coord_info(crs=CRS.from_epsg(TARGET_EPSG))

    # OBS package build
    ssobs, tobs = elk_obs.main(SIM_WS, gwf.name, False, gen_plots=False)
    wells = ssobs.obsprefix.unique()
    ss_hd_list = []
    for well in wells:
        subset = ssobs[ssobs.obsprefix==well].iloc[0,:]
        if idomain[subset.k, subset.i, subset.j] == 1:
            ss_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
    trans_hd_list = []
    trans_sites = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'))
    wells = trans_sites.obsprefix.unique()
    for well in wells:
        subset = trans_sites[trans_sites.obsprefix==well].iloc[0,:]
        if idomain[subset.k, subset.i, subset.j] == 1:
            trans_hd_list.append((subset.obsprefix,'HEAD',(subset.k, subset.i, subset.j)))
    hd_obs = {f'{mnm}.ss_head.obs.output': ss_hd_list, f'{mnm}.trans_head.obs.output': trans_hd_list}
    flopy.mf6.ModflowUtlobs(gwf, pname="head_obs", filename=f"{mnm}.obs", continuous=hd_obs)

    gwf.simulation.write_simulation()

    icell_files = [f for f in os.listdir(SIM_WS) if f.startswith('npf_icelltype')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(SIM_WS, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(SIM_WS, file), icelltype_data, fmt='%i')

    success, buff = sim.run_simulation()
    print("Run success:", success)
    
    # pdf_path = build_and_plot_group_summary(
    #     sim_ws=os.path.join("master_flow_08_highdim_restrict_bcs_flood_full_final_forward_run_base"),
    #     model_name="elk_2lay",
    #     out_pdf=os.path.join("figures", "master_flow_08_highdim_restrict_bcs_flood_full_final_forward_run_base.pdf"),
    #     annual_flag=False
    # )
    pdf_path = build_and_plot_group_summary(
        sim_ws=SIM_WS,
        model_name="elk_2lay",
        out_pdf=os.path.join("figures", SIM_WS),
        annual_flag=False
    )
    print("Wrote:", pdf_path)
    
    sim.set_all_data_external()
    sim.write_simulation()

    # Post-write QA for static RIV/DRN files (unchanged)
    riv_stress_files = [
        os.path.join(SIM_WS, f)
        for f in os.listdir(SIM_WS)
        if f.startswith(f"{mnm}.riv") and "stress" in f
    ]
    drn_stress_files = [
        os.path.join(SIM_WS, f)
        for f in os.listdir(SIM_WS)
        if f.startswith(f"{mnm}.drn_") and "stress" in f
    ]
    drn_wl_stress_files = [os.path.join(SIM_WS, f) for f in os.listdir(SIM_WS)
                        if f.startswith(f"{mnm}.drn_wl_stress")]
    drn_ag_stress_files = [os.path.join(SIM_WS, f) for f in os.listdir(SIM_WS)
                        if f.startswith(f"{mnm}.drn_ag_stress")]

    botm = np.asarray(gwf.dis.botm.array)

    # base DRN
    for drn_file in drn_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True, header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1, df['row'].values-1, df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0, 'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)

    # WL drains
    for drn_file in drn_wl_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True, header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1, df['row'].values-1, df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0, 'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)

    # AG drains (same logic)
    for drn_file in drn_ag_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True, header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1, df['row'].values-1, df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0, 'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)
    for riv_file in riv_stress_files:
        df = pd.read_csv(riv_file, delim_whitespace=True, header=None)
        df.columns = ['ly','row','col','stage','cond','rbot']
        bot = botm[df['ly'].values-1, df['row'].values-1, df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
        df.loc[df['diff'] < 0,'rbot']  = df['mbot'] + 0.1
        df['diff'] = df['rbot'] - df['mbot']
        df.loc[df['diff'] < 0,'rbot']  = df['mbot'] + 0.1
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(riv_file, sep=' ', index=False, header=False)

    # Add AUTO_FLOW_REDUCE_CSV to WEL
    wel_file = os.path.join(SIM_WS, f"{mnm}.wel")
    with open(wel_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('END options'):
            new_lines.append('  AUTO_FLOW_REDUCE_CSV FILEOUT auto_flow_reduce.csv\n')
        new_lines.append(line)
    with open(wel_file, 'w') as f:
        f.writelines(new_lines)

    # Run
    icell_files = [f for f in os.listdir(SIM_WS) if f.startswith('npf_icelltype')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(SIM_WS, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(SIM_WS, file), icelltype_data, fmt='%i')

    success, buff = sim.run_simulation()
    print("Run success:", success)
    clean_mf6(SIM_WS, mnm)
    run_zb_by_layer(SIM_WS+'_clean', modnm=mnm)
    return SIM_WS+'_clean'


def add_wel_head_obs(sim_ws):
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    gwf = sim.get_model()
    wel = gwf.get_package("wel")
    print(len(wel.stress_period_data.array))
    dwel = gwf.get_package("def_wel")
    wel = gwf.get_package("wel")
    ukijs = set()
    for i,arr in enumerate(wel.stress_period_data.array):
        if arr is None:
            continue
        df = pd.DataFrame.from_records(arr)
        kij = set(df.cellid.to_list())
        ukijs.update(kij)

    dukijs = set()
    for i,arr in enumerate(dwel.stress_period_data.array):
        if arr is None:
            continue
        df = pd.DataFrame.from_records(arr)
        kij = set(df.cellid.to_list())
        dukijs.update(kij)
    print(len(ukijs),len(dukijs))
    fname = os.path.join(sim_ws,"elk_2lay.obs_continuous_elk_2lay.trans_head.obs.output.txt")
    assert os.path.exists(fname)
    with open(fname,'a') as f:
        for kij in ukijs:
            name = "exwel-k{0}-i{1}-j{2}".format(kij[0],kij[1],kij[2])
            f.write(" {0}  {1}  {2} {3} {4}\n".format(name,"HEAD",kij[0]+1,kij[1]+1,kij[2]+1))
        for kij in dukijs:
            name = "defwel-k{0}-i{1}-j{2}".format(kij[0],kij[1],kij[2])
            f.write(" {0}  {1}  {2} {3} {4}\n".format(name,"HEAD",kij[0]+1,kij[1]+1,kij[2]+1))
    pyemu.os_utils.run("mf6",cwd=sim_ws)

if __name__ == "__main__":
    #print('building elk mf6 model...')
    #main()
    add_wel_head_obs(os.path.join("model_ws","temp"))
