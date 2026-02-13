import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import sys
import pandas as pd
from pyparsing import col
sys.path.insert(0, os.path.abspath(os.path.join("..", "dependencies")))
sys.path.insert(1, os.path.abspath(os.path.join("..", "dependencies", "flopy")))
import flopy
import geopandas as gpd
from shapely.geometry import Polygon
import contextily as cx

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime, timedelta
from matplotlib.ticker import StrMethodFormatter
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize

# ============================================================
# USER SETTINGS
# ============================================================

# ---- plot toggles ----
PLOT_THICKNESS = False
PLOT_K = False
PLOT_STORAGE = False
PLOT_RECHARGE = False
PLOT_BUDGET = True
PLOT_BC = False
PLOT_HEADS = False
PLOT_BUDGET_PIES = True

elk_ws = os.path.join("elk", "master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base")
sw_ww_ws = os.path.join("spirit_war", "post_ies_scn01_baseline_ensemble_forward_run_base")
wahp_ws = os.path.join("wahp", "post_ies_scn01_baseline_clean")

SW_WW_XOFF = 2388853.44242084
SW_WW_YOFF = 260219.09632163

MODEL_CRS = "EPSG:2265"
BASEMAP_CRS = "EPSG:3857"

OUTDIR_THK = os.path.join("figures", "thickness_tiles")
OUTDIR_K = os.path.join("figures", "k_tiles")
OUTDIR_STO = os.path.join("figures", "storage_tiles")
OUTDIR_RCH = os.path.join("figures", "recharge_tiles")
OUTDIR_BUDGET = os.path.join("figures", "budget_timeseries")
OUTDIR_BC = os.path.join("figures", "bc_stage_cond_tiles")
OUTDIR_HEADS = os.path.join("figures", "head_contour_tiles")
os.makedirs(OUTDIR_THK, exist_ok=True)
os.makedirs(OUTDIR_K, exist_ok=True)
os.makedirs(OUTDIR_STO, exist_ok=True)
os.makedirs(OUTDIR_RCH, exist_ok=True)
os.makedirs(OUTDIR_BUDGET, exist_ok=True)
os.makedirs(OUTDIR_BC, exist_ok=True)
os.makedirs(OUTDIR_HEADS, exist_ok=True)

DPI_THK = 350
DPI_K = 350
DPI_STO = 350

# -------------------------
# Basemap toggle
# -------------------------
USE_USGS_TOPO_BASEMAP = False

BASEMAP_ZOOM = 10
BASE_XYZ = cx.providers.CartoDB.Positron
USGS_TOPO_PROVIDER = cx.providers.USGS.USTopo

USE_NATIONALMAP_HYDRO_EXPORT = True
USGS_HYDRO_MAPSERVER = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer"

USE_NATIONALMAP_SHADED_EXPORT = False
USGS_SHADED_MAPSERVER = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer"

WAIT_BETWEEN_BASEMAP_CALLS = False
BASEMAP_WAIT_SECONDS = 0.5

HIDE_BASEMAP_ATTRIBUTION = True

# North arrow PNG
NORTH_ARROW_PNG = os.path.join(".", "N_arrow.png")
NORTH_ARROW_ZOOM = 0.013
NORTH_ARROW_XY = (0.95, 0.10)  # axes fraction

# Scale bar
ADD_SCALE_BAR = True
SCALE_BAR_MILES = 5.0
SCALE_BAR_COLOR = "black"
SCALE_BAR_LW = 3
SCALE_BAR_FONTSIZE = 10
SCALE_BAR_PAD_FRAC = 0.06

BUDGET_PREDICTIVE_START_YEAR = 2024   # grey shading starts Jan 1 of this year
BUDGET_YEARTICK_STEP = 2
BUDGET_FIGSIZE = (14, 6)
BUDGET_DPI = 350

BC_KPER = 0   # 0 for first stress period, or "mean" for time-weighted mean
DPI_BC = 500

DPI_HEADS = 350
HEAD_CONTOUR_INTERVAL_FT = 10.0
HEAD_CMAP_NAME = "viridis"   # swap to your custom if desired

# Compare SP0 to December 2023
HEAD_REF_KPER = 0
HEAD_TARGET_DATE = datetime(2023, 12, 15)  # mid-month = robust for monthly SPs


# - Wahpeton: water table (first non-dry) + layer 5
# - Spiritwood–Warwick: layer 1 + layer 3
# - Elk: water table
WAHP_BV_LAYER = 5
SWWW_WARWICK_LAYER = 1
SWWW_SPIRITWOOD_LAYER = 3

# Overlay transparency
OVERLAY_ALPHA = 0.82

# Legend formatting
LEGEND_DECIMALS = 2
LEGEND_SCI_THRESHOLD = 0.01

# Optional: show active-area outline (if you want it)
HEAD_SHOW_ACTIVE_OUTLINE = True
HEAD_ACTIVE_OUTLINE_STYLE = {"lw": 0.9, "color": "black", "alpha": 0.5}

# K settings
NBINS_HK = 5
NBINS_VK = 5
K_UNITS_LABEL = "(feet/day)"
VK_STORED_AS_RATIO = True
PLOT_K_ONE_FIG_PER_LAYER = True

ELK_LAYERS = None
WAHP_LAYERS = [1, 2, 3, 4, 5, 6]  # (no layer 7)
SW_WW_LAYERS = None

# Spiritwood zone array settings
SW_ZONE_NPY = "zone_array 2.npy"
SW_ZONE_L2_WINDOWS = {8, 9}
SW_ZONE_L3_BARRIER = 7
SW_ZONE_LAYER2 = 2
SW_ZONE_LAYER3 = 3

# Storage settings
NBINS_SS = 5
NBINS_SY = 5
SS_UNITS_LABEL = "(1/feet)"   # if SS is specific storage in 1/length
SY_UNITS_LABEL = "(-)"

# -------------------------
# User specified bins (K)
# -------------------------
ELK_BINS_HK = {1: [0, 40, 80, 120, 180, 300], 2: [0, 50, 100, 200, 350, 500]}
ELK_BINS_VK = {1: [0, 4, 8, 12, 20, 30], 2: [0, 5, 10, 20, 35, 50]}

WAHP_BINS_HK = {
    1: [0, 0.10, 0.30, 0.60, 1.50, 6.00],
    2: [0, 0.20, 0.50, 1.00, 3.00, 14.00],
    3: [0, 3.00, 6.00, 12.00, 25.00, 60.00],
    4: [0, 10 ** (-2.5), 1e-2, 10 ** (-1.5), 1e-1],
    5: [0, 80.00, 160.00, 240.00, 350.00],
    6: [0, 0.02, 0.05, 0.08, 0.15, 0.43],
}
WAHP_BINS_VK = {
    1: [0, 0.05, 0.10, 0.30, 1.00, 6.00],
    2: [0, 0.02, 0.05, 0.10, 0.30, 1.40],
    3: [0, 0.30, 0.60, 1.20, 2.50, 6.00],
    4: [0, 2e-05, 4e-05, 7e-05, 1.5e-04],
    5: [0, 8.00, 16.00, 24.00, 35.00],
    6: [0, 2e-04, 5e-04, 1e-03, 3e-03, 1.6e-02],
}

# Spiritwood fallback bins (non-zoned layers)
SW_BINS_HK = {1: [50, 100, 150, 200, 250, 280]}
SW_BINS_VK = {1: [0, 10, 15, 20, 30, 36.4]}

# Spiritwood zone-split bins
SW_L2_HK_CONFINING_BINS = [0.00087, 0.002, 0.005, 0.010, 0.020, 0.040]
SW_L2_HK_WINDOWS_BINS   = [0.18,    0.30,  0.45,  0.60,  0.72,  0.80]
SW_L2_VK_CONFINING_BINS = [0.00011, 0.0003, 0.0007, 0.0013, 0.0025, 0.0052]
SW_L2_VK_WINDOWS_BINS   = [0.0136,  0.025,  0.050,  0.070,  0.090,  0.102]

SW_L3_HK_SPIRITWOOD_BINS = [61, 120, 160, 200, 240, 280]
SW_L3_HK_BARRIER_BINS    = [0.11, 0.150, 0.180]
SW_L3_VK_SPIRITWOOD_BINS = [6.1, 12, 16, 20, 24, 28]
SW_L3_VK_BARRIER_BINS    = [1.1E-2, 1.5E-2, 1.8E-2]

# -------------------------
# User-specific bins (Storage) derived from your summaries
# Using edges = [p0, p10, p25, p50, p75, p100]  (=> 5 bins)
# -------------------------
ELK_BINS_SS = {
    1: [0.01960379, 0.039250602, 0.060320255, 0.09307575, 0.14991815, 0.2],
    2: [1.8e-05, 4.6e-05, 8.0e-05, 1.2e-04, 2.0e-04, 5.9e-04],
}
ELK_BINS_SY = {
    1: [0.1481553, 0.16793991, 0.20327205, 0.262953,  0.3016334],
    2: [0.22, 0.26, 0.28, 0.3014106],
}

SW_BINS_SS = {
    1: [0.01321278, 0.025, 0.05,0.1, 0.15, 0.2],
    2: [3.2e-05, 7.7e-05, 1.3e-04, 1.8e-04, 2.5e-04, 6.1e-04],
    3: [3.8e-06, 4.9e-06, 6.8e-06, 9.4e-06, 1.3e-05, 3.0e-05],
}
SW_BINS_SY = {
    1: [0.1387934, 0.1413647, 0.143283225, 0.1456226],
    # NOTE: per your instruction, do NOT plot SY for Spiritwood layers 2 & 3
}

WAHP_BINS_SS = {
    1: [6.1e-05, 1.4e-04, 2.0e-04, 3.0e-04, 4.0e-04, 8.5e-04],
    2: [3.1e-05, 5.6e-05, 7.9e-05, 1.1e-04, 1.5e-04, 3.2e-04],
    3: [2.2e-05, 3.9e-05, 5.2e-05, 7.6e-05, 1.1e-04, 2.2e-04],
    4: [8.3e-06, 2.0e-05, 2.8e-05, 4.2e-05, 5.6e-05, 9.8e-05],
    5: [1.5e-05, 2.8e-05, 3.7e-05, 5.1e-05, 7.5e-05, 2.0e-04],
    6: [3.5e-05, 5.4e-05, 7.0e-05, 1.0e-04, 1.6e-04, 4.5e-04],
}
WAHP_BINS_SY = {
    1: [0.2, 0.25],
    2: [0.08657961, 0.12203679, 0.15291939, 0.2, 0.24288945],
    # NOTE: per your instruction, do NOT plot SY for Wahpeton layers 3–6 (and no layer 7)
}

RIV_BINS_STAGE = None
RIV_BINS_COND  = None
DRN_BINS_ELEV  = None
DRN_BINS_COND  = None

# ============================================================
# BUDGET COLUMN MAPPINGS 
# ============================================================

# NOTE:
# - Provide columns exactly as they appear in each budget.csv.
# - Any missing columns are treated as zeros (so scripts won't crash).
# - Signs:
#   * Items in "in_items" plot positive stacked bars
#   * Items in "out_items" plot negative stacked bars

WAHP_BUDGET_ITEMS = {
    "in_items": {
        "Recharge": ["rcha_in"],
        "Gaining River": ["riv_in", "otriv_in"],
        "Edge GHB Inflow": ["ghb_in"],
        "WBV GHB Inflow": ["ghb_wbv_in"],
        "Interlayer flow from L4": ["layer4_exch_in"],
        "Interlayer flow from L6": ["layer6_exch_in"],
    },
    "out_items": {
        "Pumping": ["wel_car_out", "wel_malt_out", "wel_cow_out", "wel_minn_out", "wel_cob_out"],
        "Drains": ["drn_out"],
        "Losing River": ["riv_out", "otriv_out"],
        "Edge GHB Outflow": ["ghb_out"],
        "WBV GHB Outflow": ["ghb_wbv_out"],
        "Interlayer flow to L4": ["layer4_exch_out"],
        "Interlayer flow to L6": ["layer6_exch_out"],
    },
}

WAHP_PIE_BUDGET_ITEMS = {
    "in_items": {
        "Recharge": ["rcha_in"],
        "River In (RIV)": ["riv_in"],          # keep separate
        "River In (Otter Tail RIV)": ["otriv_in"],  # keep separate
        "Edge GHB Inflow": ["ghb_in"],
        "WBV GHB Inflow": ["ghb_wbv_in"],
        "Interlayer flow from L4": ["layer4_exch_in"],
        "Interlayer flow from L6": ["layer6_exch_in"],
    },
    "out_items": {
        # wells: keep separate (one wedge each)
        "Pumping — Cargill": ["wel_car_out"],
        "Pumping — Malt": ["wel_malt_out"],
        "Pumping — Wahpeton": ["wel_cow_out"],
        "Pumping — Minn-Dak": ["wel_minn_out"],
        "Pumping — Breckenridge": ["wel_cob_out"],

        "Drains": ["drn_out"],

        # rivers: keep separate (one wedge each)
        "River Out (RIV)": ["riv_out"],
        "River Out (Otter Tail RIV)": ["otriv_out"],

        "Edge GHB Outflow": ["ghb_out"],
        "WBV GHB Outflow": ["ghb_wbv_out"],
        "Interlayer flow to L4": ["layer4_exch_out"],
        "Interlayer flow to L6": ["layer6_exch_out"],
    },
}

SWWW_BUDGET_ITEMS = {
    "in_items": {
        "Recharge": ["rcha_in"],
        "Gaining River": ["riv_in"],
        "GHB Inflow": ["ghb_in"],
        "Interlayer flow from L2": ["layer2_exch_in"],
    },
    "out_items": {
        "Pumping": ["wel_wel_0_out"],
        "Drains": ["drn_out"],
        "Losing River": ["riv_out"],
        "GHB Outflow": ["ghb_out"],
        "Interlayer flow to L2": ["layer2_exch_out"],
    },
}

ELK_BUDGET_ITEMS = {
    "in_items": {
        "Recharge": ["rcha_in"],
        "Gaining River": ["riv_hazen_in", "riv_goose_in", "riv_turtle_in"],
    },
    "out_items": {
        "Pumping": ["wel_wel_out"],
        "Drains": ["drn_out"],
        "Losing River": ["riv_hazen_out", "riv_goose_out", "riv_turtle_out"],
    },
}



# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _to_array(obj):
    return obj.array if hasattr(obj, "array") else obj

def _read_mf6_heads_3d(sim_ws, sim_name=None, headfile_name=None, kstpkper=(0, 0)):
    """
    Read MF6 heads at a given (kstp,kper) from a .hds file.

    If headfile_name is None, tries to detect from the OC package output control.
    Fallback: looks for '*.hds' in sim_ws and picks the first.

    Returns: heads3d (nlay,nrow,ncol), gwf model object, simulation object
    """
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    gwf = sim.get_model(sim_name) if sim_name else sim.get_model()

    # Try OC to find headfile
    hds_path = None
    if headfile_name is not None:
        hds_path = os.path.join(sim_ws, headfile_name)
    else:
        try:
            oc = getattr(gwf, "oc", None)
            if oc is not None:
                # MF6 OC can store head file name in saverecord / head_filerecord
                try:
                    hfr = oc.head_filerecord.get_data()
                    if hfr is not None:
                        # hfr can be list like [('file.hds',)]
                        hds_path = os.path.join(sim_ws, str(hfr[0][0]))
                except Exception:
                    pass
        except Exception:
            pass

    if hds_path is None or not os.path.exists(hds_path):
        # fallback: first *.hds in workspace
        cand = [f for f in os.listdir(sim_ws) if f.lower().endswith(".hds")]
        if not cand:
            raise FileNotFoundError(f"No .hds found in {sim_ws}")
        hds_path = os.path.join(sim_ws, cand[0])

    hds = flopy.utils.HeadFile(hds_path)
    heads3d = np.array(hds.get_data(kstpkper=kstpkper), dtype=float)
    return heads3d, gwf, sim


def _mask_heads(heads3d, idomain=None, huge=1.0e20):
    """
    Convert common no-data/dry sentinels to NaN and mask idomain<=0.
    """
    h = np.array(heads3d, dtype=float)

    # common MF6 sentinels can be very large magnitude
    h = np.where(np.isfinite(h) & (np.abs(h) < huge), h, np.nan)

    if idomain is not None:
        idom = np.asarray(idomain)
        if idom.ndim == 3 and h.ndim == 3 and idom.shape == h.shape:
            h = np.where(idom > 0, h, np.nan)
        elif idom.ndim == 2 and h.ndim == 3 and idom.shape == h.shape[1:]:
            h = np.where(idom[None, :, :] > 0, h, np.nan)

    return h


def _water_table_from_heads(heads3d):
    """
    First non-NaN head down the layers (top -> bottom) for each (row,col).
    """
    h = np.array(heads3d, dtype=float)
    nlay, nrow, ncol = h.shape
    wt = np.full((nrow, ncol), np.nan, dtype=float)

    # find first finite along axis=0
    finite = np.isfinite(h)
    any_f = finite.any(axis=0)
    if not np.any(any_f):
        return wt

    first_k = np.argmax(finite, axis=0)  # gives 0 where all False too, so use mask
    rr, cc = np.where(any_f)
    wt[rr, cc] = h[first_k[rr, cc], rr, cc]
    return wt

def _plus_one_year(label):
    """
    Add 1 year to a label like '2000' or 'Dec 2023'.
    """
    parts = label.split()
    if len(parts) == 1:
        # "2000"
        return str(int(parts[0]) + 1)
    else:
        # "Dec 2023"
        return f"{parts[0]} {int(parts[1]) + 1}"


def _stress_period_midpoint_years(sim):
    """
    Return array of decimal years at the midpoint of each stress period.
    """
    from datetime import timedelta
    import numpy as np

    start_dates, end_dates, _ = _build_sp_start_end_dates(sim)
    years = np.zeros(len(start_dates), dtype=float)

    for i, (s, e) in enumerate(zip(start_dates, end_dates)):
        mid = s + (e - s) / 2
        years[i] = mid.year + (mid.timetuple().tm_yday - 1) / 365.25

    return years

def _heads_2d_from_mode(heads3d, mode="layer", layer_num=1):
    """
    mode:
      - "layer": use heads3d[layer_num-1,:,:]
      - "water_table": use first finite head down the stack
    """
    mode = str(mode).lower().strip()
    if mode == "layer":
        k0 = int(layer_num) - 1
        return np.array(heads3d[k0, :, :], dtype=float)
    elif mode in ("watertable", "water_table", "water table"):
        return _water_table_from_heads(heads3d)
    else:
        raise ValueError(f"Unknown mode='{mode}'. Use 'layer' or 'water_table'.")

# %%
def plot_wahp_asr_well_tile(
    wahp_ws,
    out_png,
    asr_dir=os.path.join("..", "gis", "input_shps", "wahp", "asr_shps"),
    shp5_name="asr5well_sys.shp",
    shp10_name="asr10wells.shp",
    sim_name="wahp7ly",
    model_name_for_title="Wahpeton",
    model_crs=MODEL_CRS,
    basemap=True,
    dpi=350,
    figsize=(14.5, 6.5),
    point_size=36,
    active_layer_num=5,
):
    """
    Two-panel tile plot (MODEL_CRS):
      Left  = 5-well ASR layout
      Right = 10-well ASR layout

    Includes:
      - dashed model boundary
      - active-area outline (layer 5)
      - ASR wells: transparent red fill + solid red outline
      - basemap, north arrow, scale bar, neatline
    """
    import os
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import flopy

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    shp5 = os.path.join(asr_dir, shp5_name)
    shp10 = os.path.join(asr_dir, shp10_name)

    if not os.path.exists(shp5):
        raise FileNotFoundError(f"Missing ASR shapefile: {shp5}")
    if not os.path.exists(shp10):
        raise FileNotFoundError(f"Missing ASR shapefile: {shp10}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    sim = flopy.mf6.MFSimulation.load(sim_ws=wahp_ws, exe_name="mf6")
    m = sim.get_model(sim_name) if sim_name else sim.get_model()
    dis = m.dis

    # --------------------------------------------------
    # Model outline + extent (MODEL_CRS)
    # --------------------------------------------------
    model_outline = _grid_outline_gdf(m)
    extent = _get_extent_from_gdf(model_outline, pad_frac=0.02)

    # --------------------------------------------------
    # Active-area outline (layer N, default = 5)
    # --------------------------------------------------
    active_outline = None
    idomain = _get_idomain(dis)
    if idomain is not None:
        idom = np.asarray(idomain)
        k0 = int(active_layer_num) - 1
        if idom.ndim == 3 and 0 <= k0 < idom.shape[0]:
            active_outline = _active_area_outline_gdf(
                m, idom[k0, :, :], crs=model_crs
            )
        elif idom.ndim == 2:
            active_outline = _active_area_outline_gdf(
                m, idom, crs=model_crs
            )

    # --------------------------------------------------
    # Read ASR shapefiles → MODEL_CRS
    # --------------------------------------------------
    g5 = gpd.read_file(shp5)
    g10 = gpd.read_file(shp10)

    if g5.crs is None:
        g5 = g5.set_crs(model_crs)
    else:
        g5 = g5.to_crs(model_crs)

    if g10.crs is None:
        g10 = g10.set_crs(model_crs)
    else:
        g10 = g10.to_crs(model_crs)

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"{model_name_for_title} — ASR Injection Locations",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(top=0.90, wspace=0.04)

    panels = [
        (ax1, g5, "5-Well ASR Configuration"),
        (ax2, g10, "10-Well ASR Configuration"),
    ]

    for ax, pts, ttl in panels:
        ax.set_title(ttl, fontsize=13)

        # extent first
        _apply_extent(ax, extent)

        # basemap
        if basemap:
            add_basemap(ax)

        # dashed model boundary
        model_outline.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
            zorder=18,
        )

        # active-area outline (solid)
        if active_outline is not None:
            active_outline.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=1.4,
                alpha=0.75,
                zorder=22,
            )

        # ----------------------------------------------
        # ASR injection points (transparent fill + red edge)
        # ----------------------------------------------
        if len(pts) > 0:
            pts.plot(
                ax=ax,
                markersize=point_size,
                facecolor=(1.0, 0.0, 0.0, 0.35),
                edgecolor="red",
                linewidth=1.5,
                zorder=30,
            )

        # map furniture
        add_north_arrow(ax)
        if ADD_SCALE_BAR:
            add_scale_bar(ax)
        add_neatline(ax)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"Wrote: {out_png}")


def _add_head_contours(
    ax,
    x2d,
    y2d,
    z2d,
    contour_interval=20.0,
    lw=0.6,
    color="k",
    label_every=None,     # None or int stride for levels
    fontsize=7,
    add_apostrophe=True,
    zorder=120,
):
    """
    Draw contours + white-halo labels.
    """
    z = np.array(z2d, dtype=float)
    z = np.where(np.isfinite(z), z, np.nan)
    if not np.isfinite(z).any():
        return None, []

    zmin = np.nanmin(z)
    zmax = np.nanmax(z)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return None, []

    # build levels rounded to interval
    lo = np.floor(zmin / contour_interval) * contour_interval
    hi = np.ceil(zmax / contour_interval) * contour_interval
    levels = np.arange(lo, hi + contour_interval, contour_interval)

    if label_every is not None and int(label_every) > 1:
        levels_lab = levels[::int(label_every)]
    else:
        levels_lab = levels

    cs = ax.contour(x2d, y2d, z, levels=levels, colors=color, linewidths=lw, zorder=zorder)
    texts = ax.clabel(cs, levels=levels_lab, fmt="%.0f", inline=True, fontsize=fontsize, colors=color)

    for t in texts:
        if add_apostrophe:
            t.set_text(t.get_text() + "'")
        t.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])

    return cs, texts

def _get_rch_3d_from_gwf(gwf) -> np.ndarray:
    """
    Returns recharge as (nper, nrow, ncol) in ft/day.
    Accepts shapes:
      (nrow, ncol)
      (nper, nrow, ncol)
      (nper, nlay, nrow, ncol) -> summed over layers
    """
    if not hasattr(gwf, "rch") or gwf.rch is None:
        raise ValueError("Model has no RCH package (gwf.rch missing).")

    rch = gwf.rch
    full = None

    if hasattr(rch, "recharge") and hasattr(rch.recharge, "array"):
        full = _to_array(rch.recharge)
    elif hasattr(rch, "rech") and hasattr(rch.rech, "array"):
        full = _to_array(rch.rech)
    else:
        raise ValueError("RCH package does not have a recognizable recharge array (.recharge/.rech).")

    full = np.asarray(full, dtype=float)

    if full.ndim == 2:
        full = full[None, ...]  # (1, nrow, ncol)
    elif full.ndim == 3:
        pass  # (nper, nrow, ncol)
    elif full.ndim == 4:
        full = full.sum(axis=1)  # (nper, nrow, ncol)
    else:
        raise ValueError(f"Unsupported RCH shape: {full.shape}")

    return full


def _parse_mf6_start_datetime(sim) -> datetime:
    raw = None
    if hasattr(sim, "tdis") and hasattr(sim.tdis, "start_date_time"):
        raw = sim.tdis.start_date_time.get_data()
    if raw is None:
        raise ValueError("Cannot read sim.tdis.start_date_time")

    s = str(raw).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    raise ValueError(f"Unrecognized start_date_time: {raw!r}")


def _get_tdis_perlen_days(sim) -> np.ndarray:
    pd = sim.tdis.perioddata.array
    return np.atleast_1d(pd["perlen"]).astype(float)


def _build_sp_start_end_dates(sim):
    start0 = _parse_mf6_start_datetime(sim)
    perlen_days = _get_tdis_perlen_days(sim)
    sp_start = []
    sp_end = []
    cur = start0
    for d in perlen_days:
        sp_start.append(cur)
        cur2 = cur + timedelta(days=float(d))
        sp_end.append(cur2)
        cur = cur2
    return np.array(sp_start, dtype=object), np.array(sp_end, dtype=object), perlen_days

def _kper_label_from_tdis(sim, kper, style="year"):
    """
    style:
      - "year"      -> "2000"
      - "monthyear" -> "Dec 2023"
    Label is based on the midpoint date of the stress period.
    """
    sp_start, sp_end, _ = _build_sp_start_end_dates(sim)
    s = sp_start[int(kper)]
    e = sp_end[int(kper)]
    mid = s + (e - s) / 2

    if style == "year":
        return f"{mid.year}"
    if style in ("monthyear", "monyear"):
        return mid.strftime("%b %Y")
    raise ValueError(f"Unknown style='{style}'")

def find_kper_for_datetime(sim_ws, target_dt, sim_name=None):
    """
    Return the MF6 stress period index (kper) that contains target_dt, based on TDIS start_date_time + perlen.

    We treat a stress period as [start, end). If target_dt falls on an exact boundary,
    it will belong to the later stress period (except at the very end).
    """
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    # ensure TDIS is loaded; _build_sp_start_end_dates uses sim.tdis
    sp_start, sp_end, _ = _build_sp_start_end_dates(sim)

    t = target_dt
    # Find first kper where start <= t < end
    for kper, (s, e) in enumerate(zip(sp_start, sp_end)):
        if (t >= s) and (t < e):
            return kper

    # If we didn't find it, give a helpful error
    raise ValueError(
        f"target_dt={t} is outside simulation period "
        f"({sp_start[0]} to {sp_end[-1]})."
    )

def _find_rch_package(model):
    """
    Return the recharge package object if present.
    Tries common MF6 names/types.
    """
    # direct attribute (common)
    for name in ("rch", "rcha"):
        if hasattr(model, name):
            pkg = getattr(model, name)
            if pkg is not None:
                return pkg

    # search packages
    try:
        for pkg in model.packagelist:
            try:
                ptype = pkg.package_type.lower()
            except Exception:
                ptype = ""
            if "rch" in ptype:
                return pkg
    except Exception:
        pass

    return None


def _get_rch_2d_for_kper(rch_pkg, kper, nrow, ncol):
    """
    Robustly pull a 2D recharge array (rate) for stress period kper.
    """
    if rch_pkg is None:
        raise ValueError("Recharge package not found on model.")

    # MF6 RCH package usually exposes: rch_pkg.recharge
    var = None
    for attr in ("recharge", "rch"):
        if hasattr(rch_pkg, attr):
            var = getattr(rch_pkg, attr)
            break
    if var is None:
        raise ValueError("Could not find recharge array on RCH package (expected .recharge or .rch).")

    # Transient: var.array may be (nper, nrow, ncol) or (nrow, ncol) or dict-like
    try:
        arr = np.array(var.array, dtype=float)
        if arr.ndim == 3:
            return arr[kper, :, :]
        if arr.ndim == 2:
            return arr
    except Exception:
        pass

    # dict-like transient accessor
    try:
        data = var.get_data(kper=kper)
        arr = np.array(data, dtype=float)
        if arr.shape == (nrow, ncol):
            return arr
    except Exception:
        pass

    raise ValueError("Unable to extract recharge 2D array for stress period.")

def _active_area_outline_gdf(model, id2d, crs=MODEL_CRS):
    """
    Build a dissolved polygon outline for the active area (idomain>0) for a given layer.
    Returns None if everything is active (active area == full grid extent) OR if id2d is None.
    """
    if id2d is None:
        return None

    id2d = np.asarray(id2d)
    if id2d.ndim != 2:
        return None

    # If all cells are active, active outline == grid outline -> skip
    if np.all(id2d > 0):
        return None

    mg = model.modelgrid
    nrow, ncol = mg.nrow, mg.ncol

    # Collect polygons for boundary active cells (active that touch inactive or outside)
    geoms = []
    for i in range(nrow):
        for j in range(ncol):
            if id2d[i, j] <= 0:
                continue

            touches_inactive = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if ii < 0 or ii >= nrow or jj < 0 or jj >= ncol:
                    touches_inactive = True
                    break
                if id2d[ii, jj] <= 0:
                    touches_inactive = True
                    break

            if not touches_inactive:
                continue

            verts = mg.get_cell_vertices(i, j)
            if verts and len(verts) >= 3:
                geoms.append(Polygon(verts))

    if not geoms:
        return None

    merged = unary_union(geoms)
    if merged is None or merged.is_empty:
        return None

    return gpd.GeoDataFrame({"name": ["active_area"]}, geometry=[merged], crs=crs)

def _mask_non_bc_cells(cond2d):
    """Keep only positive conductance cells; everything else -> NaN (so it won't plot)."""
    c = np.asarray(cond2d, dtype=float)
    return np.where(np.isfinite(c) & (c > 0.0), c, np.nan)


def _active_area_outline_gdf(model, id2d, crs=MODEL_CRS):
    """
    Return a GeoDataFrame containing ONLY the exterior outline
    of the active area (idomain > 0).

    - No internal cell edges
    - Skips if entire grid is active
    """
    if id2d is None:
        return None

    id2d = np.asarray(id2d)
    if id2d.ndim != 2:
        return None

    # If everything is active, skip (same as grid outline)
    if np.all(id2d > 0):
        return None

    mg = model.modelgrid
    nrow, ncol = mg.nrow, mg.ncol

    polys = []

    # Build polygons for all active cells
    for i in range(nrow):
        for j in range(ncol):
            if id2d[i, j] <= 0:
                continue
            verts = mg.get_cell_vertices(i, j)
            if verts and len(verts) >= 3:
                polys.append(Polygon(verts))

    if not polys:
        return None

    merged = unary_union(polys)
    if merged.is_empty:
        return None

    # ---- EXTRACT ONLY EXTERIOR BOUNDARY ----
    if merged.geom_type == "Polygon":
        outline = merged.exterior

    elif merged.geom_type == "MultiPolygon":
        outlines = [p.exterior for p in merged.geoms if not p.is_empty]
        outline = MultiLineString(outlines)

    else:
        return None

    return gpd.GeoDataFrame(
        {"name": ["active_area_outline"]},
        geometry=[outline],
        crs=crs,
    )

def _list_mf6_packages_by_type(model, ptype: str):
    """Return list of MF6 package objects whose package_type matches ptype (e.g., 'riv', 'drn')."""
    out = []
    ptype = ptype.lower()
    try:
        for pkg in model.packagelist:
            try:
                if str(pkg.package_type).lower() == ptype:
                    out.append(pkg)
            except Exception:
                continue
    except Exception:
        pass
    return out

def _normalize_spd_records(rec, kper=None):
    """
    Normalize MF6 stress_period_data.get_data() output to an iterable of records.

    Handles:
      - recarray / list -> returned as-is
      - dict: {kper: recarray} or wrappers -> pull the recarray/value
      - scalar record -> returns [record]
    """
    if rec is None:
        return None

    # If dict, pull the appropriate value
    if isinstance(rec, dict):
        if len(rec) == 0:
            return None

        # Most common: {kper: recarray}
        if kper is not None and kper in rec:
            rec = rec[kper]
        else:
            # wrapper keys sometimes appear
            for key in ("stress_period_data", "data", "records"):
                if key in rec:
                    rec = rec[key]
                    break
            else:
                # fallback: first value
                rec = next(iter(rec.values()))

    # If scalar record, make it list-like
    try:
        _ = len(rec)
        return rec
    except Exception:
        return [rec]


def _layers_with_any(cond3d):
    """Return 1-based layer numbers where any finite positive conductance exists."""
    out = []
    for k0 in range(cond3d.shape[0]):
        c = cond3d[k0, :, :]
        if np.isfinite(c).any() and np.nanmax(c) > 0:
            out.append(k0 + 1)
    return out

def _accumulate_list_bc_to_arrays(
    dis,
    pkgs,
    kper=0,
    mode="kper",   # "kper" or "mean"
    sim=None,      # required if mode=="mean"
    bc_type="riv", # "riv" or "drn"
):
    """
    Build (nlay,nrow,ncol) arrays:
      - value3d: stage (riv) or elev (drn)
      - cond3d: conductance

    Robust to:
      - cellid stored as (k,i,j) tuple
      - cellid stored as integer node id
      - get_data() returning dict wrappers

    Overlaps:
      cond sums; value is conductance-weighted average.
    """
    nlay = int(dis.nlay.get_data())
    nrow = int(dis.nrow.get_data())
    ncol = int(dis.ncol.get_data())
    nrc = nrow * ncol

    bc_type = str(bc_type).lower().strip()
    if bc_type not in ("riv", "drn"):
        raise ValueError("bc_type must be 'riv' or 'drn'")

    def _node_to_kij(node):
        """Convert integer node id -> (k,i,j) for structured grids (DIS). Handles 0-based or 1-based nodes."""
        node = int(node)

        # heuristic: if node in [1, nlay*nrow*ncol], treat as 1-based
        if 1 <= node <= nlay * nrc:
            node0 = node - 1
        else:
            node0 = node

        k = node0 // nrc
        rem = node0 % nrc
        i = rem // ncol
        j = rem % ncol

        if k < 0 or k >= nlay or i < 0 or i >= nrow or j < 0 or j >= ncol:
            return None
        return k, i, j

    def _get_field(rec, r, name):
        """Safely fetch field from recarray record, object, or dict-like."""
        # recarray
        if rec is not None and hasattr(rec, "dtype") and rec.dtype.names and name in rec.dtype.names:
            try:
                return r[name]
            except Exception:
                pass
        # attribute access
        if hasattr(r, name):
            return getattr(r, name)
        # dict-like
        try:
            return r[name]
        except Exception:
            return None

    def _extract_kij(rec, r):
        """
        Returns (k,i,j) 0-based or None.
        Supports:
          - cellid=(k,i,j)
          - cellid=node(int)
          - fallback fields
        """
        cellid = _get_field(rec, r, "cellid")

        # tuple/list cellid
        if isinstance(cellid, (tuple, list, np.ndarray)):
            if len(cellid) == 3:
                return int(cellid[0]), int(cellid[1]), int(cellid[2])
            if len(cellid) == 1:
                return _node_to_kij(cellid[0])

        # integer node id
        if isinstance(cellid, (int, np.integer)):
            return _node_to_kij(cellid)

        # fallback separate fields
        for a, b, c in [
            ("cellid_layer", "cellid_row", "cellid_column"),
            ("k", "i", "j"),
            ("layer", "row", "column"),
        ]:
            ka = _get_field(rec, r, a)
            ib = _get_field(rec, r, b)
            jc = _get_field(rec, r, c)
            if ka is not None and ib is not None and jc is not None:
                return int(ka), int(ib), int(jc)

        # last resort: record itself is tuple/list
        if isinstance(r, (tuple, list)) and len(r) >= 1:
            cid = r[0]
            if isinstance(cid, (tuple, list, np.ndarray)) and len(cid) == 3:
                return int(cid[0]), int(cid[1]), int(cid[2])
            if isinstance(cid, (int, np.integer)):
                return _node_to_kij(cid)

        return None

    def _accumulate_from_rec(rec_in, kper_for_dict, cond_sum, val_wsum):
        """Accumulate cond and cond*value from one stress-period record set."""
        rec = _normalize_spd_records(rec_in, kper=kper_for_dict)
        if rec is None:
            return

        for r in rec:
            kij = _extract_kij(rec, r)
            if kij is None:
                continue
            k, i, j = kij

            c = _get_field(rec, r, "cond")
            if c is None:
                continue
            c = float(c)
            if not np.isfinite(c) or c <= 0:
                continue

            if bc_type == "riv":
                v = _get_field(rec, r, "stage")
            else:
                v = _get_field(rec, r, "elev")

            if v is None:
                continue
            v = float(v)
            if not np.isfinite(v):
                continue

            cond_sum[k, i, j] += c
            val_wsum[k, i, j] += c * v

    # ---- mean mode ----
    if str(mode).lower() == "mean":
        if sim is None:
            raise ValueError("sim is required when mode='mean' (for perlen weighting).")

        perlen_days = _get_tdis_perlen_days(sim)
        nper = len(perlen_days)

        cond_dt = np.zeros((nlay, nrow, ncol), dtype=float)
        valc_dt = np.zeros((nlay, nrow, ncol), dtype=float)

        for kp in range(nper):
            dt = float(perlen_days[kp])
            if dt <= 0:
                continue

            ctmp = np.zeros((nlay, nrow, ncol), dtype=float)
            vctmp = np.zeros((nlay, nrow, ncol), dtype=float)

            for pkg in pkgs:
                try:
                    rec = pkg.stress_period_data.get_data(kper=kp)
                except Exception:
                    rec = None
                _accumulate_from_rec(rec, kp, ctmp, vctmp)

            cond_dt += ctmp * dt
            valc_dt += vctmp * dt

        value = np.full((nlay, nrow, ncol), np.nan, dtype=float)
        m = cond_dt > 0
        value[m] = valc_dt[m] / cond_dt[m]

        dt_sum = float(np.sum(perlen_days))
        cond_mean = np.where(dt_sum > 0, cond_dt / dt_sum, 0.0)

        return value, cond_mean

    # ---- kper mode ----
    cond_sum = np.zeros((nlay, nrow, ncol), dtype=float)
    val_wsum = np.zeros((nlay, nrow, ncol), dtype=float)

    kper_i = int(kper)
    for pkg in pkgs:
        try:
            rec = pkg.stress_period_data.get_data(kper=kper_i)
        except Exception:
            rec = None
        _accumulate_from_rec(rec, kper_i, cond_sum, val_wsum)

    value = np.full((nlay, nrow, ncol), np.nan, dtype=float)
    m = cond_sum > 0
    value[m] = val_wsum[m] / cond_sum[m]

    return value, cond_sum


def compute_avg_annual_recharge_2000_2023(sim_ws, sim_name=None, year0=2000, year1=2023):
    """
    Returns:
      avg_depth_ft_per_yr (2D): mean annual recharge depth (ft/yr) over year0..year1
      avg_rate_ft_per_day (2D): time-weighted mean recharge rate (ft/day) over the same interval
    """
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    gwf = sim.get_model(sim_name) if sim_name else sim.get_model()

    # RCH stack (nper, nrow, ncol) in ft/day
    rch3d = _get_rch_3d_from_gwf(gwf)
    nper, nrow, ncol = rch3d.shape

    sp_start, sp_end, perlen_days = _build_sp_start_end_dates(sim)
    if perlen_days.size != nper:
        raise ValueError(f"TDIS nper ({perlen_days.size}) != RCH nper ({nper})")

    # Create year accumulators (depth in feet, and total days)
    years = list(range(year0, year1 + 1))
    yearly_depth = {y: np.zeros((nrow, ncol), dtype=float) for y in years}
    yearly_days = {y: 0.0 for y in years}

    win_start = datetime(year0, 1, 1)
    win_end = datetime(year1 + 1, 1, 1)

    for kper in range(nper):
        s = sp_start[kper]
        e = sp_end[kper]
        if e <= win_start or s >= win_end:
            continue

        r2d = rch3d[kper, :, :]
        r2d = np.where(np.isfinite(r2d), r2d, 0.0)

        cur_s = max(s, win_start)
        cur_e = min(e, win_end)

        # Split any SP that crosses a calendar year boundary
        while cur_s < cur_e:
            y = cur_s.year
            y_end = datetime(y + 1, 1, 1)
            seg_e = min(cur_e, y_end)
            dt_days = (seg_e - cur_s).total_seconds() / 86400.0
            if y in yearly_depth and dt_days > 0:
                yearly_depth[y] += r2d * dt_days  # ft/day * day = ft
                yearly_days[y] += dt_days
            cur_s = seg_e

    # Convert to mean annual depth (ft/yr) averaged across years
    # We normalize partial years to a full-year equivalent via 365.25/days_covered.
    avg_depth_ft_per_yr = np.zeros((nrow, ncol), dtype=float)
    total_depth = np.zeros((nrow, ncol), dtype=float)
    total_days = 0.0
    n_years_used = 0

    for y in years:
        if yearly_days[y] <= 0:
            continue
        depth_full_year = yearly_depth[y] * (365.25 / yearly_days[y])
        avg_depth_ft_per_yr += depth_full_year
        n_years_used += 1

        total_depth += yearly_depth[y]
        total_days += yearly_days[y]

    if n_years_used == 0 or total_days <= 0:
        raise ValueError("No stress periods overlapped the requested 2000–2023 window.")

    avg_depth_ft_per_yr /= float(n_years_used)
    avg_rate_ft_per_day = total_depth / float(total_days)

    return avg_depth_ft_per_yr, avg_rate_ft_per_day

def _read_budget_csv(sim_ws, budget_csv_path):
    """
    Read a budget-like CSV. If 'datetime' is missing (common in some zbud outputs),
    synthesize it from TDIS stress-period start dates.
    """
    import pandas as pd

    if not os.path.exists(budget_csv_path):
        raise FileNotFoundError(f"budget CSV not found: {budget_csv_path}")

    df = pd.read_csv(budget_csv_path)

    # If datetime missing, synthesize from TDIS stress-period start dates
    if "datetime" not in df.columns:
        sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
        sp_start, sp_end, perlen_days = _build_sp_start_end_dates(sim)

        nper = len(perlen_days)
        if len(df) != nper:
            raise ValueError(
                f"CSV rows ({len(df)}) != TDIS nper ({nper}) and no 'datetime' column present, "
                "so I cannot align rows to stress periods."
            )

        df["datetime"] = pd.to_datetime([d for d in sp_start])

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def _wahp_zbud_zone_to_canonical(df, zone_id):
    """
    Wahp zbud.csv columns look like:
      zbly_rcha-in, zbly_car-wel-out, zbly_riv-riv-in, ...
    and have zbly_zone as the zone selector.

    We filter zone and build canonical columns that match WAHP_BUDGET_ITEMS, plus totals/storage.
    """
    import pandas as pd
    import numpy as np

    if "zbly_zone" not in df.columns:
        raise ValueError("Wahp zbud file must have 'zbly_zone' column.")

    d = df.loc[df["zbly_zone"] == zone_id, :].copy()
    if d.empty:
        raise ValueError(f"No rows found for Wahp zbud zone {zone_id}.")

    # --- helper to pull a column safely
    def col(name):
        if name not in d.columns:
            return np.zeros(len(d), dtype=float)
        return pd.to_numeric(d[name], errors="coerce").fillna(0.0).to_numpy(float)

    out = pd.DataFrame({"datetime": pd.to_datetime(d["datetime"]) if "datetime" in d.columns else d["datetime"]})

    # storage (your net_storage calc expects these exact names)
    out["sto-ss_in"]  = col("zbly_sto-ss-in")
    out["sto-sy_in"]  = col("zbly_sto-sy-in")
    out["sto-ss_out"] = col("zbly_sto-ss-out")
    out["sto-sy_out"] = col("zbly_sto-sy-out")

    # flux terms in/out
    out["rcha_in"]      = col("zbly_rcha-in")
    out["rcha_out"]     = col("zbly_rcha-out")

    out["drn_in"]       = col("zbly_drn-in")
    out["drn_out"]      = col("zbly_drn-out")

    out["riv_in"]       = col("zbly_riv-riv-in")
    out["riv_out"]      = col("zbly_riv-riv-out")

    out["otriv_in"]     = col("zbly_otriv-riv-in")
    out["otriv_out"]    = col("zbly_otriv-riv-out")

    out["ghb_in"]       = col("zbly_ghb-ghb-in")
    out["ghb_out"]      = col("zbly_ghb-ghb-out")

    out["ghb_wbv_in"]   = col("zbly_ghb_wbv-ghb-in")
    out["ghb_wbv_out"]  = col("zbly_ghb_wbv-ghb-out")

    # wells (keep separate)
    out["wel_car_in"]   = col("zbly_car-wel-in")
    out["wel_car_out"]  = col("zbly_car-wel-out")

    out["wel_malt_in"]  = col("zbly_malt-wel-in")
    out["wel_malt_out"] = col("zbly_malt-wel-out")

    out["wel_cow_in"]   = col("zbly_cow-wel-in")
    out["wel_cow_out"]  = col("zbly_cow-wel-out")

    out["wel_minn_in"]  = col("zbly_minn-wel-in")
    out["wel_minn_out"] = col("zbly_minn-wel-out")

    out["wel_cob_in"]   = col("zbly_cob-wel-in")
    out["wel_cob_out"]  = col("zbly_cob-wel-out")
    
    # ---- interlayer exchange (layer-zone only), keep separate
    out["layer4_exch_in"]  = col("zbly_from_zone_4")
    out["layer4_exch_out"] = col("zbly_to_zone_4")

    out["layer6_exch_in"]  = col("zbly_from_zone_6")
    out["layer6_exch_out"] = col("zbly_to_zone_6")

    # totals: include storage so net_storage math stays consistent with your formula
    in_cols  = [c for c in out.columns if c.endswith("_in") or c.endswith("-in")]
    out_cols = [c for c in out.columns if c.endswith("_out") or c.endswith("-out")]
    out["total_in"]  = out[in_cols].sum(axis=1)
    out["total_out"] = out[out_cols].sum(axis=1)

    return out

def _swww_zbud_zone_to_canonical(df, zone_id, sim_ws):
    """
    Spiritwood zbud columns look like:
      totim,kstp,kper,zone, STO-SS-IN, WEL-OUT, DRN_RIV-DRN-OUT, RIV-IN, ...
    Usually NO datetime column. We'll synthesize datetime from TDIS in _read_budget_csv,
    or you can pass an already-synthesized df.
    """
    import pandas as pd
    import numpy as np

    # normalize column names for easier matching
    def norm(s):
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")

    cols_map = {c: norm(c) for c in df.columns}
    d0 = df.rename(columns=cols_map).copy()

    if "zone" not in d0.columns:
        raise ValueError("Spiritwood zbud file must have 'zone' column.")
    d = d0.loc[d0["zone"] == zone_id, :].copy()
    if d.empty:
        raise ValueError(f"No rows found for Spiritwood zbud zone {zone_id}.")

    def col(name_norm):
        if name_norm not in d.columns:
            return np.zeros(len(d), dtype=float)
        return pd.to_numeric(d[name_norm], errors="coerce").fillna(0.0).to_numpy(float)

    # ensure datetime exists (if you called _read_budget_csv(sim_ws, zbud_path) it will)
    if "datetime" not in d.columns:
        # synthesize here as fallback
        sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
        sp_start, _, perlen_days = _build_sp_start_end_dates(sim)
        if len(d) != len(perlen_days):
            raise ValueError("Cannot synthesize datetime: zbud rows != nper.")
        d["datetime"] = pd.to_datetime([x for x in sp_start])

    out = pd.DataFrame({"datetime": pd.to_datetime(d["datetime"])})

    # storage
    out["sto-ss_in"]  = col("sto_ss_in")
    out["sto-sy_in"]  = col("sto_sy_in")
    out["sto-ss_out"] = col("sto_ss_out")
    out["sto-sy_out"] = col("sto_sy_out")

    # canonical terms expected by your SWWW_BUDGET_ITEMS
    # Recharge
    out["rcha_in"]  = col("rcha_in")
    out["rcha_out"] = col("rcha_out")

    # River
    out["riv_in"]   = col("riv_in")
    out["riv_out"]  = col("riv_out")

    # GHB
    out["ghb_in"]   = col("ghb_in")
    out["ghb_out"]  = col("ghb_out")

    # Wells (map WEL-* into your existing "wel_wel_0_*")
    out["wel_wel_0_in"]  = col("wel_in")
    out["wel_wel_0_out"] = col("wel_out")
    
    out["layer2_exch_in"]  = col("from_zone_2")
    out["layer2_exch_out"] = col("to_zone_2")

    # # Drains: sum the drain components into one drn_in/drn_out to match your items dict
    # drn_in = col("drn_drn_in") + col("drn_valley_drn_in") + col("drn_riv_drn_in")
    # drn_out = col("drn_drn_out") + col("drn_valley_drn_out") + col("drn_riv_drn_out")
    # out["drn_in"]  = drn_in
    # out["drn_out"] = drn_out

    # totals: include storage so net_storage math stays consistent
    in_cols  = [c for c in out.columns if c.endswith("_in") or c.endswith("-in")]
    out_cols = [c for c in out.columns if c.endswith("_out") or c.endswith("-out")]
    out["total_in"]  = out[in_cols].sum(axis=1)
    out["total_out"] = out[out_cols].sum(axis=1)

    return out

def load_budget_df(
    sim_ws,
    use_zbud=False,
    zbud_zone_id=None,
    budget_fname="budget.csv",
    zbud_fname="zbud.csv",
    model_kind=None,  # "wahp" or "swww"
):
    """
    Returns canonical budget df.
    - Full model: reads budget.csv (already has datetime)
    - Zonal:
        * wahp: parse funky datetime strings in zbud
        * swww: create datetime using totim + tdis.start_date_time
    """
    if not use_zbud:
        p = os.path.join(sim_ws, budget_fname)
        return _read_budget_csv(sim_ws, p)  # your normal budget.csv reader

    if zbud_zone_id is None:
        raise ValueError("use_zbud=True requires zbud_zone_id")

    pz = os.path.join(sim_ws, zbud_fname)

    mk = (model_kind or "").lower().strip()
    if mk == "wahp":
        dfz = read_wahp_zbud_with_parsed_datetime(pz)
        return _wahp_zbud_zone_to_canonical(dfz, zone_id=zbud_zone_id)

    elif mk in ("swww", "spiritwood", "spiritwood_warwick"):
        dfz = read_swww_zbud_with_totim_datetime(sim_ws=sim_ws, zbud_path=pz)
        return _swww_zbud_zone_to_canonical(dfz, zone_id=zbud_zone_id, sim_ws=sim_ws)

    else:
        raise ValueError("model_kind must be 'wahp' or 'swww' when use_zbud=True")



def add_neatline(ax, lw=1.2, color="black"):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)


def add_north_arrow(ax, img_path=NORTH_ARROW_PNG, zoom=NORTH_ARROW_ZOOM, xy=NORTH_ARROW_XY):
    try:
        img = mpimg.imread(img_path)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy, xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)
    except Exception:
        ax.text(
            xy[0], xy[1], "N",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )


def add_scale_bar(ax, miles=SCALE_BAR_MILES, pad_frac=SCALE_BAR_PAD_FRAC):
    length_m = float(miles) * 1609.344
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w = x1 - x0
    h = y1 - y0
    xs = x0 + pad_frac * w
    ys = y0 + pad_frac * h
    xe = xs + length_m
    ax.plot([xs, xe], [ys, ys], color=SCALE_BAR_COLOR, linewidth=SCALE_BAR_LW, zorder=60)
    ax.text(
        (xs + xe) / 2.0,
        ys + 0.02 * h,
        f"{miles:g} Miles",
        fontsize=SCALE_BAR_FONTSIZE,
        ha="center",
        va="bottom",
        color=SCALE_BAR_COLOR,
        zorder=61,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
    )


def add_basemap(ax):
    """
    IMPORTANT: call only AFTER axis extent is set.
    """
    if USE_USGS_TOPO_BASEMAP:
        cx.add_basemap(
            ax=ax,
            source=USGS_TOPO_PROVIDER,
            crs=MODEL_CRS,
            zorder=1,
            alpha=1.0,
            reset_extent=False,
        )
    else:
        cx.add_basemap(
            ax=ax,
            source="https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}",
            crs=MODEL_CRS,
            zorder=4,
            alpha=1.0,
            attribution="",
        )

        time.sleep(2.0)

        cx.add_basemap(
            ax=ax,
            source="https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}",
            crs=MODEL_CRS,
            zorder=1,
            alpha=1.0,
            attribution="",
        )


def _get_idomain(dis):
    try:
        return dis.idomain.array
    except Exception:
        return None


def _grid_to_polygons_gdf(model, values_2d, crs=MODEL_CRS, value_field="val"):
    mg = model.modelgrid
    nrow, ncol = mg.nrow, mg.ncol

    geoms, vals = [], []
    for i in range(nrow):
        for j in range(ncol):
            v = values_2d[i, j]
            if not np.isfinite(v):
                continue
            verts = mg.get_cell_vertices(i, j)
            if not verts or len(verts) < 3:
                continue
            geoms.append(Polygon(verts))
            vals.append(float(v))

    return gpd.GeoDataFrame({value_field: vals}, geometry=geoms, crs=crs)


def _grid_outline_gdf(model):
    mg = model.modelgrid
    nrow, ncol = mg.nrow, mg.ncol

    top = [mg.get_cell_vertices(0, j)[0] for j in range(ncol)] + [mg.get_cell_vertices(0, ncol - 1)[1]]
    right = [mg.get_cell_vertices(i, ncol - 1)[1] for i in range(nrow)] + [mg.get_cell_vertices(nrow - 1, ncol - 1)[2]]
    bottom = [mg.get_cell_vertices(nrow - 1, j)[2] for j in range(ncol - 1, -1, -1)] + [mg.get_cell_vertices(nrow - 1, 0)[3]]
    left = [mg.get_cell_vertices(i, 0)[3] for i in range(nrow - 1, -1, -1)] + [mg.get_cell_vertices(0, 0)[0]]

    ring = top + right + bottom + left
    poly = Polygon(ring)
    gdf = gpd.GeoDataFrame({"name": ["grid_extent"]}, geometry=[poly], crs=MODEL_CRS)
    return gdf


def add_grid_outline(ax, grid_outline_3857, lw=1.5, color="black", alpha=0.9, zorder=40):
    try:
        grid_outline_3857.plot(
            ax=ax, facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha, zorder=zorder, linestyle="--"
        )
    except Exception:
        pass


def _get_extent_from_gdf(gdf_3857, pad_frac=0.02):
    xmin, ymin, xmax, ymax = gdf_3857.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    return (xmin - pad_frac * dx, xmax + pad_frac * dx, ymin - pad_frac * dy, ymax + pad_frac * dy)


def _apply_extent(ax, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _apply_coord_overrides(model, xoff_override=None, yoff_override=None, angrot_override=None, crs=MODEL_CRS):
    try:
        model.modelgrid.set_coord_info(
            xoff=xoff_override if xoff_override is not None else model.modelgrid.xoffset,
            yoff=yoff_override if yoff_override is not None else model.modelgrid.yoffset,
            angrot=angrot_override if angrot_override is not None else model.modelgrid.angrot,
            crs=crs,
        )
    except Exception:
        pass

# ============================================================
# BUDGET HELPERS (PASTE INTO YOUR HELPER FUNCTIONS SECTION)
# ============================================================

def read_wahp_zbud_with_parsed_datetime(zbud_path):
    """
    Wahp zbud has datetime strings like '1970-01-01_zn-1'.
    We parse only the date part before '_' and ignore the suffix.

    Returns df with datetime as pandas datetime64.
    """
    import pandas as pd

    df = pd.read_csv(zbud_path)

    if "datetime" not in df.columns:
        raise ValueError("WAHP zbud expected a 'datetime' column, but it was not found.")

    # keep only the part before underscore
    dt_raw = df["datetime"].astype(str).str.split("_").str[0]
    df["datetime"] = pd.to_datetime(dt_raw, errors="raise")

    # Helpful warning if all dates are identical (often means it's not a real timestamp)
    if df["datetime"].nunique() == 1:
        print("[WARN] WAHP zbud datetime parsed but all rows have the same date. "
              "If you need real model dates, map by zbly_kper to TDIS instead.")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def read_swww_zbud_with_totim_datetime(sim_ws, zbud_path):
    """
    Spiritwood zbud has totim (model time). Create datetime as:
      datetime = sim.tdis.start_date_time + totim (days)
    """
    import pandas as pd

    df = pd.read_csv(zbud_path)

    # normalize column names a bit
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    if "totim" not in df.columns:
        raise ValueError("Spiritwood zbud expected a 'totim' column, but it was not found.")

    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    start0 = _parse_mf6_start_datetime(sim)

    # totim in MF6 is typically in time units of the model (for you: days)
    totim_days = pd.to_numeric(df["totim"], errors="coerce")
    if totim_days.isna().any():
        raise ValueError("Some totim values could not be parsed as numeric.")

    df["datetime"] = pd.to_datetime(start0) + pd.to_timedelta(totim_days, unit="D")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _safe_col(df, col):
    # return a numeric series; missing -> zeros
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return x


def _budget_rates_to_annual_volumes_acft(
    sim_ws,
    budget_df,
    year0=None,
    year1=None,
    use_datetime_col="datetime",
):
    """
    Convert stress-period rates (cfs or cfd?) to annual volumes (acre-ft/yr) by:
      volume_cf = rate_(cfd) * perlen_days
      volume_acft = volume_cf / 43560

    IMPORTANT: You said budget.csv rates are cubic feet/day (cfd). We use perlen_days from TDIS.
    We also split any stress period that crosses a calendar-year boundary (same logic as your recharge calc).
    """
    import pandas as pd

    # load tdis perlen and build exact SP start/end timestamps
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    sp_start, sp_end, perlen_days = _build_sp_start_end_dates(sim)

    nper = len(perlen_days)
    if len(budget_df) != nper:
        # try to align by datetime if the csv includes only a subset or has extra rows
        # We assume the budget rows correspond to stress periods in order; if not, this will need custom alignment.
        raise ValueError(
            f"budget.csv rows ({len(budget_df)}) != TDIS nper ({nper}). "
            "If your budget file is not 1-row-per-stress-period in order, tell me how it aligns."
        )

    if year0 is None:
        year0 = int(pd.to_datetime(sp_start[0]).year)
    if year1 is None:
        year1 = int(pd.to_datetime(sp_end[-1]).year)

    years = list(range(year0, year1 + 1))
    win_start = datetime(year0, 1, 1)
    win_end = datetime(year1 + 1, 1, 1)

    # We’ll return a structure you can fill per item:
    # annual_volumes[item] -> array(len(years)) in acre-ft/yr
    def allocate():
        return {y: 0.0 for y in years}

    return years, allocate, sp_start, sp_end, perlen_days, win_start, win_end

def debug_budget_alignment(sim_ws, df, label="MODEL", n=6):
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    sp_start, sp_end, perlen_days = _build_sp_start_end_dates(sim)

    dt0 = pd.to_datetime(df["datetime"].iloc[0])
    print("\n" + "="*80)
    print(f"[ALIGN DEBUG] {label}")
    print(f"  df rows = {len(df)} | tdis nper = {len(perlen_days)}")
    print(f"  df first datetime = {dt0}")
    print(f"  sp_start[0] = {sp_start[0]} (perlen={perlen_days[0]} d)")
    print(f"  sp_start[1] = {sp_start[1]} (perlen={perlen_days[1]} d)")
    print(f"  sp_end[0]   = {sp_end[0]}")
    print(f"  matches sp_start[0]? {dt0 == sp_start[0]}")
    print(f"  matches sp_start[1]? {dt0 == sp_start[1]}")
    print(f"  matches sp_end[0]?   {dt0 == sp_end[0]}")

    print("  First few df datetimes:")
    for i in range(min(n, len(df))):
        print(f"    row {i:3d}: {pd.to_datetime(df['datetime'].iloc[i])}")

    print("  First few sp_start:")
    for k in range(min(n, len(perlen_days))):
        print(f"    kper {k:3d}: {sp_start[k]} -> {sp_end[k]} (perlen={perlen_days[k]} d)")
    print("="*80 + "\n")
    
def align_budget_df_to_tdis(sim_ws, df):
    """
    Returns a copy of df with an integer column 'kper' that matches TDIS stress periods
    by datetime == sp_start[kper]. Rows that don't match get NaN and are dropped.
    """
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    sp_start, sp_end, perlen_days = _build_sp_start_end_dates(sim)

    # Build lookup from start datetime to kper
    start_to_kper = {pd.Timestamp(s): k for k, s in enumerate(sp_start)}

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["kper"] = out["datetime"].map(start_to_kper)

    n_bad = out["kper"].isna().sum()
    if n_bad > 0:
        print(f"[WARN] {n_bad} budget rows did not match any TDIS sp_start and will be dropped.")
        # helpful: show first few mismatches
        bad = out.loc[out["kper"].isna(), "datetime"].head(5).tolist()
        print("       example unmatched datetimes:", bad)

    out = out.dropna(subset=["kper"]).copy()
    out["kper"] = out["kper"].astype(int)

    # sort by kper just in case
    out = out.sort_values("kper").reset_index(drop=True)
    return out

def debug_first_row_recharge_units(df):
    r = float(pd.to_numeric(df["rcha_in"].iloc[0], errors="coerce"))
    acftyr = r * 365.0 / 43560.0
    print(f"[UNIT CHECK] row0 rcha_in={r:,.3f} (assumed cfd) => {acftyr:,.1f} ac-ft/yr")


def build_annual_budget_series(
    sim_ws,
    budget_csv_path=None,
    items_dict=None,
    year0=None,
    year1=None,
    budget_df=None,
    datetime_col="datetime",
):
    """
    Annualize MF6 budget rates using ONLY the budget file's datetime column, aligned by ROW ORDER.

    Convention:
      - df[datetime] is the END timestamp of each row interval.
        Row i represents (t1 - dt_days[i], t1], where t1 = datetime[i].
      - dt_days[i] = datetime[i] - datetime[i-1] for i>=1
      - dt_days[0] estimated from median of first ~10 positive deltas (fallback 365.25)

    IMPORTANT FIX:
      - If year0 is None, we derive it from the START of the first inferred interval
        (so the first "steady-state" row contributes to the prior year bin).
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime as _dt

    if items_dict is None:
        raise ValueError("items_dict is required.")

    # ---- load dataframe
    if budget_df is None:
        if budget_csv_path is None:
            raise ValueError("Provide budget_csv_path or budget_df.")
        if not os.path.exists(budget_csv_path):
            raise FileNotFoundError(f"budget file not found: {budget_csv_path}")
        df = pd.read_csv(budget_csv_path)
    else:
        df = budget_df.copy()

    if datetime_col not in df.columns:
        raise ValueError(f"budget_df must include '{datetime_col}' column.")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # ---- compute dt_days as END-stamped intervals: dt[i] = t[i] - t[i-1]
    t = df[datetime_col].to_numpy()
    dt_days = np.zeros(len(df), dtype=float)

    if len(df) >= 2:
        d = (t[1:] - t[:-1]).astype("timedelta64[s]").astype(float) / 86400.0
        dt_days[1:] = d

        d_pos = d[np.isfinite(d) & (d > 0)]
        if d_pos.size:
            take = min(10, d_pos.size)
            dt_days[0] = float(np.median(d_pos[:take]))
        else:
            dt_days[0] = 365.25
    else:
        dt_days[0] = 365.25

    bad = ~(np.isfinite(dt_days) & (dt_days > 0))
    if np.any(bad):
        med = float(np.nanmedian(dt_days[np.isfinite(dt_days) & (dt_days > 0)])) if np.any(dt_days > 0) else 365.25
        dt_days[bad] = med if (np.isfinite(med) and med > 0) else 365.25

    # ---- FIX: derive year window from FIRST interval START, not first END stamp
    t1_first = df[datetime_col].iloc[0].to_pydatetime()
    t0_first = t1_first - pd.to_timedelta(float(dt_days[0]), unit="D").to_pytimedelta()

    t1_last = df[datetime_col].iloc[-1].to_pydatetime()

    if year0 is None:
        year0 = int(t0_first.year)   # <-- key fix (captures "missing" first SP)
    if year1 is None:
        year1 = int(t1_last.year)

    years = list(range(int(year0), int(year1) + 1))
    win_start = _dt(int(year0), 1, 1)
    win_end = _dt(int(year1) + 1, 1, 1)

    # ---- needed columns
    needed_cols = set()
    for dct in (items_dict.get("in_items", {}), items_dict.get("out_items", {})):
        for _, cols in dct.items():
            needed_cols.update(cols)
    needed_cols.update(["total_in", "total_out", "sto-ss_in", "sto-sy_in", "sto-ss_out", "sto-sy_out"])

    def _safe_col(df_, col):
        if col not in df_.columns:
            return np.zeros(len(df_), dtype=float)
        return pd.to_numeric(df_[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    rates = {c: _safe_col(df, c) for c in needed_cols}

    # ---- annual accumulators (acre-ft)
    def allocate():
        return {y: 0.0 for y in years}

    ann_in = {lab: allocate() for lab in items_dict.get("in_items", {}).keys()}
    ann_out = {lab: allocate() for lab in items_dict.get("out_items", {}).keys()}
    ann_net = allocate()

    def add_split(year_bucket, row_idx, rate_cfd):
        t1 = df[datetime_col].iloc[row_idx].to_pydatetime()
        t0 = t1 - pd.to_timedelta(float(dt_days[row_idx]), unit="D").to_pytimedelta()

        if t1 <= win_start or t0 >= win_end:
            return

        cur_s = max(t0, win_start)
        cur_e = min(t1, win_end)

        while cur_s < cur_e:
            y = cur_s.year
            y_end = _dt(y + 1, 1, 1)
            seg_e = min(cur_e, y_end)

            seg_days = (seg_e - cur_s).total_seconds() / 86400.0
            if y in year_bucket and seg_days > 0:
                vol_cf = float(rate_cfd) * seg_days
                year_bucket[y] += vol_cf / 43560.0
            cur_s = seg_e

    # ---- accumulate by ROW ORDER
    for i in range(len(df)):
        for lab, cols in items_dict.get("in_items", {}).items():
            rsum = sum(float(rates[c][i]) for c in cols)
            add_split(ann_in[lab], i, rsum)

        for lab, cols in items_dict.get("out_items", {}).items():
            rsum = sum(float(rates[c][i]) for c in cols)
            add_split(ann_out[lab], i, rsum)

        ti = float(rates["total_in"][i])
        to = float(rates["total_out"][i])
        ssin = float(rates["sto-ss_in"][i])
        syin = float(rates["sto-sy_in"][i])
        ssout = float(rates["sto-ss_out"][i])
        syout = float(rates["sto-sy_out"][i])

        net_cfd = (ti - ssin - syin) - (to - ssout - syout)
        add_split(ann_net, i, net_cfd)

    def to_arr(d):
        return np.array([d[y] for y in years], dtype=float)

    annual_pos = {lab: to_arr(b) for lab, b in ann_in.items()}
    annual_neg = {lab: to_arr(b) for lab, b in ann_out.items()}
    net_storage = to_arr(ann_net)

    return years, annual_pos, annual_neg, net_storage




def plot_bc_stage_cond_tiles_per_layer(
    sim_ws,
    outdir,
    model_name_for_title,
    bc_type="riv",                 # "riv" or "drn"
    sim_name=None,
    xoff_override=None,
    yoff_override=None,
    angrot_override=None,
    kper=0,                        # int or "mean"
    bins_value=None,               # stage/elev bins (dict or list) or None
    bins_cond=None,                # cond bins (dict or list) or None
    dpi=1000,
):
    """
    One figure per layer that has this BC:
      left: stage (RIV) or elev (DRN)
      right: conductance
    """
    os.makedirs(outdir, exist_ok=True)

    # load full sim so we can time-average if requested
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    m = sim.get_model(sim_name) if sim_name else sim.get_model()
    dis = m.dis

    _apply_coord_overrides(m, xoff_override, yoff_override, angrot_override, crs=MODEL_CRS)

    # find pkgs
    pkgs = _list_mf6_packages_by_type(m, bc_type)
    if len(pkgs) == 0:
        print(f"[INFO] No {bc_type.upper()} packages found in {model_name_for_title}. Skipping.")
        return

    mode = "mean" if isinstance(kper, str) and kper.lower() == "mean" else "kper"

    value3d, cond3d = _accumulate_list_bc_to_arrays(
        dis=dis,
        pkgs=pkgs,
        kper=0 if mode == "mean" else int(kper),
        mode=mode,
        sim=sim,
        bc_type=bc_type.lower(),
    )

    # mask inactive
    idomain = _get_idomain(dis)
    if idomain is not None:
        value3d = _mask_inactive(value3d, idomain)
        cond3d  = _mask_inactive(cond3d, idomain)

    # plot only layers that actually have BC cells
    layers = _layers_with_any(cond3d)
    if not layers:
        print(f"[INFO] {model_name_for_title}: {bc_type.upper()} has no active cells (after masking).")
        return

    grid_outline_3857 = _grid_outline_gdf(m)
    model_extent = _get_extent_from_gdf(grid_outline_3857, pad_frac=0.02)

    safe = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")
    bc_tag = "riv" if bc_type.lower() == "riv" else "drn"

    for layer_num in layers:
        k0 = layer_num - 1

        v2d = value3d[k0, :, :]
        c2d = cond3d[k0, :, :]

        # Conductance: only show actual BC cells (avoid coloring whole active domain)
        c2d = _mask_non_bc_cells(c2d)

        # Active-area outline for this layer (skip if fully active => equals grid outline)
        id2d_layer = None
        if idomain is not None:
            id2d_layer = idomain[k0, :, :] if np.asarray(idomain).ndim == 3 else idomain
        active_outline = _active_area_outline_gdf(m, id2d_layer, crs=MODEL_CRS)

        # value bins
        edges_v = _resolve_bins_user(bins_value, v2d, layer_num)
        if edges_v is None:
            # allow negatives (stage/elev could be anything), so positive_only=False
            edges_v = _quantile_bins(v2d, positive_only=False)

        # cond bins
        edges_c = _resolve_bins_user(bins_cond, c2d, layer_num)
        if edges_c is None:
            edges_c = _quantile_bins(c2d, positive_only=True)

        v_cat, edges_v = _categorize(v2d, edges_v)
        c_cat, edges_c = _categorize(c2d, edges_c)

        # colormaps (discrete)
        cmap_v = mpl.cm.get_cmap("viridis", (edges_v.size - 1) if edges_v is not None else 5)
        cmap_c = mpl.cm.get_cmap("magma",   (edges_c.size - 1) if edges_c is not None else 5)

        fig, (ax_v, ax_c) = plt.subplots(1, 2, figsize=(14.5, 6.5))
        fig.suptitle(f"{model_name_for_title} — Layer {layer_num} {bc_tag.upper()} boundary condition", fontsize=16, y=0.98)
        fig.subplots_adjust(top=0.90)

        extra_artists = []

        # left panel: stage/elev
        left_title = "Stage (ft)" if bc_type.lower() == "riv" else "Drain Elevation (ft)"
        extra_artists += _plot_panel_with_legend(
            ax=ax_v,
            model=m,
            cat2d=v_cat,
            edges=edges_v,
            cmap=cmap_v,
            title=left_title,
            legend_title="Stage" if bc_type.lower() == "riv" else "Elev",
            legend_side="left",
            grid_outline_3857=grid_outline_3857,
            model_extent=model_extent,
            active_outline_3857=active_outline,
        )

        # right panel: conductance
        extra_artists += _plot_panel_with_legend(
            ax=ax_c,
            model=m,
            cat2d=c_cat,
            edges=edges_c,
            cmap=cmap_c,
            title="Conductance",
            legend_title="Cond",
            legend_side="right",
            grid_outline_3857=grid_outline_3857,
            model_extent=model_extent,
            active_outline_3857=active_outline,
        )

        fig.subplots_adjust(left=0.12, right=0.88, wspace=0.02)

        kper_tag = "mean" if mode == "mean" else f"kper{int(kper):03d}"
        out_png = os.path.join(outdir, f"{safe}_{bc_tag}_layer{layer_num:02d}_{kper_tag}_stage_cond.png")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.55, bbox_extra_artists=extra_artists)
        plt.close(fig)
        print(f"Wrote: {out_png}")


def plot_annual_budget_bars(
    years,
    annual_pos,
    annual_neg,
    net_storage,
    out_png,
    title,
    predictive_start_year=BUDGET_PREDICTIVE_START_YEAR,
    figsize=BUDGET_FIGSIZE,
    dpi=BUDGET_DPI,
):
    """
    Stacked bars:
      - positive inflows above zero
      - negative outflows below zero
    Dashed line:
      net_storage (already in acre-ft/yr)
    """
    x = np.array(years, dtype=int)
    annual_pos = {k: v for k, v in annual_pos.items() if np.any(np.abs(v) > 0)}
    annual_neg = {k: v for k, v in annual_neg.items() if np.any(np.abs(v) > 0)}

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # predictive shading
    if predictive_start_year is not None:
        ax.axvspan(predictive_start_year - 0.5, x.max() + 0.5, alpha=0.25, color="grey", zorder=0)
        ax.annotate(
            "Predictive Period",
            xy=(predictive_start_year + 0.5, 0.97),
            xycoords=("data", "axes fraction"),
            ha="left",
            va="top",
            fontsize=12,
        )

    # stacked positives
    bottom_pos = np.zeros_like(x, dtype=float)
    pos_handles = []
    for lab, vals in annual_pos.items():
        h = ax.bar(x, vals, bottom=bottom_pos, width=0.9, label=lab, zorder=3)
        bottom_pos += vals
        pos_handles.append(h)

    # stacked negatives (stack downward)
    bottom_neg = np.zeros_like(x, dtype=float)
    neg_handles = []
    for lab, vals in annual_neg.items():
        vneg = -np.abs(vals)  # ensure negative
        h = ax.bar(x, vneg, bottom=bottom_neg, width=0.9, label=lab, zorder=3)
        bottom_neg += vneg
        neg_handles.append(h)

    # net storage dashed line
    ax.plot(x, net_storage, linestyle="--", linewidth=2.0, color="black", label="Net Storage", zorder=4)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Flux (acre-ft/year)")
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    # cleaner ticks
    step = int(BUDGET_YEARTICK_STEP) if BUDGET_YEARTICK_STEP else 2
    xt = x[::step]
    ax.set_xticks(xt)
    ax.set_xticklabels([str(v) for v in xt], rotation=45, ha="right")

    ax.grid(True, axis="y", alpha=0.3, zorder=1)

    # Legend: put below plot area, multi-column
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


def plot_model_budget_timeseries(
    sim_ws,
    outdir,
    model_name_for_title,
    items_dict,
    year0=None,
    year1=None,
    budget_df=None,
    out_tag=None,
):
    os.makedirs(outdir, exist_ok=True)

    if budget_df is None:
        budget_csv = os.path.join(sim_ws, "budget.csv")
        years, annual_pos, annual_neg, net_storage = build_annual_budget_series(
            sim_ws=sim_ws,
            budget_csv_path=budget_csv,
            items_dict=items_dict,
            year0=year0,
            year1=year1,
        )
    else:
        years, annual_pos, annual_neg, net_storage = build_annual_budget_series(
            sim_ws=sim_ws,
            items_dict=items_dict,
            year0=year0,
            year1=year1,
            budget_df=budget_df,
        )

    safe = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")
    tag = f"_{out_tag}" if out_tag else ""
    out_png = os.path.join(outdir, f"{safe}_annual_budget{tag}.png")

    plot_annual_budget_bars(
        years=years,
        annual_pos=annual_pos,
        annual_neg=annual_neg,
        net_storage=net_storage,
        out_png=out_png,
        title=f"{model_name_for_title} — Annual Water Budget" + (f" ({out_tag})" if out_tag else ""),
        predictive_start_year=BUDGET_PREDICTIVE_START_YEAR,
    )



# ============================================================
# BIN HELPERS (FIXES YOUR "MISSING LEGEND BIN" ISSUE)
# ============================================================

def _enforce_strictly_increasing(edges, eps=None):
    """
    Ensure edges are strictly increasing WITHOUT dropping any edges.
    This prevents bins disappearing when quantiles repeat (avoid np.unique()).
    """
    e = np.asarray(edges, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2:
        return None

    if eps is None:
        span = float(np.nanmax(e) - np.nanmin(e))
        eps = max(1e-12, 1e-9 * max(span, 1.0))

    out = e.copy()
    for i in range(1, out.size):
        if not np.isfinite(out[i]) or out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps

    return out if out.size >= 2 else None

def _quantile_bins(vals2d, q=(0.0, 0.10, 0.25, 0.50, 0.75, 1.0), positive_only=True):
    """
    Build 5-bin edges (6 edges) from quantiles of vals2d.
    Uses your existing _normalize_bins to enforce strictly increasing edges.
    """
    v = np.asarray(vals2d, dtype=float)
    v = v[np.isfinite(v)]
    if positive_only:
        v = v[v > 0]
    if v.size < 10:
        return None
    edges = np.quantile(v, q)
    return _normalize_bins(edges)


def _normalize_bins(bin_edges):
    if bin_edges is None:
        return None
    return _enforce_strictly_increasing(bin_edges)


def _finite_pos(vals2d):
    v = np.asarray(vals2d, dtype=float)
    v = v[np.isfinite(v)]
    v = v[v > 0]
    return v


def _snap_bins_to_data_minmax(bin_edges, vals2d):
    if bin_edges is None:
        return None
    edges = np.array(bin_edges, dtype=float)
    v = _finite_pos(vals2d)
    if v.size == 0:
        return None
    edges[0] = float(np.nanmin(v))
    edges[-1] = float(np.nanmax(v))
    return _enforce_strictly_increasing(edges)


def _categorize(values, bin_edges):
    edges = _normalize_bins(bin_edges)
    if edges is None:
        return np.full(values.shape, np.nan, dtype=float), None
    cats = np.full(values.shape, np.nan, dtype=float)
    m = np.isfinite(values)
    idx = np.digitize(values[m], edges, right=False) - 1
    idx[idx < 0] = 0
    idx[idx >= (edges.size - 1)] = edges.size - 2
    cats[m] = idx.astype(float)
    return cats, edges


def _resolve_bins_user(bins_spec, vals2d, layer_num):
    if isinstance(bins_spec, dict) and layer_num in bins_spec:
        return _normalize_bins(_snap_bins_to_data_minmax(bins_spec[layer_num], vals2d))
    if isinstance(bins_spec, (list, tuple, np.ndarray)):
        return _normalize_bins(_snap_bins_to_data_minmax(bins_spec, vals2d))
    return None


def _legend_patches(cmap, edges, alpha=1.0, decimals=2):
    """
    Drop-in replacement.

    Ensures the colormap has exactly one color per bin (len(edges)-1),
    so legend entries never 'miss' a bin due to cmap.N mismatch.
    """
    if edges is None:
        return []
    n = edges.size - 1
    if n <= 0:
        return []

    # Ensure discrete cmap has exactly n colors
    try:
        cmap_use = mpl.cm.get_cmap(cmap.name, n)
    except Exception:
        # If cmap is not a named colormap, fall back
        cmap_use = cmap

    def fmt(v):
        if not np.isfinite(v):
            return "nan"
        av = abs(v)
        if av > 0 and av <= LEGEND_SCI_THRESHOLD:
            # use more precision in sci for tiny edges
            return f"{v:.{decimals}e}".replace("e", "E")
        return f"{v:.{decimals}f}"

    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        out.append(Patch(facecolor=cmap_use(i), edgecolor="none", alpha=alpha, label=f"{fmt(lo)} – {fmt(hi)}"))
    return out


# ============================================================
# K HELPERS (UNCHANGED LOGIC)
# ============================================================

def _load_mf6_model_dis_npf(sim_ws):
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, load_only=["dis", "npf"])
    m = sim.get_model()
    return sim, m, m.dis, m.npf


def _get_k_arrays(npf, vk_stored_as_ratio=True):
    hk = np.array(npf.k.array, dtype=float)
    vraw = None
    for attr in ["k33", "vka", "kv", "kz"]:
        if hasattr(npf, attr):
            try:
                vraw = np.array(getattr(npf, attr).array, dtype=float)
                break
            except Exception:
                pass
    if vraw is None:
        raise ValueError("Could not read vertical array (npf.k33 / npf.vka / npf.kv / npf.kz not found/readable).")
    vk = hk * vraw if vk_stored_as_ratio else vraw
    return hk, vk


def _mask_inactive(arr3d, idomain):
    if idomain is None:
        return arr3d
    out = arr3d.copy()
    out[idomain <= 0] = np.nan
    return out


def _load_zone_array(sim_ws, fname=SW_ZONE_NPY):
    p = os.path.join(sim_ws, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Zone array not found: {p}")
    return np.load(p)


def _zone_for_layer(zone_arr, layer_index0, nlay):
    if zone_arr.ndim == 2:
        return zone_arr
    if zone_arr.ndim == 3:
        if zone_arr.shape[0] != nlay:
            raise ValueError(f"Zone array nlay={zone_arr.shape[0]} != model nlay={nlay}")
        return zone_arr[layer_index0, :, :]
    raise ValueError(f"Unexpected zone array dims: {zone_arr.shape}")


def _plot_panel_with_legend(
    ax, model, cat2d, edges, cmap, title, legend_title, legend_side,
    grid_outline_3857, model_extent, active_outline_3857=None
):
    """
    Panel plot with discrete bins + legend, basemap, grid outline, optional active outline.
    """
    ax.set_title(title, fontsize=13)

    if edges is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        add_neatline(ax)
        return []

    n_bins = edges.size - 1
    try:
        cmap_use = mpl.cm.get_cmap(cmap.name, n_bins)
    except Exception:
        cmap_use = cmap

    norm = mpl.colors.BoundaryNorm(
        boundaries=np.arange(-0.5, n_bins + 0.5, 1.0),
        ncolors=cmap_use.N if hasattr(cmap_use, "N") else n_bins,
    )

    gdf = _grid_to_polygons_gdf(model, cat2d, crs=MODEL_CRS, value_field="cat")
    if len(gdf) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        add_neatline(ax)
        return []

    _apply_extent(ax, model_extent)

    gdf.plot(
        column="cat",
        ax=ax,
        cmap=cmap_use,
        norm=norm,
        linewidth=0.0,
        edgecolor="none",
        alpha=OVERLAY_ALPHA,
        zorder=10,
    )

    # outlines
    add_grid_outline(ax, grid_outline_3857)
    add_active_area_outline(ax, active_outline_3857)

    add_north_arrow(ax)
    if ADD_SCALE_BAR:
        add_scale_bar(ax)
    add_neatline(ax)

    if legend_side == "left":
        bbox = (-0.02, 1.0); loc = "upper right"
    else:
        bbox = (1.02, 1.0); loc = "upper left"

    add_basemap(ax)

    leg = ax.legend(
        handles=_legend_patches(cmap_use, edges, alpha=OVERLAY_ALPHA, decimals=LEGEND_DECIMALS),
        title=legend_title,
        loc=loc,
        bbox_to_anchor=bbox,
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        framealpha=0.9,
    )
    return [leg]


def _plot_zone_split_panel(
    ax,
    model,
    vals2d,
    zone_mask_a,
    zone_mask_b,
    bins_a,
    bins_b,
    cmap_a,
    cmap_b,
    title,
    legend_title_a,
    legend_title_b,
    legend_side,
    grid_outline_3857,
    model_extent,
):
    ax.set_title(title, fontsize=13)

    a_vals = np.where(zone_mask_a, vals2d, np.nan)
    b_vals = np.where(zone_mask_b, vals2d, np.nan)

    # If bins not provided, derive from data (5 bins) per zone
    if bins_a is None:
        edges_a = _quantile_bins(a_vals, positive_only=True)
    else:
        edges_a = _normalize_bins(_snap_bins_to_data_minmax(bins_a, a_vals))

    if bins_b is None:
        edges_b = _quantile_bins(b_vals, positive_only=True)
    else:
        edges_b = _normalize_bins(_snap_bins_to_data_minmax(bins_b, b_vals))

    a_cat, edges_a = _categorize(a_vals, edges_a)
    b_cat, edges_b = _categorize(b_vals, edges_b)

    _apply_extent(ax, model_extent)
    add_basemap(ax)

    legends = []

    if edges_a is not None:
        nA = edges_a.size - 1
        try:
            cmap_a_use = mpl.cm.get_cmap(cmap_a.name, nA)
        except Exception:
            cmap_a_use = cmap_a
        norm_a = mpl.colors.BoundaryNorm(
            boundaries=np.arange(-0.5, nA + 0.5, 1.0),
            ncolors=nA,
        )
        gdf_a = _grid_to_polygons_gdf(model, a_cat, crs=MODEL_CRS, value_field="cat")
        if len(gdf_a):
            gdf_a.plot(
                column="cat", ax=ax, cmap=cmap_a_use, norm=norm_a,
                linewidth=0.0, edgecolor="none", alpha=OVERLAY_ALPHA, zorder=10
            )
    else:
        cmap_a_use = cmap_a

    if edges_b is not None:
        nB = edges_b.size - 1
        try:
            cmap_b_use = mpl.cm.get_cmap(cmap_b.name, nB)
        except Exception:
            cmap_b_use = cmap_b
        norm_b = mpl.colors.BoundaryNorm(
            boundaries=np.arange(-0.5, nB + 0.5, 1.0),
            ncolors=nB,
        )
        gdf_b = _grid_to_polygons_gdf(model, b_cat, crs=MODEL_CRS, value_field="cat")
        if len(gdf_b):
            gdf_b.plot(
                column="cat", ax=ax, cmap=cmap_b_use, norm=norm_b,
                linewidth=0.0, edgecolor="none", alpha=OVERLAY_ALPHA, zorder=11
            )
    else:
        cmap_b_use = cmap_b

    add_grid_outline(ax, grid_outline_3857)
    add_north_arrow(ax)
    if ADD_SCALE_BAR:
        add_scale_bar(ax)
    add_neatline(ax)

    if legend_side == "left":
        loc = "upper right"; bbox = (-0.02, 1.0)
    else:
        loc = "upper left"; bbox = (1.02, 1.0)

    if edges_a is not None:
        leg1 = ax.legend(
            handles=_legend_patches(cmap_a_use, edges_a, alpha=OVERLAY_ALPHA, decimals=LEGEND_DECIMALS),
            title=legend_title_a,
            loc=loc,
            bbox_to_anchor=bbox,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            framealpha=0.9,
        )
        ax.add_artist(leg1)
        legends.append(leg1)

    if edges_b is not None:
        bbox2 = (bbox[0], 0.55)
        leg2 = ax.legend(
            handles=_legend_patches(cmap_b_use, edges_b, alpha=OVERLAY_ALPHA, decimals=LEGEND_DECIMALS),
            title=legend_title_b,
            loc=loc,
            bbox_to_anchor=bbox2,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            framealpha=0.9,
        )
        legends.append(leg2)

    return legends

def plot_avg_annual_recharge_tile(
    sim_ws,
    outdir,
    model_name_for_title,
    sim_name=None,
    xoff_override=None,
    yoff_override=None,
    angrot_override=None,
    dpi=350,
    year0=2000,
    year1=2023,
    bins=None,
    units="in/yr",        # "in/yr" (default) or "in/day"
    mask_zero=True,       # True => show zeros as blank (no color)
    eps=0.0,              # optional: treat <= eps as zero (e.g. 1e-12)
):
    """
    Plot average annual recharge map for year0-year1.

    Uses compute_avg_annual_recharge_2000_2023() which returns:
      avg_depth_ft_per_yr (ft/yr) and avg_rate_ft_per_day (ft/day).

    This function converts to:
      - in/yr: avg_depth_ft_per_yr * 12
      - in/day: avg_rate_ft_per_day * 12
    """
    os.makedirs(outdir, exist_ok=True)

    avg_depth_ftyr, avg_rate_ftd = compute_avg_annual_recharge_2000_2023(
        sim_ws, sim_name=sim_name, year0=year0, year1=year1
    )

    # Convert to requested units
    units_l = str(units).strip().lower()
    if units_l in ("in/yr", "in_per_year", "in/year", "inyr"):
        arr = avg_depth_ftyr * 12.0
        units_label = "in/yr"
    elif units_l in ("in/day", "in_per_day", "in/day", "inday"):
        arr = avg_rate_ftd * 12.0
        units_label = "in/day"
    else:
        raise ValueError(f"Unknown units='{units}'. Use 'in/yr' or 'in/day'.")

    # Mask 0 recharge as blank
    if mask_zero:
        arr = np.where(arr > float(eps), arr, np.nan)

    # Load model for grid geometry + idomain
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    m = sim.get_model(sim_name) if sim_name else sim.get_model()
    dis = m.dis

    _apply_coord_overrides(m, xoff_override, yoff_override, angrot_override, crs=MODEL_CRS)

    idomain = _get_idomain(dis)
    if idomain is not None:
        id2d = idomain[0] if np.asarray(idomain).ndim == 3 else idomain
        arr = np.where(id2d > 0, arr, np.nan)

    # Bins: use user bins if given; otherwise quantiles
    edges = _resolve_bins_user(bins, arr, layer_num=1)
    if edges is None:
        edges = _quantile_bins(arr, positive_only=True)

    cat2d, edges = _categorize(arr, edges)
    cmap = mpl.cm.get_cmap("cividis", (edges.size - 1) if edges is not None else 5)

    grid_outline_3857 = _grid_outline_gdf(m)
    model_extent = _get_extent_from_gdf(grid_outline_3857, pad_frac=0.02)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.5))

    st = fig.suptitle(
        f"{model_name_for_title}\nMean Annual Recharge ({year0}–{year1})",
        fontsize=16,
        y=0.99,
        linespacing=1.25,
    )

    # reserve a bit more space for the title
    fig.subplots_adjust(top=0.82)

    extra_artists = _plot_panel_with_legend(
        ax=ax,
        model=m,
        cat2d=cat2d,
        edges=edges,
        cmap=cmap,
        title=f"Recharge ({units_label})",
        legend_title="RCH",
        legend_side="right",
        grid_outline_3857=grid_outline_3857,
        model_extent=model_extent,
    )

    # IMPORTANT: include suptitle in bbox_extra_artists for reliable tight bounding
    extra = []
    if extra_artists:
        extra.extend(extra_artists)
    extra.append(st)

    safe = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")
    out_png = os.path.join(outdir, f"{safe}_avg_recharge_{year0}_{year1}_{units_label.replace('/','')}.png")

    fig.savefig(
        out_png,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.55,
        bbox_extra_artists=extra,
    )
    plt.close(fig)
    print(f"Wrote: {out_png}")

def write_zones_by_layer_dat(sim_ws, model_name=None, out_path="zones_byly.dat"):
    """
    Create MF6 zone file with:
      - Layer 1 => zone 1
      - Layer 2 => zone 2
      - Layer 3 => zone 3
    Writes one zone integer per cell in MF6 node order (layer-major).
    Works for DIS and DISV (common in these projects).
    """

    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")

    # get model
    if model_name is None:
        m = sim.get_model()
    else:
        m = sim.get_model(model_name)

    # detect discretization type and get cell counts
    if hasattr(m, "dis") and m.dis is not None:
        dis = m.dis
        nlay = int(dis.nlay.get_data())
        nrow = int(dis.nrow.get_data())
        ncol = int(dis.ncol.get_data())
        ncpl = nrow * ncol
        ncells = nlay * ncpl

        # layer-major order: [layer1(all rows/cols), layer2(...), ...]
        zones = np.repeat(np.arange(1, nlay + 1, dtype=int), ncpl)

    elif hasattr(m, "disv") and m.disv is not None:
        disv = m.disv
        nlay = int(disv.nlay.get_data())
        ncpl = int(disv.ncpl.get_data())
        ncells = nlay * ncpl

        zones = np.repeat(np.arange(1, nlay + 1, dtype=int), ncpl)

    else:
        raise ValueError(
            "Could not find DIS or DISV on the model. "
            "If this is DISU, we need a slightly different approach."
        )

    if zones.size != ncells:
        raise ValueError(f"zones length ({zones.size}) != NCELLS ({ncells})")

    out_path = os.path.join(sim_ws, out_path) if not os.path.isabs(out_path) else out_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        f.write("BEGIN DIMENSIONS\n")
        f.write(f"  NCELLS {ncells}\n")
        f.write("END DIMENSIONS\n\n")
        f.write("BEGIN GRIDDATA\n")
        f.write("  IZONE\n")
        f.write("  INTERNAL\n")
        for z in zones:
            f.write(f"{int(z)}\n")
        f.write("END GRIDDATA\n")

    print(f"Wrote: {out_path}")
    print(f"nlay={nlay}, ncpl={ncpl}, ncells={ncells}")


def plot_model_k_figs_per_layer(
    sim_ws,
    outdir,
    model_name_for_title=None,
    xoff_override=None,
    yoff_override=None,
    angrot_override=None,
    model_crs=MODEL_CRS,
    drop_idomain_le_zero=True,
    layers_to_plot=None,
    nbins_hk=5,
    nbins_vk=5,
    bins_hk=None,
    bins_vk=None,
    units_label="(model units)",
    dpi=DPI_K,
    vk_stored_as_ratio=True,
):
    os.makedirs(outdir, exist_ok=True)

    sim, m, dis, npf = _load_mf6_model_dis_npf(sim_ws)
    _apply_coord_overrides(m, xoff_override, yoff_override, angrot_override, crs=model_crs)

    grid_outline_3857 = _grid_outline_gdf(m)
    model_extent = _get_extent_from_gdf(grid_outline_3857, pad_frac=0.02)

    idomain = _get_idomain(dis)
    hk3d, vk3d = _get_k_arrays(npf, vk_stored_as_ratio=vk_stored_as_ratio)

    hk3d = np.where(hk3d > 0, hk3d, np.nan)
    vk3d = np.where(vk3d > 0, vk3d, np.nan)

    if drop_idomain_le_zero and idomain is not None:
        hk3d = _mask_inactive(hk3d, idomain)
        vk3d = _mask_inactive(vk3d, idomain)

    nlay = hk3d.shape[0]
    layer_inds = list(range(nlay)) if layers_to_plot is None else [k - 1 for k in layers_to_plot]

    if model_name_for_title is None:
        model_name_for_title = os.path.basename(os.path.normpath(sim_ws))
    safe_name = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")

    cmap_hk = mpl.cm.get_cmap("viridis", nbins_hk)
    cmap_vk = mpl.cm.get_cmap("magma", nbins_vk)

    is_spiritwood = "Spiritwood" in (model_name_for_title or "")
    zone_arr = None
    if is_spiritwood:
        try:
            zone_arr = _load_zone_array(sim_ws, SW_ZONE_NPY)
        except Exception as e:
            print(f"[WARN] Spiritwood zone array not loaded; falling back to single-scale bins. Reason: {e}")
            zone_arr = None

    for k0 in layer_inds:
        layer_num = k0 + 1
        hk_layer = hk3d[k0, :, :]
        vk_layer = vk3d[k0, :, :]

        fig, (ax_hk, ax_vk) = plt.subplots(1, 2, figsize=(14.5, 6.5))
        fig.suptitle(f"{model_name_for_title} — Layer {layer_num} conductivity", fontsize=16, y=0.98)
        fig.subplots_adjust(top=0.90)

        extra_artists = []

        if is_spiritwood and (zone_arr is not None) and (layer_num in [SW_ZONE_LAYER2, SW_ZONE_LAYER3]):
            z = _zone_for_layer(zone_arr, layer_index0=k0, nlay=nlay)
            valid = np.isfinite(hk_layer)

            if layer_num == SW_ZONE_LAYER2:
                windows = np.isin(z, list(SW_ZONE_L2_WINDOWS)) & valid
                confining = (~windows) & valid

                extra_artists += _plot_zone_split_panel(
                    ax=ax_hk, model=m, vals2d=hk_layer,
                    zone_mask_a=confining, zone_mask_b=windows,
                    bins_a=SW_L2_HK_CONFINING_BINS, bins_b=SW_L2_HK_WINDOWS_BINS,
                    cmap_a=mpl.cm.get_cmap("viridis", nbins_hk),
                    cmap_b=mpl.cm.get_cmap("vanimo", nbins_hk),
                    title=f"HK {units_label}",
                    legend_title_a="HK (confining)",
                    legend_title_b="HK (windows)",
                    legend_side="left",
                    grid_outline_3857=grid_outline_3857,
                    model_extent=model_extent,
                )

                extra_artists += _plot_zone_split_panel(
                    ax=ax_vk, model=m, vals2d=vk_layer,
                    zone_mask_a=confining, zone_mask_b=windows,
                    bins_a=SW_L2_VK_CONFINING_BINS, bins_b=SW_L2_VK_WINDOWS_BINS,
                    cmap_a=mpl.cm.get_cmap("magma", nbins_vk),
                    cmap_b=mpl.cm.get_cmap("berlin", nbins_vk),
                    title=f"VK {units_label}",
                    legend_title_a="VK (confining)",
                    legend_title_b="VK (windows)",
                    legend_side="right",
                    grid_outline_3857=grid_outline_3857,
                    model_extent=model_extent,
                )

                fig.subplots_adjust(left=0.10, right=0.78, wspace=0.02)

            elif layer_num == SW_ZONE_LAYER3:
                barrier = (z == SW_ZONE_L3_BARRIER) & valid
                spiritwood = (~barrier) & valid

                extra_artists += _plot_zone_split_panel(
                    ax=ax_hk, model=m, vals2d=hk_layer,
                    zone_mask_a=spiritwood, zone_mask_b=barrier,
                    bins_a=SW_L3_HK_SPIRITWOOD_BINS, bins_b=SW_L3_HK_BARRIER_BINS,
                    cmap_a=mpl.cm.get_cmap("viridis", nbins_hk),
                    cmap_b=mpl.cm.get_cmap("vanimo", nbins_hk),
                    title=f"HK {units_label}",
                    legend_title_a="HK (Spiritwood)",
                    legend_title_b="HK (barrier)",
                    legend_side="left",
                    grid_outline_3857=grid_outline_3857,
                    model_extent=model_extent,
                )

                extra_artists += _plot_zone_split_panel(
                    ax=ax_vk, model=m, vals2d=vk_layer,
                    zone_mask_a=spiritwood, zone_mask_b=barrier,
                    bins_a=SW_L3_VK_SPIRITWOOD_BINS, bins_b=SW_L3_VK_BARRIER_BINS,
                    cmap_a=mpl.cm.get_cmap("magma", nbins_vk),
                    cmap_b=mpl.cm.get_cmap("berlin", nbins_vk),
                    title=f"VK {units_label}",
                    legend_title_a="VK (Spiritwood)",
                    legend_title_b="VK (barrier)",
                    legend_side="right",
                    grid_outline_3857=grid_outline_3857,
                    model_extent=model_extent,
                )

                fig.subplots_adjust(left=0.10, right=0.78, wspace=0.02)

        else:
            edges_hk = _resolve_bins_user(bins_hk, hk_layer, layer_num)
            edges_vk = _resolve_bins_user(bins_vk, vk_layer, layer_num)

            hk_cat, edges_hk = _categorize(hk_layer, edges_hk)
            vk_cat, edges_vk = _categorize(vk_layer, edges_vk)

            extra_artists += _plot_panel_with_legend(
                ax=ax_hk, model=m, cat2d=hk_cat, edges=edges_hk, cmap=cmap_hk,
                title=f"HK {units_label}", legend_title="HK", legend_side="left",
                grid_outline_3857=grid_outline_3857, model_extent=model_extent,
            )
            extra_artists += _plot_panel_with_legend(
                ax=ax_vk, model=m, cat2d=vk_cat, edges=edges_vk, cmap=cmap_vk,
                title=f"VK {units_label}", legend_title="VK", legend_side="right",
                grid_outline_3857=grid_outline_3857, model_extent=model_extent,
            )

            fig.subplots_adjust(left=0.12, right=0.88, wspace=0.02)

        out_png = os.path.join(outdir, f"{safe_name}_layer{layer_num:02d}_hk_vk.png")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.55, bbox_extra_artists=extra_artists)
        plt.close(fig)
        print(f"Wrote: {out_png}")


# ============================================================
# THICKNESS PLOTTING
# ============================================================

def _load_mf6_model_dis(sim_ws):
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, load_only=["dis"])
    m = sim.get_model()
    return sim, m, m.dis


def _get_thickness_3d(dis):
    top2d = np.array(dis.top.array, dtype=float)
    botm3d = np.array(dis.botm.array, dtype=float)  # (nlay, nrow, ncol)
    nlay = botm3d.shape[0]
    thk = np.full_like(botm3d, np.nan, dtype=float)
    for k in range(nlay):
        upper = top2d if k == 0 else botm3d[k - 1, :, :]
        lower = botm3d[k, :, :]
        t = upper - lower
        thk[k, :, :] = np.where(t > 0, t, np.nan)
    return thk


def plot_model_thickness_figs_per_layer(
    sim_ws,
    outdir,
    model_name_for_title=None,
    xoff_override=None,
    yoff_override=None,
    angrot_override=None,
    model_crs=MODEL_CRS,
    layers_to_plot=None,
    dpi=DPI_THK,
):
    os.makedirs(outdir, exist_ok=True)
    sim, m, dis = _load_mf6_model_dis(sim_ws)
    _apply_coord_overrides(m, xoff_override, yoff_override, angrot_override, crs=model_crs)

    grid_outline_3857 = _grid_outline_gdf(m)
    model_extent = _get_extent_from_gdf(grid_outline_3857, pad_frac=0.02)

    idomain = _get_idomain(dis)
    thk3d = _get_thickness_3d(dis)
    if idomain is not None:
        thk3d = _mask_inactive(thk3d, idomain)

    nlay = thk3d.shape[0]
    layer_inds = list(range(nlay)) if layers_to_plot is None else [k - 1 for k in layers_to_plot]

    if model_name_for_title is None:
        model_name_for_title = os.path.basename(os.path.normpath(sim_ws))
    safe_name = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")

    for k0 in layer_inds:
        layer_num = k0 + 1
        vals2d = thk3d[k0, :, :]

        # bins: quantiles into 5 bins
        v = vals2d[np.isfinite(vals2d)]
        if v.size == 0:
            edges = None
        else:
            qs = np.quantile(v, [0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
            edges = _normalize_bins(qs)

        cat2d, edges = _categorize(vals2d, edges)
        cmap = mpl.cm.get_cmap("cividis", (edges.size - 1) if edges is not None else 5)

        fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.5))
        fig.suptitle(f"{model_name_for_title} — Layer {layer_num} thickness", fontsize=16, y=0.98)
        fig.subplots_adjust(top=0.90)

        extra_artists = _plot_panel_with_legend(
            ax=ax, model=m, cat2d=cat2d, edges=edges, cmap=cmap,
            title="Thickness (feet)", legend_title="Thickness", legend_side="right",
            grid_outline_3857=grid_outline_3857, model_extent=model_extent
        )

        out_png = os.path.join(outdir, f"{safe_name}_layer{layer_num:02d}_thickness.png")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.55, bbox_extra_artists=extra_artists)
        plt.close(fig)
        print(f"Wrote: {out_png}")


# ============================================================
# STORAGE PLOTTING (SS + SY side-by-side)
# ============================================================

def _load_mf6_model_dis_sto(sim_ws):
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, load_only=["dis", "sto"])
    m = sim.get_model()
    sto = getattr(m, "sto", None)
    return sim, m, m.dis, sto


def _as_3d(arr, nlay, nrow, ncol):
    a = np.array(arr, dtype=float)
    if a.ndim == 3:
        return a
    if a.ndim == 2:
        return np.repeat(a[np.newaxis, :, :], nlay, axis=0)
    if a.ndim == 1 and a.size == nlay:
        out = np.zeros((nlay, nrow, ncol), dtype=float)
        for k in range(nlay):
            out[k, :, :] = a[k]
        return out
    raise ValueError(f"Cannot coerce array to 3D: shape={a.shape}")


def _get_storage_arrays(sto, dis):
    if sto is None:
        return None, None
    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()

    ss = None
    sy = None

    if hasattr(sto, "ss"):
        try:
            ss = _as_3d(sto.ss.array, nlay, nrow, ncol)
        except Exception:
            ss = None

    if hasattr(sto, "sy"):
        try:
            sy = _as_3d(sto.sy.array, nlay, nrow, ncol)
        except Exception:
            sy = None

    return ss, sy

def plot_budget_piecharts_for_year(
    sim_ws,
    outdir,
    model_name_for_title,
    items_dict,
    pie_year=2023,
    out_fmt=("png", "pdf"),
    dpi=350,
    budget_df=None,
    out_tag=None,
):
    os.makedirs(outdir, exist_ok=True)

    if budget_df is None:
        budget_csv = os.path.join(sim_ws, "budget.csv")
        years, annual_pos, annual_neg, net_storage = build_annual_budget_series(
            sim_ws=sim_ws,
            budget_csv_path=budget_csv,
            items_dict=items_dict,
            year0=pie_year,
            year1=pie_year,
        )
    else:
        years, annual_pos, annual_neg, net_storage = build_annual_budget_series(
            sim_ws=sim_ws,
            items_dict=items_dict,
            year0=pie_year,
            year1=pie_year,
            budget_df=budget_df,
        )

    if len(years) != 1 or int(years[0]) != int(pie_year):
        raise ValueError(f"Unexpected years returned: {years}")
    iy = 0

    def _make_pie(kind):
        if kind == "in":
            data_dict = annual_pos
            title = f"{model_name_for_title}\nInflows in {pie_year} (acre-ft)"
        else:
            data_dict = annual_neg
            title = f"{model_name_for_title}\nOutflows in {pie_year} (acre-ft)"

        if out_tag:
            title += f"\n({out_tag})"

        labels, values = [], []
        for lab, arr in data_dict.items():
            v = abs(float(arr[iy]))
            if np.isfinite(v) and v > 0:
                labels.append(lab)
                values.append(v)

        if not values:
            print(f"[INFO] No nonzero {kind} values for {model_name_for_title} in {pie_year} ({out_tag}).")
            return

        values = np.array(values, dtype=float)
        cmap = mpl.cm.get_cmap("tab20", len(values))
        colors = [cmap(i) for i in range(len(values))]

        fig, ax = plt.subplots(1, 1, figsize=(10.5, 8.5))
        wedges, texts, autotexts = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct="%1.1f%%",
            pctdistance=0.75,
            wedgeprops=dict(edgecolor="black", linewidth=0.8),
        )

        for t in autotexts:
            t.set_color("white")
            t.set_fontsize(10)

        labels2 = [f"{lab}\n{val:,.0f}" for lab, val in zip(labels, values)]
        ax.legend(
            wedges,
            labels2,
            title="Component (acre-ft)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=0.95,
            fontsize=9,
            title_fontsize=10,
        )

        ax.set_title(title, fontsize=16, loc="left")
        ax.set_aspect("equal")

        safe = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")
        tag = f"_{out_tag}" if out_tag else ""
        base = os.path.join(outdir, f"{safe}_budget_pie_{kind}_{pie_year}{tag}")

        for fmt in out_fmt:
            out_path = f"{base}.{fmt}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Wrote: {out_path}")
        plt.close(fig)

    _make_pie("in")
    _make_pie("out")




def plot_spiritwood_riv_stage_change_external_fast(
    sim_ws,
    sim_name="swww",
    kper_ref=0,
    out_png=None,
    riv_pkg_name=None,        # optional: match package_name like "riv_rivers"
    figsize=(10, 4),
    dpi=300,
):
    """
    FAST RIV stage-change plot for MF6 RIV packages where PERIOD data is in external files.

    Strategy:
      - Parse the RIV package file to find PERIOD blocks.
      - For each PERIOD, if it uses OPEN/CLOSE (or similar), read the external file and extract
        the first numeric stage.
      - If stage is constant/no PERIOD blocks, tries to extract one numeric stage from any data-like line.
      - Carries forward stage for periods with no explicit update.

    Assumes stage change is uniform across cells (your case).
    """

    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import flopy

    # -------------------------
    # load sim/model + find riv package file
    # -------------------------
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    m = sim.get_model(sim_name)
    nper = int(sim.tdis.nper.get_data())

    riv_pkgs = _list_mf6_packages_by_type(m, "riv")
    if not riv_pkgs:
        raise ValueError("No RIV packages found in the model.")

    if riv_pkg_name is not None:
        cand = [p for p in riv_pkgs if str(p.package_name).lower() == str(riv_pkg_name).lower()]
        if not cand:
            raise ValueError(f"Requested riv_pkg_name='{riv_pkg_name}' not found. "
                             f"Available: {[p.package_name for p in riv_pkgs]}")
        pkg = cand[0]
    else:
        pkg = riv_pkgs[0]

    pkg_fn = None
    for attr in ("filename", "fn_path", "_filename", "package_filename"):
        if hasattr(pkg, attr):
            val = getattr(pkg, attr)
            if isinstance(val, str) and val.strip():
                pkg_fn = val
                break
    if pkg_fn is None:
        raise ValueError("Could not resolve RIV package filename from flopy package object.")

    riv_path = pkg_fn if os.path.isabs(pkg_fn) else os.path.join(sim_ws, pkg_fn)
    if not os.path.exists(riv_path):
        raise FileNotFoundError(f"RIV package file not found: {riv_path}")

    riv_dir = os.path.dirname(riv_path)

    # -------------------------
    # helpers to extract one stage from a text line / external file
    # -------------------------
    def _extract_stage_from_data_line(line):
        """
        Heuristic extraction of stage from an MF6 RIV list record line.
        Works for:
          - "(k i j) stage cond rbot"
          - "k i j stage cond rbot"
          - "node stage cond rbot"
        Ignores comments and non-numeric tokens.
        """
        line = line.split("#", 1)[0].strip()
        if not line:
            return None

        low = line.lower()
        if low.startswith(("begin", "end", "period", "options", "dimensions", "packagedata")):
            return None
        if low.startswith(("open/close", "open", "filein", "external")):
            return None

        # normalize parentheses/commas
        cleaned = line.replace("(", " ").replace(")", " ").replace(",", " ")
        toks = cleaned.split()

        nums = []
        for t in toks:
            try:
                nums.append(float(t))
            except Exception:
                continue

        # need at least [cellid + stage]
        if len(nums) < 2:
            return None

        # structured cellid => stage is 4th numeric; node => stage is 2nd numeric
        if len(nums) >= 4:
            return float(nums[3])
        return float(nums[1])

    _stage_cache = {}

    def _stage_from_external_file(path):
        """
        Read just enough of an external period file to get the first numeric stage.
        Cache by absolute path.
        """
        ap = os.path.abspath(path)
        if ap in _stage_cache:
            return _stage_cache[ap]

        if not os.path.exists(ap):
            _stage_cache[ap] = None
            return None

        st = None
        with open(ap, "r") as f:
            for raw in f:
                st = _extract_stage_from_data_line(raw)
                if st is not None and np.isfinite(st):
                    break

        _stage_cache[ap] = st
        return st

    # -------------------------
    # parse RIV package file for PERIOD blocks and OPEN/CLOSE pointers
    # -------------------------
    re_begin_period = re.compile(r"^\s*begin\s+period\s+(\d+)", re.IGNORECASE)
    re_period       = re.compile(r"^\s*period\s+(\d+)", re.IGNORECASE)
    re_openclose    = re.compile(r"^\s*open/close\s+(.+)$", re.IGNORECASE)

    stage_by_kper = {}
    cur_kper = None
    saw_any_period = False

    # also allow a constant stage (if no periods)
    constant_stage = None

    with open(riv_path, "r") as f:
        for raw in f:
            s = raw.strip()
            low = s.lower()

            m0 = re_begin_period.match(s)
            if m0:
                saw_any_period = True
                cur_kper = int(m0.group(1)) - 1
                continue

            m1 = re_period.match(s)
            # avoid catching words like PERIODDATA; keep it simple
            if m1 and not low.startswith("perioddata"):
                saw_any_period = True
                cur_kper = int(m1.group(1)) - 1
                continue

            if low.startswith("end period"):
                cur_kper = None
                continue

            # If inside a period: look for OPEN/CLOSE first, else try inline data
            if cur_kper is not None:
                mo = re_openclose.match(s)
                if mo:
                    # filename may be quoted; may include extra tokens (like FACTOR)
                    rhs = mo.group(1).strip().strip('"').strip("'")
                    # take first token as filename
                    fname = rhs.split()[0].strip('"').strip("'")
                    ext_path = fname if os.path.isabs(fname) else os.path.join(riv_dir, fname)
                    st = _stage_from_external_file(ext_path)
                    if st is not None and np.isfinite(st):
                        stage_by_kper[cur_kper] = float(st)
                    continue

                # inline record (rare if truly externalized, but safe)
                if cur_kper not in stage_by_kper:
                    st = _extract_stage_from_data_line(s)
                    if st is not None and np.isfinite(st):
                        stage_by_kper[cur_kper] = float(st)
                continue

            # No period blocks: try to discover a constant stage from any record-like line
            if not saw_any_period and constant_stage is None:
                st = _extract_stage_from_data_line(s)
                if st is not None and np.isfinite(st):
                    constant_stage = float(st)

    # -------------------------
    # build stage series (carry forward)
    # -------------------------
    stage = np.full(nper, np.nan, dtype=float)

    if stage_by_kper:
        last = np.nan
        for kper in range(nper):
            if kper in stage_by_kper and np.isfinite(stage_by_kper[kper]):
                last = stage_by_kper[kper]
            stage[kper] = last

    elif constant_stage is not None and np.isfinite(constant_stage):
        stage[:] = constant_stage

    else:
        raise ValueError(
            f"Could not find numeric stage in RIV package or external period files.\n"
            f"RIV file: {riv_path}\n"
            "If your RIV stage is driven by TS6/timeseries names (non-numeric), "
            "we’ll need to parse the TS file instead."
        )

    if not np.isfinite(stage[kper_ref]):
        raise ValueError(f"No stage found for kper_ref={kper_ref}. Parsed file: {riv_path}")

    dstage = stage - stage[kper_ref]
    print(dstage)
    print(stage[:20])
    print("min/max stage:", np.nanmin(stage), np.nanmax(stage))
    print("unique (first 50):", np.unique(stage[np.isfinite(stage)])[:50])

    # -------------------------
    # plot
    # -------------------------
    years = _stress_period_midpoint_years(sim)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(years, dstage, lw=2.0, color="black")
    ax.axhline(0.0, color="0.5", lw=1.0, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Δ RIV Stage (ft)")
    ax.set_title(f"Spiritwood RIV Stage Change (relative to kper {kper_ref})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out_png}")
    else:
        plt.show()




def plot_model_storage_figs_per_layer(
    sim_ws,
    outdir,
    model_name_for_title=None,
    xoff_override=None,
    yoff_override=None,
    angrot_override=None,
    model_crs=MODEL_CRS,
    drop_idomain_le_zero=True,
    layers_to_plot=None,
    nbins_ss=5,
    nbins_sy=5,
    bins_ss=None,          # dict or list; optional (non-zoned)
    bins_sy=None,          # dict or list; optional (non-zoned)
    dpi=DPI_STO,
):
    os.makedirs(outdir, exist_ok=True)

    sim, m, dis, sto = _load_mf6_model_dis_sto(sim_ws)
    _apply_coord_overrides(m, xoff_override, yoff_override, angrot_override, crs=model_crs)

    grid_outline_3857 = _grid_outline_gdf(m)
    model_extent = _get_extent_from_gdf(grid_outline_3857, pad_frac=0.02)

    idomain = _get_idomain(dis)
    ss3d, sy3d = _get_storage_arrays(sto, dis)

    if ss3d is None:
        raise ValueError(f"No SS found in STO for: {sim_ws}")

    # Clean up
    ss3d = np.where(ss3d > 0, ss3d, np.nan)
    if sy3d is not None:
        sy3d = np.where(sy3d >= 0, sy3d, np.nan)

    if drop_idomain_le_zero and idomain is not None:
        ss3d = _mask_inactive(ss3d, idomain)
        if sy3d is not None:
            sy3d = _mask_inactive(sy3d, idomain)

    nlay = ss3d.shape[0]
    layer_inds = list(range(nlay)) if layers_to_plot is None else [k - 1 for k in layers_to_plot]

    if model_name_for_title is None:
        model_name_for_title = os.path.basename(os.path.normpath(sim_ws))
    safe_name = model_name_for_title.lower().replace("–", "-").replace("—", "-").replace(" ", "_")

    is_spiritwood = "Spiritwood" in (model_name_for_title or "")
    is_wahp = "Wahpeton" in (model_name_for_title or "")

    # Load zone array if Spiritwood
    zone_arr = None
    if is_spiritwood:
        try:
            zone_arr = _load_zone_array(sim_ws, SW_ZONE_NPY)
        except Exception as e:
            print(f"[WARN] Spiritwood zone array not loaded for storage; falling back to single-scale SS. Reason: {e}")
            zone_arr = None

    # Colormaps "same as K" style
    cmap_a_ss = mpl.cm.get_cmap("viridis", nbins_ss)
    cmap_b_ss = mpl.cm.get_cmap("vanimo", nbins_ss)
    cmap_a_sy = mpl.cm.get_cmap("magma", nbins_sy)
    cmap_b_sy = mpl.cm.get_cmap("berlin", nbins_sy)

    for k0 in layer_inds:
        layer_num = k0 + 1
        ss_layer = ss3d[k0, :, :]

        # Determine whether SY exists for this layer
        sy_layer = None
        if sy3d is not None:
            sy_layer = sy3d[k0, :, :]

        # Your rules for missing SY
        if is_spiritwood and layer_num in [2, 3]:
            sy_layer = None
        if is_wahp and layer_num in [3, 4, 5, 6]:
            sy_layer = None

        has_sy = (sy_layer is not None) and np.isfinite(sy_layer).any()

        # ---- Create figure: 1 panel if no SY, else 2 panels
        if has_sy:
            fig, (ax_ss, ax_sy) = plt.subplots(1, 2, figsize=(14.5, 6.5))
        else:
            fig, ax_ss = plt.subplots(1, 1, figsize=(8.0, 6.5))
            ax_sy = None

        fig.suptitle(f"{model_name_for_title} — Layer {layer_num} storage", fontsize=16, y=0.98)
        fig.subplots_adjust(top=0.90)

        extra_artists = []

        # ---- Spiritwood zoned storage for layers 2 & 3 (same zones as K)
        do_zoned = is_spiritwood and (zone_arr is not None) and (layer_num in [SW_ZONE_LAYER2, SW_ZONE_LAYER3])

        if do_zoned:
            z = _zone_for_layer(zone_arr, layer_index0=k0, nlay=nlay)
            valid = np.isfinite(ss_layer)

            if layer_num == SW_ZONE_LAYER2:
                windows = np.isin(z, list(SW_ZONE_L2_WINDOWS)) & valid
                confining = (~windows) & valid

                # SS zoned (bins derived per zone unless you pass explicit bins)
                extra_artists += _plot_zone_split_panel(
                    ax=ax_ss, model=m, vals2d=ss_layer,
                    zone_mask_a=confining, zone_mask_b=windows,
                    bins_a=None, bins_b=None,   # auto quantile bins per zone
                    cmap_a=cmap_a_ss, cmap_b=cmap_b_ss,
                    title=f"SS {SS_UNITS_LABEL}",
                    legend_title_a="SS (confining)",
                    legend_title_b="SS (windows)",
                    legend_side="right" if not has_sy else "left",
                    grid_outline_3857=grid_outline_3857,
                    model_extent=model_extent,
                )

                if has_sy:
                    extra_artists += _plot_zone_split_panel(
                        ax=ax_sy, model=m, vals2d=sy_layer,
                        zone_mask_a=confining, zone_mask_b=windows,
                        bins_a=None, bins_b=None,
                        cmap_a=cmap_a_sy, cmap_b=cmap_b_sy,
                        title=f"SY {SY_UNITS_LABEL}",
                        legend_title_a="SY (confining)",
                        legend_title_b="SY (windows)",
                        legend_side="right",
                        grid_outline_3857=grid_outline_3857,
                        model_extent=model_extent,
                    )

            elif layer_num == SW_ZONE_LAYER3:
                barrier = (z == SW_ZONE_L3_BARRIER) & valid
                spiritwood = (~barrier) & valid

                extra_artists += _plot_zone_split_panel(
                    ax=ax_ss, model=m, vals2d=ss_layer,
                    zone_mask_a=spiritwood, zone_mask_b=barrier,
                    bins_a=None, bins_b=None,
                    cmap_a=cmap_a_ss, cmap_b=cmap_b_ss,
                    title=f"SS {SS_UNITS_LABEL}",
                    legend_title_a="SS (Spiritwood)",
                    legend_title_b="SS (barrier)",
                    legend_side="right" if not has_sy else "left",
                    grid_outline_3857=grid_outline_3857,
                    model_extent=model_extent,
                )

                if has_sy:
                    extra_artists += _plot_zone_split_panel(
                        ax=ax_sy, model=m, vals2d=sy_layer,
                        zone_mask_a=spiritwood, zone_mask_b=barrier,
                        bins_a=None, bins_b=None,
                        cmap_a=cmap_a_sy, cmap_b=cmap_b_sy,
                        title=f"SY {SY_UNITS_LABEL}",
                        legend_title_a="SY (Spiritwood)",
                        legend_title_b="SY (barrier)",
                        legend_side="right",
                        grid_outline_3857=grid_outline_3857,
                        model_extent=model_extent,
                    )

            if has_sy:
                fig.subplots_adjust(left=0.10, right=0.78, wspace=0.02)
            else:
                fig.subplots_adjust(left=0.10, right=0.90)

        else:
            # ---- Non-zoned storage: use user bins if supplied, else quantile bins
            edges_ss = _resolve_bins_user(bins_ss, ss_layer, layer_num)
            if edges_ss is None:
                edges_ss = _quantile_bins(ss_layer, positive_only=True)

            ss_cat, edges_ss = _categorize(ss_layer, edges_ss)
            cmap_ss = mpl.cm.get_cmap("viridis", (edges_ss.size - 1) if edges_ss is not None else nbins_ss)

            extra_artists += _plot_panel_with_legend(
                ax=ax_ss, model=m, cat2d=ss_cat, edges=edges_ss, cmap=cmap_ss,
                title=f"SS {SS_UNITS_LABEL}", legend_title="SS",
                legend_side="right" if not has_sy else "left",
                grid_outline_3857=grid_outline_3857, model_extent=model_extent,
            )

            if has_sy:
                edges_sy = _resolve_bins_user(bins_sy, sy_layer, layer_num)
                if edges_sy is None:
                    edges_sy = _quantile_bins(sy_layer, positive_only=False)
                sy_cat, edges_sy = _categorize(sy_layer, edges_sy)
                cmap_sy = mpl.cm.get_cmap("magma", (edges_sy.size - 1) if edges_sy is not None else nbins_sy)

                extra_artists += _plot_panel_with_legend(
                    ax=ax_sy, model=m, cat2d=sy_cat, edges=edges_sy, cmap=cmap_sy,
                    title=f"SY {SY_UNITS_LABEL}", legend_title="SY",
                    legend_side="right",
                    grid_outline_3857=grid_outline_3857, model_extent=model_extent,
                )

            if has_sy:
                fig.subplots_adjust(left=0.12, right=0.88, wspace=0.02)
            else:
                fig.subplots_adjust(left=0.12, right=0.88)

        out_png = os.path.join(outdir, f"{safe_name}_layer{layer_num:02d}_ss" + ("_sy" if has_sy else "") + ".png")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.55, bbox_extra_artists=extra_artists)
        plt.close(fig)
        print(f"Wrote: {out_png}")

def plot_head_contour_tiles_two_kpers(
    sim_ws,
    out_png,
    model_name_for_title,
    sim_name=None,
    kstpkper0=(0, 0),
    kstpkper1=(0, 0),
    mode="layer",          # "layer" or "water_table"
    layer_num=1,           # used if mode="layer"
    xoff_override=None,
    yoff_override=None,
    angrot_override=None,
    contour_interval=20.0,
    cmap_name="viridis",
    dpi=350,
    show_active_outline=True,
    active_outline_style=None,
    panel_labels=None,     # (label0, label1) optional
):
    """
    Two-panel tiled head + contours.

    panel_labels:
      - None -> defaults to "Stress period <kper>"
      - tuple/list of 2 strings -> used as panel titles, e.g. ("2000", "Dec 2023")
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # ---- read heads
    h3d_0, m, sim = _read_mf6_heads_3d(sim_ws, sim_name=sim_name, kstpkper=kstpkper0)
    h3d_1, _, _   = _read_mf6_heads_3d(sim_ws, sim_name=sim_name, kstpkper=kstpkper1)

    # ---- apply coord overrides for correct map placement
    _apply_coord_overrides(m, xoff_override, yoff_override, angrot_override, crs=MODEL_CRS)

    # ---- mask dry/inactive
    dis = m.dis
    idomain = _get_idomain(dis)
    h3d_0 = _mask_heads(h3d_0, idomain=idomain)
    h3d_1 = _mask_heads(h3d_1, idomain=idomain)

    # ---- pick 2D heads
    z0 = _heads_2d_from_mode(h3d_0, mode=mode, layer_num=layer_num)
    z1 = _heads_2d_from_mode(h3d_1, mode=mode, layer_num=layer_num)

    # ---- grid centers for contours
    mg = m.modelgrid
    x2d = np.array(mg.xcellcenters, dtype=float)
    y2d = np.array(mg.ycellcenters, dtype=float)

    # ---- extent from full grid outline
    grid_outline = _grid_outline_gdf(m)
    model_extent = _get_extent_from_gdf(grid_outline, pad_frac=0.02)

    # ---- active outline (layer-specific if idomain 3D)
    active_outline = None
    if show_active_outline and (idomain is not None):
        if np.asarray(idomain).ndim == 3:
            if str(mode).lower().strip() == "layer":
                k0 = int(layer_num) - 1
                id2d = idomain[k0, :, :]
            else:
                id2d = (np.any(idomain > 0, axis=0)).astype(int)
        else:
            id2d = idomain
        active_outline = _active_area_outline_gdf(m, id2d, crs=MODEL_CRS)

    # ---- shared colorbar range (across both panels)
    v = np.concatenate([z0[np.isfinite(z0)].ravel(), z1[np.isfinite(z1)].ravel()])
    if v.size == 0:
        raise ValueError("No finite head values found for either snapshot.")
    vmin = np.floor(np.nanmin(v) / 10.0) * 10.0
    vmax = np.ceil(np.nanmax(v) / 10.0) * 10.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap(cmap_name)

    # ---- panel titles
    if panel_labels is not None and len(panel_labels) == 2:
        t0, t1 = panel_labels[0], panel_labels[1]
    else:
        t0 = f"Stress period {kstpkper0[1]}"
        t1 = f"Stress period {kstpkper1[1]}"

    # ---- figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.5, 6.5))

    # Big title (single line) + subtitle line (prevents overlap)
    supt = fig.suptitle(model_name_for_title, fontsize=18, y=0.99)
    subtitle = fig.text(
        0.5, 0.955,
        f"Hydraulic head + {int(contour_interval)}-ft contours",
        ha="center", va="top", fontsize=14
    )

    # Give the plots more top room and keep space at right for colorbar
    # rect = [left, bottom, right, top] in figure coords
    fig.tight_layout(rect=[0.0, 0.0, 0.93, 0.92])

    # ---- draw panels
    for ax, z, ttl in [(ax0, z0, t0), (ax1, z1, t1)]:
        ax.set_title(ttl, fontsize=14, pad=10)

        _apply_extent(ax, model_extent)
        add_basemap(ax)

        gdf = _grid_to_polygons_gdf(m, z, crs=MODEL_CRS, value_field="head")
        if len(gdf) > 0:
            gdf.plot(
                column="head",
                ax=ax,
                cmap=cmap,
                norm=norm,
                linewidth=0.0,
                edgecolor="none",
                alpha=0.85,
                zorder=10,
            )

        _add_head_contours(
            ax=ax,
            x2d=x2d,
            y2d=y2d,
            z2d=z,
            contour_interval=contour_interval,
            lw=0.6,
            color="k",
            fontsize=7,
            add_apostrophe=True,
            zorder=120,
        )

        add_grid_outline(ax, grid_outline)

        if active_outline is not None:
            style = active_outline_style or {"lw": 0.9, "color": "black", "alpha": 0.5}
            add_active_area_outline(ax, active_outline, **style)

        add_north_arrow(ax)
        if ADD_SCALE_BAR:
            add_scale_bar(ax)
        add_neatline(ax)

    # ---- shared colorbar (kept outside with rect right=0.93)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Dedicated colorbar axis (stable placement)
    cax = fig.add_axes([0.94, 0.12, 0.02, 0.76])  # [left, bottom, width, height]
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("Hydraulic head (ft amsl)", fontsize=12)
    cb.set_ticks([vmin, vmax])
    cb.set_ticklabels([f"Low: {vmin:.0f}", f"High: {vmax:.0f}"])

    # Save (include suptitle/subtitle in bbox)
    fig.savefig(
        out_png,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=[supt, subtitle],
    )
    plt.close(fig)
    print(f"Wrote: {out_png}")


# ============================================================
# MAIN RUNNER (kept: thickness + K calls remain, behind flags)
# ============================================================

def main():
    # ---- thickness ----
    if PLOT_THICKNESS:
        plot_model_thickness_figs_per_layer(
            sim_ws=elk_ws, outdir=OUTDIR_THK, model_name_for_title="Elk Valley",
            layers_to_plot=ELK_LAYERS, dpi=DPI_THK
        )
        plot_model_thickness_figs_per_layer(
            sim_ws=wahp_ws, outdir=OUTDIR_THK, model_name_for_title="Wahpeton",
            layers_to_plot=WAHP_LAYERS, dpi=DPI_THK
        )
        plot_model_thickness_figs_per_layer(
            sim_ws=sw_ww_ws, outdir=OUTDIR_THK, model_name_for_title="Spiritwood–Warwick",
            xoff_override=SW_WW_XOFF, yoff_override=SW_WW_YOFF, angrot_override=0.0,
            layers_to_plot=SW_WW_LAYERS, dpi=DPI_THK
        )

    # ---- K ----
    if PLOT_K and PLOT_K_ONE_FIG_PER_LAYER:
        plot_model_k_figs_per_layer(
            sim_ws=elk_ws,
            outdir=OUTDIR_K,
            model_name_for_title="Elk Valley",
            layers_to_plot=ELK_LAYERS,
            nbins_hk=NBINS_HK,
            nbins_vk=NBINS_VK,
            bins_hk=ELK_BINS_HK,
            bins_vk=ELK_BINS_VK,
            units_label=K_UNITS_LABEL,
            dpi=DPI_K,
            vk_stored_as_ratio=VK_STORED_AS_RATIO,
        )

        plot_model_k_figs_per_layer(
            sim_ws=wahp_ws,
            outdir=OUTDIR_K,
            model_name_for_title="Wahpeton",
            layers_to_plot=WAHP_LAYERS,
            nbins_hk=NBINS_HK,
            nbins_vk=NBINS_VK,
            bins_hk=WAHP_BINS_HK,
            bins_vk=WAHP_BINS_VK,
            units_label=K_UNITS_LABEL,
            dpi=DPI_K,
            vk_stored_as_ratio=VK_STORED_AS_RATIO,
        )

        plot_model_k_figs_per_layer(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_K,
            model_name_for_title="Spiritwood–Warwick",
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            layers_to_plot=SW_WW_LAYERS,
            nbins_hk=NBINS_HK,
            nbins_vk=NBINS_VK,
            bins_hk=SW_BINS_HK,
            bins_vk=SW_BINS_VK,
            units_label=K_UNITS_LABEL,
            dpi=DPI_K,
            vk_stored_as_ratio=VK_STORED_AS_RATIO,
        )

    # ---- storage ----
    if PLOT_STORAGE:
        plot_model_storage_figs_per_layer(
            sim_ws=elk_ws,
            outdir=OUTDIR_STO,
            model_name_for_title="Elk Valley",
            layers_to_plot=ELK_LAYERS,
            nbins_ss=NBINS_SS,
            nbins_sy=NBINS_SY,
            bins_ss=ELK_BINS_SS,
            bins_sy=ELK_BINS_SY,
            #no_sy_layers=set(),  # Elk has SY for both layers
            dpi=DPI_STO,
        )

        plot_model_storage_figs_per_layer(
            sim_ws=wahp_ws,
            outdir=OUTDIR_STO,
            model_name_for_title="Wahpeton",
            layers_to_plot=WAHP_LAYERS,  # 1–6 only (no 7)
            nbins_ss=NBINS_SS,
            nbins_sy=NBINS_SY,
            bins_ss=WAHP_BINS_SS,
            bins_sy=WAHP_BINS_SY,
            #no_sy_layers={3, 4, 5, 6},  # per your instruction
            dpi=DPI_STO,
        )

        plot_model_storage_figs_per_layer(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_STO,
            model_name_for_title="Spiritwood–Warwick",
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            layers_to_plot=SW_WW_LAYERS,
            nbins_ss=NBINS_SS,
            nbins_sy=NBINS_SY,
            bins_ss=SW_BINS_SS,
            bins_sy=SW_BINS_SY,
            #no_sy_layers={2, 3},  # per your instruction
            dpi=DPI_STO,
        )

    # ---- recharge ----
    if PLOT_RECHARGE:
        plot_avg_annual_recharge_tile(
            sim_ws=elk_ws,
            outdir=OUTDIR_RCH,
            model_name_for_title="Elk Valley",
            sim_name="elk_2lay",   # set if needed; else remove
            year0=2000,
            year1=2023,
            units="in/yr",
            eps=0.0,
        )

        plot_avg_annual_recharge_tile(
            sim_ws=wahp_ws,
            outdir=OUTDIR_RCH,
            model_name_for_title="Wahpeton",
            sim_name="wahp7ly",  # set if needed; else remove
            units="in/yr",
            year0=2000,
            year1=2023,
            eps=0.0
        )

        plot_avg_annual_recharge_tile(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_RCH,
            model_name_for_title="Spiritwood–Warwick",
            sim_name="swww",  # set if needed; else remove
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            units="in/yr",
            year0=2000,
            year1=2023,
            eps=0.0
        )
        
    # ---- budget ----
    if PLOT_BUDGET:
        plot_model_budget_timeseries(
            sim_ws=wahp_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Wahpeton",
            items_dict=WAHP_BUDGET_ITEMS,
            year0=None, year1=None,
            budget_df=None,
            out_tag="Entire Model",
        )
        # zonal Layer 5 (zbud.csv filtered)
        wahp_l5_df = load_budget_df(
            sim_ws=wahp_ws,
            use_zbud=True,
            zbud_zone_id=5,
            zbud_fname="zbud.csv",
            model_kind="wahp",
        )
        plot_model_budget_timeseries(
            sim_ws=wahp_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Wahpeton",
            items_dict=WAHP_PIE_BUDGET_ITEMS,
            year0=None, year1=None,
            budget_df=wahp_l5_df,
            out_tag="WBV - Layer 5",
        )
        plot_model_budget_timeseries(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Spiritwood–Warwick",
            items_dict=SWWW_BUDGET_ITEMS,
            year0=None, year1=None,
            budget_df=None,
            out_tag="Entire Model",
        )

        swww_l3_df = load_budget_df(
            sim_ws=sw_ww_ws,
            use_zbud=True,
            zbud_zone_id=3,
            zbud_fname="zbud.csv",
            model_kind="swww",
        )
        plot_model_budget_timeseries(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Spiritwood–Warwick",
            items_dict=SWWW_BUDGET_ITEMS,
            year0=None, year1=None,
            budget_df=swww_l3_df,
            out_tag="Spiritwood - Layer 3",
        )

        plot_model_budget_timeseries(
            sim_ws=elk_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Elk Valley",
            items_dict=ELK_BUDGET_ITEMS,
            year0=None,
            year1=None,
        )

    # ---- BC stage/cond ----
    if PLOT_BC:
        # Elk
        plot_bc_stage_cond_tiles_per_layer(
            sim_ws=elk_ws,
            outdir=OUTDIR_BC,
            model_name_for_title="Elk Valley",
            bc_type="riv",
            sim_name="elk_2lay",
            kper=BC_KPER,
            bins_value=RIV_BINS_STAGE,
            bins_cond=RIV_BINS_COND,
            dpi=DPI_BC,
        )
        plot_bc_stage_cond_tiles_per_layer(
            sim_ws=elk_ws,
            outdir=OUTDIR_BC,
            model_name_for_title="Elk Valley",
            bc_type="drn",
            sim_name="elk_2lay",
            kper=BC_KPER,
            bins_value=DRN_BINS_ELEV,
            bins_cond=DRN_BINS_COND,
            dpi=DPI_BC,
        )

        # Wahpeton
        plot_bc_stage_cond_tiles_per_layer(
            sim_ws=wahp_ws,
            outdir=OUTDIR_BC,
            model_name_for_title="Wahpeton",
            bc_type="riv",
            sim_name="wahp7ly",
            kper=BC_KPER,
            bins_value=RIV_BINS_STAGE,
            bins_cond=RIV_BINS_COND,
            dpi=DPI_BC,
        )
        plot_bc_stage_cond_tiles_per_layer(
            sim_ws=wahp_ws,
            outdir=OUTDIR_BC,
            model_name_for_title="Wahpeton",
            bc_type="drn",
            sim_name="wahp7ly",
            kper=BC_KPER,
            bins_value=DRN_BINS_ELEV,
            bins_cond=DRN_BINS_COND,
            dpi=DPI_BC,
        )

        # Spiritwood–Warwick (coord override)
        plot_bc_stage_cond_tiles_per_layer(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_BC,
            model_name_for_title="Spiritwood–Warwick",
            bc_type="riv",
            sim_name="swww",
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            kper=BC_KPER,
            bins_value=RIV_BINS_STAGE,
            bins_cond=RIV_BINS_COND,
            dpi=DPI_BC,
        )
        plot_bc_stage_cond_tiles_per_layer(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_BC,
            model_name_for_title="Spiritwood–Warwick",
            bc_type="drn",
            sim_name="swww",
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            kper=BC_KPER,
            bins_value=DRN_BINS_ELEV,
            bins_cond=DRN_BINS_COND,
            dpi=DPI_BC,
        )
        
        
    # ---- heads + contours ----
    if PLOT_HEADS:
        # Find Dec 2023 kper using TDIS for each model
        elk_kper_dec2023  = find_kper_for_datetime(elk_ws,  HEAD_TARGET_DATE, sim_name="elk_2lay")
        swww_kper_dec2023 = find_kper_for_datetime(sw_ww_ws, HEAD_TARGET_DATE, sim_name="swww")
        wahp_kper_dec2023 = find_kper_for_datetime(wahp_ws, HEAD_TARGET_DATE, sim_name="wahp7ly")

        print("[HEADS] Dec-2023 kper indices:",
              "Elk =", elk_kper_dec2023,
              "SWWW =", swww_kper_dec2023,
              "Wahp =", wahp_kper_dec2023)
        
        sim_elk  = flopy.mf6.MFSimulation.load(sim_ws=elk_ws, exe_name="mf6")
        sim_swww = flopy.mf6.MFSimulation.load(sim_ws=sw_ww_ws, exe_name="mf6")
        sim_wahp = flopy.mf6.MFSimulation.load(sim_ws=wahp_ws, exe_name="mf6")

        elk_labels  = (_plus_one_year(_kper_label_from_tdis(sim_elk, HEAD_REF_KPER, "year")),
                    _kper_label_from_tdis(sim_elk,  elk_kper_dec2023, "monthyear"))

        swww_labels = (_kper_label_from_tdis(sim_swww, HEAD_REF_KPER, "year"),
                    _kper_label_from_tdis(sim_swww, swww_kper_dec2023, "monthyear"))

        wahp_labels = (_plus_one_year(_kper_label_from_tdis(sim_wahp, HEAD_REF_KPER, "year")),
                    "2023")

        # -------------------------
        # Wahpeton Shallow Sand = Water Table (first non-dry)
        # -------------------------
        plot_head_contour_tiles_two_kpers(
            sim_ws=wahp_ws,
            out_png=os.path.join(OUTDIR_HEADS, "wahpeton_shallow_sand_watertable_sp0_vs_dec2023.png"),
            model_name_for_title="Wahpeton Shallow Sand (Water Table)",
            sim_name="wahp7ly",
            kstpkper0=(0, HEAD_REF_KPER),
            kstpkper1=(0, wahp_kper_dec2023),
            mode="water_table",
            contour_interval=HEAD_CONTOUR_INTERVAL_FT,
            cmap_name=HEAD_CMAP_NAME,
            dpi=DPI_HEADS,
            show_active_outline=HEAD_SHOW_ACTIVE_OUTLINE,
            active_outline_style=HEAD_ACTIVE_OUTLINE_STYLE,
            panel_labels=wahp_labels,
        )

        # -------------------------
        # Wahpeton Buried Valley = Layer 5
        # -------------------------
        plot_head_contour_tiles_two_kpers(
            sim_ws=wahp_ws,
            out_png=os.path.join(OUTDIR_HEADS, "wahpeton_buried_valley_layer05_sp0_vs_dec2023.png"),
            model_name_for_title="Wahpeton Buried Valley (Layer 5)",
            sim_name="wahp7ly",
            kstpkper0=(0, HEAD_REF_KPER),
            kstpkper1=(0, wahp_kper_dec2023),
            mode="layer",
            layer_num=WAHP_BV_LAYER,
            contour_interval=HEAD_CONTOUR_INTERVAL_FT,
            cmap_name=HEAD_CMAP_NAME,
            dpi=DPI_HEADS,
            show_active_outline=HEAD_SHOW_ACTIVE_OUTLINE,
            active_outline_style=HEAD_ACTIVE_OUTLINE_STYLE,
            panel_labels=wahp_labels,
        )

        # -------------------------
        # Spiritwood–Warwick: Warwick = Layer 1
        # -------------------------
        plot_head_contour_tiles_two_kpers(
            sim_ws=sw_ww_ws,
            out_png=os.path.join(OUTDIR_HEADS, "spiritwood_warwick_layer01_sp0_vs_dec2023.png"),
            model_name_for_title="Spiritwood–Warwick — Warwick (Layer 1)",
            sim_name="swww",
            kstpkper0=(0, HEAD_REF_KPER),
            kstpkper1=(0, swww_kper_dec2023),
            mode="layer",
            layer_num=SWWW_WARWICK_LAYER,
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            contour_interval=HEAD_CONTOUR_INTERVAL_FT,
            cmap_name=HEAD_CMAP_NAME,
            dpi=DPI_HEADS,
            show_active_outline=HEAD_SHOW_ACTIVE_OUTLINE,
            active_outline_style=HEAD_ACTIVE_OUTLINE_STYLE,
            panel_labels=swww_labels,
        )

        # -------------------------
        # Spiritwood–Warwick: Spiritwood = Layer 3
        # -------------------------
        plot_head_contour_tiles_two_kpers(
            sim_ws=sw_ww_ws,
            out_png=os.path.join(OUTDIR_HEADS, "spiritwood_spiritwood_layer03_sp0_vs_dec2023.png"),
            model_name_for_title="Spiritwood–Warwick — Spiritwood (Layer 3)",
            sim_name="swww",
            kstpkper0=(0, HEAD_REF_KPER),
            kstpkper1=(0, swww_kper_dec2023),
            mode="layer",
            layer_num=SWWW_SPIRITWOOD_LAYER,
            xoff_override=SW_WW_XOFF,
            yoff_override=SW_WW_YOFF,
            angrot_override=0.0,
            contour_interval=HEAD_CONTOUR_INTERVAL_FT,
            cmap_name=HEAD_CMAP_NAME,
            dpi=DPI_HEADS,
            show_active_outline=HEAD_SHOW_ACTIVE_OUTLINE,
            active_outline_style=HEAD_ACTIVE_OUTLINE_STYLE,
            panel_labels=swww_labels,
        )

        # -------------------------
        # Elk Valley = Water Table (first non-dry)
        # -------------------------
        plot_head_contour_tiles_two_kpers(
            sim_ws=elk_ws,
            out_png=os.path.join(OUTDIR_HEADS, "elk_water_table_sp0_vs_dec2023.png"),
            model_name_for_title="Elk Valley (Water Table)",
            sim_name="elk_2lay",
            kstpkper0=(0, HEAD_REF_KPER),
            kstpkper1=(0, elk_kper_dec2023),
            mode="water_table",
            contour_interval=HEAD_CONTOUR_INTERVAL_FT,
            cmap_name=HEAD_CMAP_NAME,
            dpi=DPI_HEADS,
            show_active_outline=HEAD_SHOW_ACTIVE_OUTLINE,
            active_outline_style=HEAD_ACTIVE_OUTLINE_STYLE,
            panel_labels=elk_labels,
        )
        
    # ---- budget pie charts (2023 only) ----
    if PLOT_BUDGET_PIES:
        plot_budget_piecharts_for_year(
            sim_ws=wahp_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Wahpeton",
            items_dict=WAHP_PIE_BUDGET_ITEMS,
            pie_year=2023,
            out_fmt=("png", "pdf"),
            dpi=BUDGET_DPI,
            budget_df=None,
            out_tag="full_model",
        )
        
        wahp_l5_df = load_budget_df(
            sim_ws=wahp_ws,
            use_zbud=True,
            zbud_zone_id=5,
            zbud_fname="zbud.csv",
            model_kind="wahp",
        )
        plot_budget_piecharts_for_year(
            sim_ws=wahp_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Wahpeton",
            items_dict=WAHP_PIE_BUDGET_ITEMS,
            pie_year=2023,
            out_fmt=("png", "pdf"),
            dpi=BUDGET_DPI,
            budget_df=wahp_l5_df,
            out_tag="layer5_zone5",
        )

        plot_budget_piecharts_for_year(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Spiritwood–Warwick",
            items_dict=SWWW_BUDGET_ITEMS,
            pie_year=2023,
            out_fmt=("png", "pdf"),
            dpi=BUDGET_DPI,
            budget_df=None,
            out_tag="full_model",
        )

        swww_l3_df = load_budget_df(
            sim_ws=sw_ww_ws,
            use_zbud=True,
            zbud_zone_id=3,
            zbud_fname="zbud.csv",
            model_kind="swww",
        )
        plot_budget_piecharts_for_year(
            sim_ws=sw_ww_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Spiritwood–Warwick",
            items_dict=SWWW_BUDGET_ITEMS,
            pie_year=2023,
            out_fmt=("png", "pdf"),
            dpi=BUDGET_DPI,
            budget_df=swww_l3_df,
            out_tag="layer3_zone3",
        )

        plot_budget_piecharts_for_year(
            sim_ws=elk_ws,
            outdir=OUTDIR_BUDGET,
            model_name_for_title="Elk Valley",
            items_dict=ELK_BUDGET_ITEMS,
            pie_year=2023,
            out_fmt=("png", "pdf"),
            dpi=BUDGET_DPI,
        )


    plot_wahp_asr_well_tile(
        wahp_ws=wahp_ws,
        out_png=os.path.join(
            "figures", "asr_tiles", "wahpeton_asr_5_vs_10_tile.png"
        ),
        asr_dir=os.path.join("..", "gis", "input_shps", "wahp", "asr_shps"),
        shp5_name="asr5well_sys.shp",
        shp10_name="asr10wells.shp",
        sim_name="wahp7ly",
        basemap=True,
        point_size=36,
        active_layer_num=5,
        dpi=350,
    )


if __name__ == "__main__":
    main()