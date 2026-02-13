import os
import pandas as pd
import geopandas as gpd
import numpy as np
import re


# -------------------------
# Paths
# -------------------------
POD_SHP = os.path.join(
    "..", "..", "gis", "input_shps", "elk", "water_use",
    "Elk_Valley_aquifer_PermitPOD_withDeferred.shp"
)

MEMBER_SUMMARY_CSV = os.path.join(
    "pest_mou_output", "seasonal mar output", "output_shapefiles","mou_wells", "member_summaries",
    "member_gen_98_member_23253_pso_pod_permit_summary.csv"
)

OUT_DIR = os.path.join("pest_mou_output", "seasonal mar output", "output_shapefiles", "mou_wells")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_BASE = os.path.join(OUT_DIR, "deferred_permits_with_mou_results_mar")
NEW_COL = "opt_AFR_base"  # <=10 chars, shapefile-safe


# -------------------------
# Helpers
# -------------------------
def norm_str(x):
    """Normalize join keys to reduce mismatch from int/float/space issues."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    # remove trailing .0 if pandas read numeric as float
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s

def remove_shp_set(path_no_ext: str):
    for e in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = path_no_ext + e
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

def polygons_to_cell_center_points(gdf: gpd.GeoDataFrame, out_crs=None) -> gpd.GeoDataFrame:
    """
    Convert polygon grid cells to point features at cell centers (centroids).
    If the input CRS is geographic, project to a planar CRS first for accurate centroids.
    If out_crs is provided, output is reprojected to that CRS.
    """
    if gdf.empty:
        return gdf.copy()

    gdf_in = gdf.copy()

    # Choose a planar CRS for centroid calculations if current CRS is geographic
    # If already projected, keep it.
    calc_crs = gdf_in.crs
    if calc_crs is None:
        # No CRS info: centroid will still work but may be questionable.
        # Keep as-is.
        calc_gdf = gdf_in
    else:
        try:
            if calc_crs.is_geographic:
                # Use a local UTM zone inferred from data bounds
                # (good default for centroid computations)
                calc_crs = gdf_in.estimate_utm_crs()
                calc_gdf = gdf_in.to_crs(calc_crs)
            else:
                calc_gdf = gdf_in
        except Exception:
            # Fallback: just compute centroids in-place
            calc_gdf = gdf_in

    # Compute centroids
    pts = calc_gdf.copy()
    pts["geometry"] = calc_gdf.geometry.centroid
    pts = pts.set_geometry("geometry")

    # Reproject to desired output CRS if requested
    if out_crs is not None and pts.crs is not None:
        pts = pts.to_crs(out_crs)
    elif gdf_in.crs is not None and pts.crs is not None and pts.crs != gdf_in.crs:
        # Return to original CRS if we temporarily projected
        pts = pts.to_crs(gdf_in.crs)

    return pts


# -------------------------
# Read shapefile and filter statuses
# -------------------------
gdf = gpd.read_file(POD_SHP)

# expected fields: permit_num, pod, status
# (if your actual names differ slightly, change here)
PERMIT_FLD = "permit_num"
POD_FLD = "pod"
STATUS_FLD = "status"

for f in (PERMIT_FLD, POD_FLD, STATUS_FLD):
    if f not in gdf.columns:
        raise ValueError(
            f"Expected field '{f}' not found in {POD_SHP}. "
            f"Available fields: {list(gdf.columns)}"
        )

wanted_status = {"Deferred", "Conditionally Approved"}
gdf_f = gdf[gdf[STATUS_FLD].astype(str).isin(wanted_status)].copy()

print(f"Input features: {len(gdf)}")
print(f"Filtered (Deferred/Conditionally Approved): {len(gdf_f)}")


# -------------------------
# Read member summary CSV, determine qualifying (pod,permit) combos
# -------------------------
ms = pd.read_csv(MEMBER_SUMMARY_CSV)

need = {"pod_id", "permit", "cells_applied_total", "dvpop1_cells"}
missing = need - set(ms.columns)
if missing:
    raise ValueError(
        f"Member summary CSV missing columns: {missing}. "
        f"Found: {list(ms.columns)}"
    )

ms["pod_key"] = ms["pod_id"].apply(norm_str)
ms["permit_key"] = ms["permit"].apply(norm_str)
ms["cells_applied_total"] = pd.to_numeric(ms["cells_applied_total"], errors="coerce").fillna(0).astype(int)
ms["dvpop1_cells"] = pd.to_numeric(ms["dvpop1_cells"], errors="coerce").fillna(0).astype(int)

# Qualify: dvpop1_cells > 0 and equals total
qual = ms[(ms["dvpop1_cells"] > 0) & (ms["dvpop1_cells"] == ms["cells_applied_total"])].copy()

qual_keys = set(zip(qual["pod_key"], qual["permit_key"]))
print(f"Qualifying pod/permit combos for {NEW_COL}: {len(qual_keys)}")


# -------------------------
# Add new column and populate
# -------------------------
gdf_f["pod_key"] = gdf_f[POD_FLD].apply(norm_str)
gdf_f["permit_key"] = gdf_f[PERMIT_FLD].apply(norm_str)

gdf_f[NEW_COL] = [
    1 if (pk, rk) in qual_keys else 0
    for pk, rk in zip(gdf_f["pod_key"].to_numpy(), gdf_f["permit_key"].to_numpy())
]

# Optional: keep the join keys or drop them
gdf_f = gdf_f.drop(columns=["pod_key", "permit_key"], errors="ignore")

print(f"{NEW_COL}=1 count: {int(gdf_f[NEW_COL].sum())} of {len(gdf_f)}")


# -------------------------
# Write outputs
# -------------------------
remove_shp_set(OUT_BASE)

# Shapefile
gdf_f.to_file(OUT_BASE + ".shp")
print("Wrote:", OUT_BASE + ".shp")

# GeoPackage (nice for accumulating future scenario columns)
gpkg_path = OUT_BASE + ".gpkg"
if os.path.exists(gpkg_path):
    try:
        os.remove(gpkg_path)
    except Exception:
        pass
gdf_f.to_file(gpkg_path, driver="GPKG")
print("Wrote:", gpkg_path)
