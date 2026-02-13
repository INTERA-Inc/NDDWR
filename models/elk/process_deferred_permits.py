"""
process_deferred_permits.py

Investigate "Deferred" + "Conditionally Approved" PODs and how they relate to:
- existing PODs
- existing permits
- existing permit holders
- existing wells (spatial)
- existing POD polygons (spatial overlap)

PLUS:
- build deferred_requested_per_well.csv from req_acft
- add a NEW MF6 WEL package ("def_wel") to an existing model
  using deferred requested per-well flows, starting at SP 325 (Jan 2024),
  optionally applying irrigation seasonal weights using use_type.

Outputs:
- data/processed/water_use/deferred_investigation/deferred_candidates.csv
- data/processed/water_use/deferred_investigation/deferred_summary_counts.csv
- data/processed/water_use/deferred_investigation/deferred_overlap_pairs.csv
- data/processed/water_use/deferred_investigation/deferred_requested_per_well.csv
- model_ws/elk_2lay_monthly_clean/def_wel.wel (new package files in workspace)
"""

import os
import re
from pathlib import Path
from typing import Optional, Iterable, Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import nearest_points
import sys

# local deps (as in your pasted version)
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "dependencies")))
sys.path.insert(1, os.path.abspath(os.path.join("..", "..", "dependencies", "flopy")))
sys.path.insert(2, os.path.abspath(os.path.join("..", "..", "dependencies", "pyemu")))
import flopy
import calendar

# -----------------------------------------------------------------------------#
# Paths (adjust as needed)
# -----------------------------------------------------------------------------#
GIS_PATH   = os.path.join("..", "..", "gis", "input_shps", "elk", "water_use")

WELLS_SHP  = os.path.join(GIS_PATH, "Elk_Valley_aquifer_water_use_wells.shp")
POD_SHP    = os.path.join(GIS_PATH, "2025-09-22_Elk_Valley_aquifer_PermitPOD.shp")
POD_DEFERRED_SHP = os.path.join(GIS_PATH, "Elk_Valley_aquifer_PermitPOD_withDeferred.shp")

OUT_DIR = os.path.join("data", "processed", "water_use", "deferred_investigation")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
OUT_CANDIDATES_SHP = os.path.join(OUT_DIR, "deferred_candidates.shp")
OUT_CANDIDATES_GPKG = os.path.join(OUT_DIR, "deferred_candidates.gpkg")  # optional but recommended

MODEL_DOMAIN_SHP = os.path.join("..", "..", "gis", "input_shps", "elk", "elk_boundary_lf.shp")

# existing “reported use” allocation file (used only to classify wells)
EXISTING_ALLOC_CSV = os.path.join("data", "processed", "water_use", "per_well_allocation_main_monthlyTrue.csv")

OUT_REQ_PER_WELL_CSV = os.path.join(OUT_DIR, "deferred_requested_per_well.csv")
OUT_REQ_WELLS_SHP    = os.path.join(OUT_DIR, "deferred_requested_wells.shp")
OUT_REQ_WELLS_GPKG   = os.path.join(OUT_DIR, "deferred_requested_wells.gpkg")

AF_TO_CF = 43_560.0  # acre-ft -> ft^3

VERBOSE = True

# -----------------------------------------------------------------------------#
# Utilities (kept consistent with your existing style)
# -----------------------------------------------------------------------------#
def log(msg: str):
    if VERBOSE:
        print(msg)

def first_existing(colnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(colnames)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def fix_invalid_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        gdf = gdf.copy()
        gdf.loc[gdf.geometry.notna(), "geometry"] = gdf.loc[gdf.geometry.notna(), "geometry"].buffer(0)
    return gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

def safe_to_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if target_crs is None or gdf.crs == target_crs:
        return gdf
    if gdf.crs is None:
        # Don’t guess; caller should set CRS if truly missing.
        return gdf
    return gdf.to_crs(target_crs)

def norm_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def norm_key(x) -> str:
    """More aggressive normalization for matching IDs/permits."""
    s = norm_str(x)
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_holder(x) -> str:
    """Normalize holder names for matching (permit_hol-like fields)."""
    s = norm_key(x).lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\b(inc|llc|ltd|co|corp|corporation|company)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_pod_cols(pod_g: gpd.GeoDataFrame) -> Tuple[str, str]:
    pod_id_col = first_existing(pod_g.columns, ["pod", "pod_id", "podnum", "pod_num", "podname", "pod_name"])
    permit_col = first_existing(pod_g.columns, ["permit_number", "permit_num", "permit", "permit_no", "per_num"])
    if pod_id_col is None:
        raise ValueError(f"Could not find POD id column in {list(pod_g.columns)}")
    if permit_col is None:
        raise ValueError(f"Could not find permit column in {list(pod_g.columns)}")
    return pod_id_col, permit_col

def detect_well_id_col(wells_g: gpd.GeoDataFrame) -> str:
    well_id_col = first_existing(wells_g.columns, ["site_locat", "well_no", "wellid", "well_id", "wellname"])
    if well_id_col is None:
        wells_g["_autowellid"] = np.arange(len(wells_g)).astype(int).astype(str)
        return "_autowellid"
    return well_id_col

def pick_status_col(gdf: gpd.GeoDataFrame) -> str:
    c = first_existing(gdf.columns, ["status", "pod_status", "permit_status"])
    if c is None:
        raise ValueError(f"Could not find a 'status' column in deferred POD shapefile. Columns={list(gdf.columns)}")
    return c

def pick_holder_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    return first_existing(gdf.columns, ["permit_hol", "permit_hold", "permit_holder", "holder", "owner", "name"])

def write_deferred_candidates_vector(dep_keep: gpd.GeoDataFrame, out_shp: str, out_gpkg: Optional[str] = None):
    """
    Write deferred candidate info to a shapefile (and optionally a geopackage).

    Notes:
    - Shapefile limits field names to 10 chars; we rename columns.
    - Shapefile can't store list objects; we stringify list fields.
    """
    gdf = dep_keep.copy()

    list_cols = [c for c in ["wells_in_deferred_poly", "overlap_current_pod_ids"] if c in gdf.columns]
    for c in list_cols:
        gdf[c] = gdf[c].apply(lambda v: "" if v is None else "|".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v))

    bool_cols = [c for c in gdf.columns if gdf[c].dtype == bool]
    for c in bool_cols:
        gdf[c] = gdf[c].astype(int)

    keep = []
    for c in [
        "_status_norm",
        "pod_id_exists_in_current",
        "permit_exists_in_current",
        "holder_exists_in_current",
        "n_wells_in_deferred_poly",
        "wells_in_deferred_poly",
        "overlaps_current_pod",
        "overlap_current_pod_ids",
        "linkage_signals",
    ]:
        if c in gdf.columns:
            keep.append(c)

    id_like = [c for c in gdf.columns if c.lower() in ("pod", "pod_id", "podnum", "pod_num", "podname", "pod_name")]
    permit_like = [c for c in gdf.columns if c.lower() in ("permit", "permit_no", "permit_num", "permit_number", "per_num")]
    holder_like = [c for c in gdf.columns if c.lower() in ("permit_hol", "permit_hold", "permit_holder", "holder", "owner", "name")]
    for c in (id_like + permit_like + holder_like):
        if c not in keep and c != "geometry":
            keep.insert(1, c)

    keep = [c for c in keep if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]].copy()

    if out_gpkg:
        gdf.to_file(out_gpkg, layer="deferred_candidates", driver="GPKG")

    rename = {}
    for c in gdf.columns:
        if c == "geometry":
            continue
        if c == "_status_norm":
            rename[c] = "status"
        elif c == "pod_id_exists_in_current":
            rename[c] = "podmatch"
        elif c == "permit_exists_in_current":
            rename[c] = "permatch"
        elif c == "holder_exists_in_current":
            rename[c] = "holmatch"
        elif c == "n_wells_in_deferred_poly":
            rename[c] = "nwells"
        elif c == "wells_in_deferred_poly":
            rename[c] = "well_ids"
        elif c == "overlaps_current_pod":
            rename[c] = "ovrlp_pod"
        elif c == "overlap_current_pod_ids":
            rename[c] = "ov_podids"
        elif c == "linkage_signals":
            rename[c] = "signals"
        else:
            rename[c] = c[:10]

    used = {}
    for k, v in list(rename.items()):
        vv = v
        if vv in used:
            used[vv] += 1
            suffix = str(used[vv])
            vv = (vv[: 10 - len(suffix)] + suffix)[:10]
            rename[k] = vv
        else:
            used[vv] = 0

    gdf_shp = gdf.rename(columns=rename)
    gdf_shp.to_file(out_shp, driver="ESRI Shapefile")

# -----------------------------------------------------------------------------#
# Duplicate collapse for requested allocations
# -----------------------------------------------------------------------------#
def collapse_duplicate_requested_allocations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per Well by summing use_acft and recomputing cfd.
    Merge Permits/Pods bracket lists like the main allocator.
    """
    if df.empty:
        return df.copy()

    d = df.copy()

    def _merge_bracket_lists(values: pd.Series) -> str:
        def _parse(s: str) -> list:
            if pd.isna(s):
                return []
            s = str(s).strip()
            if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
                inner = s[1:-1].strip()
                if not inner:
                    return []
                return [p.strip() for p in inner.split(",") if p.strip()]
            return [s] if s else []

        acc = set()
        for v in values.dropna():
            acc.update(_parse(v))

        def _key(x):
            try:
                return (0, float(x))
            except Exception:
                return (1, str(x))
        merged = sorted(acc, key=_key)
        return f"[{', '.join(map(str, merged))}]"

    def _merge_values_to_list(values: pd.Series) -> str:
        vals = sorted(set([str(v).strip() for v in values.dropna() if str(v).strip() != ""]))
        if not vals:
            return ""
        if len(vals) == 1:
            return vals[0]
        return f"[{', '.join(vals)}]"

    def _first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    def _merge_flag(series: pd.Series) -> str:
        vals = sorted(set([str(v) for v in series.dropna() if str(v).strip() != ""]))
        if not vals:
            return ""
        if len(vals) == 1:
            return vals[0]
        if any("dummy" in v.lower() for v in vals):
            return "mixed (includes dummy)"
        return "mixed"

    agg = (
        d.groupby(["Well"], dropna=False)
         .apply(lambda g: pd.Series({
             "Permits": _merge_bracket_lists(g["Permits"]) if "Permits" in g else "[]",
             "Pods":    _merge_bracket_lists(g["Pods"]) if "Pods" in g else "[]",
             "use_type": _merge_values_to_list(g.get("use_type", pd.Series(dtype=object))),
             "x_2265":  _first_non_null(g.get("x_2265", pd.Series(dtype=float))),
             "y_2265":  _first_non_null(g.get("y_2265", pd.Series(dtype=float))),
             "x_2266":  _first_non_null(g.get("x_2266", pd.Series(dtype=float))),
             "y_2266":  _first_non_null(g.get("y_2266", pd.Series(dtype=float))),
             "total_dept": _first_non_null(g.get("total_dept", pd.Series(dtype=float))),
             "top_screen": _first_non_null(g.get("top_screen", pd.Series(dtype=float))),
             "bottom_scr": _first_non_null(g.get("bottom_scr", pd.Series(dtype=float))),
             "use_acft": g["use_acft"].sum(min_count=1),
             "days": _first_non_null(g.get("days", pd.Series(dtype=float))),
             "well_flag": _merge_flag(g.get("well_flag", pd.Series(dtype=object))),
             "status": _merge_flag(g.get("status", pd.Series(dtype=object))),
         }))
         .reset_index()
    )

    agg["days"] = pd.to_numeric(agg["days"], errors="coerce").fillna(365.0)
    agg["cfd"] = (pd.to_numeric(agg["use_acft"], errors="coerce").fillna(0.0) * AF_TO_CF) / agg["days"].replace({0: np.nan})

    cols_order = [
        "Permits", "Pods", "Well",
        "use_type",
        "x_2265", "y_2265", "x_2266", "y_2266",
        "total_dept", "top_screen", "bottom_scr",
        "use_acft", "days", "cfd",
        "well_flag", "status",
    ]
    cols_order = [c for c in cols_order if c in agg.columns] + [c for c in agg.columns if c not in cols_order]
    return agg[cols_order]

# -----------------------------------------------------------------------------#
# Core investigation (unchanged)
# -----------------------------------------------------------------------------#
def investigate_deferred(
    wells_shp: str = WELLS_SHP,
    pod_shp: str = POD_SHP,
    deferred_shp: str = POD_DEFERRED_SHP,
    out_dir: str = OUT_DIR,
    status_keep = ("Deferred", "Conditionally Approved"),
    overlap_area_frac_thresh: float = 0.02,
):
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log("Loading layers...")
    wells = gpd.read_file(wells_shp)
    pod   = gpd.read_file(pod_shp)
    dep   = gpd.read_file(deferred_shp)

    pod = fix_invalid_polygons(pod)
    dep = fix_invalid_polygons(dep)

    target_crs = pod.crs if pod.crs is not None else dep.crs
    if target_crs is None:
        raise ValueError("Both POD CRS and deferred POD CRS are None. Please define CRS before proceeding.")
    pod = safe_to_crs(pod, target_crs)
    dep = safe_to_crs(dep, target_crs)
    wells = safe_to_crs(wells, target_crs)

    pod_id_col, pod_permit_col = detect_pod_cols(pod)
    dep_id_col, dep_permit_col = detect_pod_cols(dep)

    status_col = pick_status_col(dep)
    holder_col_dep = pick_holder_col(dep)
    holder_col_pod = pick_holder_col(pod)

    well_id_col = detect_well_id_col(wells)

    log(f"Detected columns:")
    log(f"  Current POD: id='{pod_id_col}', permit='{pod_permit_col}', holder='{holder_col_pod}'")
    log(f"  Deferred POD: id='{dep_id_col}', permit='{dep_permit_col}', status='{status_col}', holder='{holder_col_dep}'")
    log(f"  Wells: well_id='{well_id_col}'")

    dep["_status_norm"] = dep[status_col].astype(str).str.strip()
    dep_keep = dep[dep["_status_norm"].isin(list(status_keep))].copy()
    log(f"Deferred candidates kept: {len(dep_keep)} of {len(dep)} (status in {status_keep})")

    pod["_pod_id_key"]    = pod[pod_id_col].apply(norm_key)
    pod["_permit_key"]    = pod[pod_permit_col].apply(norm_key)
    dep_keep["_pod_id_key"] = dep_keep[dep_id_col].apply(norm_key)
    dep_keep["_permit_key"] = dep_keep[dep_permit_col].apply(norm_key)

    if holder_col_pod:
        pod["_holder_key"] = pod[holder_col_pod].apply(norm_holder)
    else:
        pod["_holder_key"] = ""

    if holder_col_dep:
        dep_keep["_holder_key"] = dep_keep[holder_col_dep].apply(norm_holder)
    else:
        dep_keep["_holder_key"] = ""

    existing_pod_ids = set(pod["_pod_id_key"].dropna().astype(str))
    existing_permits = set(pod["_permit_key"].dropna().astype(str))
    existing_holders = set(pod["_holder_key"].dropna().astype(str))

    dep_keep["pod_id_exists_in_current"] = dep_keep["_pod_id_key"].isin(existing_pod_ids)
    dep_keep["permit_exists_in_current"] = dep_keep["_permit_key"].isin(existing_permits)
    dep_keep["holder_exists_in_current"] = dep_keep["_holder_key"].isin(existing_holders) & (dep_keep["_holder_key"] != "")

    log("Spatial join: wells ↔ deferred PODs ...")
    wells_min = wells[[well_id_col, "geometry"]].copy()
    dep_min = dep_keep[["_pod_id_key", "_permit_key", "_holder_key", "_status_norm", "geometry"]].copy()

    wdep = gpd.sjoin(wells_min, dep_min, how="inner", predicate="intersects")
    wells_by_dep = (
        wdep.groupby("_pod_id_key")[well_id_col]
            .apply(lambda s: sorted(set(map(str, s.astype(str)))))
            .to_dict()
    )
    dep_keep["n_wells_in_deferred_poly"] = dep_keep["_pod_id_key"].map(lambda k: len(wells_by_dep.get(k, [])))
    dep_keep["wells_in_deferred_poly"]   = dep_keep["_pod_id_key"].map(lambda k: wells_by_dep.get(k, []))

    log("Spatial overlap: deferred PODs ↔ current POD polygons ...")
    pod_min = pod[["_pod_id_key", "_permit_key", "_holder_key", "geometry"]].copy()
    pod_diss = pod_min.dissolve(by="_pod_id_key", as_index=False)

    inter = gpd.overlay(
        dep_min[["_pod_id_key", "geometry"]],
        pod_diss[["_pod_id_key", "geometry"]].rename(columns={"_pod_id_key": "_pod_id_key_current"}),
        how="intersection",
        keep_geom_type=False
    )

    if inter.empty:
        log("No spatial intersections found between deferred and current POD polygons.")
        overlap_pairs = pd.DataFrame(columns=["dep_pod_id", "current_pod_id", "intersect_area", "dep_area", "area_frac"])
        dep_keep["overlaps_current_pod"] = False
        dep_keep["overlap_current_pod_ids"] = [[] for _ in range(len(dep_keep))]
    else:
        dep_area = dep_min.copy()
        dep_area["dep_area"] = dep_area.geometry.area
        dep_area = dep_area[["_pod_id_key", "dep_area"]]

        inter["intersect_area"] = inter.geometry.area
        inter = inter.merge(dep_area, on="_pod_id_key", how="left")
        inter["area_frac"] = inter["intersect_area"] / inter["dep_area"].replace({0.0: np.nan})

        overlap_pairs = (
            inter.loc[inter["area_frac"].fillna(0.0) >= overlap_area_frac_thresh,
                      ["_pod_id_key", "_pod_id_key_current", "intersect_area", "dep_area", "area_frac"]]
                .rename(columns={"_pod_id_key": "dep_pod_id", "_pod_id_key_current": "current_pod_id"})
                .sort_values(["dep_pod_id", "area_frac"], ascending=[True, False])
                .reset_index(drop=True)
        )

        ov_map = (
            overlap_pairs.groupby("dep_pod_id")["current_pod_id"]
            .apply(lambda s: sorted(set(s.astype(str))))
            .to_dict()
        )
        dep_keep["overlaps_current_pod"] = dep_keep["_pod_id_key"].isin(set(ov_map.keys()))
        dep_keep["overlap_current_pod_ids"] = dep_keep["_pod_id_key"].map(lambda k: ov_map.get(k, []))

    def linkage_tag(r):
        tags = []
        if bool(r.get("pod_id_exists_in_current")): tags.append("POD_ID_MATCH")
        if bool(r.get("permit_exists_in_current")): tags.append("PERMIT_MATCH")
        if bool(r.get("holder_exists_in_current")): tags.append("HOLDER_MATCH")
        if int(r.get("n_wells_in_deferred_poly", 0)) > 0: tags.append("WELLS_IN_POLY")
        if bool(r.get("overlaps_current_pod")): tags.append("OVERLAPS_CURRENT_POD")
        return "|".join(tags) if tags else "NO_MATCH_SIGNALS"

    dep_keep["linkage_signals"] = dep_keep.apply(linkage_tag, axis=1)

    out_candidates = os.path.join(out_dir, "deferred_candidates.csv")
    keep_cols = [
        "_status_norm",
        dep_id_col, dep_permit_col,
        holder_col_dep if holder_col_dep else None,
        "pod_id_exists_in_current", "permit_exists_in_current", "holder_exists_in_current",
        "n_wells_in_deferred_poly", "wells_in_deferred_poly",
        "overlaps_current_pod", "overlap_current_pod_ids",
        "linkage_signals",
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in dep_keep.columns]
    dep_keep_out = dep_keep[keep_cols].copy()
    dep_keep_out.to_csv(out_candidates, index=False)
    log(f"Wrote: {out_candidates}  (rows={len(dep_keep_out)})")

    out_pairs = os.path.join(out_dir, "deferred_overlap_pairs.csv")
    overlap_pairs.to_csv(out_pairs, index=False)
    log(f"Wrote: {out_pairs}  (rows={len(overlap_pairs)})")

    summary = pd.DataFrame({
        "metric": [
            "n_deferred_total",
            "n_deferred_with_pod_id_match",
            "n_deferred_with_permit_match",
            "n_deferred_with_holder_match",
            "n_deferred_with_wells_in_poly",
            "n_deferred_overlapping_current_pod",
            "n_deferred_no_match_signals",
        ],
        "value": [
            int(len(dep_keep)),
            int(dep_keep["pod_id_exists_in_current"].sum()),
            int(dep_keep["permit_exists_in_current"].sum()),
            int(dep_keep["holder_exists_in_current"].sum()),
            int((dep_keep["n_wells_in_deferred_poly"] > 0).sum()),
            int(dep_keep["overlaps_current_pod"].sum()),
            int((dep_keep["linkage_signals"] == "NO_MATCH_SIGNALS").sum()),
        ]
    })
    out_summary = os.path.join(out_dir, "deferred_summary_counts.csv")
    summary.to_csv(out_summary, index=False)
    log(f"Wrote: {out_summary}")

    out_shp = os.path.join(out_dir, "deferred_candidates.shp")
    out_gpkg = os.path.join(out_dir, "deferred_candidates.gpkg")
    write_deferred_candidates_vector(dep_keep, out_shp=out_shp, out_gpkg=out_gpkg)
    log(f"Wrote: {out_shp}")
    log(f"Wrote: {out_gpkg}")

    return {
        "candidates_csv": out_candidates,
        "summary_csv": out_summary,
        "overlap_pairs_csv": out_pairs,
    }

# -----------------------------------------------------------------------------#
# Domain + geometry helpers for dummy wells
# -----------------------------------------------------------------------------#
def _load_domain_inner(domain_shp: str, target_crs, interior_buffer_ft: float = 300.0):
    dom = gpd.read_file(domain_shp)
    dom = dom[~dom.geometry.isna() & ~dom.geometry.is_empty].copy()
    if dom.empty:
        raise ValueError(f"Empty model domain shapefile: {domain_shp}")

    if dom.crs is None:
        raise ValueError(f"Model domain shapefile CRS is None: {domain_shp}")

    if target_crs is not None and dom.crs != target_crs:
        dom = dom.to_crs(target_crs)

    geom = dom.unary_union
    inner = geom.buffer(-float(interior_buffer_ft))

    if inner.is_empty:
        inner = geom.buffer(-10.0)
    if inner.is_empty:
        inner = geom

    return geom, inner

def _move_point_inside_with_buffer(pt, domain_inner, domain_full):
    if pt is None:
        return None
    if domain_inner.contains(pt):
        return pt

    p_near = nearest_points(pt, domain_inner)[1]
    cen = domain_inner.centroid
    dx = cen.x - p_near.x
    dy = cen.y - p_near.y
    mag = (dx*dx + dy*dy) ** 0.5
    if mag > 0:
        p_near = type(pt)(p_near.x + dx/mag, p_near.y + dy/mag)

    if domain_inner.contains(p_near):
        return p_near
    if domain_full.contains(p_near):
        return p_near

    return domain_inner.representative_point()

def _detect_cols_for_wells(wells_gdf: gpd.GeoDataFrame):
    well_id = first_existing(wells_gdf.columns, ["site_locat", "well_no", "wellid", "well_id"])
    if well_id is None:
        wells_gdf["_autowellid"] = np.arange(len(wells_gdf)).astype(int).astype(str)
        well_id = "_autowellid"

    total_dept = first_existing(wells_gdf.columns, ["total_dept", "total_depth", "tot_depth"])
    top_scr    = first_existing(wells_gdf.columns, ["top_screen", "top_scr", "screen_top"])
    bot_scr    = first_existing(wells_gdf.columns, ["bottom_scr", "bot_scr", "screen_bot", "bottom_screen"])

    return {
        "well_id": well_id,
        "total_dept": total_dept,
        "top_screen": top_scr,
        "bottom_scr": bot_scr,
    }

def _add_xy(gdf: gpd.GeoDataFrame):
    g = gdf.copy()
    if g.crs is None:
        raise ValueError("GeoDataFrame CRS is None; cannot compute x/y.")

    g2265 = g.to_crs(2265)
    g["x_2265"] = g2265.geometry.x
    g["y_2265"] = g2265.geometry.y

    g2266 = g.to_crs(2266)
    g["x_2266"] = g2266.geometry.x
    g["y_2266"] = g2266.geometry.y

    return g

def _load_reported_use_well_set(existing_alloc_csv: str) -> set:
    if not os.path.exists(existing_alloc_csv):
        return set()
    df = pd.read_csv(existing_alloc_csv, dtype={"Well": str})
    if "use_acft" in df.columns:
        df["use_acft"] = pd.to_numeric(df["use_acft"], errors="coerce").fillna(0.0)
        return set(df.loc[df["use_acft"] > 0, "Well"].astype(str).unique())
    return set()

def _ensure_dummy_well_in_domain_for_pod(
    pod_row: gpd.GeoSeries,
    wells_aug: gpd.GeoDataFrame,
    well_id_col: str,
    domain_full,
    domain_inner,
    dummy_prefix: str = "DUMMY_",
):
    geom = pod_row.geometry
    if geom is None or geom.is_empty:
        return None, wells_aug

    rep_pt = geom.representative_point()
    rep_pt2 = _move_point_inside_with_buffer(rep_pt, domain_inner, domain_full)

    pod_id = norm_key(pod_row.get("_pod_id_key", ""))
    if pod_id == "":
        pod_id = str(int(np.random.randint(1_000_000)))
    dummy_id = f"{dummy_prefix}{pod_id}"

    existing_ids = set(wells_aug[well_id_col].astype(str)) if well_id_col in wells_aug.columns else set()
    if dummy_id in existing_ids:
        k = 2
        while f"{dummy_id}_{k}" in existing_ids:
            k += 1
        dummy_id = f"{dummy_id}_{k}"

    new_row = {well_id_col: dummy_id, "geometry": rep_pt2, "is_dummy": 1}
    wells_aug2 = pd.concat([wells_aug, gpd.GeoDataFrame([new_row], crs=wells_aug.crs)], ignore_index=True)
    return dummy_id, wells_aug2

# -----------------------------------------------------------------------------#
# Build deferred requested per-well table
# -----------------------------------------------------------------------------#
def build_deferred_requested_per_well(
    pod_deferred_shp: str = POD_DEFERRED_SHP,
    wells_shp: str = WELLS_SHP,
    pod_current_shp: str = POD_SHP,
    model_domain_shp: str = MODEL_DOMAIN_SHP,
    existing_alloc_csv: str = EXISTING_ALLOC_CSV,
    out_csv: str = OUT_REQ_PER_WELL_CSV,
    out_wells_shp: str = OUT_REQ_WELLS_SHP,
    out_wells_gpkg: str = OUT_REQ_WELLS_GPKG,
    status_keep=("Deferred", "Conditionally Approved"),
    req_col_candidates=("req_acft", "requested_af", "requested_acft", "req_af"),
):
    pod_cur = gpd.read_file(pod_current_shp)
    dep = gpd.read_file(pod_deferred_shp)
    wells = gpd.read_file(wells_shp)

    pod_cur = fix_invalid_polygons(pod_cur)
    dep = fix_invalid_polygons(dep)

    target_crs = pod_cur.crs if pod_cur.crs is not None else dep.crs
    if target_crs is None:
        raise ValueError("Could not determine CRS from POD layers.")

    dep = safe_to_crs(dep, target_crs)
    wells = safe_to_crs(wells, target_crs)

    dep_pod_id_col, dep_permit_col = detect_pod_cols(dep)
    status_col = pick_status_col(dep)
    holder_col = pick_holder_col(dep)
    req_col = first_existing(dep.columns, list(req_col_candidates))
    use_type_col = first_existing(dep.columns, ["use_type", "usetype", "use_typ", "useclass"])
    if use_type_col is None:
        raise ValueError(f"Could not find use_type column in deferred POD shapefile. Columns={list(dep.columns)}")
    if req_col is None:
        raise ValueError(f"Could not find requested-use column. Tried {req_col_candidates}. Columns={list(dep.columns)}")

    dep["_status_norm"] = dep[status_col].astype(str).str.strip()
    dep = dep[dep["_status_norm"].isin(list(status_keep))].copy()

    dep["_pod_id_key"] = dep[dep_pod_id_col].apply(norm_key)
    dep["_permit_key"] = dep[dep_permit_col].apply(norm_key)
    dep["_holder_key"] = dep[holder_col].apply(norm_holder) if holder_col else ""

    dep["_req_acft"] = pd.to_numeric(dep[req_col], errors="coerce").fillna(0.0)
    dep["_use_type"] = dep[use_type_col].astype(str).str.strip()

    # --- NEW: apply use-type scaling (Irrigation -> 20%, others -> 100%) ---
    def _use_scale(u: str) -> float:
        u0 = str(u).strip().lower()
        if u0 == "irrigation":
            return 0.20
        # keep municipal/rural/industrial/unknown at full requested
        return 1.00

    dep["_use_scale"] = dep["_use_type"].apply(_use_scale)
    dep["_req_acft_raw"] = dep["_req_acft"]                 # keep original for QA
    dep["_req_acft"] = dep["_req_acft"] * dep["_use_scale"] # scaled requested AF


    domain_full, domain_inner = _load_domain_inner(model_domain_shp, target_crs, interior_buffer_ft=300.0)

    wells_aug = wells.copy()
    wcols = _detect_cols_for_wells(wells_aug)

    if "is_dummy" not in wells_aug.columns:
        wells_aug["is_dummy"] = 0

    wells_reported_use = _load_reported_use_well_set(existing_alloc_csv)

    wells_min = wells_aug[[wcols["well_id"], "geometry", "is_dummy"]].copy()
    dep_min = dep[["_pod_id_key", "_permit_key", "_status_norm", "_req_acft", "geometry"]].copy()

    wdep = gpd.sjoin(wells_min, dep_min, how="inner", predicate="intersects")
    wells_by_pod = (
        wdep.groupby("_pod_id_key")[wcols["well_id"]]
            .apply(lambda s: sorted(set(map(str, s.astype(str)))))
            .to_dict()
    )

    out_rows = []

    for _, row in dep.iterrows():
        pod_id = str(row["_pod_id_key"])
        permit = str(row["_permit_key"])
        req_total = float(row["_req_acft"])
        use_type = str(row["_use_type"])

        # If requested == 0, we do nothing (no well record)
        if req_total <= 0:
            continue

        well_ids = wells_by_pod.get(pod_id, [])

        if len(well_ids) == 0:
            dummy_id, wells_aug = _ensure_dummy_well_in_domain_for_pod(
                row, wells_aug, wcols["well_id"], domain_full=domain_full, domain_inner=domain_inner
            )
            if dummy_id is not None:
                well_ids = [dummy_id]

        if len(well_ids) == 0:
            continue

        per_well_af = req_total / float(len(well_ids))
        days = 365.0
        cfd = (per_well_af * AF_TO_CF) / days

        wells_aug = _add_xy(wells_aug)

        for wid in well_ids:
            wrec = wells_aug[wells_aug[wcols["well_id"]].astype(str) == str(wid)]
            if wrec.empty:
                x2265=y2265=x2266=y2266=total_dept=top_scr=bot_scr=np.nan
                is_dummy = 0
            else:
                wrec0 = wrec.iloc[0]
                x2265 = wrec0.get("x_2265", np.nan)
                y2265 = wrec0.get("y_2265", np.nan)
                x2266 = wrec0.get("x_2266", np.nan)
                y2266 = wrec0.get("y_2266", np.nan)
                total_dept = wrec0.get(wcols["total_dept"], np.nan) if wcols["total_dept"] else np.nan
                top_scr    = wrec0.get(wcols["top_screen"], np.nan) if wcols["top_screen"] else np.nan
                bot_scr    = wrec0.get(wcols["bottom_scr"], np.nan) if wcols["bottom_scr"] else np.nan
                is_dummy   = int(wrec0.get("is_dummy", 0))
                req_total_raw = float(row["_req_acft_raw"]) if "_req_acft_raw" in dep.columns else float(row["_req_acft"])
                req_total     = float(row["_req_acft"])  # scaled value
                scale         = float(row.get("_use_scale", 1.0))

            if str(wid).startswith("DUMMY_") or is_dummy == 1:
                well_flag = "dummy well"
            else:
                well_flag = "existing well (reported use)" if str(wid) in wells_reported_use else "existing well with new flow"

            out_rows.append({
                "Permits": f"[{permit}]",
                "Pods": f"[{pod_id}]",
                "Well": str(wid),
                "use_type": use_type,
                "x_2265": x2265, "y_2265": y2265,
                "x_2266": x2266, "y_2266": y2266,
                "total_dept": total_dept,
                "top_screen": top_scr,
                "bottom_scr": bot_scr,
                "use_acft": per_well_af,
                "days": days,
                "cfd": cfd,
                "well_flag": well_flag,
                "status": str(row["_status_norm"]),
                "req_acft_raw": req_total_raw,
                "req_acft_scaled": req_total,
                "use_scale": scale,
            })

    out_df = pd.DataFrame(out_rows)

    col_order = [
        "Permits", "Pods", "Well",
        "use_type",
        "x_2265", "y_2265", "x_2266", "y_2266",
        "total_dept", "top_screen", "bottom_scr",
        "use_acft", "days", "cfd",
        "well_flag", "status",
    ]
    for c in col_order:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[col_order].copy()

    out_df = collapse_duplicate_requested_allocations(out_df)

    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote deferred requested per-well CSV: {out_csv}  (rows={len(out_df)})")

    used_wells = set(out_df["Well"].astype(str).unique())
    wells_used = wells_aug[wells_aug[wcols["well_id"]].astype(str).isin(used_wells)].copy()
    if len(wells_used) > 0:
        wells_used = _add_xy(wells_used)
        wells_used["Well"] = wells_used[wcols["well_id"]].astype(str)
        wells_used = wells_used.merge(
            out_df[["Well", "well_flag"]].drop_duplicates(),
            on="Well",
            how="left"
        )

        wells_used.to_file(out_wells_shp, driver="ESRI Shapefile")
        print(f"Wrote deferred wells shapefile: {out_wells_shp}")

        try:
            wells_used.to_file(out_wells_gpkg, layer="deferred_requested_wells", driver="GPKG")
            print(f"Wrote deferred wells geopackage: {out_wells_gpkg}")
        except Exception as e:
            print(f"[warn] Could not write gpkg: {e}")

    return out_df

# -----------------------------------------------------------------------------#
# NEW: MF6 WEL creation utilities
# -----------------------------------------------------------------------------#
def _pick_layer_from_screen_midpoint(gwf, top_screen, bottom_scr, total_dept, i, j):
    top = np.asarray(gwf.dis.top.array, float)
    botm = np.asarray(gwf.dis.botm.array, float)
    nlay = botm.shape[0]

    ts = float(top_screen) if np.isfinite(top_screen) else 0.0
    bs = float(bottom_scr) if np.isfinite(bottom_scr) else 0.0
    td = float(total_dept) if np.isfinite(total_dept) else 0.0

    if bs == 0.0 and td > 0.0:
        bs = td
    if bs == 0.0 and td == 0.0:
        return 1 if nlay >= 2 else 0

    cell_top = float(top[int(i), int(j)])
    mid_depth = 0.5 * (ts + bs)
    mid_elev = cell_top - mid_depth

    for k in range(nlay):
        top_k = cell_top if k == 0 else float(botm[k - 1, int(i), int(j)])
        bot_k = float(botm[k, int(i), int(j)])
        if (mid_elev <= top_k) and (mid_elev >= bot_k):
            return k

    if mid_elev > cell_top:
        return 0
    return nlay - 1

# -----------------------------------------------------------------------------#
# Irrigation signal (matches your existing logic)
# -----------------------------------------------------------------------------#
def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1]

def _normalize_month_weights(w: dict) -> dict:
    s = float(sum(w.values()))
    if s <= 0:
        return {m: 1.0/12.0 for m in range(1, 13)}
    return {int(k): float(v)/s for k, v in w.items()}

def irrigation_weights() -> dict:
    w = {
        1: 0.00, 2: 0.00, 3: 0.05, 4: 0.10, 5: 0.12,
        6: 0.20, 7: 0.22, 8: 0.18, 9: 0.08, 10: 0.04,
        11: 0.01, 12: 0.00
    }
    return _normalize_month_weights(w)

def even_weights() -> dict:
    return {m: 1.0/12.0 for m in range(1, 13)}

def use_type_monthly_weights(use_type: str) -> dict:
    if str(use_type).strip().lower() == "irrigation":
        return irrigation_weights()
    return even_weights()

def monthly_cfd_from_annual_mean_cfd(
    annual_mean_cfd: float,
    *,
    use_type: str,
    year: int,
    month: int,
) -> float:
    if annual_mean_cfd is None:
        return 0.0
    annual_mean_cfd = float(annual_mean_cfd)
    if annual_mean_cfd == 0.0:
        return 0.0

    w = use_type_monthly_weights(use_type)
    wm = float(w.get(int(month), 0.0))
    dim = float(_days_in_month(year, month))
    return annual_mean_cfd * 365.0 * wm / dim

def _sp_to_year_month(sp: int, sp_start: int, base_year: int, base_month: int) -> Tuple[int, int]:
    """
    Map stress period index to (year, month) assuming monthly stress periods.
    sp_start corresponds to base_year/base_month.
    """
    offset = int(sp) - int(sp_start)
    if offset < 0:
        raise ValueError("sp_to_year_month called with sp < sp_start")

    m0 = (base_year * 12 + (base_month - 1)) + offset
    year = m0 // 12
    month = (m0 % 12) + 1
    return int(year), int(month)

def _choose_xy_columns_for_modelgrid(df: pd.DataFrame, mgrid) -> Tuple[str, str]:
    """
    Heuristic to pick which XY columns to use for intersect(), based on model grid extents.
    """
    # model bounds (in model CRS units)
    try:
        xmin, xmax, ymin, ymax = mgrid.extent  # (xmin,xmax,ymin,ymax)
    except Exception:
        # fallback from x/ycellcenters
        xcc = np.asarray(mgrid.xcellcenters)
        ycc = np.asarray(mgrid.ycellcenters)
        xmin, xmax = float(np.nanmin(xcc)), float(np.nanmax(xcc))
        ymin, ymax = float(np.nanmin(ycc)), float(np.nanmax(ycc))

    candidates = []
    for xs, ys in [("x_2265", "y_2265"), ("x_2266", "y_2266")]:
        if xs in df.columns and ys in df.columns:
            x = pd.to_numeric(df[xs], errors="coerce")
            y = pd.to_numeric(df[ys], errors="coerce")
            if x.notna().any() and y.notna().any():
                inx = ((x >= xmin) & (x <= xmax)).mean()
                iny = ((y >= ymin) & (y <= ymax)).mean()
                candidates.append((inx + iny, xs, ys))

    if not candidates:
        return "x_2265", "y_2265"

    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]

# -----------------------------------------------------------------------------#
# UPDATED: Add deferred requested wells as NEW MF6 WEL package (post-build)
# -----------------------------------------------------------------------------#
def add_deferred_wel_package_to_existing_model(
    model_ws=os.path.join("model_ws", "elk_2lay_monthly_clean"),
    deferred_per_well_csv=None,
    package_name="def_wel",
    package_filename="def_wel.wel",
    sp_start=325,
    drop_zero_records=True,
    apply_use_type_monthly_signal=True,
    base_year=2024,
    base_month=1,
    verbose=True,
):
    """
    Load an existing MF6 model and add a new WEL package for deferred requested wells.

    - deferred_per_well_csv: output from build_deferred_requested_per_well()
      Must contain: Well, x_2265, y_2265, cfd, top_screen, bottom_scr, total_dept
      Optional: use_type (used if apply_use_type_monthly_signal=True)
    - If apply_use_type_monthly_signal=True:
        cfd is treated as ANNUAL-MEAN rate (annual_volume/365) and is converted to
        month-specific cfd using use_type weights + days-in-month.
    - Applies pumping from stress period sp_start through end of simulation.
    - Combines duplicates at the MODEL CELL level (k,i,j) by summing Q.
    """

    if deferred_per_well_csv is None:
        raise ValueError("deferred_per_well_csv is required")

    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, verbosity_level=0)
    gwf = sim.get_model()
    if gwf is None:
        raise RuntimeError(f"No GWF model found in simulation at: {model_ws}")

    mgrid = gwf.modelgrid
    idomain = np.asarray(gwf.dis.idomain.array, int)

    df = pd.read_csv(deferred_per_well_csv, dtype={"Well": str})
    req_cols = ["Well", "x_2265", "y_2265", "cfd", "top_screen", "bottom_scr", "total_dept"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Deferred CSV missing required column '{c}'. Columns={list(df.columns)}")

    # optional use_type
    if "use_type" not in df.columns:
        df["use_type"] = ""

    for c in ["x_2265", "y_2265", "x_2266", "y_2266", "cfd", "top_screen", "bottom_scr", "total_dept"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # pick best XY columns for this model grid (handles the "y looks off" cases)
    xcol, ycol = _choose_xy_columns_for_modelgrid(df, mgrid)

    df = df.dropna(subset=[xcol, ycol]).copy()
    if df.empty:
        raise ValueError(f"Deferred CSV has no valid coordinates after filtering (using {xcol}/{ycol}).")

    if drop_zero_records:
        df = df.loc[df["cfd"].fillna(0.0) != 0.0].copy()
        if df.empty:
            raise ValueError("All deferred records have cfd == 0. Nothing to add.")

    # map to (i,j)
    ii, jj, keep_idx = [], [], []
    for idx, r in df.iterrows():
        x = float(r[xcol]); y = float(r[ycol])
        try:
            i, j = mgrid.intersect(x, y)
        except Exception:
            i, j = None, None
        if i is None or j is None:
            continue
        keep_idx.append(idx)
        ii.append(int(i))
        jj.append(int(j))

    df = df.loc[keep_idx].copy()
    df["i"] = ii
    df["j"] = jj

    if df.empty:
        raise ValueError("No deferred wells intersected the model grid (check CRS/extents).")

    # assign layer k
    ks = []
    for _, r in df.iterrows():
        k = _pick_layer_from_screen_midpoint(
            gwf,
            r["top_screen"], r["bottom_scr"], r["total_dept"],
            r["i"], r["j"]
        )
        ks.append(int(k))
    df["k"] = ks

    # drop inactive
    active_mask = []
    for _, r in df.iterrows():
        k, i, j = int(r["k"]), int(r["i"]), int(r["j"])
        active_mask.append(idomain[k, i, j] == 1)
    df = df.loc[active_mask].copy()

    if df.empty:
        raise ValueError("All deferred wells mapped to inactive idomain cells.")

    # number of stress periods
    tdis = sim.get_package("tdis")
    nper = int(tdis.nper.get_data())

    if sp_start < 0 or sp_start >= nper:
        raise ValueError(f"sp_start={sp_start} is outside model range (nper={nper})")

    # build stress_period_data
    wel_spd = {}
    for sp in range(int(sp_start), int(nper)):
        y, m = _sp_to_year_month(sp, sp_start=int(sp_start), base_year=int(base_year), base_month=int(base_month))

        if apply_use_type_monthly_signal:
            df_sp = df.copy()
            df_sp["cfd_eff"] = [
                monthly_cfd_from_annual_mean_cfd(float(r.cfd), use_type=str(r.use_type), year=y, month=m)
                for r in df_sp.itertuples(index=False)
            ]
        else:
            df_sp = df.copy()
            df_sp["cfd_eff"] = df_sp["cfd"].astype(float)

        # MF6 convention: extraction negative
        df_sp["q_cfd"] = -df_sp["cfd_eff"].astype(float).fillna(0.0)

        # sum by cell
        grouped = (
            df_sp.groupby(["k", "i", "j"], dropna=False)["q_cfd"]
                .sum(min_count=1)
                .reset_index()
        )

        per_recs = [((int(r.k), int(r.i), int(r.j)), float(r.q_cfd)) for r in grouped.itertuples(index=False)]
        per_recs = [rec for rec in per_recs if np.isfinite(rec[1])]
        wel_spd[sp] = per_recs

    if verbose:
        # quick summary using first SP
        g0 = pd.DataFrame(wel_spd[int(sp_start)], columns=["cell", "q_cfd"])
        print(f"[{package_name}] XY columns used: {xcol}, {ycol}")
        print(f"[{package_name}] raw wells: {len(df):,}")
        print(f"[{package_name}] unique cells in first SP: {len(g0):,}")
        print(f"[{package_name}] applying SP {sp_start} .. {nper-1} ({nper-sp_start} periods)")
        print(f"[{package_name}] total Q first SP (cfd, negative=extraction): {g0['q_cfd'].sum():,.3f}")
        print(f"[{package_name}] monthly signal enabled: {apply_use_type_monthly_signal} (base {base_year}-{base_month:02d})")

    # replace if present
    existing = gwf.get_package(package_name)
    if existing is not None:
        gwf.remove_package(existing)

    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd,
        pname=package_name,
        filename=package_filename,
        save_flows=True,
        auto_flow_reduce=0.1,
    )

    sim.set_all_data_external()
    sim.write_simulation()

    # Add AUTO_FLOW_REDUCE_CSV to WEL
    wel_file = os.path.join(model_ws, "def_wel.wel")
    with open(wel_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('END options'):
            new_lines.append('  AUTO_FLOW_REDUCE_CSV FILEOUT defwel_auto_flow_reduce.csv\n')
        new_lines.append(line)
    with open(wel_file, 'w') as f:
        f.writelines(new_lines)

    return df

# -----------------------------------------------------------------------------#
# CLI / main
# -----------------------------------------------------------------------------#
def main(model_ws=os.path.join("model_ws", "elk_2lay_monthly")):
    # 1) build the deferred requested per-well table
    out_df = build_deferred_requested_per_well()

    # 2) add the MF6 WEL package to the existing model (monthly signal on)
    add_deferred_wel_package_to_existing_model(
        model_ws=model_ws,
        deferred_per_well_csv=OUT_REQ_PER_WELL_CSV,
        package_name="def_wel",
        package_filename="def_wel.wel",
        sp_start=325,                  # Jan 2024
        drop_zero_records=True,
        apply_use_type_monthly_signal=True,
        base_year=2024,
        base_month=1,
        verbose=True,
    )

if __name__ == "__main__":
    main()