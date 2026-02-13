# -*- coding: utf-8 -*-
"""
Elk Valley water-use → per-well flow-rate allocation

What this script does
---------------------
1) Loads wells, PODs, Elk boundary, monthly (permit-level) and annual (POD-level) water-use.
2) Repairs annual TSVs that may have newline-continuations in a text column.
3) Builds robust joins (CRS alignment, geometry fixes) and a POD→Wells map,
   handling PODs with no intersecting wells using `data/raw/water_use/pod_with_no_wells.csv`.
   - If assigned_pod_1..5 are provided, use ALL wells from those assigned PODs.
   - If none are provided, creates a dummy well at the centroid of the POD polygon.
4) Computes water-use series:
   - Monthly (by PERMIT) → AF per (permit, year, [month?])
   - Annual (by POD)     → AF per (permit, pod, year, use_type)
5) Allocation logic:
   - Prefer MONTHLY values for a (permit, year) **if** the monthly total >= annual total across its PODs.
     If monthly total < annual total for that (permit, year), fall back to annual.
   - If MONTHLY=True:
       * years ≤ 1999 → annual rates (Month=NaN)
       * years ≥ 2000 → monthly rates
     (but still subject to the monthly-vs-annual comparison above)
   - If MONTHLY=False:
       * always use annual (with annual→monthly expansion only for the “monthly_all” export described below).
6) Distribute use to wells:
   - For monthly-by-permit: find all PODs for that permit; union their wells; divide evenly among those wells.
   - For annual-by-pod: divide the POD’s yearly AF evenly among the wells mapped to that POD
     (including assigned POD wells or dummy well if necessary).
7) Annual→Monthly expansion (for the “monthly_all” export):
   - use_type ∈ {Municipal, Rural Water, Industrial} → even split over 12 months
   - use_type == Irrigation → weighted schedule (no use in Dec/Jan/Feb; heavier in Jun/Jul/Aug)
8) Outputs
   A) Main allocation table honoring MONTHLY flag & rules (pre-2000 annual, 2000+ monthly if MONTHLY=True).
   B) If MONTHLY_ALL=True, a separate *fully monthly* table over all years, expanding annuals via use_type schedule.
   Both include columns:
      Permits (list), Pods (list), Year, Month (NaN for true-annual rows), Well, x_2265, y_2265, x_2266, y_2266,
      total_dept, top_screen, bottom_scr, use_acft, days, cfd

Notes:
- Assumes EPSG:2265 and EPSG:2266 are desired outputs for well coordinates.
- Dummy wells get Well IDs like "DUMMY_<podid>" and blank attributes where unknown.
- Be liberal with column-name detection via `first_existing`.
"""

import os
import io
import math
import calendar
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from glob import glob
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
# Config flags (you can also pass from main())
# -----------------------------------------------------------------------------#
MONTHLY = True      # if True: annual pre-2000 (Month=NaN), monthly 2000+ (subject to monthly-vs-annual rule)
MONTHLY_ALL = True  # if True: also write a fully-monthly export across all years, expanding annuals
VERBOSE = True

# Constants
AF_GAL = 325_851.429                # gallons per acre-foot
AF_TO_CF = 43_560.0                 # cubic feet per acre-foot
GPM_PER_CFD = 7.48051948 / 1440.0  # convert cubic feet/day -> gallons/min
CRS_2265 = "EPSG:2265"
CRS_2266 = "EPSG:2266"

# -----------------------------------------------------------------------------#
# Paths (adjust as needed)
# -----------------------------------------------------------------------------#
GIS_PATH   = os.path.join("..", "..", "gis", "input_shps", "elk", "water_use")
WELLS_SHP  = os.path.join(GIS_PATH, "Elk_Valley_aquifer_water_use_wells.shp")
POD_SHP    = os.path.join(GIS_PATH, "2025-09-22_Elk_Valley_aquifer_PermitPOD.shp")
ELK_BOUND  = os.path.join("..", "..", "gis", "input_shps", "elk", "elk_boundary.shp")

DATA_PATH         = os.path.join("data", "raw", "water_use")
MONTHLY_PATH      = os.path.join(DATA_PATH, "Elk Valley_Water use monthly gallons PerPermit.txt")
ANNUAL_PATH       = os.path.join(DATA_PATH, "Elk Valley_Water use annual Acft PerPOD.txt")
PODS_NOWELLS_CSV  = os.path.join(DATA_PATH, "pods_with_no_wells.csv")  

OUT_DIR           = os.path.join("data", "processed", "water_use")
FIG_TS_DIR        = os.path.join("figs", "permit_vs_pod")
FIG_MAP_DIR       = os.path.join("figs", "permit_pod_maps")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(FIG_TS_DIR).mkdir(parents=True, exist_ok=True)
Path(FIG_MAP_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------#
# Utils
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

def fix_invalid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        return fix_invalid_polygons(gdf)
    return gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

def safe_to_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if target_crs is None or gdf.crs == target_crs:
        return gdf
    if gdf.crs is None:
        # We cannot safely assume a CRS; require user to pre-define if truly unknown.
        # Here, we pass-through to avoid accidental wrong reprojection.
        return gdf
    return gdf.to_crs(target_crs)

def repair_tsv_with_continuations(src_path, dst_path=None, sep="\t", encoding="utf-8-sig") -> Tuple[str, int]:
    """
    Repairs a TSV where the last column may contain unquoted newlines.
    """
    p = Path(src_path)
    if dst_path is None:
        dst_path = p.with_suffix(p.suffix + ".cleaned")

    raw = p.read_bytes().replace(b"\x00", b"")
    text = raw.decode(encoding, errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or (len(lines) == 1 and lines[0] == ""):
        raise ValueError("Input appears empty.")

    header = lines[0]
    expected_cols = header.count(sep) + 1
    expected_seps = expected_cols - 1
    maxsplit = expected_seps

    out_lines = [header + "\n"]
    i = 1
    while i < len(lines):
        buf = lines[i]
        if buf == "" and i == len(lines) - 1:
            break

        j = i + 1
        while True:
            have_cols = len(buf.split(sep, maxsplit)) >= expected_cols
            if j >= len(lines):
                break

            next_line = lines[j]
            next_seps = next_line.count(sep)

            if next_line != header and next_seps < expected_seps:
                buf = buf + "\n" + next_line
                j += 1
                continue

            if not have_cols:
                buf = buf + "\n" + next_line
                j += 1
                continue
            break

        parts = buf.split(sep, maxsplit)
        if len(parts) < expected_cols:
            parts += [""] * (expected_cols - len(parts))

        fixed = [p_.strip() for p_ in parts[:expected_cols - 1]]
        comment = parts[-1].strip().replace("\n", " ")
        out_lines.append(sep.join(fixed + [comment]) + "\n")
        i = j

    Path(dst_path).write_text("".join(out_lines), encoding=encoding, newline="\n")
    return str(dst_path), expected_cols

def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(int(year)) else 365

def month_len(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1]

def to_num_or_nan(x) -> float:
    return pd.to_numeric(str(x).replace(",", "").strip(), errors="coerce")

def list_bracket(items: Iterable) -> str:
    # Render as [a, b, c]
    arr = [str(i) for i in items if pd.notna(i)]
    return f"[{', '.join(arr)}]"

def _parse_bracket_list(s: str) -> list:
    """Turn strings like "[1563, 1795]" into list[str]."""
    if pd.isna(s):
        return []
    s = str(s).strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [p.strip() for p in inner.split(",") if p.strip()]
    return [s] if s else []

def _merge_bracket_lists(values: pd.Series) -> str:
    """Union+sort all bracket-list strings in a series and return as bracket string."""
    acc = set()
    for v in values.dropna():
        acc.update(_parse_bracket_list(v))
    def _key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))
    merged = sorted(acc, key=_key)
    return f"[{', '.join(map(str, merged))}]"

def _first_non_null(series: pd.Series):
    notna = series.dropna()
    return notna.iloc[0] if len(notna) else np.nan

def _days_for_row(year: int, month):
    return days_in_year(int(year)) if pd.isna(month) else month_len(int(year), int(month))

def collapse_duplicate_allocations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per (Well, Year, Month) by summing AF and recomputing cfd.
    Also merges 'Permits' and 'Pods' lists and carries first non-null attributes.
    """
    if df.empty:
        return df.copy()

    d = df.copy()
    # treat Month NaN as 0 for grouping, then restore
    d["_MonthKey"] = d["Month"].fillna(0).astype(int)

    agg = (
        d.groupby(["Well", "Year", "_MonthKey"], dropna=False)
         .apply(lambda g: pd.Series({
             "Permits": _merge_bracket_lists(g["Permits"]) if "Permits" in g else "[]",
             "Pods":    _merge_bracket_lists(g["Pods"]) if "Pods" in g else "[]",
             "x_2265":  _first_non_null(g.get("x_2265", pd.Series(dtype=float))),
             "y_2265":  _first_non_null(g.get("y_2265", pd.Series(dtype=float))),
             "x_2266":  _first_non_null(g.get("x_2266", pd.Series(dtype=float))),
             "y_2266":  _first_non_null(g.get("y_2266", pd.Series(dtype=float))),
             "total_dept": _first_non_null(g.get("total_dept", pd.Series(dtype=float))),
             "top_screen": _first_non_null(g.get("top_screen", pd.Series(dtype=float))),
             "bottom_scr": _first_non_null(g.get("bottom_scr", pd.Series(dtype=float))),
             "use_acft": g["use_acft"].sum(min_count=1)
         }))
         .reset_index()
    )

    agg["Month"] = agg["_MonthKey"].replace(0, np.nan)
    agg.drop(columns=["_MonthKey"], inplace=True)
    agg["Year"] = pd.to_numeric(agg["Year"], errors="coerce").astype("Int64")
    agg["days"] = [
        _days_for_row(int(y), m) if pd.notna(y) else np.nan
        for y, m in zip(agg["Year"], agg["Month"])
    ]
    agg["cfd"] = (agg["use_acft"] * AF_TO_CF) / agg["days"].replace({0: np.nan})

    cols_order = ["Permits","Pods","Year","Month","Well",
                  "x_2265","y_2265","x_2266","y_2266",
                  "total_dept","top_screen","bottom_scr",
                  "use_acft","days","cfd"]
    cols_order = [c for c in cols_order if c in agg.columns] + [c for c in agg.columns if c not in cols_order]
    return agg[cols_order]

# -----------------------------------------------------------------------------#
# Load & prep data
# -----------------------------------------------------------------------------#
def load_layers() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    elk_bound = gpd.read_file(ELK_BOUND)
    wells = gpd.read_file(WELLS_SHP)
    pod = gpd.read_file(POD_SHP)

    wells = fix_invalid(wells)
    pod = fix_invalid_polygons(pod)

    if wells.crs is None or pod.crs is None:
        raise ValueError("wells.crs or pod.crs is None. Please set CRS before joining.")

    # Align to elk boundary CRS if available, else leave as-is
    target_crs = elk_bound.crs if elk_bound is not None else pod.crs
    elk_bound = safe_to_crs(elk_bound, target_crs)
    pod = safe_to_crs(pod, target_crs)
    wells = safe_to_crs(wells, target_crs)

    return elk_bound, wells, pod

def load_monthly_and_annual() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # monthly (permit-level gallons)
    monthly = pd.read_csv(MONTHLY_PATH, sep="\t")
    # annual (pod-level AF) w/ repair step
    cleaned, _ = repair_tsv_with_continuations(ANNUAL_PATH)
    annual = pd.read_csv(cleaned, sep="\t", dtype=str, engine="python")
    return monthly, annual

def load_pods_no_wells_csv() -> pd.DataFrame:
    if not Path(PODS_NOWELLS_CSV).exists():
        return pd.DataFrame(columns=["pod", "permit_num",
                                     "assigned_pod_1", "assigned_pod_2", "assigned_pod_3",
                                     "assigned_pod_4", "assigned_pod_5"])
    df = pd.read_csv(PODS_NOWELLS_CSV, dtype=str)
    # normalize column names
    needed = ["pod", "permit_num",
              "assigned_pod_1", "assigned_pod_2", "assigned_pod_3", "assigned_pod_4", "assigned_pod_5"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    for c in needed:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "nan", "None"]), c] = np.nan
    return df[needed].copy()

# -----------------------------------------------------------------------------#
# Column name discovery
# -----------------------------------------------------------------------------#
def detect_pod_cols(pod_g: gpd.GeoDataFrame) -> Tuple[str, str]:
    pod_id_col = first_existing(pod_g.columns, ["pod", "pod_id", "podnum", "pod_num", "podname", "pod_name"])
    permit_col = first_existing(pod_g.columns, ["permit_number", "permit_num", "permit", "permit_no", "per_num"])
    if pod_id_col is None:
        raise ValueError(f"Could not find a POD identifier column in POD shapefile. Columns={list(pod_g.columns)}")
    if permit_col is None:
        raise ValueError(f"Could not find a PERMIT column in POD shapefile. Columns={list(pod_g.columns)}")
    return pod_id_col, permit_col

def detect_well_cols(wells_g: gpd.GeoDataFrame) -> Dict[str, Optional[str]]:
    return {
        "well_id": first_existing(wells_g.columns, ["site_locat", "well_no", "wellid", "well_id", "wellname"]),
        "total_dept": first_existing(wells_g.columns, ["total_dept", "total_depth", "tot_depth", "totaldepth"]),
        "top_screen": first_existing(wells_g.columns, ["top_screen", "top_scr", "topscreen"]),
        "bottom_scr": first_existing(wells_g.columns, ["bottom_scr", "bottom_screen", "botscreen", "bot_scr"]),
    }

def _ensure_dummy_well_for_pod(
    pod_g: gpd.GeoDataFrame, pod_id: str, wells_aug: gpd.GeoDataFrame, well_id_col: str, pod_id_col: str
) -> Tuple[List[str], gpd.GeoDataFrame]:
    if VERBOSE:
        print(f"[diagnose:dummy] REQUEST to create dummy for POD='{pod_id}'")

    sel = pod_g[pod_g[pod_id_col].astype(str).str.strip() == str(pod_id).strip()]

    if VERBOSE:
        print(f"[diagnose:dummy]  POD lookup by '{pod_id_col}': "
              f"rows={len(sel)}, any_geom_na={sel.geometry.isna().any() if not sel.empty else 'n/a'}")

    if sel.empty or sel.geometry.isna().all():
        if VERBOSE:
            print(f"[diagnose:dummy]  ABORT: POD '{pod_id}' not found or geometry missing.")
        return [], wells_aug

    # Try to fix invalid polys (in case centroid fails)
    if VERBOSE and not sel.is_valid.all():
        print(f"[diagnose:dummy]  WARNING: invalid geometry for POD '{pod_id}'. Attempting buffer(0).")
    try:
        geom_union = sel.geometry.buffer(0).unary_union if not sel.is_valid.all() else sel.geometry.unary_union
    except Exception as e:
        if VERBOSE:
            print(f"[diagnose:dummy]  ERROR: union/buffer(0) failed for POD '{pod_id}': {e}")
        geom_union = sel.geometry.unary_union  # fallback

    try:
        centroid = geom_union.centroid
    except Exception as e:
        if VERBOSE:
            print(f"[diagnose:dummy]  ERROR: centroid() failed for POD '{pod_id}': {e}")
        return [], wells_aug

    dummy_id = f"DUMMY_{pod_id}"
    if VERBOSE:
        try:
            x, y = float(centroid.x), float(centroid.y)
        except Exception:
            x = y = float("nan")
        print(f"[diagnose:dummy]  Creating dummy well id='{dummy_id}' at centroid=({x:.3f}, {y:.3f}), "
              f"CRS={wells_aug.crs}")

    new_row = {well_id_col: dummy_id, "geometry": centroid}
    for extra in ["total_dept", "top_screen", "bottom_scr", "site_locat", "well_no"]:
        if extra in wells_aug.columns:
            new_row[extra] = np.nan

    try:
        wells_aug = pd.concat(
            [wells_aug, gpd.GeoDataFrame([new_row], geometry="geometry", crs=wells_aug.crs)],
            ignore_index=True
        )
    except Exception as e:
        if VERBOSE:
            print(f"[diagnose:dummy]  ERROR: appending dummy well '{dummy_id}' failed: {e}")
        return [], wells_aug

    if VERBOSE:
        print(f"[diagnose:dummy]  SUCCESS: dummy well '{dummy_id}' appended. New wells count={len(wells_aug)}")
    return [dummy_id], wells_aug
# -----------------------------------------------------------------------------#
# Build POD→Wells mapping (including assigned PODs & dummy wells)
# -----------------------------------------------------------------------------#

def distribute_monthly_permit_annualized(
    per_month_af: pd.DataFrame,      # columns: permit, year, month, af (subset for 1 permit+year)
    permit_to_pods: Dict[str, Set[str]],
    pod_to_wells: Dict[str, List[str]],
    wells_aug: gpd.GeoDataFrame,
    well_cols: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Collapse monthly-permit AF to ANNUAL per well (Month=NaN, days=365/366).
    Evenly divides the total annual AF across all wells in any POD for that permit.
    """
    if per_month_af.empty:
        return pd.DataFrame(columns=["Permits","Pods","Year","Month","Well","x_2265","y_2265","x_2266","y_2266",
                                     "total_dept","top_screen","bottom_scr","use_acft","days","cfd"])

    wcols = detect_output_well_cols(wells_aug, well_cols)
    permit = str(per_month_af["permit"].iloc[0])
    year   = int(per_month_af["year"].iloc[0])

    # PODs and union of wells like the monthly path
    pods = sorted(permit_to_pods.get(permit, set()))
    wells_u: Set[str] = set()
    for p in pods:
        wells_u.update(pod_to_wells.get(str(p), []))
    wells_list = sorted(wells_u)
    if len(wells_list) == 0:
        return pd.DataFrame(columns=["Permits","Pods","Year","Month","Well","x_2265","y_2265","x_2266","y_2266",
                                     "total_dept","top_screen","bottom_scr","use_acft","days","cfd"])

    total_af = float(per_month_af["af"].fillna(0).sum())
    if total_af <= 0:
        # nothing to allocate
        return pd.DataFrame(columns=["Permits","Pods","Year","Month","Well","x_2265","y_2265","x_2266","y_2266",
                                     "total_dept","top_screen","bottom_scr","use_acft","days","cfd"])

    n = len(wells_list)
    per_well_af = total_af / n
    dyear = days_in_year(year)
    cfd = (per_well_af * AF_TO_CF) / dyear

    out_rows = []
    for wid in wells_list:
        wrec = wells_aug[wells_aug[wcols["well_id"]].astype(str) == str(wid)]
        if wrec.empty:
            x2265=y2265=x2266=y2266=total_dept=top_scr=bot_scr=np.nan
        else:
            wrec = wrec.iloc[0]
            x2265 = wrec.get("x_2265", np.nan)
            y2265 = wrec.get("y_2265", np.nan)
            x2266 = wrec.get("x_2266", np.nan)
            y2266 = wrec.get("y_2266", np.nan)
            total_dept = wrec.get(wcols["total_dept"], np.nan)
            top_scr    = wrec.get(wcols["top_screen"], np.nan)
            bot_scr    = wrec.get(wcols["bottom_scr"], np.nan)

        out_rows.append({
            "Permits": list_bracket([permit]),
            "Pods": list_bracket(pods),
            "Year": year,
            "Month": np.nan,          # annualized
            "Well": str(wid),
            "x_2265": x2265, "y_2265": y2265, "x_2266": x2266, "y_2266": y2266,
            "total_dept": total_dept, "top_screen": top_scr, "bottom_scr": bot_scr,
            "use_acft": per_well_af,
            "days": dyear,
            "cfd": cfd
        })
    return pd.DataFrame(out_rows)


def build_pod_to_wells_map(
    wells_g: gpd.GeoDataFrame,
    pod_g: gpd.GeoDataFrame,
    pods_no_wells: pd.DataFrame,
    well_cols: Dict[str, Optional[str]],
    pod_id_col: str,
    base_join_predicate: str = "intersects",   # keep your current predicate
) -> Tuple[Dict[str, List[str]], gpd.GeoDataFrame]:
    if VERBOSE:
        print(f"[diagnose:map] Building POD→Wells map with predicate='{base_join_predicate}'")

    wells_aug = wells_g.copy()
    well_id_col = well_cols["well_id"] if well_cols["well_id"] else "_autowellid"
    if well_cols["well_id"] is None:
        wells_aug = wells_aug.reset_index().rename(columns={"index": "_autowellid"})
        if VERBOSE:
            print("[diagnose:map] No explicit well-id column; using '_autowellid'")

    # Base mapping by spatial join
    pod_geom = pod_g[[pod_id_col, "geometry"]].copy()
    join = gpd.sjoin(
        wells_aug[[well_id_col, "geometry"]],
        pod_geom,
        how="left",
        predicate=base_join_predicate,
    )

    # Summarize base hits
    pod_to_wells: Dict[str, List[str]] = {}
    for pod_id, g in join.groupby(pod_id_col, dropna=True):
        pod_to_wells[str(pod_id).strip()] = sorted(g[well_id_col].astype(str).unique())

    if VERBOSE:
        n_with_hits = sum(1 for k, v in pod_to_wells.items() if len(v) > 0)
        print(f"[diagnose:map] Base map built: PODs with ≥1 well={n_with_hits}, "
              f"PODs total(with key)={len(pod_to_wells)}")

    # Normalize the CSV columns lightly (no logic change — just diagnostics)
    pods_no_wells = pods_no_wells.copy()
    for c in ["pod", "permit_num", "assigned_pod_1", "assigned_pod_2", "assigned_pod_3", "assigned_pod_4", "assigned_pod_5"]:
        if c not in pods_no_wells.columns:
            pods_no_wells[c] = np.nan
        pods_no_wells[c] = pods_no_wells[c].astype(str).str.strip()

    created_dummy = []
    reassigned_from_assigned = []
    rows_total = len(pods_no_wells)

    if VERBOSE:
        print(f"[diagnose:map] pods_with_no_wells.csv rows={rows_total}")

    for idx, row in pods_no_wells.iterrows():
        p = str(row["pod"]).strip()
        assigned = [str(row[c]).strip() for c in [f"assigned_pod_{i}" for i in range(1,6)]
                    if c in pods_no_wells.columns and row[c] not in [None, "", "nan", "NaN"]]

        if VERBOSE:
            print(f"[diagnose:map] Row {idx+1}/{rows_total}: POD='{p}', assigned={assigned if assigned else '[]'}")

        if len(assigned) > 0:
            # gather wells from all assigned pods
            wells_list = []
            for ap in assigned:
                wells_list.extend(pod_to_wells.get(str(ap).strip(), []))
            wells_list = sorted(set(wells_list))

            if VERBOSE:
                print(f"[diagnose:map]  Assigned → gathered wells={len(wells_list)} "
                      f"(from {len(assigned)} assigned pods)")

            if len(wells_list) == 0:
                if VERBOSE:
                    print(f"[diagnose:map]  Assigned pods had no wells; creating dummy for POD='{p}'")
                wells_list, wells_aug = _ensure_dummy_well_for_pod(pod_g, p, wells_aug, well_id_col, pod_id_col)
                if wells_list and wells_list[0].startswith("DUMMY_"):
                    created_dummy.append(p)
            else:
                reassigned_from_assigned.append(p)

            pod_to_wells[p] = wells_list

        else:
            # No assigned pods → force dummy
            if VERBOSE:
                print(f"[diagnose:map]  No assigned pods for '{p}' — creating dummy.")
            wells_list, wells_aug = _ensure_dummy_well_for_pod(pod_g, p, wells_aug, well_id_col, pod_id_col)
            if wells_list and wells_list[0].startswith("DUMMY_"):
                created_dummy.append(p)
            pod_to_wells[p] = wells_list

    if VERBOSE:
        print(f"[diagnose:map] SUMMARY: created_dummy_for={created_dummy}")
        print(f"[diagnose:map]          reassigned_from_assigned={reassigned_from_assigned}")
        print(f"[diagnose:map]          total_dummy_created={len(created_dummy)}")

    return pod_to_wells, wells_aug


def _ensure_dummy_well_for_pod(
    pod_g: gpd.GeoDataFrame, pod_id: str, wells_aug: gpd.GeoDataFrame, well_id_col: str, pod_id_col: str
) -> Tuple[List[str], gpd.GeoDataFrame]:
    sel = pod_g[pod_g[pod_id_col].astype(str).str.strip() == str(pod_id).strip()]
    if sel.empty or sel.geometry.isna().all():
        return [], wells_aug

    # use representative_point so it's inside the polygon
    try:
        geom = sel.geometry
        if not getattr(geom, "is_valid", pd.Series([True]*len(sel))).all():
            geom = geom.buffer(0)
        rep_pt = geom.representative_point().unary_union
    except Exception:
        # fallback to centroid
        rep_pt = sel.geometry.unary_union.centroid

    dummy_id = f"DUMMY_{pod_id}"
    new_row = {well_id_col: dummy_id, "geometry": rep_pt}

    # Only fill extras that are NOT the id column
    for extra in ["total_dept", "top_screen", "bottom_scr", "site_locat", "well_no"]:
        if extra != well_id_col and extra in wells_aug.columns:
            new_row[extra] = np.nan

    wells_aug = pd.concat(
        [wells_aug, gpd.GeoDataFrame([new_row], geometry="geometry", crs=wells_aug.crs)],
        ignore_index=True
    )
    return [dummy_id], wells_aug

# -----------------------------------------------------------------------------#
# Monthly (permit-level) AF by (permit, year[, month])
# -----------------------------------------------------------------------------#
def compute_monthly_af_by_permit(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()

    # columns
    permit_col = first_existing(m.columns, ["permit_number", "permit_num", "permit", "permit_no"])
    year_col   = first_existing(m.columns, ["use_year", "year"])
    total_use  = first_existing(m.columns, ["total_use", "annual_total_gallons"])

    month_cols = [c for c in [
        "use_january","use_february","use_march","use_april","use_may","use_june",
        "use_july","use_august","use_september","use_october","use_november","use_december"
    ] if c in m.columns]

    if permit_col is None or year_col is None:
        raise ValueError("Could not find permit/year columns in monthly table.")

    # normalize keys
    m["permit_number_key"] = m[permit_col].astype(str).str.strip()
    m["use_year"] = pd.to_numeric(m[year_col], errors="coerce").astype("Int64")

    # numericize months & total_use (gallons)
    for c in month_cols:
        m[c + "_num"] = m[c].apply(to_num_or_nan)
    if total_use:
        m["total_use_num"] = m[total_use].apply(to_num_or_nan)
    else:
        m["total_use_num"] = np.nan

    # Annual gallons from months (fallback to total_use)
    m["sum_months_gal"] = m[[c + "_num" for c in month_cols]].sum(axis=1, min_count=1) if month_cols else np.nan
    m["annual_gal"] = m["sum_months_gal"].fillna(m["total_use_num"])
    m["annual_af_from_monthly"] = m["annual_gal"] / AF_GAL

    # PERMIT-YEAR aggregated AF (if multiple rows exist)
    monthly_af_by_permit_year = (
        m.groupby(["permit_number_key","use_year"], dropna=False)["annual_af_from_monthly"]
         .sum()
         .reset_index()
         .rename(columns={"annual_af_from_monthly":"permit_af_from_months"})
    )

    # Optional per-month breakdown if all 12 month columns are present
    per_month_long = None
    if len(month_cols) > 0:
        month_map = {
            "use_january": 1, "use_february": 2, "use_march": 3, "use_april": 4,
            "use_may": 5, "use_june": 6, "use_july": 7, "use_august": 8,
            "use_september": 9, "use_october": 10, "use_november": 11, "use_december": 12
        }
        present_months = [c for c in month_cols if c in month_map]
        wide = m[["permit_number_key", "use_year"] + [c + "_num" for c in present_months]].copy()
        # melt gallons; convert to AF
        long = wide.melt(id_vars=["permit_number_key","use_year"], var_name="month_col", value_name="gal")
        long["month"] = long["month_col"].str.replace("_num","",regex=False).map(month_map)
        long["af"] = long["gal"] / AF_GAL
        per_month_long = long.dropna(subset=["month"]) \
                             .drop(columns=["month_col","gal"]) \
                             .rename(columns={"permit_number_key":"permit", "use_year":"year"})

    return monthly_af_by_permit_year, per_month_long

# -----------------------------------------------------------------------------#
# Annual (pod-level) AF by (permit, pod, year, use_type)
# -----------------------------------------------------------------------------#
def compute_annual_pod_af(annual: pd.DataFrame) -> pd.DataFrame:
    a = annual.copy()
    # detect columns
    permit_col  = first_existing(a.columns, ["permit_number", "permit_num", "permit", "permit_no"])
    pod_col     = first_existing(a.columns, ["pod", "pod_id", "podnum", "pod_num", "podname", "pod_name"])
    year_col    = first_existing(a.columns, ["use_year", "year"])
    acft_col    = first_existing(a.columns, ["reported_acft", "acft", "reported_acft_num"])
    use_typecol = first_existing(a.columns, ["use_type", "usetype", "usecategory"])

    if permit_col is None or pod_col is None or year_col is None or acft_col is None:
        raise ValueError("Annual table is missing required columns.")

    a["permit_number_key"] = a[permit_col].astype(str).str.strip()
    a["pod_key"] = a[pod_col].astype(str).str.strip()
    a["use_year"] = pd.to_numeric(a[year_col], errors="coerce").astype("Int64")

    rep = a[acft_col].astype(str).str.replace(",", "", regex=False)
    rep_num = pd.to_numeric(rep.str.extract(r"(-?\d*\.?\d+)", expand=False), errors="coerce")
    a["reported_acft_num"] = rep_num.fillna(0.0)

    if use_typecol is None:
        a["use_type"] = "Unknown"
    else:
        a["use_type"] = a[use_typecol].astype(str).str.strip().replace({"": "Unknown"})

    # Keep only needed cols
    annual_pod_af = a[["permit_number_key","pod_key","use_year","reported_acft_num","use_type"]].dropna(subset=["use_year"])
    return annual_pod_af

# -----------------------------------------------------------------------------#
# Choose monthly vs annual per (permit, year)
# -----------------------------------------------------------------------------#
def build_monthly_vs_annual_preference(monthly_af_by_permit_year: pd.DataFrame,
                                       annual_pod_af: pd.DataFrame) -> pd.DataFrame:
    # sum of POD AF by (permit, year)
    sum_pods = (
        annual_pod_af.groupby(["permit_number_key","use_year"], dropna=False)["reported_acft_num"]
                     .sum()
                     .reset_index()
                     .rename(columns={"reported_acft_num":"sum_pods_af"})
    )
    comp = monthly_af_by_permit_year.merge(sum_pods, how="inner", on=["permit_number_key","use_year"])
    comp["prefer_monthly"] = comp["permit_af_from_months"] >= comp["sum_pods_af"]
    return comp  # columns: permit_number_key, use_year, permit_af_from_months, sum_pods_af, prefer_monthly

# -----------------------------------------------------------------------------#
# Annual → monthly weighting schedule
# -----------------------------------------------------------------------------#
def irrigation_weights() -> Dict[int, float]:
    """
    Month weights for irrigation (sum to 1).
    No use in Dec, Jan, Feb. Heaviest in Jun/Jul/Aug.
    Adjust as needed.
    """
    w = {
        1: 0.00, 2: 0.00, 3: 0.05, 4: 0.10, 5: 0.12,
        6: 0.20, 7: 0.22, 8: 0.18, 9: 0.08, 10: 0.04,
        11: 0.01, 12: 0.00
    }
    # normalize to sum=1 (guard if edited)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 0 else {m: (1/12) for m in range(1,13)}

def even_weights() -> Dict[int, float]:
    return {m: 1.0/12.0 for m in range(1,13)}

def use_type_monthly_weights(use_type: str) -> Dict[int, float]:
    if str(use_type).strip().lower() == "irrigation":
        return irrigation_weights()
    # Municipal, Rural Water, Industrial, Unknown → even split
    return even_weights()

# -----------------------------------------------------------------------------#
# Coordinate helpers
# -----------------------------------------------------------------------------#
def add_xy_columns(wells_g: gpd.GeoDataFrame, cols_map: Dict[str, Optional[str]]) -> gpd.GeoDataFrame:
    """
    Adds x_2265, y_2265, x_2266, y_2266 columns based on well geometry.
    """
    g = wells_g.copy()
    # 2265
    try:
        g_2265 = safe_to_crs(g, CRS_2265)
        x2265 = g_2265.geometry.x
        y2265 = g_2265.geometry.y
    except Exception:
        x2265 = pd.Series([np.nan]*len(g), index=g.index)
        y2265 = pd.Series([np.nan]*len(g), index=g.index)

    # 2266
    try:
        g_2266 = safe_to_crs(g, CRS_2266)
        x2266 = g_2266.geometry.x
        y2266 = g_2266.geometry.y
    except Exception:
        x2266 = pd.Series([np.nan]*len(g), index=g.index)
        y2266 = pd.Series([np.nan]*len(g), index=g.index)

    g["x_2265"] = x2265
    g["y_2265"] = y2265
    g["x_2266"] = x2266
    g["y_2266"] = y2266
    return g

# -----------------------------------------------------------------------------#
# Distribution builders
# -----------------------------------------------------------------------------#
def distribute_monthly_permit(
    per_month_af: pd.DataFrame,      # columns: permit, year, month, af
    permit_to_pods: Dict[str, Set[str]],
    pod_to_wells: Dict[str, List[str]],
    wells_aug: gpd.GeoDataFrame,
    well_cols: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Returns rows with per-well monthly allocations for all (permit,year,month) in per_month_af.
    """
    wcols = detect_output_well_cols(wells_aug, well_cols)
    out_rows = []
    for (permit, year), g1 in per_month_af.groupby(["permit","year"]):
        pods = sorted(permit_to_pods.get(str(permit), set()))
        # union wells across pods
        wells_u: Set[str] = set()
        for p in pods:
            wells_u.update(pod_to_wells.get(str(p), []))
        wells_list = sorted(wells_u)
        if len(wells_list) == 0:
            continue

        for _, r in g1.iterrows():
            month = int(r["month"])
            af = float(r["af"]) if pd.notna(r["af"]) else 0.0
            if af <= 0:
                continue
            n = len(wells_list)
            per_well_af = af / n
            days = month_len(int(year), int(month))
            cfd = (per_well_af * AF_TO_CF) / days
            for wid in wells_list:
                wrec = wells_aug[wells_aug[wcols["well_id"]].astype(str) == str(wid)]
                if wrec.empty:
                    # Shouldn't happen, but guard
                    x2265=y2265=x2266=y2266=total_dept=top_scr=bot_scr=np.nan
                else:
                    wrec = wrec.iloc[0]
                    x2265 = wrec.get("x_2265", np.nan)
                    y2265 = wrec.get("y_2265", np.nan)
                    x2266 = wrec.get("x_2266", np.nan)
                    y2266 = wrec.get("y_2266", np.nan)
                    total_dept = wrec.get(wcols["total_dept"], np.nan)
                    top_scr    = wrec.get(wcols["top_screen"], np.nan)
                    bot_scr    = wrec.get(wcols["bottom_scr"], np.nan)

                out_rows.append({
                    "Permits": list_bracket([permit]),
                    "Pods": list_bracket(pods),
                    "Year": int(year),
                    "Month": int(month),
                    "Well": str(wid),
                    "x_2265": x2265, "y_2265": y2265, "x_2266": x2266, "y_2266": y2266,
                    "total_dept": total_dept, "top_screen": top_scr, "bottom_scr": bot_scr,
                    "use_acft": per_well_af,
                    "days": days,
                    "cfd": cfd
                })
    return pd.DataFrame(out_rows)

def distribute_annual_pod(
    annual_pod_af: pd.DataFrame,   # columns: permit_number_key, pod_key, use_year, reported_acft_num, use_type
    pod_to_wells: Dict[str, List[str]],
    wells_aug: gpd.GeoDataFrame,
    well_cols: Dict[str, Optional[str]],
    monthize: bool = False
) -> pd.DataFrame:
    """
    If monthize=False: returns annual rows (Month=NaN; days=365/366).
    If monthize=True: expand annual AF into 12 monthly rows using use_type schedule.
    """
    wcols = detect_output_well_cols(wells_aug, well_cols)
    out_rows = []

    for (permit, year, pod), g in annual_pod_af.groupby(["permit_number_key","use_year","pod_key"]):
        total_af = float(g["reported_acft_num"].sum())
        use_type = str(g["use_type"].iloc[0]) if "use_type" in g.columns else "Unknown"

        wells_list = pod_to_wells.get(str(pod), [])
        if len(wells_list) == 0:
            # As a last resort this pod has no wells; create a dummy now?
            # Keeping consistent with earlier logic, skip silently if none.
            continue

        if not monthize:
            n = len(wells_list)
            per_well_af = total_af / n if n > 0 else 0.0
            dyear = days_in_year(int(year)) if pd.notna(year) else 365
            cfd = (per_well_af * AF_TO_CF) / dyear
            for wid in wells_list:
                wrec = wells_aug[wells_aug[wcols["well_id"]].astype(str) == str(wid)]
                if wrec.empty:
                    x2265=y2265=x2266=y2266=total_dept=top_scr=bot_scr=np.nan
                else:
                    wrec = wrec.iloc[0]
                    x2265 = wrec.get("x_2265", np.nan)
                    y2265 = wrec.get("y_2265", np.nan)
                    x2266 = wrec.get("x_2266", np.nan)
                    y2266 = wrec.get("y_2266", np.nan)
                    total_dept = wrec.get(wcols["total_dept"], np.nan)
                    top_scr    = wrec.get(wcols["top_screen"], np.nan)
                    bot_scr    = wrec.get(wcols["bottom_scr"], np.nan)

                out_rows.append({
                    "Permits": list_bracket([permit]),
                    "Pods": list_bracket([pod]),
                    "Year": int(year),
                    "Month": np.nan,
                    "Well": str(wid),
                    "x_2265": x2265, "y_2265": y2265, "x_2266": x2266, "y_2266": y2266,
                    "total_dept": total_dept, "top_screen": top_scr, "bottom_scr": bot_scr,
                    "use_acft": per_well_af,
                    "days": dyear,
                    "cfd": cfd
                })
        else:
            weights = use_type_monthly_weights(use_type)
            for mth in range(1, 13):
                af_m = total_af * weights[mth]
                n = len(wells_list)
                per_well_af = af_m / n if n > 0 else 0.0
                d = month_len(int(year), int(mth))
                cfd = (per_well_af * AF_TO_CF) / d
                for wid in wells_list:
                    wrec = wells_aug[wells_aug[wcols["well_id"]].astype(str) == str(wid)]
                    if wrec.empty:
                        x2265=y2265=x2266=y2266=total_dept=top_scr=bot_scr=np.nan
                    else:
                        wrec = wrec.iloc[0]
                        x2265 = wrec.get("x_2265", np.nan)
                        y2265 = wrec.get("y_2265", np.nan)
                        x2266 = wrec.get("x_2266", np.nan)
                        y2266 = wrec.get("y_2266", np.nan)
                        total_dept = wrec.get(wcols["total_dept"], np.nan)
                        top_scr    = wrec.get(wcols["top_screen"], np.nan)
                        bot_scr    = wrec.get(wcols["bottom_scr"], np.nan)

                    out_rows.append({
                        "Permits": list_bracket([permit]),
                        "Pods": list_bracket([pod]),
                        "Year": int(year),
                        "Month": int(mth),
                        "Well": str(wid),
                        "x_2265": x2265, "y_2265": y2265, "x_2266": x2266, "y_2266": y2266,
                        "total_dept": total_dept, "top_screen": top_scr, "bottom_scr": bot_scr,
                        "use_acft": per_well_af,
                        "days": d,
                        "cfd": cfd
                    })
    return pd.DataFrame(out_rows)

def detect_output_well_cols(wells_aug: gpd.GeoDataFrame, well_cols: Dict[str, Optional[str]]) -> Dict[str, str]:
    # ensure final usable columns
    out = {}
    out["well_id"]     = well_cols["well_id"]     if well_cols["well_id"]     in wells_aug.columns else first_existing(wells_aug.columns, ["_autowellid"])
    out["total_dept"]  = well_cols["total_dept"]  if well_cols["total_dept"]  in wells_aug.columns else None
    out["top_screen"]  = well_cols["top_screen"]  if well_cols["top_screen"]  in wells_aug.columns else None
    out["bottom_scr"]  = well_cols["bottom_scr"]  if well_cols["bottom_scr"]  in wells_aug.columns else None
    return out

# -----------------------------------------------------------------------------#
# Permits→PODs mapping
# -----------------------------------------------------------------------------#
def build_permit_to_pods(pod_g: gpd.GeoDataFrame, pod_id_col: str, permit_col: str) -> Dict[str, Set[str]]:
    d: Dict[str, Set[str]] = {}
    for permit, g in pod_g.groupby(permit_col):
        key = str(permit).strip()
        pods = set(g[pod_id_col].astype(str).str.strip().unique())
        d[key] = pods
    return d

# -----------------------------------------------------------------------------#
# Main orchestration
# -----------------------------------------------------------------------------#
def build_allocation(MONTHLY: bool = True, MONTHLY_ALL: bool = True):
    log("Loading layers...")
    elk_bound, wells, pod = load_layers()
    pod_id_col, pod_permit_col = detect_pod_cols(pod)
    well_cols = detect_well_cols(wells)

    log("Loading use tables...")
    monthly, annual = load_monthly_and_annual()

    log("Computing monthly (permit) AF and annual (pod) AF...")
    monthly_af_by_permit_year, monthly_permit_monthly_af = compute_monthly_af_by_permit(monthly)
    annual_pod_af = compute_annual_pod_af(annual)

    # Build simple lookups of monthly and annual totals per (permit, year)
    if monthly_permit_monthly_af is not None and not monthly_permit_monthly_af.empty:
        monthly_nonzero = (
            monthly_permit_monthly_af
            .groupby(["permit","year"], dropna=False)["af"]
            .sum()
            .reset_index()
            .rename(columns={"af":"monthly_total_af"})
        )
        monthly_sum_lut = {(str(r["permit"]), int(r["year"])): float(r["monthly_total_af"])
                           for _, r in monthly_nonzero.iterrows()}
    else:
        monthly_nonzero = pd.DataFrame(columns=["permit","year","monthly_total_af"])
        monthly_sum_lut = {}

    annual_total_by_permit_year = (
        annual_pod_af.groupby(["permit_number_key","use_year"], dropna=False)["reported_acft_num"]
                     .sum()
                     .reset_index()
    )
    annual_sum_lut = {(str(r["permit_number_key"]), int(r["use_year"])): float(r["reported_acft_num"])
                      for _, r in annual_total_by_permit_year.iterrows()}

    # load pods-no-wells mapping and build pod→wells map (including dummy wells)
    log("Building POD→Wells map (handling assigned pods & dummy wells)...")
    pods_nowells = load_pods_no_wells_csv()

    # Prepare well coords columns BEFORE dummy creation
    wells_aug = add_xy_columns(wells, well_cols)

    # Build {pod -> wells} (creates dummies where needed)
    pod_to_wells, wells_aug = build_pod_to_wells_map(wells_aug, pod, pods_nowells, well_cols, pod_id_col)

    # RECOMPUTE XY so newly-added dummies get coordinates
    wells_aug = add_xy_columns(wells_aug, well_cols)

    # Permit→PODs map
    permit_to_pods = build_permit_to_pods(pod, pod_id_col, pod_permit_col)

    # ---- Allocation A (main) ----
    log("Allocating main output according to MONTHLY flag and new rules...")
    rows_main = []

    permits_in_monthly = set(monthly_af_by_permit_year["permit_number_key"].dropna().astype(str))
    permits_in_annual  = set(annual_pod_af["permit_number_key"].dropna().astype(str))
    all_permits = sorted(permits_in_monthly | permits_in_annual)

    # Years to consider: union of years from both sources
    years = sorted(
        set(annual_pod_af["use_year"].dropna().astype(int))
        | set(monthly_af_by_permit_year["use_year"].dropna().astype(int))
    )

    # MAIN: For each permit & year, choose monthly vs annual under the new policy
    for permit in all_permits:
        for y in years:
            if not pd.notna(y):
                continue
            y = int(y)

            m_total = monthly_sum_lut.get((str(permit), y), None)  # None => no monthly rows
            a_total = annual_sum_lut.get((str(permit), y), 0.0)

            has_monthly_rows = (m_total is not None)
            monthly_positive = (m_total is not None) and (m_total > 0.0)

            if MONTHLY:
                # Use monthly whenever available:
                if y >= 2000:
                    if monthly_positive:
                        pm = monthly_permit_monthly_af[
                            (monthly_permit_monthly_af["permit"] == permit) &
                            (monthly_permit_monthly_af["year"] == y)
                        ]
                        if not pm.empty:
                            alloc = distribute_monthly_permit(pm, permit_to_pods, pod_to_wells, wells_aug, well_cols)
                            rows_main.append(alloc)
                    else:
                        # monthly missing or monthly total == 0 -> fall back to annual if > 0
                        ann_y = annual_pod_af[(annual_pod_af["permit_number_key"] == permit) &
                                              (annual_pod_af["use_year"] == y)]
                        if not ann_y.empty and a_total > 0:
                            alloc = distribute_annual_pod(ann_y, pod_to_wells, wells_aug, well_cols, monthize=True)
                            rows_main.append(alloc)
                else:
                    # y < 2000 → if monthly exists, aggregate to annual; else use annual
                    if has_monthly_rows and m_total is not None and m_total > 0:
                        pm = monthly_permit_monthly_af[
                            (monthly_permit_monthly_af["permit"] == permit) &
                            (monthly_permit_monthly_af["year"] == y)
                        ]
                        if not pm.empty:
                            alloc = distribute_monthly_permit_annualized(pm, permit_to_pods, pod_to_wells, wells_aug, well_cols)
                            rows_main.append(alloc)
                    else:
                        ann_y = annual_pod_af[(annual_pod_af["permit_number_key"] == permit) &
                                              (annual_pod_af["use_year"] == y)]
                        if not ann_y.empty and a_total > 0:
                            alloc = distribute_annual_pod(ann_y, pod_to_wells, wells_aug, well_cols, monthize=False)
                            rows_main.append(alloc)
            else:
                # MONTHLY == False → always produce ANNUAL rows
                if has_monthly_rows:
                    pm = monthly_permit_monthly_af[
                        (monthly_permit_monthly_af["permit"] == permit) &
                        (monthly_permit_monthly_af["year"] == y)
                    ]
                    if not pm.empty and m_total > 0:
                        # monthly exists → aggregate to annual per well
                        alloc = distribute_monthly_permit_annualized(pm, permit_to_pods, pod_to_wells, wells_aug, well_cols)
                        rows_main.append(alloc)
                    else:
                        # monthly total == 0 or no rows usable → if annual > 0 use annual; else nothing
                        ann_y = annual_pod_af[(annual_pod_af["permit_number_key"] == permit) &
                                              (annual_pod_af["use_year"] == y)]
                        if not ann_y.empty and a_total > 0:
                            alloc = distribute_annual_pod(ann_y, pod_to_wells, wells_aug, well_cols, monthize=False)
                            rows_main.append(alloc)
                else:
                    # no monthly rows → use annual if present
                    ann_y = annual_pod_af[(annual_pod_af["permit_number_key"] == permit) &
                                          (annual_pod_af["use_year"] == y)]
                    if not ann_y.empty and a_total > 0:
                        alloc = distribute_annual_pod(ann_y, pod_to_wells, wells_aug, well_cols, monthize=False)
                        rows_main.append(alloc)

    main_df = pd.concat(rows_main, ignore_index=True) if rows_main else pd.DataFrame(
        columns=["Permits","Pods","Year","Month","Well","x_2265","y_2265","x_2266","y_2266",
                 "total_dept","top_screen","bottom_scr","use_acft","days","cfd"]
    )
    
    # NEW: collapse duplicates per (Well, Year, Month)
    main_df = collapse_duplicate_allocations(main_df)
    
    out_main = os.path.join(OUT_DIR, f"per_well_allocation_main_{'monthlyTrue' if MONTHLY else 'monthlyFalse'}.csv")
    main_df.to_csv(out_main, index=False)
    log(f"Main allocation written: {out_main}  (rows={len(main_df)})")

    # ---- Allocation B (monthly_all) ----
    if MONTHLY_ALL:
        log("Building fully-monthly allocation (monthly_all=True)...")
        rows_mall = []

        # 1) Use monthly-permit rows wherever available (all years in table)
        if monthly_permit_monthly_af is not None and not monthly_permit_monthly_af.empty:
            alloc_m = distribute_monthly_permit(monthly_permit_monthly_af, permit_to_pods, pod_to_wells, wells_aug, well_cols)
            rows_mall.append(alloc_m)

        # 2) For (permit,year) pairs without monthly rows, expand ANNUAL via use_type schedule
        have_pm = set()
        if monthly_permit_monthly_af is not None and not monthly_permit_monthly_af.empty:
            have_pm = set(
                monthly_permit_monthly_af[["permit","year"]]
                .dropna()
                .apply(lambda r: (str(r["permit"]), int(r["year"])), axis=1)
            )

        for (permit, y), grp in annual_pod_af.groupby(["permit_number_key","use_year"]):
            key = (str(permit), int(y))
            if key in have_pm:
                continue  # already covered by monthly
            alloc_ann_m = distribute_annual_pod(grp, pod_to_wells, wells_aug, well_cols, monthize=True)
            rows_mall.append(alloc_ann_m)

        monthly_all_df = pd.concat(rows_mall, ignore_index=True) if rows_mall else pd.DataFrame(
            columns=["Permits","Pods","Year","Month","Well","x_2265","y_2265","x_2266","y_2266",
                     "total_dept","top_screen","bottom_scr","use_acft","days","cfd"]
        )
        
        # NEW: collapse duplicates per (Well, Year, Month)
        monthly_all_df = collapse_duplicate_allocations(monthly_all_df)
        
        out_mall = os.path.join(OUT_DIR, "per_well_allocation_monthly_all.csv")
        monthly_all_df.to_csv(out_mall, index=False)
        log(f"Monthly-all allocation written: {out_mall}  (rows={len(monthly_all_df)})")

    # Optional: still write a comparison CSV (QA only; not used for logic)
    comp_path = os.path.join(OUT_DIR, "permit_vs_pods_annual_comparison.csv")
    try:
        comp = monthly_af_by_permit_year.merge(
            annual_total_by_permit_year,
            left_on=["permit_number_key","use_year"],
            right_on=["permit_number_key","use_year"],
            how="inner"
        )
        comp = comp.rename(columns={"reported_acft_num":"sum_pods_af"})
        comp["prefer_monthly_old_rule"] = comp["permit_af_from_months"] >= comp["sum_pods_af"]
        comp.to_csv(comp_path, index=False)
        log(f"Comparison written: {comp_path}  (rows={len(comp)})")
    except Exception as e:
        log(f"Comparison write skipped due to error: {e}")

    # Debug: ensure dummy wells appear in allocation (optional)
    print(f"[debug:alloc] dummy rows in main_df: {main_df['Well'].astype(str).str.startswith('DUMMY_').sum()}")

    log("Exporting well shapefile with annual AF as stress periods...")
    export_well_annual_sp_shapefile(
        allocation_df=main_df,
        wells_aug=wells_aug,
        well_cols=well_cols,
        pod_g=pod,                 # pass POD layer for any backfill
        pod_id_col=pod_id_col,     # pass POD id col for any backfill
        out_base=os.path.join(OUT_DIR, "well_annual_AF_SP"),
        write_gpkg=True
    )

    return main_df

# -----------------------------------------------------------------------------#
# Optional: small plotting helpers (timeseries & POD map) — if you’d like to keep
# -----------------------------------------------------------------------------#
def plot_permit_timeseries(permit_id: str,
                           monthly_af_by_permit_year: pd.DataFrame,
                           annual_pod_af: pd.DataFrame,
                           out_dir: str = FIG_TS_DIR):
    import matplotlib.pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pm = monthly_af_by_permit_year[monthly_af_by_permit_year["permit_number_key"] == permit_id]
    pa = annual_pod_af[annual_pod_af["permit_number_key"] == permit_id]
    if pm.empty or pa.empty:
        return None
    pivot = pa.pivot_table(index="use_year", columns="pod_key", values="reported_acft_num", aggfunc="sum").sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pm["use_year"], pm["permit_af_from_months"], marker="o", label=f"Permit {permit_id} (monthly → AF)")
    for pod_col in pivot.columns:
        ax.plot(pivot.index, pivot[pod_col], marker="o", linestyle="--", label=f"POD {pod_col}")
    ax.set_title(f"Permit {permit_id}: monthly-derived AF vs annual POD AF")
    ax.set_xlabel("Year")
    ax.set_ylabel("Acre-feet")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    out_path = Path(out_dir) / f"permit_{permit_id}_timeseries.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)

def plot_permit_pods(permit_id: str, elk_bound: gpd.GeoDataFrame, pod: gpd.GeoDataFrame, wells: gpd.GeoDataFrame,
                     pod_id_col: str, pod_permit_col: str, out_dir: str = FIG_MAP_DIR):
    import matplotlib.pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sel = pod[pod[pod_permit_col].astype(str).str.strip() == str(permit_id).strip()].copy()
    if sel.empty:
        return None
    # wells by permit if available; else spatial join
    wperm_col = first_existing(wells.columns, ["permit_number","permit_num","permit","permit_no"])
    if wperm_col:
        wells_sel = wells[wells[wperm_col].astype(str).str.strip() == str(permit_id).strip()].copy()
    else:
        try:
            wells_sel = gpd.sjoin(wells, sel[[pod_id_col,"geometry"]], predicate="within", how="inner")
        except Exception:
            wells_sel = gpd.GeoDataFrame(geometry=[], crs=wells.crs)
    sel["_label_pt"] = sel.representative_point()
    labels = sel[[pod_id_col,"_label_pt"]].drop_duplicates(subset=[pod_id_col]).set_geometry("_label_pt")
    fig, ax = plt.subplots(figsize=(9, 9))
    elk_bound.boundary.plot(ax=ax, linewidth=1.5, alpha=0.7)
    sel.plot(ax=ax, edgecolor="black", linewidth=0.8, alpha=0.35)
    if not wells_sel.empty:
        wells_sel.plot(ax=ax, markersize=12, marker="o", alpha=0.9)
    for _, r in labels.iterrows():
        ax.text(r["_label_pt"].x, r["_label_pt"].y, str(r[pod_id_col]), fontsize=9, ha="center", va="center")
    ax.set_title(f"Permit {permit_id}: PODs within Elk Valley", pad=12)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    if elk_bound is not None and not elk_bound.empty:
        minx, miny, maxx, maxy = elk_bound.total_bounds
        dx, dy = (maxx-minx)*0.03, (maxy-miny)*0.03
        ax.set_xlim(minx-dx, maxx+dx); ax.set_ylim(miny-dy, maxy+dy)
    fig.tight_layout()
    out_path = Path(out_dir) / f"permit_{permit_id}_pods_map.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def export_well_annual_sp_shapefile(
    allocation_df: pd.DataFrame,
    wells_aug: gpd.GeoDataFrame,
    well_cols: Dict[str, Optional[str]],
    pod_g: gpd.GeoDataFrame,         # <— add
    pod_id_col: str,                  # <— add
    out_base: str = os.path.join(OUT_DIR, "well_annual_AF_SP"),
    write_gpkg: bool = True
):
    if allocation_df.empty:
        log("export_well_annual_sp_shapefile: allocation_df is empty; skipping.")
        return None

    # Ensure XY exist for ANY wells added after base load (esp. dummies)
    wells_aug = add_xy_columns(wells_aug, well_cols)

    # 0) Quick counts
    n_wells_g = len(wells_aug)
    n_dummy_g = wells_aug[
        detect_output_well_cols(wells_aug, well_cols)["well_id"]
    ].astype(str).str.startswith("DUMMY_").sum()
    print(f"[export:debug] wells_aug rows={n_wells_g}, dummy_wells_in_geom={n_dummy_g}")

    # --- 1) Well id column present in wells_aug
    wcols = detect_output_well_cols(wells_aug, well_cols)
    well_id_col = wcols["well_id"]
    if not well_id_col or well_id_col not in wells_aug.columns:
        raise ValueError(f"Cannot determine well identifier column for export. Have cols={list(wells_aug.columns)}")

    # --- 2) Collapse allocation to annual (Well, Year) AF
    df = allocation_df.copy()
    for col in ["Well", "Year", "use_acft"]:
        if col not in df.columns:
            raise ValueError(f"allocation_df missing '{col}'")
    df["Well"] = df["Well"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"])

    # Which wells appear in allocation? (should include DUMMY_* if they got AF)
    alloc_wells = set(df["Well"].unique().tolist())
    n_dummy_alloc = sum(1 for w in alloc_wells if str(w).startswith("DUMMY_"))
    print(f"[export:debug] unique wells in allocation={len(alloc_wells)}, dummy_in_alloc={n_dummy_alloc}")

    annual = (
        df.groupby(["Well", "Year"], dropna=False)["use_acft"]
          .sum()
          .reset_index()
    )
    if annual.empty:
        log("No annual totals to export; skipping.")
        return None

    # --- 3) Wide pivot
    years = sorted(annual["Year"].dropna().astype(int).unique().tolist())
    sp_map = {f"SP{i+1}": y for i, y in enumerate(years)}
    af_cols_sp = [f"SP{i+1}" for i in range(len(years))]
    af_cols_yr = [f"AF_{y}" for y in years]

    wide = annual.pivot(index="Well", columns="Year", values="use_acft").fillna(0.0)
    wide_sp = wide.copy(); wide_sp.columns = af_cols_sp
    wide_yr = wide.copy(); wide_yr.columns = af_cols_yr

    # --- 4) Prepare wells table (ensure dummies are present with XY)
    base_cols = [c for c in [well_id_col, "x_2265","y_2265","x_2266","y_2266","total_dept","top_screen","bottom_scr"] if c in wells_aug.columns]
    wmin = wells_aug[base_cols + ["geometry"]].drop_duplicates(subset=[well_id_col]).copy()
    wmin[well_id_col] = wmin[well_id_col].astype(str).str.strip()

    # Missing XY? recompute just in case (no-op if already set)
    if not {"x_2265","y_2265","x_2266","y_2266"}.issubset(set(wmin.columns)):
        wmin = add_xy_columns(wmin, well_cols)

    # --- 4a) DIAGNOSTIC: Which alloc wells aren’t in geometry table?
    missing_geom_ids = sorted(set(alloc_wells) - set(wmin[well_id_col].unique().tolist()))
    if missing_geom_ids:
        print(f"[export:warn] {len(missing_geom_ids)} Well IDs in allocation not found in wells_aug. "
              f"First few: {missing_geom_ids[:10]}")

    # --- 5) Merge
    wide_sp = wide_sp.reset_index().rename(columns={"Well": well_id_col})
    wide_yr = wide_yr.reset_index().rename(columns={"Well": well_id_col})

    g_sp = wmin.merge(wide_sp, on=well_id_col, how="right")   # <-- right join ensures every alloc well appears
    g_sp[af_cols_sp] = g_sp[af_cols_sp].fillna(0.0)

    g_yr = wmin.merge(wide_yr, on=well_id_col, how="right")
    g_yr[af_cols_yr] = g_yr[af_cols_yr].fillna(0.0)

    # --- 5a) DIAGNOSTIC: any geometries still missing after right-join?
    missing_geom_after = g_sp["geometry"].isna().sum()
    if missing_geom_after > 0:
        mask = g_sp["geometry"].isna() & g_sp[well_id_col].astype(str).str.startswith("DUMMY_")
        if mask.any():
            print(f"[export:fix] backfilling geometry for {mask.sum()} dummy wells missing geometry.")
            # Build lookup: DUMMY_<podid> -> representative point of POD
            pod_lookup = pod_g[[pod_id_col, "geometry"]].copy()
            pod_lookup[pod_id_col] = pod_lookup[pod_id_col].astype(str).str.strip()
            pod_lookup["_dummy_id"] = "DUMMY_" + pod_lookup[pod_id_col]
            pod_lookup["_pt"] = pod_lookup.geometry.representative_point()

            rep_map = dict(zip(pod_lookup["_dummy_id"], pod_lookup["_pt"]))
            g_sp.loc[mask, "geometry"] = g_sp.loc[mask, well_id_col].map(rep_map)
            
        # --- 5b) Ensure XY attributes are populated for all rows (esp. dummies)
    g_sp_out = gpd.GeoDataFrame(g_sp, geometry="geometry", crs=wells_aug.crs)
    g_sp_out = add_xy_columns(g_sp_out, well_cols)   # overwrites/creates x_2265,y_2265,x_2266,y_2266

    # Do the same for GPKG version's attribute table
    g_yr_out = gpd.GeoDataFrame(g_yr, geometry="geometry", crs=wells_aug.crs)
    g_yr_out = add_xy_columns(g_yr_out, well_cols)

    # --- 6) Write outputs
    shp_path = f"{out_base}.shp"
    g_sp_out.to_file(shp_path)
    log(f"Wrote shapefile: {shp_path}  (fields: {['...'] + af_cols_sp[:5]} ...)")

    legend_path = f"{out_base}_legend.csv"
    pd.DataFrame({"SP": list(sp_map.keys()), "Year": list(sp_map.values())}).to_csv(legend_path, index=False)
    log(f"Wrote legend:    {legend_path}")

    if write_gpkg:
        gpkg_path = f"{out_base}.gpkg"
        g_yr_out.to_file(gpkg_path, layer="well_annual_AF", driver="GPKG")
        log(f"Wrote geopackage: {gpkg_path}  (fields include {af_cols_yr[:5]} ... )")

    return {
        "shapefile": shp_path,
        "legend_csv": legend_path,
        "gpkg": f"{out_base}.gpkg" if write_gpkg else None,
        "sp_to_year": sp_map
    }

def _read_alloc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Well": str})
    # robust numeric
    for c in ["Year", "Month", "use_acft", "days", "cfd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # compute GPM (prefer cfd if present, else derive from AF & days)
    if "cfd" in df.columns and df["cfd"].notna().any():
        df["gpm"] = df["cfd"] * GPM_PER_CFD
    else:
        # derive cfd from AF / days if needed
        # cfd = AF * 43560 / days
        df["gpm"] = (df["use_acft"] * 43560.0 / df["days"]) * GPM_PER_CFD
    return df

def plot_well_timeseries_gpm(
    out_dir_root: str = OUT_DIR,
    figs_dir: str = os.path.join("figs", "well_timeseries_gpm"),
    wells_subset: Optional[Iterable[str]] = None,   # restrict to a set/list of Well IDs
    figsize: Tuple[int, int] = (11, 5)
):
    """
    Loads allocation outputs and plots GPM per well:
      - Monthly-All: per_well_allocation_monthly_all.csv (monthly points across entire record)
      - Annual: rows where Month is NaN from any main file(s) (flat yearly GPM)
    """
    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    # Find files
    monthly_all_path = os.path.join(out_dir_root, "per_well_allocation_monthly_all.csv")
    main_candidates = sorted(glob(os.path.join(out_dir_root, "per_well_allocation_main_*.csv")))
    if not os.path.exists(monthly_all_path):
        print("[gpm-plots] Monthly-All file not found; expected:", monthly_all_path)
    if not main_candidates:
        print("[gpm-plots] No 'main' allocation files found in", out_dir_root)

    # Load
    df_mall = _read_alloc_csv(monthly_all_path) if os.path.exists(monthly_all_path) else pd.DataFrame()
    df_main_all = pd.concat((_read_alloc_csv(p) for p in main_candidates), ignore_index=True) if main_candidates else pd.DataFrame()

    # Keep only annual rows from "main" (Month is NaN)
    if not df_main_all.empty:
        df_annual = df_main_all[df_main_all["Month"].isna()].copy()
    else:
        df_annual = pd.DataFrame()

    if df_mall.empty and df_annual.empty:
        print("[gpm-plots] Nothing to plot.")
        return None

    # Build timestamps
    def _to_month_date(frame: pd.DataFrame) -> pd.Series:
        # safe integer conversion
        yy = pd.to_numeric(frame["Year"], errors="coerce").astype("Int64")
        mm = pd.to_numeric(frame["Month"], errors="coerce").astype("Int64")
        # fallback: if month missing, keep NaT
        s = pd.to_datetime(
            dict(year=yy.fillna(1).astype("float").astype("Int64"),
                 month=mm.fillna(1).astype("float").astype("Int64"),
                 day=1),
            errors="coerce"
        )
        # enforce monthly first-of-month for monthly-all
        return s

    if not df_mall.empty:
        df_mall = df_mall.copy()
        df_mall["date"] = _to_month_date(df_mall)
        df_mall = df_mall.dropna(subset=["date"])

    if not df_annual.empty:
        df_annual = df_annual.copy()
        # place annual points mid-year (July 1) for readability
        yy = pd.to_numeric(df_annual["Year"], errors="coerce").astype("Int64")
        df_annual["date"] = pd.to_datetime(
            dict(year=yy.fillna(1).astype("float").astype("Int64"),
                 month=7, day=1),
            errors="coerce"
        )
        df_annual = df_annual.dropna(subset=["date"])

    # Restrict wells if requested
    wells_all = set()
    if not df_mall.empty:
        wells_all |= set(df_mall["Well"].astype(str).unique())
    if not df_annual.empty:
        wells_all |= set(df_annual["Well"].astype(str).unique())

    if wells_subset is not None:
        wells_all = wells_all & set(map(str, wells_subset))

    wells_all = sorted(wells_all)
    if not wells_all:
        print("[gpm-plots] No wells matched selection.")
        return None

    index_rows = []

    for w in wells_all:
        m = df_mall[df_mall["Well"].astype(str) == w].copy() if not df_mall.empty else pd.DataFrame()
        a = df_annual[df_annual["Well"].astype(str) == w].copy() if not df_annual.empty else pd.DataFrame()

        if m.empty and a.empty:
            continue

        # Prepare series
        m = m.sort_values("date") if not m.empty else m
        a = a.sort_values("date") if not a.empty else a

        fig, ax = plt.subplots(figsize=figsize)

        # Monthly-All (line with markers)
        if not m.empty:
            ax.plot(m["date"], m["gpm"], marker="o", linewidth=1.6, label="Monthly-All (GPM)")

        # Annual (plot as step-like flat rate per year; here as markers+line)
        if not a.empty:
            # Average rate is already constant in "gpm" computed above
            ax.plot(a["date"], a["gpm"], marker="s", linestyle="--", linewidth=1.2, label="Annual (GPM)")

        ax.set_title(f"Well {w} — Flow Rate (GPM): Monthly-All vs Annual")
        ax.set_xlabel("Date")
        ax.set_ylabel("GPM")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        # filename
        safe_w = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(w))
        out_png = os.path.join(figs_dir, f"well_{safe_w}_gpm.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

        # record (robust to empty frames lacking a 'date' column)
        date_series = []
        if not m.empty and "date" in m.columns:
            date_series.append(m["date"])
        if not a.empty and "date" in a.columns:
            date_series.append(a["date"])

        if date_series:
            dates_combined = pd.concat(date_series)
            min_date = dates_combined.min()
            max_date = dates_combined.max()
        else:
            min_date = pd.NaT
            max_date = pd.NaT

        index_rows.append({
            "Well": w,
            "has_monthly_all": (not m.empty),
            "has_annual": (not a.empty),
            "start": min_date,
            "end": max_date,
            "png": out_png
        })


    # write an index CSV
    idx_df = pd.DataFrame(index_rows)
    idx_path = os.path.join(figs_dir, "index_well_timeseries_gpm.csv")
    idx_df.to_csv(idx_path, index=False)
    print(f"[gpm-plots] Wrote {len(idx_df)} plots to: {figs_dir}")
    print(f"[gpm-plots] Index: {idx_path}")

    return {"plots_dir": figs_dir, "index_csv": idx_path}
# -----------------------------------------------------------------------------#
# Entry point
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    # You can toggle here or pass via environment/args if you prefer
    MONTHLY = bool(MONTHLY)
    MONTHLY_ALL = bool(MONTHLY_ALL)
    df = build_allocation(MONTHLY=MONTHLY, MONTHLY_ALL=MONTHLY_ALL)
    log("Done.")
    #plot_well_timeseries_gpm()
    log("Done.")