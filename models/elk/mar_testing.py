# -*- coding: utf-8 -*-
"""
Elk Valley MF6: Update/replace an existing MAR WEL package IN-PLACE in a base model folder.

This version:
- Avoids ANY (row,col) that is used by ANY existing WEL/RIV/DRN package in ANY LAYER
  (and any stress period), excluding the MAR package itself.
- Avoids "flood-prone" cells where simulated head > DIS top (tol) using an existing .hds.
- Places MAR sites ONLY within a hard max distance (in cells) to deferred-permit wells
  (from deferred_requested_per_well.csv). If not enough eligible cells exist, it will
  place fewer than requested sites (and prints a warning).
- Enforces a minimum spacing between MAR sites (in cells) to avoid clustering.
- Writes MAR locations to CSV + shapefile.
- Optionally RUN_MODEL = False to skip running MF6 after writing.

Notes
-----
- Seasonal monthly injection is implemented with explicit [] for OFF months so MF6
  does not persist list-package entries.
- To keep MAR files in the workspace, set EXTERNALIZE_MAR_ONLY=False.
"""

import os
import sys
import platform
import glob
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict

import numpy as np
import pandas as pd

# Keep dependency import pattern consistent with your repo
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "dependencies")))
sys.path.insert(1, os.path.abspath(os.path.join("..", "..", "dependencies", "flopy")))
sys.path.insert(2, os.path.abspath(os.path.join("..", "..", "dependencies", "pyemu")))

import flopy  # noqa: E402

# Optional deps for shapefile output
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None

try:
    from scipy.spatial import cKDTree  # fast nearest-neighbor
except Exception:
    cKDTree = None


# --------------------------
# UTILS
# --------------------------
def find_mf6_exe(bin_folder=os.path.join("..", "..", "bin")) -> str:
    """Find mf6 executable in a repo-style bin folder structure."""
    osys = platform.system().lower()
    if "windows" in osys:
        exe_path = os.path.join(bin_folder, "win", "mf6.exe")
    elif "linux" in osys:
        exe_path = os.path.join(bin_folder, "linux", "mf6")
    elif "darwin" in osys:
        exe_path = os.path.join(bin_folder, "mac", "mf6")
    else:
        raise OSError(f"Unsupported platform: {osys}")

    exe_path = os.path.abspath(exe_path)
    if not os.path.isfile(exe_path):
        raise OSError(f"Expected mf6 binary not found: {exe_path}")
    return exe_path


def acftyr_to_cfd(acft_per_year: float) -> float:
    """Convert acre-feet/year to cubic-feet/day (WEL uses L^3/T rate)."""
    return float(acft_per_year) * 43560.0 / 365.0


def tail_text(buff, n=80) -> str:
    """Pretty tail for mf6 run output."""
    if buff is None:
        return ""
    if isinstance(buff, (list, tuple)):
        return "\n".join([str(x) for x in buff[-n:]])
    return str(buff)


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


# --------------------------
# MF6 LIST-PACKAGE HELPERS (WEL/RIV/DRN)
# --------------------------
def _iter_spd_records(spd_obj):
    """
    Yield MF6 list-package records from stress_period_data regardless of whether
    FloPy returns:
      - dict {kper: list-of-recs}
      - MFData with get_data()
      - numpy recarray / structured array
      - MFData with .data or .array
    """
    if spd_obj is None:
        return

    # Case 1: plain dict {kper: list-of-recs}
    if isinstance(spd_obj, dict):
        for _, recs in spd_obj.items():
            if recs is None:
                continue
            for r in recs:
                yield r
        return

    # Case 2: MFData that supports get_data()
    try:
        gd = spd_obj.get_data()
        if isinstance(gd, dict):
            for _, recs in gd.items():
                if recs is None:
                    continue
                for r in recs:
                    yield r
            return
        if gd is not None:
            try:
                for r in gd:
                    yield r
                return
            except TypeError:
                pass
    except Exception:
        pass

    # Case 3: MFData / MFList: .data
    try:
        data = spd_obj.data
        if isinstance(data, dict):
            for _, recs in data.items():
                if recs is None:
                    continue
                for r in recs:
                    yield r
            return
        if data is not None:
            try:
                for r in data:
                    yield r
                return
            except TypeError:
                pass
    except Exception:
        pass

    # Case 4: MFData: .array
    try:
        arr = spd_obj.array
        if arr is not None:
            for r in arr:
                yield r
            return
    except Exception:
        pass

    # Case 5: last resort: treat as iterable
    try:
        for r in spd_obj:
            yield r
    except Exception:
        return


def _extract_cellid_from_record(rec):
    """
    Robustly extract cellid (k,i,j) from a stress period record that may be:
      - tuple/list like ((k,i,j), q, ...)
      - numpy void / structured record with field 'cellid'
      - object with attribute 'cellid'
    Returns cellid tuple or None.
    """
    cellid = None

    # tuple/list style: ((k,i,j), q, ...)
    if isinstance(rec, (list, tuple)) and len(rec) >= 1:
        cellid = rec[0]

    # numpy record / recarray style: has field 'cellid'
    if cellid is None:
        try:
            if hasattr(rec, "dtype") and rec.dtype.names and "cellid" in rec.dtype.names:
                cellid = rec["cellid"]
        except Exception:
            pass

    # attribute style
    if cellid is None:
        try:
            cellid = getattr(rec, "cellid", None)
        except Exception:
            cellid = None

    if cellid is None:
        return None

    # normalize to tuple
    try:
        if isinstance(cellid, np.ndarray):
            cellid = tuple(cellid.tolist())
        elif isinstance(cellid, list):
            cellid = tuple(cellid)
    except Exception:
        return None

    if isinstance(cellid, tuple) and len(cellid) == 3:
        k = _safe_int(cellid[0])
        i = _safe_int(cellid[1])
        j = _safe_int(cellid[2])
        if k is None or i is None or j is None:
            return None
        return (k, i, j)

    return None


def collect_forbidden_ij_any_layer(
    gwf,
    pkg_types=("wel", "riv", "drn"),
    include_pnames: Optional[Set[str]] = None,
    exclude_pnames: Optional[Set[str]] = None,
    verbose: bool = False,
) -> Set[Tuple[int, int]]:
    """
    Collect forbidden (i,j) cell indices from ANY LAYER across ALL packages
    of requested types, across ALL stress periods.
    """
    forbidden: Set[Tuple[int, int]] = set()
    pkg_types_l = set(t.lower() for t in pkg_types)

    for pkg in gwf.packagelist:
        ptype = ""
        try:
            ptype = (pkg.package_type or "").lower()
        except Exception:
            ptype = ""

        if ptype not in pkg_types_l:
            continue

        pname = ""
        try:
            pname = str(pkg.package_name)
        except Exception:
            pname = ""

        if include_pnames is not None and pname not in include_pnames:
            continue
        if exclude_pnames is not None and pname in exclude_pnames:
            continue

        spd = getattr(pkg, "stress_period_data", None)
        if spd is None:
            if verbose:
                print(f"[MAR] pkg '{pname}' ({ptype}) has no stress_period_data attribute")
            continue

        n_added = 0
        for rec in _iter_spd_records(spd):
            cellid = _extract_cellid_from_record(rec)
            if cellid is None:
                continue
            _, i, j = cellid
            forbidden.add((int(i), int(j)))
            n_added += 1

        if verbose:
            print(f"[MAR] Forbidden ij from pkg '{pname}' ({ptype}): +{n_added} records")

    if verbose:
        print(f"[MAR] Unique forbidden (i,j) cells from WEL/RIV/DRN (any layer): {len(forbidden)}")

    return forbidden


# --------------------------
# FLOOD MASK (HDS) HELPERS
# --------------------------
def find_hds_file(sim_ws: str, model_name: str) -> str:
    """Locate a MODFLOW 6 head file in sim_ws."""
    cand1 = os.path.join(sim_ws, f"{model_name}.hds")
    if os.path.isfile(cand1):
        return cand1
    cands = sorted(glob.glob(os.path.join(sim_ws, "*.hds")))
    if len(cands) == 0:
        raise FileNotFoundError(f"No .hds file found in {sim_ws}")
    return cands[0]


def collect_flooded_ij_from_hds(
    sim_ws: str,
    *,
    model_name: str,
    gwf,
    hds_path: str = None,
    head_above_top_tol: float = 0.01,
    verbose: bool = True,
) -> Set[Tuple[int, int]]:
    """
    Return set of (i,j) where simulated head exceeds DIS top elevation
    by > head_above_top_tol at ANY checked time and ANY layer (conservative).
    """
    if hds_path is None:
        hds_path = find_hds_file(sim_ws, model_name=model_name)

    dis = gwf.dis
    top = np.asarray(dis.top.array, float)  # (nrow,ncol)

    hf = flopy.utils.HeadFile(hds_path)
    kstpkper_list = hf.get_kstpkper()
    times = hf.get_times()

    flooded: Set[Tuple[int, int]] = set()
    n_checked = 0

    for rec_idx, (kstp, kper) in enumerate(kstpkper_list):
        try:
            h3d = hf.get_data(kstpkper=(kstp, kper))
        except Exception:
            t = times[rec_idx] if rec_idx < len(times) else None
            h3d = hf.get_data(totim=t)

        h3d = np.asarray(h3d, float)
        if h3d.ndim == 2:
            h3d = h3d.reshape((1,) + h3d.shape)

        hmax = np.nanmax(h3d, axis=0)  # (nrow,ncol)
        mask = np.isfinite(hmax) & (hmax > (top + float(head_above_top_tol)))

        ij = np.argwhere(mask)
        for (i, j) in ij:
            flooded.add((int(i), int(j)))

        n_checked += 1

    if verbose:
        print(f"[MAR] HDS flooding screen using: {hds_path}")
        print(f"[MAR] Records checked: {n_checked}")
        print(f"[MAR] Flooded unique (i,j) cells where head > top+{head_above_top_tol}: {len(flooded)}")

    return flooded


# --------------------------
# MAR SPD BUILDERS
# --------------------------
def _cell_recs(mar_cells_1idx: List[Tuple[int, int]], layer_0idx: int, q_cfd_each_site: float):
    recs = []
    for (r1, c1) in mar_cells_1idx:
        i = int(r1) - 1
        j = int(c1) - 1
        k = int(layer_0idx)
        recs.append(((k, i, j), float(q_cfd_each_site)))
    return recs


def build_mar_spd_seasonal_monthly(
    nper: int,
    mar_cells_1idx: List[Tuple[int, int]],
    *,
    mar_start_sp_1idx: int,
    layer_0idx: int = 1,
    q_cfd_each_site: float = 0.0,
    pred_start_year: int = 2024,
    pred_start_month: int = 1,
    active_months=(6, 7, 8),
    verbose: bool = False,
) -> Dict[int, list]:
    """
    Seasonal MAR for MONTHLY stress periods with explicit [] for OFF months.
    """
    if mar_start_sp_1idx < 1:
        raise ValueError("mar_start_sp_1idx must be >= 1 (1-indexed).")
    if not (1 <= pred_start_month <= 12):
        raise ValueError("pred_start_month must be in 1..12")

    active_months = set(int(m) for m in active_months)
    start_kper = int(mar_start_sp_1idx) - 1
    if start_kper >= nper:
        raise ValueError(f"MAR start SP {mar_start_sp_1idx} => kper {start_kper}, but nper={nper}.")

    recs = _cell_recs(mar_cells_1idx, layer_0idx, q_cfd_each_site)

    spd: Dict[int, list] = {}
    on_kpers = []
    off_kpers = []

    for kper in range(start_kper, int(nper)):
        offset_months = kper - start_kper
        m0 = (pred_start_month - 1 + offset_months) % 12
        month = m0 + 1

        if month in active_months:
            spd[kper] = list(recs)
            on_kpers.append(kper)
        else:
            spd[kper] = []  # explicit OFF
            off_kpers.append(kper)

    if verbose:
        print(f"[MAR] Seasonal months ON: {sorted(active_months)}")
        print(f"[MAR] ON kpers count={len(on_kpers)} first few={on_kpers[:12]}")
        print(f"[MAR] OFF kpers count={len(off_kpers)} first few={off_kpers[:12]}")

    return spd


# --------------------------
# DISPERSED LOCATION SELECTION + DEFERRED BIAS + MIN SPACING
# --------------------------
def get_active_ij_from_layer(gwf, layer_0idx: int = 1) -> np.ndarray:
    """
    Return array of allowed (i,j) where idomain for layer_0idx is active (>0) if available.
    """
    dis = gwf.dis
    nrow = int(dis.nrow.data)
    ncol = int(dis.ncol.data)

    idomain = None
    try:
        idomain = dis.idomain.array
    except Exception:
        idomain = None

    if idomain is None:
        ii, jj = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
        return np.column_stack([ii.ravel(), jj.ravel()]).astype(int)

    layer = int(layer_0idx)
    mask = np.asarray(idomain[layer, :, :]) > 0
    return np.argwhere(mask).astype(int)


def _choose_xy_columns_for_modelgrid(df: pd.DataFrame, mgrid) -> Tuple[str, str]:
    """Pick which XY columns to use for intersect() based on grid extents."""
    try:
        xmin, xmax, ymin, ymax = mgrid.extent
    except Exception:
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


def load_deferred_targets_as_ij(
    gwf,
    deferred_csv: str,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load deferred_requested_per_well.csv and map to model (i,j) using modelgrid.intersect.
    Returns unique array of target ij (0-indexed).
    """
    if deferred_csv is None or not os.path.isfile(deferred_csv):
        raise FileNotFoundError(f"Deferred CSV not found: {deferred_csv}")

    df = pd.read_csv(deferred_csv, dtype={"Well": str})
    mgrid = gwf.modelgrid

    xcol, ycol = _choose_xy_columns_for_modelgrid(df, mgrid)
    df = df.dropna(subset=[xcol, ycol]).copy()
    if df.empty:
        raise ValueError(f"Deferred CSV has no usable coordinates after filtering ({xcol}/{ycol}).")

    ii, jj = [], []
    for r in df.itertuples(index=False):
        x = float(getattr(r, xcol))
        y = float(getattr(r, ycol))
        try:
            i, j = mgrid.intersect(x, y)
        except Exception:
            i, j = None, None
        if i is None or j is None:
            continue
        ii.append(int(i))
        jj.append(int(j))

    if len(ii) == 0:
        raise ValueError("No deferred targets intersected the model grid (CRS/extents mismatch?).")

    targets = np.unique(np.column_stack([np.array(ii, int), np.array(jj, int)]), axis=0)

    if verbose:
        print(f"[MAR] Deferred targets loaded: {len(df):,} rows -> {targets.shape[0]:,} unique (i,j) targets")
        print(f"[MAR] Deferred XY columns used: {xcol}, {ycol}")

    return targets


def compute_min_dist_cells(candidate_ij: np.ndarray, target_ij: np.ndarray) -> np.ndarray:
    """For each candidate (i,j), compute min Euclidean distance to any target (i,j) in cell units."""
    cand = np.asarray(candidate_ij, int)
    targ = np.asarray(target_ij, int)
    if cand.size == 0 or targ.size == 0:
        return np.full((cand.shape[0],), np.inf, dtype=float)

    if cKDTree is not None:
        tree = cKDTree(targ.astype(float))
        d, _ = tree.query(cand.astype(float), k=1)
        return np.asarray(d, float)

    dmin = np.full((cand.shape[0],), np.inf, dtype=float)
    for i in range(cand.shape[0]):
        di = targ[:, 0] - cand[i, 0]
        dj = targ[:, 1] - cand[i, 1]
        dmin[i] = float(np.sqrt(np.min(di * di + dj * dj)))
    return dmin


def _weighted_choice(rng: np.random.Generator, idxs: np.ndarray, weights: np.ndarray) -> int:
    """Choose one element from idxs with probability proportional to weights."""
    w = np.asarray(weights, float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return int(rng.choice(idxs))
    p = w / s
    return int(rng.choice(idxs, p=p))


def _min_dist_to_chosen(cell_ij: Tuple[int, int], chosen_ij: List[Tuple[int, int]]) -> float:
    """Min Euclidean distance (cell units) from cell to any chosen cell."""
    if not chosen_ij:
        return np.inf
    ci, cj = int(cell_ij[0]), int(cell_ij[1])
    d2min = np.inf
    for (i, j) in chosen_ij:
        di = ci - int(i)
        dj = cj - int(j)
        d2 = di * di + dj * dj
        if d2 < d2min:
            d2min = d2
    return float(np.sqrt(d2min))


def select_dispersed_sites_near_targets(
    allowed_ij: np.ndarray,
    forbidden_ij: Set[Tuple[int, int]],
    target_ij: np.ndarray,
    *,
    n_sites: int = 30,
    seed: int = 123,
    near_radius_cells: int = 3,      # HARD max distance to deferred targets
    prefer_near_power: float = 2.0,  # weights within bins (still random)
    min_sep_cells: float = 2.0,      # NEW: enforce spacing to avoid clustering
    verbose: bool = True,
) -> List[Tuple[int, int]]:
    """
    Hard constraint selection:
      - Eligible candidates MUST be within near_radius_cells of a deferred target.
      - Enforce min spacing between chosen sites (min_sep_cells).
      - If not enough eligible cells exist, returns fewer than n_sites.
    """
    rng = np.random.default_rng(seed)

    # Filter allowed by forbidden
    keep = []
    for (i, j) in allowed_ij:
        if (int(i), int(j)) in forbidden_ij:
            continue
        keep.append((int(i), int(j)))
    if len(keep) == 0:
        raise RuntimeError("[MAR] No allowable cells remain after applying forbidden mask(s).")
    keep = np.array(keep, dtype=int)

    # Hard filter by distance to deferred targets
    dmin = compute_min_dist_cells(keep, target_ij=target_ij)
    near_mask = np.isfinite(dmin) & (dmin <= float(near_radius_cells))
    pool = keep[near_mask]
    pool_dmin = dmin[near_mask]

    if verbose:
        print(f"[MAR] Candidates total (after forbidden): {keep.shape[0]:,}")
        print(f"[MAR] HARD filter: within {near_radius_cells} cells of deferred targets: {pool.shape[0]:,}")
        print(f"[MAR] Min separation between MAR sites (cells): {min_sep_cells}")

    if pool.shape[0] == 0:
        if verbose:
            print("[MAR] No eligible cells within hard distance threshold. Returning 0 sites.")
        return []

    # Bin extents based on the eligible pool
    i_all = pool[:, 0]
    j_all = pool[:, 1]
    i_min, i_max = int(i_all.min()), int(i_all.max())
    j_min, j_max = int(j_all.min()), int(j_all.max())

    # bins ~ sqrt(n_sites)
    nbin_r = max(int(np.floor(np.sqrt(n_sites))), 1)
    nbin_c = max(int(np.ceil(n_sites / max(nbin_r, 1))), 1)

    r_edges = np.linspace(i_min, i_max + 1, nbin_r + 1, dtype=int)
    c_edges = np.linspace(j_min, j_max + 1, nbin_c + 1, dtype=int)

    rb = np.searchsorted(r_edges, i_all, side="right") - 1
    cb = np.searchsorted(c_edges, j_all, side="right") - 1
    rb = np.clip(rb, 0, nbin_r - 1)
    cb = np.clip(cb, 0, nbin_c - 1)
    b_all = rb * nbin_c + cb

    bins: List[List[int]] = [[] for _ in range(nbin_r * nbin_c)]
    for idx in range(pool.shape[0]):
        bins[int(b_all[idx])].append(idx)

    # weights: closer candidates more likely within a bin
    w = 1.0 / np.power(pool_dmin + 1.0, float(prefer_near_power))

    chosen: List[Tuple[int, int]] = []
    chosen_set: Set[Tuple[int, int]] = set()

    # Pass 1: one pick per bin
    for b in rng.permutation(len(bins)):
        if len(chosen) >= n_sites:
            break
        idxs = bins[b]
        if not idxs:
            continue

        # candidates in this bin that satisfy min spacing
        avail = []
        for ii0 in idxs:
            cell = (int(pool[ii0, 0]), int(pool[ii0, 1]))
            if cell in chosen_set:
                continue
            if min_sep_cells and min_sep_cells > 0:
                if _min_dist_to_chosen(cell, chosen) < float(min_sep_cells):
                    continue
            avail.append(ii0)

        if not avail:
            continue

        idxs_arr = np.array(avail, dtype=int)
        pick_local = _weighted_choice(rng, idxs_arr, w[idxs_arr])
        cell = (int(pool[pick_local, 0]), int(pool[pick_local, 1]))
        chosen.append(cell)
        chosen_set.add(cell)

    # Pass 2: fill remaining (random across pool) while enforcing min spacing
    if len(chosen) < n_sites:
        all_idxs = np.arange(pool.shape[0], dtype=int)
        rng.shuffle(all_idxs)
        for ii0 in all_idxs:
            if len(chosen) >= n_sites:
                break
            cell = (int(pool[ii0, 0]), int(pool[ii0, 1]))
            if cell in chosen_set:
                continue
            if min_sep_cells and min_sep_cells > 0:
                if _min_dist_to_chosen(cell, chosen) < float(min_sep_cells):
                    continue
            chosen.append(cell)
            chosen_set.add(cell)

    if verbose:
        if len(chosen) < n_sites:
            print(f"[MAR] Placed {len(chosen)} of {n_sites} requested sites (hard distance / spacing constraints).")
        print(f"[MAR] Eligible-pool selection: bins={nbin_r}x{nbin_c}, chosen={len(chosen)} sites")

    # Return as 1-indexed (row,col)
    return [(i + 1, j + 1) for (i, j) in chosen]


# --------------------------
# OUTPUT: MAR LOCATIONS SHAPEFILE / CSV
# --------------------------
def write_mar_locations_outputs(
    gwf,
    mar_cells_1idx: List[Tuple[int, int]],
    out_basepath: str,
    *,
    layer_0idx: int = 1,
    verbose: bool = True,
):
    """
    Write MAR locations to:
      - CSV: <out_basepath>.csv
      - Shapefile: <out_basepath>.shp (if geopandas available)
    """
    mgrid = gwf.modelgrid

    rows = []
    for n, (r1, c1) in enumerate(mar_cells_1idx, start=1):
        i = int(r1) - 1
        j = int(c1) - 1
        try:
            x = float(np.asarray(mgrid.xcellcenters)[i, j])
            y = float(np.asarray(mgrid.ycellcenters)[i, j])
        except Exception:
            x = np.nan
            y = np.nan

        rows.append(
            {
                "site_id": n,
                "k_0idx": int(layer_0idx),
                "i_0idx": int(i),
                "j_0idx": int(j),
                "row_1idx": int(r1),
                "col_1idx": int(c1),
                "x": x,
                "y": y,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = f"{out_basepath}.csv"
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"[MAR] Wrote MAR locations CSV: {out_csv} (rows={len(df)})")

    if gpd is None or Point is None:
        if verbose:
            print("[MAR] geopandas/shapely not available; skipping shapefile output.")
        return

    geoms = [Point(float(r["x"]), float(r["y"])) if np.isfinite(r["x"]) and np.isfinite(r["y"]) else None for r in rows]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms)

    # try attach CRS if present on modelgrid
    crs = None
    try:
        crs = getattr(mgrid, "crs", None)
    except Exception:
        crs = None
    if crs is not None:
        try:
            gdf = gdf.set_crs(crs, allow_override=True)
        except Exception:
            pass

    out_shp = f"{out_basepath}.shp"
    gdf.to_file(out_shp, driver="ESRI Shapefile")
    if verbose:
        print(f"[MAR] Wrote MAR locations shapefile: {out_shp}")


# --------------------------
# CORE WORK
# --------------------------
def add_or_replace_mar_wel_package_inplace(
    sim_ws: str,
    *,
    model_name: str,
    n_mar_sites: int,
    q_acftyr_each_site: float,
    mar_start_sp_1idx: int,
    wel_pname: str = "wel_mar",
    bin_folder: str = os.path.join("..", "..", "bin"),
    # write/external controls
    externalize_mar_only: bool = False,  # set False to keep files in workspace
    external_folder: str = "external",
    overwrite_existing_mar: bool = True,
    verbose: bool = True,
    # seasonal controls
    seasonal: bool = True,
    pred_start_year: int = 2024,
    pred_start_month: int = 1,
    active_months=(6, 7, 8),
    # selection controls
    seed: int = 123,
    deferred_csv: Optional[str] = None,
    hard_max_dist_cells: int = 3,
    min_sep_cells: float = 2.0,
    prefer_near_power: float = 2.0,
    # flood mask controls
    apply_flood_mask: bool = True,
    head_above_top_tol: float = 0.01,
    hds_path: Optional[str] = None,
    # outputs
    write_locations: bool = True,
    locations_out_base: Optional[str] = None,
    # run control
    run_model: bool = True,
):
    sim_ws = str(Path(sim_ws).resolve())
    exe = find_mf6_exe(bin_folder=bin_folder)

    if verbose:
        print(f"[MAR] Loading simulation from: {sim_ws}")
        print(f"[MAR] mf6 exe: {exe}")

    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    sim.exe_name = exe

    gwf = sim.get_model(model_name)
    if gwf is None:
        raise ValueError(f"[MAR] Could not find model '{model_name}' in simulation at {sim_ws}")

    # remove existing mar package if present
    existing = gwf.get_package(wel_pname)
    if existing is not None:
        if overwrite_existing_mar:
            if verbose:
                print(f"[MAR] Removing existing package '{wel_pname}' and replacing it.")
            gwf.remove_package(wel_pname)
        else:
            raise ValueError(f"[MAR] Package '{wel_pname}' exists; set overwrite_existing_mar=True to replace.")

    # Forbidden (i,j) from all WEL/RIV/DRN, excluding MAR pname
    forbidden_ij = collect_forbidden_ij_any_layer(
        gwf,
        pkg_types=("wel", "riv", "drn"),
        exclude_pnames={wel_pname},
        verbose=verbose,
    )

    # Flood mask
    if apply_flood_mask:
        flooded_ij = collect_flooded_ij_from_hds(
            sim_ws=sim_ws,
            model_name=model_name,
            gwf=gwf,
            hds_path=hds_path,
            head_above_top_tol=head_above_top_tol,
            verbose=verbose,
        )
        forbidden_ij = set(forbidden_ij) | set(flooded_ij)
        if verbose:
            print(f"[MAR] Unique forbidden (i,j) after adding flood mask: {len(forbidden_ij)}")

    # active candidates in injection layer (layer 2 => layer_0idx=1)
    allowed_ij = get_active_ij_from_layer(gwf, layer_0idx=1)

    # deferred targets
    if deferred_csv is None:
        raise ValueError("[MAR] deferred_csv is required for hard distance constraint placement.")
    targets_ij = load_deferred_targets_as_ij(gwf, deferred_csv=deferred_csv, verbose=verbose)

    # select
    mar_cells_1idx = select_dispersed_sites_near_targets(
        allowed_ij=allowed_ij,
        forbidden_ij=forbidden_ij,
        target_ij=targets_ij,
        n_sites=int(n_mar_sites),
        seed=int(seed),
        near_radius_cells=int(hard_max_dist_cells),
        prefer_near_power=float(prefer_near_power),
        min_sep_cells=float(min_sep_cells),
        verbose=verbose,
    )

    if len(mar_cells_1idx) == 0:
        raise RuntimeError("[MAR] No MAR sites selected under hard distance + spacing constraints.")

    # write outputs
    if write_locations:
        if locations_out_base is None:
            locations_out_base = os.path.join(sim_ws, "mar_locations")
        write_mar_locations_outputs(
            gwf,
            mar_cells_1idx=mar_cells_1idx,
            out_basepath=str(locations_out_base),
            layer_0idx=1,
            verbose=verbose,
        )

    # build SPD
    nper = int(sim.tdis.nper.data)
    q_cfd = acftyr_to_cfd(q_acftyr_each_site)
    start_kper = int(mar_start_sp_1idx) - 1

    if verbose:
        print(f"[MAR] nper = {nper}")
        print(f"[MAR] MAR starts at SP {mar_start_sp_1idx} (1-indexed) => kper {start_kper} (0-indexed)")
        print(f"[MAR] N sites = {len(mar_cells_1idx)} (row,col 1-indexed)")
        print(f"[MAR] First 10 sites = {mar_cells_1idx[:10]}")
        print(f"[MAR] Q per site = {q_acftyr_each_site:,.3f} ac-ft/yr = {q_cfd:,.3f} ft^3/day")
        if seasonal:
            print(
                f"[MAR] MODE = seasonal (monthly); pred_start = {pred_start_year:04d}-{pred_start_month:02d}; "
                f"ON months = {list(active_months)}"
            )
        else:
            raise ValueError("[MAR] This script is configured for seasonal=True only.")

    spd = build_mar_spd_seasonal_monthly(
        nper=nper,
        mar_cells_1idx=mar_cells_1idx,
        mar_start_sp_1idx=mar_start_sp_1idx,
        layer_0idx=1,
        q_cfd_each_site=q_cfd,
        pred_start_year=pred_start_year,
        pred_start_month=pred_start_month,
        active_months=active_months,
        verbose=verbose,
    )

    # Create MAR WEL package
    wel_pkg = flopy.mf6.ModflowGwfwel(
        gwf,
        pname=wel_pname,
        filename=f"{model_name}.{wel_pname}",  # stays in workspace
        print_input=True,
        print_flows=True,
        save_flows=True,
        stress_period_data=spd,
        maxbound=len(mar_cells_1idx),
    )

    # Externalize MAR only (optional)
    if externalize_mar_only:
        # Put external files in the model workspace (not ./external/)
        ext_folder = str(external_folder).strip() if external_folder is not None else "."
        if ext_folder == "":
            ext_folder = "."

        # If user passed ".", do NOT create a subfolder
        if ext_folder not in (".", "./"):
            (Path(sim_ws) / ext_folder).mkdir(parents=True, exist_ok=True)

        wel_pkg.set_all_data_external(
            check_data=False,
            external_data_folder=ext_folder,   # "." means model workspace
            base_name=wel_pname,               # helps keep file names grouped
            binary=False,
        )
        if verbose:
            print(f"[MAR] Externalized '{wel_pname}' into model workspace folder '{ext_folder}'")
    else:
        if verbose:
            print("[MAR] Keeping MAR package inline (not externalized).")

    # Write (always)
    sim.write_simulation()

    # Run (optional)
    if run_model:
        success, buff = sim.run_simulation(silent=not verbose, report=True)
        if not success:
            raise RuntimeError(f"[MAR] MF6 run failed.\n--- tail ---\n{tail_text(buff)}")
        if verbose:
            print("[MAR] MF6 run completed successfully.")
    else:
        if verbose:
            print("[MAR] RUN_MODEL=False -> skipping MF6 run (files written only).")

    return sim_ws, mar_cells_1idx


# --------------------------
# MAIN
# --------------------------
def main():
    # ==========================
    # USER EDITS START HERE
    # ==========================

    # Use base model folder IN PLACE
    BASE_MODEL_DIR = os.path.join("post_ies_scen02_baseline_elk2lay")
    MODEL_NAME = "elk_2lay"

    # Existing MAR package name to replace
    MAR_WEL_PNAME = "wel_mar"

    # Predictive period start
    MAR_START_SP_1IDX = 325  # 1-indexed

    # Number of dispersed MAR injection sites (will place fewer if constraints bind)
    N_MAR_SITES = 30

    # Injection rate per site (ac-ft/yr)
    INKSTER_AREA_AC = 7.1
    TARGET_AREA_AC = 10.0
    BASE_ACFTYR = 600.0
    Q_ACFTYR_EACH_SITE = BASE_ACFTYR * (TARGET_AREA_AC / INKSTER_AREA_AC)

    BIN_FOLDER = os.path.join("..", "..", "bin")

    # Keep MAR files in workspace (NOT external/)
    EXTERNALIZE_MAR_ONLY = True
    EXTERNAL_FOLDER = "."  # ignored if EXTERNALIZE_MAR_ONLY=False

    # Seasonal monthly injection controls
    SEASONAL_MAR = True
    SUMMER_MONTHS = (5, 6, 7, 8, 9)  # May-Sep

    # Define calendar month for SP 325
    PRED_START_YEAR = 2024
    PRED_START_MONTH = 1

    # Random seed
    SEED = 123

    # Deferred-permit bias / hard distance constraint
    DEFERRED_PER_WELL_CSV = os.path.join(
        "data", "processed", "water_use", "deferred_investigation", "deferred_requested_per_well.csv"
    )
    HARD_MAX_DIST_CELLS = 3           # DO NOT place MAR if >3 cells from a deferred well-cell
    MIN_SEP_CELLS = 2.0               # prevent “on top of each other” clustering
    PREFER_NEAR_POWER = 2.0           # tie-break weighting within bins (still random)

    # Flood avoidance using heads (.hds)
    APPLY_FLOOD_MASK = True
    HEAD_ABOVE_TOP_TOL = 0.01
    HDS_PATH = None  # auto-find <MODEL_NAME>.hds or first *.hds in BASE_MODEL_DIR

    # Output products
    WRITE_LOCATIONS = True
    LOCATIONS_OUT_BASE = None  # if None, writes to <BASE_MODEL_DIR>/mar_locations.{csv,shp}

    # Run toggle
    RUN_MODEL = False  # <- set True to run MF6 after writing

    # ==========================
    # USER EDITS END HERE
    # ==========================

    base_ws = Path(BASE_MODEL_DIR).resolve()
    if not base_ws.exists():
        raise FileNotFoundError(f"[MAR] BASE_MODEL_DIR does not exist: {base_ws}")

    print(f"[MAR] Updating MAR WEL package IN PLACE in: {base_ws}")

    add_or_replace_mar_wel_package_inplace(
        sim_ws=str(base_ws),
        model_name=MODEL_NAME,
        n_mar_sites=N_MAR_SITES,
        q_acftyr_each_site=Q_ACFTYR_EACH_SITE,
        mar_start_sp_1idx=MAR_START_SP_1IDX,
        wel_pname=MAR_WEL_PNAME,
        bin_folder=BIN_FOLDER,
        externalize_mar_only=EXTERNALIZE_MAR_ONLY,
        external_folder=EXTERNAL_FOLDER,
        overwrite_existing_mar=True,
        verbose=True,
        seasonal=SEASONAL_MAR,
        pred_start_year=PRED_START_YEAR,
        pred_start_month=PRED_START_MONTH,
        active_months=SUMMER_MONTHS,
        seed=SEED,
        deferred_csv=DEFERRED_PER_WELL_CSV,
        hard_max_dist_cells=HARD_MAX_DIST_CELLS,
        min_sep_cells=MIN_SEP_CELLS,
        prefer_near_power=PREFER_NEAR_POWER,
        apply_flood_mask=APPLY_FLOOD_MASK,
        head_above_top_tol=HEAD_ABOVE_TOP_TOL,
        hds_path=HDS_PATH,
        write_locations=WRITE_LOCATIONS,
        locations_out_base=LOCATIONS_OUT_BASE,
        run_model=RUN_MODEL,
    )


if __name__ == "__main__":
    main()
