#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELK – MP7 residence time heatmap (Layer 2 seeds, exclude DRN/RIV/WEL)

Style:
- mirrors elk02_model_build.py: path constants + main() + __main__ block :contentReference[oaicite:1]{index=1}

Outputs (written under MP_WS):
- particle_start_cells.csv              (particle_index, k,i,j)
- residence_days_L2.npy                 (nrow,ncol) float, NaN where none
- residence_days_L2.csv                 (i,j,residence_days)
- residence_heatmap_L2.png              (plan-view heat map)
- MODPATH listing / endpoint / pathline files

Notes:
- Seeds ONE particle per active cell in layer 2 (0-based k=1)
- Excludes cells that appear in DRN/RIV/WEL stress_period_data at ANY kper (union)
- Releases particles once at the predictive start time
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import flopy


# =============================================================================
# USER SETTINGS (edit these)
# =============================================================================

# ---- where your FINAL calibrated MF6 model lives (or the folder you want to track)
SIM_WS = Path(r"master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base")

# ---- MF6 model name inside the simulation (e.g., "elk_2lay")
MODEL_NAME = "elk_2lay"

# ---- MODPATH 7 executable:
# Option A: full path to mp7.exe
MP7_EXE = os.path.join("..", "..", "bin", "win", "mp7.exe")
# Option B: just "mp7" if it's on PATH: MP7_EXE = Path("mp7")

# ---- output workspace for modpath run
MP_WS = SIM_WS / "mp7_residence_L2"

# ---- which layer to seed particles (1-based for humans)
SEED_LAYER_1BASED = 2  # layer 2

# ---- predictive start definition:
# Strongly recommended: explicitly set your predictive start stress period (0-based)
PRED_START_KPER: Optional[int] = None  # e.g., 540  (set this if you know it)

# If PRED_START_KPER is None, we try to auto-detect using START_DATE_TIME and this year threshold.
PRED_YEAR = 2024

# ---- MP7 porosity
POROSITY = 0.15

# ---- plotting
OVERLAY_BCS_ON_PLOT = True
FIG_DPI = 300

# =============================================================================
# Helpers
# =============================================================================

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def _find_mp7_exe(mp7_exe: Path) -> str:
    """
    Resolve MP7 executable: either a real file path or something on PATH.
    (Your notebook example runs mp7 via subprocess/cwd style) :contentReference[oaicite:2]{index=2}
    """
    mp7_exe = Path(mp7_exe)
    if mp7_exe.is_file():
        return str(mp7_exe.resolve())

    # try PATH if user provided "mp7" / "mp7.exe"
    from shutil import which
    hit = which(str(mp7_exe))
    if hit:
        return hit

    raise FileNotFoundError(f"Could not find MP7 executable: {mp7_exe}")


def _read_tdis_start_datetime(tdis_path: Path) -> Optional[pd.Timestamp]:
    """Parse START_DATE_TIME from TDIS if present."""
    if not _exists(tdis_path):
        return None
    pat = re.compile(r"^\s*START_DATE_TIME\s+(.+?)\s*$", re.IGNORECASE)
    for line in tdis_path.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            raw = m.group(1).strip().strip("'").strip('"')
            try:
                return pd.to_datetime(raw)
            except Exception:
                return None
    return None


def _compute_sp_start_totim_days(sim: flopy.mf6.MFSimulation) -> np.ndarray:
    """
    Returns t0 array length nper+1 in DAYS:
      t0[kper] = start time of stress period kper
      t0[nper] = end time of simulation
    """
    per = sim.tdis.perioddata.array
    perlen = np.array([float(r[0]) for r in per], dtype=float)
    t0 = np.concatenate([[0.0], np.cumsum(perlen)])
    return t0


def _detect_pred_start(sim: flopy.mf6.MFSimulation, sim_ws: Path, model_name: str) -> Tuple[int, float, str]:
    """
    Determine predictive start:
      - if PRED_START_KPER provided -> use it
      - else if START_DATE_TIME exists -> pick first SP with end-year >= PRED_YEAR
      - else -> fallback to 0
    """
    nper = int(sim.tdis.nper.data)
    t0 = _compute_sp_start_totim_days(sim)

    if PRED_START_KPER is not None:
        k = int(PRED_START_KPER)
        if not (0 <= k < nper):
            raise ValueError(f"PRED_START_KPER={k} out of range (0..{nper-1})")
        return k, float(t0[k]), "Predictive start set from PRED_START_KPER."

    tdis_path = sim_ws / f"{model_name}.tdis"
    start_dt = _read_tdis_start_datetime(tdis_path)
    if start_dt is None:
        return 0, 0.0, (
            "No START_DATE_TIME in TDIS and PRED_START_KPER is None. "
            "Defaulting predictive start to kper=0. Set PRED_START_KPER to be safe."
        )

    per = sim.tdis.perioddata.array
    perlen = np.array([float(r[0]) for r in per], dtype=float)
    end_days = np.cumsum(perlen)
    end_dt = start_dt + pd.to_timedelta(end_days, unit="D")
    idx = np.where(end_dt.year >= int(PRED_YEAR))[0]
    if len(idx) == 0:
        k = nper - 1
        return k, float(t0[k]), f"No stress period ends in/after {PRED_YEAR}; using last kper={k}."
    k = int(idx[0])
    return k, float(t0[k]), f"Auto-detected predictive start from START_DATE_TIME={start_dt.date()} and PRED_YEAR={PRED_YEAR}."


def _extract_cellid_from_row(row):
    """
    Robustly extract (k,i,j) from a MF6 stress_period_data row.

    Handles:
      - numpy structured recarray rows with 'cellid'
      - structured rows with separate k/i/j fields
      - tuples/lists where row == (k,i,j)
      - tuples/lists where row[0] == (k,i,j)
      - tuples/lists where first three entries are k,i,j
    """
    # 1) numpy structured record with named fields
    dtype = getattr(row, "dtype", None)
    names = getattr(dtype, "names", None) if dtype is not None else None
    if names:
        if "cellid" in names:
            cid = row["cellid"]
            return int(cid[0]), int(cid[1]), int(cid[2])
        # sometimes k/i/j exist as separate fields
        if all(n in names for n in ("k", "i", "j")):
            return int(row["k"]), int(row["i"]), int(row["j"])
        if all(n in names for n in ("layer", "row", "column")):
            return int(row["layer"]), int(row["row"]), int(row["column"])

    # 2) row itself might be (k,i,j)
    if isinstance(row, (tuple, list, np.ndarray)):
        if len(row) == 3 and all(np.isscalar(x) for x in row):
            return int(row[0]), int(row[1]), int(row[2])

        # 3) first element might be (k,i,j)
        if len(row) > 0:
            first = row[0]
            if isinstance(first, (tuple, list, np.ndarray)) and len(first) == 3:
                return int(first[0]), int(first[1]), int(first[2])

        # 4) sometimes the first three entries are k,i,j (flat)
        if len(row) >= 3 and all(np.isscalar(x) for x in row[:3]):
            return int(row[0]), int(row[1]), int(row[2])

    raise TypeError(f"Could not extract cellid from row of type {type(row)}: {row!r}")


def _collect_bc_cells_layer_union(gwf: flopy.mf6.ModflowGwf, layer_index0: int) -> Set[Tuple[int, int, int]]:
    """
    Union of all (k,i,j) cells in ALL DRN*/RIV*/WEL* packages across stress periods,
    filtered to a single layer.
    """
    bc_cells: Set[Tuple[int, int, int]] = set()

    # Grab all package names so we include drn_ag, drn_wl, etc.
    pkg_names = [p for p in gwf.get_package_list()]
    target = ("drn", "riv", "wel")

    for pname in pkg_names:
        low = pname.lower()
        if not low.startswith(target):
            continue

        pkg = gwf.get_package(pname)
        spd = getattr(pkg, "stress_period_data", None)
        if spd is None:
            continue

        # Prefer iterating the internal dict if present (less format surprise)
        data_obj = getattr(spd, "data", None)

        if isinstance(data_obj, dict):
            # data_obj: {kper: recarray/list}
            items = data_obj.items()
        else:
            # fallback: use get_data per kper
            nper = int(gwf.simulation.tdis.nper.data)
            items = ((kper, spd.get_data(kper=kper)) for kper in range(nper))

        for kper, rec in items:
            if rec is None:
                continue
            try:
                for row in rec:
                    k, i, j = _extract_cellid_from_row(row)
                    if k == layer_index0:
                        bc_cells.add((k, i, j))
            except Exception:
                # if this kper is weirdly encoded, skip rather than crashing
                continue

    return bc_cells



def _build_seed_cells(idomain: np.ndarray, layer_index0: int, bc_cells_layer: Set[Tuple[int, int, int]]):
    """Return arrays kk, ii, jj for particle starts: active in layer AND not BC."""
    active = (idomain[layer_index0] > 0)
    ii, jj = np.where(active)
    keep = np.ones(ii.shape[0], dtype=bool)
    for n in range(ii.shape[0]):
        if (layer_index0, int(ii[n]), int(jj[n])) in bc_cells_layer:
            keep[n] = False
    ii = ii[keep]
    jj = jj[keep]
    kk = np.full_like(ii, layer_index0, dtype=int)
    return kk, ii, jj


def _maybe_1based_to_0based(a: np.ndarray, max0: int) -> np.ndarray:
    """Detect and convert 1-based indices to 0-based if needed."""
    if a.size == 0:
        return a
    mn = int(np.nanmin(a))
    mx = int(np.nanmax(a))
    if mn >= 1 and mx <= (max0 + 1):
        return a - 1
    return a

def _safe_laytyp_from_mf6_npf(gwf: flopy.mf6.ModflowGwf) -> np.ndarray:
    """
    Return laytyp (length nlay) safely, avoiding MFData text-int parsing issues.

    For MF6, icelltype is per-cell (nlay,nrow,ncol).
    MODPATH laytyp is per-layer: 0=confined, >0=convertible.

    Strategy:
      - Try to read icelltype as floats, then cast to int.
      - Reduce each layer to max value in that layer.
      - If anything fails, return zeros.
    """
    nlay = int(gwf.modelgrid.nlay)

    try:
        # try multiple access patterns to avoid .array reading path
        ic = None
        try:
            ic = gwf.npf.icelltype.get_data(apply_mult=True)
        except Exception:
            ic = None

        if ic is None:
            # last resort
            ic = gwf.npf.icelltype.array

        ic = np.asarray(ic)

        # if it came in as strings like '1.000000E+00', convert via float first
        if ic.dtype.kind in ("U", "S", "O"):
            ic = ic.astype(float)

        ic = ic.astype(int)

        laytyp = np.zeros(nlay, dtype=int)
        for k in range(nlay):
            laytyp[k] = int(np.nanmax(ic[k]))
        return laytyp

    except Exception as ex:
        print(f"[warn] Could not read/parse NPF icelltype safely ({ex}); using laytyp=0 for all layers.")
        return np.zeros(nlay, dtype=int)

# =============================================================================
# Core workflow
# =============================================================================

def setup_run_modpath_and_postprocess(
    sim_ws: Path,
    model_name: str,
    mp_ws: Path,
    mp7_exe: str,
    seed_layer_1based: int,
    pred_start_totim_days: float,
    porosity: float,
):
    sim_ws = sim_ws.resolve()
    mp_ws = mp_ws.resolve()
    mp_ws.mkdir(parents=True, exist_ok=True)

    # --- load MF6
    print("[mp7] loading MF6 simulation...")
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), exe_name="mf6")
    gwf = sim.get_model(model_name)

    seed_layer0 = int(seed_layer_1based) - 1
    idomain = gwf.dis.idomain.array.copy()
    nlay = int(gwf.dis.nlay.data)
    if not (0 <= seed_layer0 < nlay):
        raise ValueError(f"Seed layer {seed_layer_1based} out of range; model has nlay={nlay}")

    # --- BC exclusion mask (union across stress periods)
    bc_cells = _collect_bc_cells_layer_union(gwf, seed_layer0)
    print(f"[mp7] BC cells excluded in layer {seed_layer_1based}: {len(bc_cells):,}")

    kk, ii, jj = _build_seed_cells(idomain, seed_layer0, bc_cells)
    print(f"[mp7] Seed particles (layer {seed_layer_1based}, active minus BCs): {len(ii):,}")

    # Save particle start key (like your example writes a key CSV) :contentReference[oaicite:3]{index=3}
    key_csv = mp_ws / "particle_start_cells.csv"
    pd.DataFrame({"k": kk, "i": ii, "j": jj}).to_csv(key_csv, index_label="particle_index")
    print(f"[mp7] wrote: {key_csv}")

    # --- Create ParticleGroup
    slocs = list(zip(kk.tolist(), ii.tolist(), jj.tolist()))
    pdat = flopy.modpath.ParticleData(
        slocs,
        structured=True,
        drape=0,
        timeoffset=float(pred_start_totim_days),  # release time in "model time" (days)
    )
    pg = flopy.modpath.ParticleGroup(
        particlegroupname="PG_L2_ACTIVE_NONBC",
        particledata=pdat,
        filename="mp7.pg.sloc",
    )

    # --- Build MP7 simulation (mirrors your setup_modpath_sim style) :contentReference[oaicite:4]{index=4}
    mp_name = f"{model_name}_mp7_residence_L{seed_layer_1based}"
    mp = flopy.modpath.Modpath7(
        modelname=mp_name,
        flowmodel=gwf,
        exe_name=mp7_exe,
        model_ws=str(mp_ws),
    )
    laytyp = _safe_laytyp_from_mf6_npf(gwf)
    print(f"[mp7] laytyp (per-layer) = {laytyp.tolist()}")

    flopy.modpath.Modpath7Bas(
        mp,
        porosity=float(porosity),
        laytyp=laytyp,
    )

    mpsim = flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="forward",
        budgetoutputoption="summary",
        referencetime=[0, 0, 0.0],
        timepointdata=[1, [0.0]],
        zonedataoption="off",
        particlegroups=[pg],
        weaksourceoption="pass_through",
        weakssinkoption="pass_through",
        stoptimeoption="extend",
    )

    mp.write_input()
    print(f"[mp7] wrote MODPATH inputs in: {mp_ws}")

    # --- Copy required MF6 support files into mp_ws (avoids path issues)
    # dis.grb is critical
    candidates = [
        sim_ws / f"{model_name}.dis.grb",
        sim_ws / f"{model_name}.hds",
    ]
    # budget file from OC, if present
    try:
        bud_rec = gwf.oc.budget_filerecord.array
        for i in range(len(bud_rec)):
            candidates.append(sim_ws / str(bud_rec[i][0]))
    except Exception:
        # common fallbacks
        candidates.extend([sim_ws / f"{model_name}.cbc", sim_ws / f"{model_name}.bud"])

    copied = 0
    for f in candidates:
        if f.exists():
            dst = mp_ws / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
    print(f"[mp7] copied {copied} MF6 support files into MP workspace")

    # --- Run MODPATH
    print("[mp7] running MODPATH 7...")
    ok, buff = mp.run_model(silent=False, report=True)
    if not ok:
        raise RuntimeError("MODPATH 7 run failed. Check the MP listing file in MP_WS.")
    print("[mp7] run complete")

    # --- Postprocess endpoints -> residence (days)
    endpoint_fname = getattr(mpsim, "endpointfilename", None) or "mp7.end"
    endpoint_path = mp_ws / str(endpoint_fname)
    if not endpoint_path.exists():
        # try also MP name based default
        # (different FloPy versions vary)
        alt = mp_ws / f"{mp_name}.end"
        if alt.exists():
            endpoint_path = alt
        else:
            raise FileNotFoundError(f"Endpoint file not found: {endpoint_path}")

    ep = flopy.utils.EndpointFile(str(endpoint_path))
    ed = ep.get_alldata()
    if ed is None or len(ed) == 0:
        raise RuntimeError("Endpoint file is empty; no particles tracked?")

    names = set(ed.dtype.names or [])
    def _pick(*cands):
        for c in cands:
            if c in names:
                return c
        raise KeyError(f"Could not find any of {cands} in endpoint dtype names: {sorted(names)}")

    k0f = _pick("k0", "kstart", "k_init")
    i0f = _pick("i0", "istart", "i_init")
    j0f = _pick("j0", "jstart", "j_init")

    # time fields vary; prefer time - time0
    if "time0" in names and "time" in names:
        t0 = np.asarray(ed["time0"], float)
        t1 = np.asarray(ed["time"], float)
        res = t1 - t0
    elif "time" in names:
        t1 = np.asarray(ed["time"], float)
        res = t1 - float(pred_start_totim_days)
    else:
        raise KeyError(f"Could not find time fields in endpoint file: {sorted(names)}")

    k0 = np.asarray(ed[k0f], float)
    i0 = np.asarray(ed[i0f], float)
    j0 = np.asarray(ed[j0f], float)

    nrow = int(gwf.modelgrid.nrow)
    ncol = int(gwf.modelgrid.ncol)

    # handle possible 1-based indexing
    k0 = _maybe_1based_to_0based(k0, max0=nlay - 1)
    i0 = _maybe_1based_to_0based(i0, max0=nrow - 1)
    j0 = _maybe_1based_to_0based(j0, max0=ncol - 1)

    # grid of residence time (days), keyed by starting cell (layer seed only)
    grid = np.full((nrow, ncol), np.nan, dtype=float)
    seed_layer0 = int(seed_layer_1based) - 1

    mask = (k0.astype(int) == seed_layer0)
    i0i = i0[mask].astype(int)
    j0j = j0[mask].astype(int)
    rr = res[mask].astype(float)

    good = (i0i >= 0) & (i0i < nrow) & (j0j >= 0) & (j0j < ncol)
    i0i, j0j, rr = i0i[good], j0j[good], rr[good]

    # if duplicates: keep max residence (conservative)
    for r_i, c_j, v in zip(i0i, j0j, rr):
        if np.isnan(grid[r_i, c_j]) or v > grid[r_i, c_j]:
            grid[r_i, c_j] = v

    npy_path = mp_ws / "residence_days_L2.npy"
    np.save(npy_path, grid)
    print(f"[post] wrote: {npy_path}")

    out_csv = mp_ws / "residence_days_L2.csv"
    ri, rj = np.where(~np.isnan(grid))
    pd.DataFrame({"i": ri, "j": rj, "residence_days": grid[ri, rj]}).to_csv(out_csv, index=False)
    print(f"[post] wrote: {out_csv}")

    # --- Plot heatmap (plan view)
    idom2d = idomain[seed_layer0] > 0
    data = np.ma.masked_where(~idom2d | np.isnan(grid), grid)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    mv = flopy.plot.PlotMapView(model=gwf, layer=seed_layer0, ax=ax)
    pm = mv.plot_array(data)
    cb = plt.colorbar(pm, ax=ax, shrink=0.85)
    cb.set_label("Residence time (days)")
    ax.set_title(f"{model_name} – residence time (days) – Layer {seed_layer_1based}", loc="left")
    ax.set_aspect("equal")

    if OVERLAY_BCS_ON_PLOT:
        # overlay package cells (if present)
        if hasattr(gwf, "riv"):
            mv.plot_bc(package=gwf.riv, color="cyan", linewidth=0.5)
        if hasattr(gwf, "drn"):
            mv.plot_bc(package=gwf.drn, color="magenta", linewidth=0.5)
        if hasattr(gwf, "wel"):
            mv.plot_bc(package=gwf.wel, color="black", linewidth=0.5)

    out_png = mp_ws / "residence_heatmap_L2.png"
    fig.savefig(out_png, dpi=FIG_DPI, facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote: {out_png}")

    return grid


# =============================================================================
# Main
# =============================================================================

def main():
    print("ELK MP7 residence time (Layer 2) – starting...")

    if not SIM_WS.exists():
        raise FileNotFoundError(f"SIM_WS not found: {SIM_WS}")

    mp7_exe = _find_mp7_exe(MP7_EXE)
    print(f"[cfg] SIM_WS  : {SIM_WS}")
    print(f"[cfg] MODEL   : {MODEL_NAME}")
    print(f"[cfg] MP7_EXE  : {mp7_exe}")
    print(f"[cfg] MP_WS    : {MP_WS}")
    print(f"[cfg] LAYER    : {SEED_LAYER_1BASED}")
    print(f"[cfg] POROSITY : {POROSITY}")

    # load once to find predictive start
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(SIM_WS), exe_name="mf6")
    pred_kper, pred_totim_days, note = _detect_pred_start(sim, SIM_WS, MODEL_NAME)
    print(f"[pred] {note}")
    print(f"[pred] pred_start_kper={pred_kper}, pred_start_totim_days={pred_totim_days:.3f}")

    setup_run_modpath_and_postprocess(
        sim_ws=SIM_WS,
        model_name=MODEL_NAME,
        mp_ws=MP_WS,
        mp7_exe=mp7_exe,
        seed_layer_1based=SEED_LAYER_1BASED,
        pred_start_totim_days=pred_totim_days,
        porosity=POROSITY,
    )

    print("Done.")


if __name__ == "__main__":
    main()
