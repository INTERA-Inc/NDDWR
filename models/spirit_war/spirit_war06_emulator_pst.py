"""
Use the base realization of the calibrated projected model
    - For the projection use the 'buisiness as usual' scenario
    - Then, we can perturb from there to determine the response at X, Y, and Z
      years into the future
"""

import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import pyemu
import flopy
import time
import platform
import shutil
import numpy as np
import warnings
import geopandas as gpd
import random
import pandas as pd
warnings.filterwarnings("ignore")

# ------------------------------------------------------- #
# Initialization Functions
# ------------------------------------------------------- #
def base_posterior_param_forward_run(m_d0, noptmax):
    modnm = 'swww'
    m_d = m_d0 + '_forward_run_base'
    print('copying dir {0} to {1}'.format(m_d0, m_d))
    shutil.copytree(m_d0, m_d, ignore=shutil.ignore_patterns('*.cbb', '*.hds', '*.log', '*.lst', '*.rec', '*.rei', '*_obs.csv','*.pdf'))
    m_d0 = None
    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    # Update parameter set
    pst.parrep(parfile=os.path.join(m_d, f'{modnm}.{noptmax}.base.par'))
    # pst.control_data.noptmax = 0
    pst.write(os.path.join(m_d, f'{modnm}.pst'), version=2)
    prep_deps(m_d)
    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst', cwd=m_d)
    # pyemu.os_utils.run('python forward_run.py', cwd=m_d)

def init_emulator_ws(m_d_flow, pred_ws):
    src = m_d_flow
    dst = os.path.join(pred_ws)

    # delete old copy if it exists
    if os.path.exists(dst):
        shutil.rmtree(dst)
    def ignore_large_files(dir, files):
        ignored = []
        for f in files:
            # full path to check if it's a directory
            full_path = os.path.join(dir, f)
            # ignore some folders:
            if os.path.isfile(full_path) and f == 'results':
                ignored.append(f)
            elif os.path.isfile(full_path) and f == 'mult':
                 ignored.append(f)             
            elif os.path.isfile(full_path) and f == 'org':
                 ignored.append(f)               
            elif os.path.isfile(full_path) and f == 'org_mws':
                 ignored.append(f)                                        
            elif f.endswith(".rec"):
                ignored.append(f)
            elif f.endswith(".rei"):
                ignored.append(f)
            elif f.endswith(".jcb"):
                ignored.append(f)
        return ignored
 
    # copytree with ignore function
    shutil.copytree(src, dst, ignore=ignore_large_files)
    
    # Copy over the all_wells_ref.csv and idom.npy
    shutil.copy(os.path.join('data','analyzed','all_wells_ref.csv'),os.path.join(dst,'all_wells_ref.csv'))
    shutil.copy(os.path.join('.','idom.npy'),os.path.join(dst,'idom.npy'))


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


# ------------------------------------------------------------------------
# --- Prep rates in all_wells_ref.csv file using baseline pumping scenario
# ------------------------------------------------------------------------
def build_use_acft_yr_from_baseline_wels(pred_ws, year_start="2024-01-01"):
    """
    Compute use_acft_yr per well from baseline predictive WEL files by summing
    monthly volumes across one calendar year starting at year_start.

    Returns DataFrame with 0-based k,i,j and use_acft_yr.
    """
    CUFT_PER_ACFT = 43_560.0
    spd = pd.read_csv(os.path.join(".", "tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["stress_period"] = pd.to_numeric(spd["stress_period"], errors="raise").astype(int)  # 1-based
    spd["year"] = spd["start_datetime"].dt.year.astype(int)
    spd["month"] = spd["start_datetime"].dt.month.astype(int)
    spd["days"] = spd["start_datetime"].dt.days_in_month.astype(int)

    # pick the 12 stress periods for the chosen year (or starting month)
    start_dt = pd.to_datetime(year_start)
    end_dt = start_dt + pd.DateOffset(months=12)

    sp_sel = spd.loc[(spd["start_datetime"] >= start_dt) & (spd["start_datetime"] < end_dt)].copy()
    if len(sp_sel) != 12:
        raise ValueError(f"Expected 12 monthly stress periods between {start_dt.date()} and {end_dt.date()}, got {len(sp_sel)}")

    # Accumulate annual ac-ft by (layer,row,column) (1-based in file)
    ann = {}

    for _, r in sp_sel.iterrows():
        sp = int(r["stress_period"])  # matches filename suffix
        days = int(r["days"])

        wel_file = os.path.join(pred_ws, f"swww.wel_stress_period_data_{sp}.txt")
        if not os.path.exists(wel_file):
            raise FileNotFoundError(wel_file)

        df = pd.read_csv(wel_file, delim_whitespace=True, header=None,
                         names=["layer", "row", "column", "flux_cfd"])

        # monthly volume in ac-ft for each entry
        # note: pumping is negative, use abs
        df["acft_month"] = df["flux_cfd"].abs() * days / CUFT_PER_ACFT

        # sum into dict
        for (lay, row, col), v in df.groupby(["layer", "row", "column"])["acft_month"].sum().items():
            key = (int(lay), int(row), int(col))
            ann[key] = ann.get(key, 0.0) + float(v)

    out = (pd.DataFrame(
            [(k, i, j, v) for (k, i, j), v in ann.items()],
            columns=["layer", "row", "column", "use_acft_yr"]
          ))

    # convert to 0-based for all_wells_ref
    out["k"] = out["layer"] - 1
    out["i"] = out["row"] - 1
    out["j"] = out["column"] - 1
    out = out[["k", "i", "j", "use_acft_yr"]]

    return out


def build_hypothetical_wells(
    idom,
    sheyenne_pairs,
    nrow=175, ncol=155,
    # how many hypothetical wells per layer (k index). Adjust to taste.
    target_per_k={0: 150, 2: 150},
    # stratification grid size (cells). Bigger = fewer bins, more uniform.
    bin_size=10,
    seed=42
    ):
    rng = np.random.default_rng(seed)

    # sheyenne_pairs should contain 0-based (i,j)
    sheyenne_pairs = set(sheyenne_pairs)

    records = []

    for k in sorted(target_per_k.keys()):
        # ---- build eligibility mask for this k
        active = (idom[k, :, :] == 1)

        # Skip Northern Spiritwood (for your k==2 rule)
        if k == 2:
            # i < 48 excluded
            active[:48, :] = False

            # Skip sw_sheyenne (only for k==2 per your note)
            if len(sheyenne_pairs) > 0:
                # build mask from pairs
                mask_shey = np.zeros((nrow, ncol), dtype=bool)
                for (i, j) in sheyenne_pairs:
                    if 0 <= i < nrow and 0 <= j < ncol:
                        mask_shey[i, j] = True
                active[mask_shey] = False

        # If you also want to skip k==1 entirely, you simply don't include it in target_per_k

        eligible = np.argwhere(active)  # (i,j) rows
        if eligible.size == 0:
            continue

        # ---- stratify by coarse bins
        bi = eligible[:, 0] // bin_size
        bj = eligible[:, 1] // bin_size

        # group indices by bin
        # key = bi * big + bj
        big = (ncol // bin_size) + 10
        keys = bi * big + bj

        # get unique bins
        uniq_keys = np.unique(keys)

        # how many to pick per bin (roughly)
        target = int(target_per_k[k])
        # cap target by eligible
        target = min(target, eligible.shape[0])

        # if there are many bins, sample bins; if few bins, take multiple per bin
        # compute bin -> list of eligible indices
        bin_to_inds = {}
        for idx, key in enumerate(keys):
            bin_to_inds.setdefault(key, []).append(idx)

        # decide how many from each bin
        # start with 1 per bin until we hit target, then distribute remainder
        chosen = []

        # shuffle bins
        bins = uniq_keys.copy()
        rng.shuffle(bins)

        # first pass: 1 per bin
        for key in bins:
            if len(chosen) >= target:
                break
            inds = bin_to_inds[key]
            chosen.append(inds[rng.integers(0, len(inds))])

        # second pass: fill remainder by cycling bins
        if len(chosen) < target:
            bins_cycle = list(bins)
            rng.shuffle(bins_cycle)
            bptr = 0
            while len(chosen) < target:
                key = bins_cycle[bptr % len(bins_cycle)]
                inds = bin_to_inds[key]
                chosen.append(inds[rng.integers(0, len(inds))])
                bptr += 1

        chosen = np.array(chosen, dtype=int)
        chosen_ij = eligible[chosen]  # (i,j) chosen

        # --- build records
        for (i, j) in chosen_ij:
            records.append({
                "Well": f"{k}_{i}_{j}",
                "Permits": ["hypothetical"],
                "cfd": 0,
                "layer": k + 1,      # 1-based
                "row": i + 1,        # 1-based
                "column": j + 1,     # 1-based
                "k": k,
                "i": int(i),
                "j": int(j),
                "use_type": "Irrigation",
                "use_acft_yr": 0
            })

    return pd.DataFrame(records)


def merge_baseline_acft_with_ref(ref_csv="all_wells_ref.csv",
                                 base_df=None,
                                 out_csv="all_wells_ref.csv"):
    """
    Merge baseline-derived use_acft_yr into all_wells_ref.csv,
    and ensure ALL baseline wells exist in the ref file.

    Update: Adding all active cells as available pumping locations

    Fails if any baseline WEL entries are missing from ref.
    """
    idom = np.load(os.path.join('idom.npy'))

    if base_df is None:
        raise ValueError("base_df must be provided")

    ref = pd.read_csv(ref_csv)

    # -----------------------------
    # Sanity checks on ref columns
    # -----------------------------
    for c in ["k", "i", "j"]:
        if c not in ref.columns:
            raise ValueError(f"{ref_csv} missing required column '{c}'")

    # Ensure integer indexing
    ref[["k", "i", "j"]] = ref[["k", "i", "j"]].astype(int)
    base_df[["k", "i", "j"]] = base_df[["k", "i", "j"]].astype(int)

    # --------------------------------------
    # CHECK: all baseline wells exist in ref
    # --------------------------------------
    ref_keys  = set(zip(ref["k"], ref["i"], ref["j"]))
    base_keys = set(zip(base_df["k"], base_df["i"], base_df["j"]))

    missing = base_keys - ref_keys
    if missing:
        missing = list(missing)
        example = missing[:10]
        raise ValueError(
            f"{len(missing)} baseline WEL wells are missing from {ref_csv}.\n"
            f"Example missing (k,i,j): {example}\n"
            f"Ref file must include ALL baseline wells for absolute-pumping emulator."
        )

    # -----------------------------
    # Merge baseline values
    # -----------------------------
    merged = ref.merge(
        base_df,
        on=["k", "i", "j"],
        how="left",
        suffixes=("", "_baseline"),
    )

    # -----------------------------
    # Assign use_acft_yr
    # -----------------------------
    if "use_acft_yr" in merged.columns:
        merged["use_acft_yr"] = (
            merged["use_acft_yr_baseline"]
            .fillna(merged["use_acft_yr"])
            .fillna(0.0)
        )
        merged = merged.drop(columns=["use_acft_yr_baseline"])
    else:
        merged["use_acft_yr"] = merged["use_acft_yr_baseline"].fillna(0.0)
        merged = merged.drop(columns=["use_acft_yr_baseline"])

    # ---------------------------------------------
    # Add more possible pumping locations
    # Trying some random seeding
    # --> Turning off for now
    # ---------------------------------------------
    # sw_sheyenne_cells = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','sw_sheyenne_cells.shp'))
    # sheyenne_pairs = set(zip(sw_sheyenne_cells["row"].astype(int),
    #                          sw_sheyenne_cells["column"].astype(int),
    #                          )
    #                      )
    # hypo_df = build_hypothetical_wells(idom, sheyenne_pairs, target_per_k={0:150, 2:150}, bin_size=10)
    
    # # Concat hypothetical wells
    # merged = pd.concat([merged, hypo_df],
    #                    ignore_index=True)     
    
    # Drop duplicates, preserving those from merged rather than new_df      
    # merged = merged.drop_duplicates(subset=['i','j','k'],
    #                                 keep='first')
    
    merged.to_csv(out_csv, index=False)

    return merged


# ----------------------------------------------------------
# --- Extrapolate IES acre-ft/year values to the WEL package
# ----------------------------------------------------------
def extrapolate_wel_acft():
    """
    Spiritwood/Warwick emulator helper.

    Assumptions:
      - all_wells_ref_with_baseline.csv includes ALL wells controlled in emulator
        * baseline wells have baseline scenario use_acft_yr
        * non-baseline/proposed wells start at 0
      - irrigation distro matches how baseline was built
      - monthly_stress_period_info.csv exists
    """
    # Need to import these at the function level for IES
    import math
    import calendar
    
    # ------------------------
    # Hard-coded configuration
    # ------------------------
    sp_start0 = 319  # 0-based predictive start stress period
    spd_path = os.path.join("..", "tables", "monthly_stress_period_info.csv")
    ref_path = "all_wells_ref_with_baseline.csv"
    
    acft_col = "use_acft_yr"
    use_type_col = "use_type"
    idx_cols = ("k", "i", "j")
    
    grow_months = (5, 6, 7, 8, 9)
    peak_month = 7
    sigma_mo = 1.2
    
    # If a well isn't in the baseline WEL file, only add it if acre/ft-yr != 0
    drop_zero_new_wells = True
    
    # Deterministic ordering
    sort_before_write = True
    
    # -------------------------
    # Constants + month weights
    # -------------------------
    CUFT_PER_ACFT = 43_560.0
    
    w_raw = [math.exp(-0.5 * ((m - peak_month) / sigma_mo) ** 2) for m in grow_months]
    w_sum = sum(w_raw)
    irr_wmap = {m: w / w_sum for m, w in zip(grow_months, w_raw)}
    
    # ------------------------
    # Load stress period table
    # ------------------------
    spd = pd.read_csv(spd_path).copy()
    if "stress_period" not in spd.columns or "start_datetime" not in spd.columns:
        raise ValueError("monthly_stress_period_info.csv must contain: stress_period, start_datetime")
        
    spd["stress_period"] = pd.to_numeric(spd["stress_period"], errors="raise").astype(int)  # 1-based
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"], errors="raise")
    
    spd["sp0"] = spd["stress_period"] - 1  # 0-based
    spd["year"] = spd["start_datetime"].dt.year.astype(int)
    spd["month"] = spd["start_datetime"].dt.month.astype(int)
    spd = spd.set_index("sp0", drop=False)
    
    sp_end0 = int(spd["sp0"].max())
    if sp_start0 < 0 or sp_start0 > sp_end0:
        raise ValueError(f"sp_start0={sp_start0} is outside [0, {sp_end0}]")
        
    # ------------------------
    # Load ref file (PEST-edited)
    # ------------------------
    ref = pd.read_csv(ref_path).copy()
    
    for c in idx_cols:
        if c not in ref.columns:
            raise ValueError(f"all_wells_ref.csv missing required column: {c}")
    if acft_col not in ref.columns:
        raise ValueError(f"all_wells_ref.csv missing required column: {acft_col}")
    if use_type_col not in ref.columns:
        raise ValueError(f"all_wells_ref.csv missing required column: {use_type_col}")

    ref[list(idx_cols)] = ref[list(idx_cols)].apply(pd.to_numeric, errors="raise").astype(int)
    ref[acft_col] = pd.to_numeric(ref[acft_col], errors="coerce").fillna(0.0).astype(float)
    ref[use_type_col] = ref[use_type_col].astype(str).str.strip()
    
    ref_k = ref["k"].to_numpy(int)
    ref_i = ref["i"].to_numpy(int)
    ref_j = ref["j"].to_numpy(int)
    ref_acft = ref[acft_col].to_numpy(float)
    is_irr = ref[use_type_col].str.lower().eq("irrigation").to_numpy(bool)
    
    # ---------------------------------------------------
    # Loop through predictive stress periods and update
    # ---------------------------------------------------
    for sp0 in range(sp_start0, sp_end0 + 1):
        if sp0 not in spd.index:
            raise ValueError(f"Stress period sp0={sp0} not found in monthly stress period table")
            
        year = int(spd.loc[sp0, "year"])
        month = int(spd.loc[sp0, "month"])
        days = calendar.monthrange(year, month)[1]
        
        irr_frac = irr_wmap.get(month, 0.0)  # 0 outside grow months
        mfrac = np.where(is_irr, irr_frac, 1.0 / 12.0)
        
        m_acft = ref_acft * mfrac
        
        q_cfd = (m_acft * CUFT_PER_ACFT) / float(days)
        q_cfd = -np.abs(q_cfd)
        
        # Map for this stress period: (k,i,j) 0-based -> q
        ref_map = {(int(k), int(i), int(j)): float(q)
                   for k, i, j, q in zip(ref_k, ref_i, ref_j, q_cfd)}
        
        # -------------------------------
        # Read existing baseline WEL file
        # -------------------------------
        sp_file = sp0 + 1  # filenames are 1-based
        wel_file = os.path.join(f"swww.wel_stress_period_data_{sp_file}.txt")
        if not os.path.exists(wel_file):
            raise FileNotFoundError(f"Missing WEL file for sp_file={sp_file} (sp0={sp0}): {wel_file}")
            
        existing = pd.read_csv(wel_file,
                               delim_whitespace=True,
                               header=None,
                               names=["layer", "row", "column", "flux_cfd"],
                               )
        
        # Ensure numeric
        existing[["layer", "row", "column"]] = existing[["layer", "row", "column"]].apply(pd.to_numeric, errors="raise").astype(int)
        existing["flux_cfd"] = pd.to_numeric(existing["flux_cfd"], errors="coerce").astype(float)
        
        # Build 0-based keys for matching (file is 1-based)
        existing["_key0"] = list(zip(existing["layer"] - 1, existing["row"] - 1, existing["column"] - 1))
        ex_index = {k: idx for idx, k in enumerate(existing["_key0"].to_list())}
        
        # -----------------------------------
        # Overwrite all baseline wells in file
        # -----------------------------------
        for key0, idx in ex_index.items():
            if key0 in ref_map:
                existing.loc[idx, "flux_cfd"] = ref_map[key0]
            else:
                pass
                
        # -----------------------------------------
        # Add wells from ref that are not in baseline
        # -----------------------------------------
        missing = [key0 for key0 in ref_map.keys() if key0 not in ex_index]
        if missing:
            rows = []
            for (k0, i0, j0) in missing:
                q = ref_map[(k0, i0, j0)]
                if drop_zero_new_wells and abs(q) == 0.0:
                    continue
                rows.append((k0 + 1, i0 + 1, j0 + 1, q))  # write 1-based
            existing = existing.drop(columns=["_key0"])
            if rows:
                add_df = pd.DataFrame(rows, columns=["layer", "row", "column", "flux_cfd"])
                existing = pd.concat([existing, add_df], ignore_index=True)
        else:
            existing = existing.drop(columns=["_key0"])

        if sort_before_write:
            existing = existing.sort_values(["layer", "row", "column"]).reset_index(drop=True)

        existing.to_csv(wel_file, sep=" ", header=False, index=False)
        
    return


# -------------------------------------
# Helper to track heads as observations
# -------------------------------------
def record_head_arrays(ws='.'):
    """
    Record active heads as 1D arrays for IES observations
    """
    import flopy
    idom = np.load('idom.npy')
    hds = flopy.utils.HeadFile(os.path.join(ws,'swww.hds'))
    heads = hds.get_alldata()
    fnames = []
    # Track heads corresponding to December of each year -> Zero based indexing
    # Only track some snapshots of all active heads
    # May 2043 (Spring high), August 2043 (Fall low), and end of model in December 2043
    times = [551, 554, 558]
    for kper in times:
        # Only track layers 1 and 3 (Warwick and Spiritwood)
        for layer in [0,2]:
            # Saving as 1-based filed
            fname = f"hds_layer{layer}_kper{kper+1}.txt"
            fnames.append(fname)
            heads_sp = heads[kper,layer,:,:].copy()
            mask = (idom[layer,:,:] == 1)
            head_active = heads_sp[mask]
            np.savetxt(os.path.join(ws,fname),head_active,fmt='%15.6e')
    return fnames


# Transient head observations
def head_targets_process():
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    # m = sim.get_model('swww')

    # Load and clean up the ss hobs
    df = pd.read_csv("swww.ss_head.obs.output.csv",index_col=0)

    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values-1,unit='d')
    df.columns = [c.lower().replace(".","-") for c in df.columns]
    df.index.name = "datetime"
    dfs = [df]
    df.to_csv("swww.ss_head.obs.output.csv")

    # Load and clean up the transient hobs
    df = pd.read_csv("swww.trans_head.obs.output.csv",index_col=0)

    # Keep only the heads at the end of the stress period - no intermediate timesteps
    df = df.loc[df.index % 1 == 0]

    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values-1,unit='d')
    df.columns = [c.lower().replace(".","-") for c in df.columns]
    df.index.name = "datetime"
    dfs.append(df)
    df.to_csv('swww.trans_head.obs.output.csv')

    return dfs

def init_head_targets_process(d):
    b_d = os.getcwd()
    os.chdir(d)
    head_targ_dfs = head_targets_process()
    os.chdir(b_d)
    return head_targ_dfs

# ----------------------- #
# Setup PEST
# ----------------------- #
def setup_pstfrom(org_d='.',
                  modnm='swww',
                  template='emulator_template',
                  num_reals=10):
    
    # Copy over org_d to temp_d
    temp_d = org_d + '_temp'
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(org_d,temp_d)
    
    if os.path.exists(template):
        shutil.rmtree(template)
    
    # Copy over the head observations
    shutil.copy2(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'),os.path.join(temp_d,'transient_well_targets_lookup_shrt.csv'))
    # shutil.copy2(os.path.join('data','raw','water_lvl_targs_manual_ly_assign.csv'),os.path.join(temp_d,'water_lvl_targs_manual_ly_assign.csv'))
    shutil.copy2(os.path.join('data', 'analyzed', 'transient_well_targets.csv'), os.path.join(temp_d, 'transient_well_targets.csv'))
    
    # load flow model and model info:
    sim = flopy.mf6.MFSimulation.load(sim_ws=temp_d, 
                                      exe_name='mf6')
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model(f'{modnm}')
    template_ws = os.path.join('emulator_template')
    
    # Remove org and mult folders
    shutil.rmtree(os.path.join(temp_d,'org'))
    shutil.rmtree(os.path.join(temp_d,'mult'))

    # instantiate PEST container
    pf = pyemu.utils.PstFrom(original_d=temp_d, 
                             new_d=template_ws,
                             remove_existing=True,
                             longnames=True,
                             spatial_reference=m.modelgrid,
                             zero_based=False, 
                             start_datetime=start_datetime
                             )
    
    pf.mod_sys_cmds.append('mf6')

    # ---- Pumping wells as parameters
    # -> Writing to a reference file then applying with extrapolate_WEL()
    wel_file = 'all_wells_ref_with_baseline.csv'

    # Sigle pumping rate for entire projected period
    pf.add_parameters(wel_file,
                      par_type="grid",      # Each well will be adjustable
                      par_style='d',        # Direct type parameter
                      pargp='welrate',
                      index_cols=[3, 4, 5], # Layer, Row, Column
                      use_cols=[10],
                      par_name_base='wel_grd',
                      transform='none',
                      upper_bound=800,     # X acre-ft/year max pumping
                      lower_bound=0,        # 0 pumping
                      mfile_skip=1,
                      )
    
    # Extrapolate IES parameterized pumping to entire predictive period.
    # Handles acre-ft/year to cfd conversion and irrigation well monthly
    # distributions.
    pf.add_py_function("spirit_war06_emulator_pst.py","extrapolate_wel_acft()",
                       is_pre_cmd=True)
    
    # head arrays at specified kpers
    pf.add_py_function("spirit_war06_emulator_pst.py","record_head_arrays()",is_pre_cmd=False)
    fnames = record_head_arrays(ws=pf.new_d) # record head arrays
    for f in fnames:
        pf.add_observations(f, 
                            insfile=f+".ins",  
                            prefix=f.split(".")[0]
                            )
   
    # Add observation wells
    hdf = init_head_targets_process(template)
    pf.add_observations('swww.trans_head.obs.output.csv',
                        index_cols=['datetime'],
                        use_cols=hdf[1].columns.to_list(),
                        obsgp='transhds',
                        ofile_sep=',',
                        prefix='transhds'
                        )
    
    #---build pst---#
    pst = pf.build_pst(version=None)
    pst.control_data.noptmax = 0
    pst.write(os.path.join(pf.new_d,'swww.pst'), version=2)
    pyemu.os_utils.run('pestpp-glm swww.pst', cwd=pf.new_d)
    pst = pyemu.Pst(os.path.join(pf.new_d,"swww.pst"))

    # draw from the prior and save the ensemble in binary format
    # pe = pf.draw(num_reals, use_specsim=False)
    # pe.to_binary(os.path.join(template, 'prior.jcb'))
    # pst.pestpp_options['ies_par_en'] = 'prior.jcb'
    # pst.pestpp_options['save_binary'] = True

    # # write the updated pest control file
    # pst.write(os.path.join(pf.new_d, 'swww.pst'),version=2)

    # shutil.copy(os.path.join(pf.new_d, 'swww.obs_data.csv'),
    #             os.path.join(pf.new_d, 'swww.obs_data_orig.csv'))
    
    return template



def check_port_number(port):
    import socket, errno

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind(('', port))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            raise ValueError(f'Port {port} is already in use, please chose a different port number')
        else:
            # something else raised the socket.error exception
            print(f'Port {port} is good to go, have fun sport')

    s.close()


def run_ies(template_ws='template_d', 
            m_d=None, 
            num_workers=12, 
            noptmax=-1, 
            num_reals=None,
            init_lam=None, 
            local=True, 
            hostname=None, 
            port=4263,
            use_condor=False,
            **kwargs):

    if m_d is None:
        m_d = template_ws.replace('template', 'master')

    pst = pyemu.Pst(os.path.join(template_ws, 'swww.pst'))

    # Set control file options:
    pst.control_data.noptmax = noptmax
    # Factor of base runtime where it gives up
    # pst.pestpp_options['overdue_giveup_fac'] = 4
    # pst.pestpp_options['ies_bad_phi_sigma'] = 1.5
    # Lowering this from 1e+20 t0 1e+8
    # pst.pestpp_options['ies_bad_phi'] = 1e+8
    # Option to reinflate
    # pst.pestpp_options["ies_n_iter_reinflate"] = [-2,999]
    # pst.pestpp_options["ies_multimodal_alpha"] = 0.99

    #pst.pestpp_options['panther_agent_freeze_on_fail'] = True

    # pst.pestpp_options['save_binary'] = True
    if num_reals is not None:
        pst.pestpp_options['ies_num_reals'] = num_reals

    # Adjust derinc for the welrate group
    par_gp = pst.parameter_groups
    derinc_dict = {
        'welrate':{
                    # Use negative (pumping out), maybe....
                    "derinc":50,
                    # multiplier type
                    "inctyp":"absolute"
                   },
               }
    
    for k,v in derinc_dict.items():
        par_gp.loc[k,'derinc'] = derinc_dict[k]["derinc"]
        par_gp.loc[k,'inctyp'] = derinc_dict[k]["inctyp"]
    
    pst.write(os.path.join(template_ws, 'swww.pst'), version=2)

    prep_worker(template_ws, template_ws + '_clean')

    master_p = None

    if hostname is None:
        pyemu.os_utils.start_workers(template_ws, 
                                     'pestpp-ies',
                                       'swww.pst',
                                     num_workers=num_workers, 
                                     worker_root='.',
                                     master_dir=m_d, 
                                     local=local,
                                     port=4269)

    elif use_condor:
        check_port_number(port)

        jobid = condor_submit(template_ws=template_ws + '_clean', pstfile='swww.pst', conda_zip='nddwrpy311.tar.gz',
                              subfile='swww.sub',
                              workerfile='worker.sh', executables=['mf6', 'pestpp-ies','mp7'], request_memory=5000,
                              request_disk='22g', port=port,
                              num_workers=num_workers)

        # jwhite - commented this out so not starting local workers on the condor submit machine # no -ross
        pyemu.os_utils.start_workers(template_ws + '_clean', 'pestpp-ies', 'swww.pst', num_workers=0, worker_root='.',
                                     port=port, local=local, master_dir=m_d)

        if jobid is not None:
            # after run master is finished clean up condor by using condor_rm
            print(f'killing condor job {jobid}')
            os.system(f'condor_rm {jobid}')

    # if a master was spawned, wait for it to finish
    if master_p is not None:
        master_p.wait()


def run_pestpp(md="emulator_master",
               td="emulator_template",
               casename="swww",
               noptmax=-1,
               num_workers=10,
               num_reals=60,
               worker_root=".",
               pestpp_version="glm",
               reuse_master=False,
               cleanup=True):
    
    pst = pyemu.Pst(os.path.join(td,f"{casename}.pst"))
    pst.control_data.noptmax = noptmax

    pst.pestpp_options['ies_num_reals'] = num_reals

    par_gp = pst.parameter_groups
    derinc_dict = {
        'welrate':{
                    # Increment by 50 acre-ft/year
                    "derinc":50,
                    # absolute since use direct type pars
                    "inctyp":"absolute"
                   },
               }
    for k,v in derinc_dict.items():
        par_gp.loc[k,'derinc'] = derinc_dict[k]["derinc"]
        par_gp.loc[k,'inctyp'] = derinc_dict[k]["inctyp"]

    pst.write(os.path.join(td, f"{casename}.pst"),version=2)
    # run
    pyemu.os_utils.start_workers(td,  # the folder which contains the "template" PEST dataset
                                f'pestpp-{pestpp_version}',  # the PEST software version we want to run
                                f'{casename}.pst',  # the control file to use with PEST
                                num_workers=num_workers,  # how many agents to deploy
                                worker_root=worker_root,
                                # where to deploy the agent directories; relative to where python is running
                                master_dir=md,  # the manager directory
                                reuse_master=reuse_master,
                                cleanup=cleanup,
                                port=4099,
                                )
    return


def prep_worker(org_d, new_d,run_flex_cond=False):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)
    exts = ['rei', 'hds', 'cbc', 'ucn', 'cbb', 'ftl', 'm3d', 'tso', 'ddn','log','rec','list']
    if run_flex_cond:
        files = [f for f in os.listdir(new_d) if f.lower().split('.')[-1] in exts]
        for f in files:
            if f != 'prior.jcb':  # need prior.jcb to run ies
                os.remove(os.path.join(new_d, f))
    else:
        files = [f for f in os.listdir(new_d) if f.lower().split('.')[-1] in exts]
        for f in files:
            if f != 'cond_post.jcb' and f != 'autocorr_noise.jcb':  # need prior.jcb to run ies
                os.remove(os.path.join(new_d, f))
    mlt_dir = os.path.join(new_d, 'mult')
    for f in os.listdir(mlt_dir)[1:]:
        os.remove(os.path.join(mlt_dir, f))
    tpst = os.path.join(new_d, 'temp.pst')
    if os.path.exists(tpst):
        os.remove(tpst)


def condor_submit(template_ws, pstfile, conda_zip='nddwrpy311.tar.gz', subfile='condor.sub', workerfile='worker.sh',
                  executables=[], request_memory=4000, request_disk='10g', port=4200, num_workers=71):
    '''
    :param template_ws: path to template_ws
    :param pstfile: name of pest control file
    :param conda_zip: conda-pack zip file
    :param subfile: condor submit file name
    :param workerfile: condor worker file name
    :param executables: any executables in the template_ws that might need permissions changed
    :param request_memory: memory to request for each job
    :param request_disk: disk space
    :param port: port number, should be same as the one used when running the master
    :param num_workers: number of workers to start
    :return:
    '''
    # template_ws = os.path.join('model_ws', 'template')

    if not os.path.join(conda_zip):
        str = 'conda-pack dir {conda_zip} does not exist\n ' + 'consider running conda-pack while in your conda env\n'
        AssertionError(str)
    conda_base = conda_zip.replace('.tar.gz', '')

    # should probably remove to remove tmp files to make copying faster...
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'temp')):
        shutil.rmtree(os.path.join(cwd, 'temp'))
    shutil.copytree(os.path.join(cwd, template_ws), 'temp')

    # zip template_ws
    os.system('tar cfvz temp.tar.gz temp')

    if not os.path.exists('log'):
        os.makedirs('log')

    # write worker file
    worker_f = open(os.path.join(cwd, workerfile), 'w')
    worker_lines = ['#!/bin/sh\n',
                    '\n',
                    '# make conda-pack dir\n',
                    f'mkdir {conda_base}\n',
                    f'tar -xf {conda_zip} -C {conda_base}\n',
                    '\n',
                    '# unzip temp\n',
                    'tar xzf temp.tar.gz\n',
                    'cd temp\n',
                    '\n',
                    '# add python to path (relative)\n',
                    f'export PATH=../{conda_base}/bin:$PATH\n',
                    'python -c "print(\'python is working\')"\n',
                    'which python',
                    '\n']

    if len(executables) > 0:
        worker_lines += ['# make sure executables have permissions\n'] + [f'chmod +x {exe}\n' for exe in executables]

    worker_lines += ['\n',
                     f'./pestpp-ies {pstfile} /h $1:$2\n']
    worker_f.writelines(worker_lines)
    worker_f.close()

    sub_f = open(os.path.join(cwd, subfile), 'w')
    sublines = ['# never ever change this!\n',
                'notification = Never\n',
                '\n',
                "# just plain'ole vanilla for us!\n",
                'universe = vanilla\n',
                '\n',
                '# this will log all the worker stdout and stderr - make sure to mkdir a "./log" dir where ever\n',
                '# the condor_submit command is issued\n',
                'log = log/worker_$(Cluster).log\n',
                'output = log/worker_$(Cluster)_$(Process).out\n',
                'error = log/worker_$(Cluster)_$(Process).err\n',
                '\n', '# define what system is required\n',
                'requirements = ( (OpSys == "LINUX") && (Arch == "X86_64"))\n',
                '# how much mem each worker needs in mb\n',
                f'request_memory = {request_memory}\n',
                '\n',
                '# how many cpus per worker\n',
                'request_cpus = 1\n',
                '\n',
                '# how much disk space each worker needs in gb (append a "g")\n',
                f'request_disk = {request_disk}\n',
                '\n',
                '# the command to execute remotely on the worker hosts to start the condor "job"\n',
                f'executable = {workerfile}\n',
                '\n',
                '# the command line args to pass to worker.sh.  These are the 0) IP address/UNC name of the master host\n',
                '# and 1) the port number for pest comms.  These must in that order as they are used in worker.sh\n',
                '# ausdata-head1.cluster or 10.99.10.30 \n',
                f'arguments = ausdata-head1.cluster {port}\n',
                '\n',
                '# stream the info back to the log files\n',
                'stream_output = True\n',
                'stream_error = True\n',
                '\n',
                '# transfer the files to start the job\n',
                'should_transfer_files = YES\n',
                'when_to_transfer_output = ON_EXIT\n',
                '\n',
                '# the files to transfer before starting the job (in addition to the executable command file)\n',
                f'transfer_input_files = temp.tar.gz, {conda_zip}\n',
                '\n',
                '# number of workers to start\n',
                f'queue {num_workers}']
    sublines += ['\n',
                '# Set job priority (higher = higher priority, default is 0, max is 20)\n',
                'priority = 10\n',  # Change this value as needed
                '\n']
    sub_f.writelines(sublines)
    sub_f.close()

    os.system(f'condor_submit {subfile} > condor_jobID.txt')

    jobfn = open('condor_jobID.txt')
    lines = jobfn.readlines()
    jobfn.close()
    jobid = lines[1].split()[-1].replace('.', '')
    print(f'{num_workers} job(s) submitted to cluster {jobid}.')

    return int(jobid)



def prepare_response_matrix(md="emulator_master",response_jcb="response.jcb"):
    pst = pyemu.Pst(os.path.join(md,"swww.pst"))
    res = pst.res
    par = pst.parameter_data
    obs = pst.observation_data

    assert os.path.exists(os.path.join(md,"swww.jcb")), "no jco"
    jco = pyemu.Jco.from_binary(os.path.join(md,"swww.jcb"))

    #TODO: select specific obsnmes?
    forecasts = obs.obsnme.tolist()

    # specify dv pars
    dv_names = par.loc[par.pargp.isin(['welrate']),"parnme"]
        
    # Save as csv for emulator
    dv_names.to_csv('dashboard/app/assets/parnme.csv')
    
    # construct response matrix
    resp_mat = jco.get(row_names=forecasts, col_names=dv_names.tolist())
    resp_mat.to_coo(os.path.join(md,response_jcb))
    print(f"response matrix written to {os.path.join(md,response_jcb)}")
    
    # forecast df
    forecast_df = res.loc[forecasts,['modelled']].copy()
    forecast_df["change"] = 0.0
    forecast_df["forecast"] = forecast_df.modelled + forecast_df.change
    forecast_df.to_csv(os.path.join(md,"forecast_response.csv"))
    print(f"forecast response written to {os.path.join(md,'forecast_response.csv')}")

    return resp_mat, forecast_df


# ---- Main
if __name__ == "__main__":
    # -- Define the predictive scenario to build the emulator off of
    scen_ws = 'post_ies_scn01_baseline_ensemble'
    forward_run_scen = True
    
    # -- Create forward base run of baseline pumping scenario
    if forward_run_scen:
        print('....Creating baseline scenario forward run')
        base_posterior_param_forward_run(m_d0=scen_ws,
                                        # noptmax of the calibration run
                                         noptmax=0
                                         )
        
    # -- Specificy vars to init directories
    use_condor = False
    post_ws = scen_ws + "_forward_run_base"  # Path to foward base run from model scenario
    pred_ws = post_ws + "_emulator"          # New workspace for IES setup
    modnm = 'swww'                           # Model name
    
    # -- Make the emulator workspace
    init_workspace = True
    if init_workspace:
        print('....Prepping predictive workspace')
        init_emulator_ws(post_ws, pred_ws)
        prep_deps(pred_ws)
      
    # -- Prep the all_wels_ref.csv file with baseline pumping values
    # build baseline ac-ft/yr from predictive WEL files
    base_df = build_use_acft_yr_from_baseline_wels(pred_ws=pred_ws,
                                                   year_start="2024-01-01",
                                                   )
    
    # Save to add to display table for Emulator app
    base_df.to_csv(os.path.join('data','analyzed','baseline_scenario_pumpage.csv'),index=False)
    base_df.to_csv(os.path.join('dashboard','app','assets','baseline_scenario_pumpage.csv'),index=False)
    
    # merge
    _ = merge_baseline_acft_with_ref(ref_csv=os.path.join(pred_ws,"all_wells_ref.csv"),
                                     base_df=base_df,
                                     out_csv=os.path.join(pred_ws,"all_wells_ref_with_baseline.csv"),
                                     )
    
    # -- Condor/worker options
    if use_condor:
        num_reals_flow = 120
        num_workers_flow = 60
        hostname = '10.99.10.30'
        pid = os.getpid()
        current_time = int(time.time())
        seed_value = (pid + current_time) % 65536
        np.random.seed(seed_value)
        port = random.randint(4001, 4999)
        print(f'port #: {port}')
    else:
        num_reals_flow = 50
        num_workers_flow = 10
        hostname = None
        port = None
        
    # -- Setup IES
    print("....Setting up pst object")
    setup_pstfrom(org_d=pred_ws,
                  modnm='swww',
                  template='emulator_template',
                  num_reals=num_reals_flow,
                  )
    
    # -- Version mostly copied from Rui's script
    print("....Running glm")
    run_pestpp(md="emulator_master",
               td="emulator_template",
               casename="swww",
               noptmax=-1,
               num_workers=num_workers_flow,
               num_reals=num_reals_flow,
               worker_root=".",
               pestpp_version="glm",
               reuse_master=False,
               cleanup=True
               )
    
    # -- Creates response matrix
    print("....Processing Response Matrix")
    prepare_response_matrix(md='emulator_master',
                            response_jcb="response.jcb",
                            )
