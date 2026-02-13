import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import pyemu
import flopy
import pypestutils
from pypestutils.pestutilslib import PestUtilsLib
from pypestutils import helpers as ppu
import platform
import pandas as pd
import geopandas as gpd
import shutil
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import random
import time 
import pathlib
import re
from flopy.utils import HeadFile
import glob

import warnings            
warnings.filterwarnings('ignore')

# set fig formats:
import elk04_process_plot_results as elkpp
elkpp.set_graph_specifications()
elkpp.set_map_specifications()

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



def build_flood_ineq_cells(ws: str, modnm: str = "elk_2lay",
                           sample_stride: int = 3,
                           layer: int = 0) -> pd.DataFrame:
    """
    Select inequality cells inside a flood polygon.

    - Intersects GRID_SHP with FLOOD_POLY_SHP.
    - Samples every `sample_stride`th intersecting cell.
    - Drops cells that are inactive or already used by a HEAD obs.
    - Writes `flood_ineq_cells.csv` to `ws`.

    Returns
    -------
    pd.DataFrame with columns ['k','i','j'].
    """
    
    # path to flood polygon and grid shapefile (edit to match your paths)

    cwd = os.getcwd()
    os.chdir(ws)
    
    FLOOD_POLY_SHP = os.path.join("..", "..", "..", "gis", "input_shps", "elk", "flooded_cells_ineq.shp")
    GRID_SHP = os.path.join("..", "..", "..", "gis", "output_shps", "elk", "elk_cell_size_660ft_epsg2265_rot20.grid.shp")

    try:
        assert os.path.exists(GRID_SHP), f"GRID_SHP not found: {GRID_SHP}"
        assert os.path.exists(FLOOD_POLY_SHP), f"FLOOD_POLY_SHP not found: {FLOOD_POLY_SHP}"

        # load model to get idomain
        sim = flopy.mf6.MFSimulation.load(sim_ws=".", exe_name="mf6", load_only=["dis"])
        m = sim.get_model(modnm)
        idomain = m.dis.idomain.array  # (nlay, nrow, ncol)

        # load grid + flood polygon
        grid = gpd.read_file(GRID_SHP)
        flood = gpd.read_file(FLOOD_POLY_SHP)

        # make sure CRS match
        if grid.crs != flood.crs:
            flood = flood.to_crs(grid.crs)

        # expect grid to have zero-based i,j (if not, adapt here)
        if not {"i", "j"}.issubset(grid.columns):
            raise ValueError("GRID_SHP must contain 'i' and 'j' columns for row/col indices.")

        # intersect grid with flood polygon (inner join)
        inter = gpd.sjoin(grid, flood, how="inner", predicate="intersects")
        if inter.empty:
            print("[flood_ineq] No grid cells intersect flood polygon; nothing to do.")
            return pd.DataFrame(columns=["k", "i", "j"])

        # sort and subsample every Nth cell
        inter = inter.sort_values(["i", "j"])
        inter = inter.iloc[::sample_stride].copy()

        # build set of existing HEAD obs cells (from .obs file)
        hobs_fname = f"{modnm}.obs"
        existing_cells = set()
        if os.path.exists(hobs_fname):
            with open(hobs_fname, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and (" head " in line.lower() or " head" in parts[1].lower()):
                        try:
                            k = int(parts[2]) - 1
                            i0 = int(parts[3]) - 1
                            j0 = int(parts[4]) - 1
                            existing_cells.add((k, i0, j0))
                        except Exception:
                            continue

        rows = []
        for _, r in inter.iterrows():
            i0 = int(r["i"])
            j0 = int(r["j"])
            k0 = int(layer)  # just top layer for now

            # skip inactive cells
            if not (0 <= k0 < idomain.shape[0] and 0 <= i0 < idomain.shape[1] and 0 <= j0 < idomain.shape[2]):
                continue
            if idomain[k0, i0, j0] != 1:
                continue

            # skip if already a head obs cell
            if (k0, i0, j0) in existing_cells:
                continue

            rows.append({"k": k0, "i": i0, "j": j0})

        df = pd.DataFrame(rows).drop_duplicates()
        if df.empty:
            print("[flood_ineq] No valid inequality cells selected.")
        else:
            df.to_csv("flood_ineq_cells.csv", index=False)
            print(f"[flood_ineq] Wrote {len(df)} inequality cells to flood_ineq_cells.csv")

        return df

    finally:
        os.chdir(cwd)


def set_initial_array_vals(org_d, modnm='elk', run_tag=''):
    # copy org_d to org_d + run_tag, remove if exists
    if os.path.exists(org_d+run_tag):
        shutil.rmtree(org_d+run_tag)
    shutil.copytree(org_d, org_d+run_tag)
    
    # load model
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_d, exe_name='mf6', )
    m = sim.get_model(modnm)

  
    k = m.npf.k.array
    aniso = m.npf.k33.array
    ss = m.sto.ss.array
    sy = m.sto.sy.array
    
    print("initial k min,max: ", np.nanmin(k), np.nanmax(k))
    print("initial aniso min,max: ", np.nanmin(aniso), np.nanmax(aniso))
    print("initial ss min,max: ", np.nanmin(ss), np.nanmax(ss))
    print("initial sy min,max: ", np.nanmin(sy), np.nanmax(sy))


    # floor recharge:
    min_rch = 0.1 / 12.0 / 365.0
    rch_files = [f for f in os.listdir(org_d) if f.startswith('rch_rech')]
    for file in rch_files:
        rch = np.loadtxt(os.path.join(org_d, file))
        # rch equal to zero set to nan:
        rch = np.where(rch <= 0.0, np.nan, rch)
        
        print(f"file: {file} min rch: {np.nanmin(rch)*4380} in/yr")
        print(f"file: {file} max rch: {np.nanmax(rch)*4380} in/yr")
        print(f"file: {file} median rch: {np.nanmedian(rch)*4380} in/yr")
        print('---'*10)
        # rch = np.where(rch < min_rch, min_rch, rch)
        
        # # make 2D mask of riv cells:
        # riv_cells = list(zip(riv_df['row'], riv_df['col']))
        # riv_mask = np.zeros_like(rch, dtype=bool)
        # for r, c in riv_cells:
        #     riv_mask[r, c] = True

        # # where riv mask is true, set rch to 0
        # rch = np.where(riv_mask, 0.0, rch)
        # np.savetxt(os.path.join(org_d, file), rch, fmt='%15.6E')


    k[0] = 10.0
    k[1] = 175.0


    # update packages and write out mf files
    m.npf.k.set_data(k)
    #m.npf.k33.set_data(aniso)
    #m.sto.ss.set_data(ss)
    #m.sto.sy.set_data(sy)

    m.write()
    if run_tag:
        # run the model
        pyemu.os_utils.run(r'mf6', cwd=org_d, verbose=True)


def budget_process(modnm='elk'):
    df = pd.read_csv("budget.csv",index_col=0)
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model(modnm)

    # change columns names that have "well" in them to "wel" to match the model:
    wcols = [c for c in df.columns if "WEL" in c]
    for c in wcols:
        nc = 'wel_'+c.split('(')[1].split(')')[0].lower()+c.split('(')[1].split(')')[1].lower()
        df = df.rename(columns={c:c.replace(c,nc)})

    # change columns names that have "riv" in them to individual riv files:
    rcols = [c for c in df.columns if "RIV" in c]
    for c in rcols:
        nc = c.split('(')[1].split(')')[0].lower()+c.split('(')[1].split(')')[1].lower()
        df = df.rename(columns={c:c.replace(c,nc)})

    # change columns names that have "GHB" in them to individual ghb files:
    rcols = [c for c in df.columns if "GHB" in c]
    for c in rcols:
        nc = c.split('(')[1].split(')')[0].lower()+c.split('(')[1].split(')')[1].lower()
        df = df.rename(columns={c:c.replace(c,nc)})

    wcols = [c for c in df.columns if "DRN" in c ]
    cols=[]
    for c in df.columns:
        if len(c.split("("))>1:
            cols.append(c.split("(")[0].lower()+c.split(")")[1].lower())
        else:
            cols.append(c.lower())
    df.columns=cols
    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    df.index.name = "datetime"
    df.to_csv("budget.csv")
    dfs = [df]
    
    return dfs


def init_budget_process(d,modnm='elk'):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = budget_process(modnm=modnm)
    os.chdir(b_d)
    return dfs


def znbud_by_ly_process(modnm='elk'):
    # run znbud
    pyemu.utils.run("zbud6")
    assert os.path.exists("zbud.csv"), "zbud.csv not found, zbud6 may have failed"
    
    df = pd.read_csv("zbud.csv",index_col=0)
    cols = df.columns
    cols = [c.lower().replace(" ","_") for c in cols]
    # add 'zbly_' to each column name
    cols = ['zbly_'+c for c in cols]
    df.columns = cols
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model(modnm)
    bot = m.dis.botm.array
    dtim = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    # add value from 'zone' column in df to each dtim val:
    dtim = [f"{d.strftime('%Y-%m-%d')}_zn-{int(z)}" for d,z in zip(dtim,df['zbly_zone'])]
 
    df.index = dtim
    df.index.name = "datetime"
    df.to_csv("zbud.csv")
    dfs = [df]
    return dfs


def init_zonbud_process(d,modnm='elk'):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = znbud_by_ly_process(modnm=modnm)
    os.chdir(b_d)
    return dfs



def flood_ineq_obs_process(modnm: str = "elk_2lay") -> pd.DataFrame:
    """
    Build inequality obs as a long table:

        datetime, obsname, simval

    where simval = max(head - top, 0) at the end of each stress period
    for each selected cell in flood_ineq_cells.csv.
    """
    import os
    import glob
    import numpy as np
    import pandas as pd
    import flopy
    from flopy.utils import HeadFile

    sim_ws = "."
    print(f"[flood_ineq] running in {os.path.abspath(sim_ws)}")

    # --- ensure cells file exists ---
    cells_path = os.path.join(sim_ws, "flood_ineq_cells.csv")
    if not os.path.exists(cells_path):
        print("[flood_ineq] flood_ineq_cells.csv not found; building it now...")
        build_flood_ineq_cells(ws=sim_ws, modnm=modnm, sample_stride=3, layer=0)

    if not os.path.exists(cells_path):
        print("[flood_ineq] still no flood_ineq_cells.csv after build; no inequality obs written.")
        return pd.DataFrame()

    cells_df = pd.read_csv(cells_path)
    if cells_df.empty:
        print("[flood_ineq] flood_ineq_cells.csv is empty; no inequality obs written.")
        return pd.DataFrame()

    cells_df["k"] = cells_df["k"].astype(int)
    cells_df["i"] = cells_df["i"].astype(int)
    cells_df["j"] = cells_df["j"].astype(int)

    # --- load model for top + time info ---
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=sim_ws, exe_name="mf6", load_only=["dis", "tdis"]
    )
    m = sim.get_model(modnm)
    top = m.dis.top.array  # (nrow, ncol)

    try:
        start_dt = pd.to_datetime(sim.tdis.start_date_time.array)
    except Exception:
        start_dt = pd.to_datetime("2000-01-01")

    # --- head file ---
    headfile = os.path.join(sim_ws, f"{modnm}.hds")
    if not os.path.exists(headfile):
        hfiles = glob.glob(os.path.join(sim_ws, "*.hds"))
        if not hfiles:
            print("[flood_ineq] No head file (*.hds) found; cannot compute inequality obs.")
            return pd.DataFrame()
        headfile = hfiles[0]
        print(f"[flood_ineq] Using head file: {headfile}")

    hds = HeadFile(headfile)
    kstpkpers = hds.get_kstpkper()
    times = hds.get_times()
    if not kstpkpers or not times:
        print("[flood_ineq] Head file has no records; cannot compute inequality obs.")
        return pd.DataFrame()
    assert len(kstpkpers) == len(times), "kstpkper and times length mismatch in head file"

    # last record index for each kper
    last_idx_for_kper = {}
    for idx, (kstp, kper) in enumerate(kstpkpers):
        last_idx_for_kper[kper] = idx

    records = []

    for kper in sorted(last_idx_for_kper.keys()):
        idx = last_idx_for_kper[kper]
        head = hds.get_data(kstpkper=kstpkpers[idx])  # (nlay, nrow, ncol)
        totim_days = times[idx]
        dt = start_dt + pd.to_timedelta(totim_days, unit="D")

        for _, row in cells_df.iterrows():
            k = int(row["k"])
            i = int(row["i"])
            j = int(row["j"])

            if not (0 <= k < head.shape[0] and 0 <= i < head.shape[1] and 0 <= j < head.shape[2]):
                continue

            h = float(head[k, i, j])
            t = float(top[i, j])

            if np.isnan(h) or np.isnan(t):
                simval = 0.0
            else:
                # TRUE inequality: only penalize head > top
                simval = max(h - t, 0.0)

            obsname = f"ineqhd_k:{k}_i:{i}_j:{j}"
            records.append(
                {"datetime": dt, "obsname": obsname, "simval": simval}
            )

    if not records:
        print("[flood_ineq] No inequality values computed for any stress period.")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df.sort_values(["datetime", "obsname"], inplace=True)
    df.to_csv("flood_ineq_obs.csv", index=False)
    print(f"[flood_ineq] Wrote flood_ineq_obs.csv with {df.shape[0]} rows.")
    return df



def head_targets_process(modnm='elk'):
    
    sim = flopy.mf6.MFSimulation.load(sim_ws='.',load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model(modnm)

    df = pd.read_csv(f'elk_2lay.ss_head.obs.output',index_col=0)
    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    df.columns = [c.lower().replace('.','-') for c in df.columns]
    df.index.name = 'datetime'
    dfs = [df]

    df.to_csv(f'{modnm}.ss_head.obs.output.csv')

    df = pd.read_csv(f'elk_2lay.trans_head.obs.output',index_col=0)
    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    df.columns = [c.lower().replace('.','-') for c in df.columns]
    df.index.name = 'datetime'
    
    dfs.append(df)

    df.to_csv(f'{modnm}.trans_head.obs.output.csv')
    
    return dfs


def init_head_targets_process(d,modnm='elk'):
    b_d = os.getcwd()
    os.chdir(d)
    ssdf = head_targets_process(modnm=modnm)
    os.chdir(b_d)
    return ssdf


def dd_targets_process(modnm: str = "elk"):
    """
    Build SP-delta (between-stress-period change) targets from transient heads.

    - Loads MF6 TDIS to turn model time (days since start) into datetimes.
    - Reads:
        <modnm>.trans_head.obs.output         -> simulated heads (df)
        transient_well_targets.csv            -> observed/target heads (t_df)
    - Keeps only columns present in BOTH df and t_df.
    - Aligns sim to obs in time (obs index drives).
    - Masks sim where obs is NaN.
    - Computes *time differences* (delta between stress periods):

        Δsim(t) = sim(t) - sim(t-1)
        Δobs(t) = obs(t) - obs(t-1)

      for each column independently.

    - First time step per column has no previous value -> NaN, then set to -9999.0.
    - Column names: "transh" → "dd" to indicate difference/drawdown-type targets.

    Outputs
    -------
    Writes:
      <modnm>.trans_sim_dd_targs.output
      <modnm>.trans_obs_dd_targs.output

    Returns
    -------
    df_diff, obs_diff : (pd.DataFrame, pd.DataFrame)
        SP-delta simulated and observed differences.
    """
    # --- load model time and sim table ---
    sim = flopy.mf6.MFSimulation.load(sim_ws=".", load_only=["dis", "tdis"])
    start_datetime = sim.tdis.start_date_time.array
    _ = sim.get_model(modnm)  # not used directly, but keeps symmetry with original

    # simulated heads: index in days since start -> convert to datetimes
    df = pd.read_csv(f"{modnm}.trans_head.obs.output", index_col=0)
    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values, unit="d")
    df.columns = [c.lower().replace(".", "-") for c in df.columns]
    df.index.name = "datetime"

    # observed/target heads
    t_df = pd.read_csv("transient_well_targets.csv")
    t_df.set_index("start_datetime", inplace=True)
    t_df.index = pd.to_datetime(t_df.index, errors="coerce")
    t_df.columns = [c.lower().replace(".", "-") for c in t_df.columns]

    # --- keep only overlapping columns between sim and obs ---
    cols_keep = sorted(set(df.columns) & set(t_df.columns))
    if not cols_keep:
        raise ValueError("No overlapping columns between simulated heads and target heads.")

    obs = t_df[cols_keep].copy()
    sim_heads = df[cols_keep].copy()

    # sort by time just in case
    obs = obs.sort_index()
    sim_heads = sim_heads.sort_index()

    # clip sim to obs time span and align to obs index (obs drives)
    sim_heads = sim_heads[sim_heads.index <= obs.index.max()]
    sim_heads = sim_heads.reindex(obs.index)

    # safety checks
    assert list(sim_heads.columns) == list(obs.columns), "Column mismatch after alignment."
    assert sim_heads.shape == obs.shape, f"Shape mismatch: sim={sim_heads.shape}, obs={obs.shape}"

    # mask sim where obs is NaN
    sim_masked = sim_heads.where(obs.notna())

    # ------------------------------------------------------------------
    # SP deltas (difference between consecutive stress periods)
    # ------------------------------------------------------------------
    # Δsim(t) = sim(t) - sim(t-1)
    df_diff = sim_masked.diff()  # first row NaN for each column

    # Δobs(t) = obs(t) - obs(t-1)
    obs_diff = obs.diff()

    # ------------------------------------------------------------------
    # Final formatting and output
    # ------------------------------------------------------------------
    df_diff.index.name = "datetime"
    obs_diff.index.name = "datetime"

    # Replace NaNs (no data / first step) with -9999.0
    df_diff_filled = df_diff.copy()
    obs_diff_filled = obs_diff.copy()
    df_diff_filled.fillna(-9999.0, inplace=True)
    obs_diff_filled.fillna(-9999.0, inplace=True)

    # Replace 'transh' with 'dd' in column names to indicate difference targets
    df_diff_filled.columns = [c.replace("transh", "dd") for c in df_diff_filled.columns]
    obs_diff_filled.columns = [c.replace("transh", "dd") for c in obs_diff_filled.columns]

    # Write out
    df_diff_filled.to_csv(f"{modnm}.trans_sim_dd_targs.output")
    obs_diff_filled.to_csv(f"{modnm}.trans_obs_dd_targs.output")

    return df_diff_filled, obs_diff_filled



def init_dd_targets_process(d,modnm='elk'):
    b_d = os.getcwd()
    os.chdir(d)
    df_diff, obs_diff = dd_targets_process(modnm=modnm)
    os.chdir(b_d)
    return df_diff


def process_listbudget_obs(mod_name='elk'):
    '''post processor to return volumetric flux and cumulative flux values from MODFLOW list file

    Args:
        mod_name (str): MODFLOW model name

    Returns:
        flx: Pandas DataFrame object of volumetric fluxes from listbudget output
        cum: Pandas DataFrame object of cumulative volumetric fluxes from listbudget output
    '''
    lst = flopy.utils.Mf6ListBudget('{0}.lst'.format(mod_name))
    flx, cum = lst.get_dataframes(diff=True, start_datetime='1969-12-31')
    flx.loc[:,'datetime'] = flx.index.strftime('%Y%m%d')
    cum.loc[:,'datetime'] = cum.index.strftime('%Y%m%d')
    flx.index = flx.pop('datetime')
    cum.index = cum.pop('datetime')

    # # get wel type packages in the order that they appear in nam file:
    sim = flopy.mf6.MFSimulation.load(sim_ws='.')
    m = sim.get_model(mod_name)
    pkg_lst = m.get_package_list()
    pkg_lst = [p.lower() for p in pkg_lst]
    
    drop_lst = ['dis',
                'ic',
                'npf',
                'sto',
                'rch',
                'oc',
                'drn',
                'riv',
                'obs',
                'ghb',
                ]
    well_pkg_lst = [pkg for pkg in pkg_lst if not any(drop in pkg for drop in drop_lst)]
    
    # replace colum names that start with 'wel' with the package name:
    cnt = 0
    
    for col in flx.columns:
        if col.startswith('wel'):
            flx = flx.rename(columns={col: well_pkg_lst[cnt]+'-uin'}) # uin = user input (from wel package)
            cnt += 1
    # get columns that startwith uin:
    wuin = [col for col in flx.columns if col.endswith('uin')]
    
    flx = flx[wuin]
    
    for pkg in well_pkg_lst:
        # join WEL input to flx df, calc rejected WEL flux
        wel_files = [f for f in os.listdir('.') if '{0}_stress_period_'.format(pkg) in f.lower()]
        sp_tags = [int(f.split('.')[-2].split('_')[-1]) for f in wel_files]
     
        # sort wel files by stress period
        wel_files = [f for _,f in sorted(zip(sp_tags,wel_files))]
        sp_tags = [s for s,_ in sorted(zip(sp_tags,wel_files))]
  
        in_tot = np.zeros((len(flx)))

        for wel_file, sp_tag in zip(wel_files, sp_tags):
            in_wel = pd.read_csv(wel_file, header=None, delim_whitespace=True).iloc[:, 3]
            in_tot[sp_tag - 1] = in_wel.sum()    

        flx.loc[:, f'{pkg}-simin'] = in_tot.tolist()

    flx.to_csv('listbudget_flx_obs.csv')
    cum.to_csv('listbudget_cum_obs.csv')
    return flx,cum


def init_listbudget_obs(d='.', mod_name='elk'):
    '''helper function to run process_listbudget_obs processor during PEST setup

    Args:
        d (str): model working directory, typically 'model_template' when building pest interface
        mod_name (str): MODFLOW model name

    Returns:
        flx: Pandas DataFrame object of volumetric fluxes from listbudget output
        cum: Pandas DataFrame object of cumulative volumetric fluxes from listbudget output
    '''
    b_d = os.getcwd()
    os.chdir(d)
    flx, cum = process_listbudget_obs(mod_name)
    os.chdir(b_d)
    return flx, cum


def process_mfinput_obs(mod_name='elk'):
    '''post processor to calculate summary statistics of MODFLOW input values

    Args:
        mod_name (str): MODFLOW model name

    Returns:
        df: Pandas DataFrame object of MODFLOW input value summary statistics
    '''
    sim = flopy.mf6.MFSimulation.load(sim_ws='.',load_only=['dis'])
    m = sim.get_model(mod_name)
    ib = m.dis.idomain.data
    
    df = pd.read_csv('mult2model_info.csv', index_col=0)
    
    arr_df = df.loc[df.model_file.apply(lambda x: 'stress_period' not in x and 'drn' not in x and 'rch' not in x and
                                                  'riv' not in x and 'ghb' not in x)].copy().groupby('org_file').first()
    
    arr_df.loc[:, 'layer'] = arr_df.loc[:, 'model_file'].apply(lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))
    # make layer zero based:
    arr_df.loc[:, 'layer'] = arr_df.loc[:, 'layer'] - 1
    
    print(arr_df.lower_bound.unique())
    arr_df.loc[:,'lower_bound'] = arr_df.loc[:,'lower_bound'].astype(float)
    arr_df.loc[:,'upper_bound'] = arr_df.loc[:,'upper_bound'].astype(float)
    arr_df.loc[:,'array'] = arr_df.loc[:, 'model_file'].apply(lambda x: np.loadtxt(x))
    arr_df.loc[:, 'array'] = [np.where(ib[lay]>0, a, np.nan) for a,lay in zip(arr_df.loc[:,'array'],arr_df.loc[:,'layer'])]
    
    arr_df.loc[:,'mean'] = arr_df.loc[:,'array'].apply(lambda x: np.nanmean(x))
    arr_df.loc[:,'std'] = arr_df.loc[:,'array'].apply(lambda x: np.nanstd(x))
    
    arr_df.loc[:,'min'] = arr_df.loc[:,'array'].apply(lambda x: np.nanquantile(x, 0.))
    arr_df.loc[:,'qnt25'] = arr_df.loc[:,'array'].apply(lambda x: np.nanquantile(x, 0.25))
    arr_df.loc[:,'qnt50'] = arr_df.loc[:,'array'].apply(lambda x: np.nanquantile(x, 0.5))
    arr_df.loc[:,'qnt75'] = arr_df.loc[:,'array'].apply(lambda x: np.nanquantile(x, 0.75))
    arr_df.loc[:,'max'] = arr_df.loc[:,'array'].apply(lambda x: np.nanquantile(x, 1.))

    arr_df.loc[:, 'near_lbnd'] = [np.sum(np.where(a < lb * 1.05, 1, 0)) for a, lb in zip(arr_df.loc[:,'array'],arr_df.loc[:,'lower_bound'])]
    arr_df.loc[:, 'near_ubnd'] = [np.sum(np.where(a > ub * 0.95, 1, 0)) for a, ub in zip(arr_df.loc[:,'array'],arr_df.loc[:,'upper_bound'])]
    arr_df.loc[:,'input'] = arr_df.loc[:,'model_file'].apply(lambda x: x.split('.')[-2])
    arr_df.loc[:,'prop'] = arr_df.input.apply(lambda x: '_'.join(x.split('_')[1:2]))
    arr_df.loc[:,'input'] = arr_df.loc[:,'prop'] + '_k:' + arr_df.loc[:,'layer'].apply(lambda x: str(x).zfill(2))

    df = arr_df.filter(['input', 'upper_bound', 'lower_bound','min','qnt25','qnt50','qnt75','max','near_lbnd','near_ubnd','mean','std'])
    df.index = df.pop('input')

    df.to_csv('mfinput_obs.csv')
 
    return df


def init_mfinput_obs(template_ws='template', mod_name='elk'):
    '''helper function to run mfinput_obs processor during PEST setup

    Args:
        template_ws (str): model working directory, typically 'model_template' when building pest interface
        mod_name (str): MODFLOW model name

    Returns:
        df: Pandas DataFrame object of MODFLOW input value summary statistics
    '''
    b_d = os.getcwd()
    os.chdir(template_ws)
    df = process_mfinput_obs(mod_name)
    os.chdir(b_d)
    return df


def riv_drn_bot_chk(model_ws='.', mnm='elk_2lay'):
    """
    Safety check for RIV/DRN/GHB stages vs cell bottoms in the *clean* workspace.

    - For all DRN-style files (static, WL, AG):
        stage >= botm + 0.1

    - For all RIV-style files:
        stage >= botm + 3.0
        rbot  >= botm + 0.1
        stage >= rbot + 0.1
    """
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=model_ws, exe_name="mf6", load_only=["dis"]
    )
    m = sim.get_model()
    botm = m.dis.botm.array

    # --- find stress files in the CLEAN workspace (no model-name prefix) ---

    # Static geographic drains: drn_s_stress..., drn_ms_stress..., drn_mn_stress..., drn_n_stress...
    # drn_stress_files = [
    #     os.path.join(model_ws, f)
    #     for f in os.listdir(model_ws)
    #     if f.startswith("drn_") and "stress" in f
    #     and not f.startswith("drn_ag")   # exclude ag drains
    #     and not f.startswith("drn_wl")   # exclude WL drains
    # ]

    # WL drains: drn_wl_stress_period_data_XXX.txt
    drns_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("drn_s_stress")
    ]

    # WL drains: drn_wl_stress_period_data_XXX.txt
    drnms_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("drn_ms_stress")
    ]
    
    drnmn_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("drn_mn_stress")
    ]

    drnn_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("drn_n_stress")
    ]
    
    # WL drains: drn_wl_stress_period_data_XXX.txt
    drn2_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("drn_wl_stress")
    ]

    # AG drains: drn_ag_stress_period_data_XXX.txt
    drn_ag_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("drn_ag_stress")
    ]

    # All RIV packages: riv_turtle_stress..., riv_goose_stress..., riv_hazen_stress...
    # riv_stress_files = [
    #     os.path.join(model_ws, f)
    #     for f in os.listdir(model_ws)
    #     if f.startswith("riv_") and "stress" in f
    # ]

    riv_turtle_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("riv_turtle_stress")
    ]

    riv_hazen_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("riv_hazen_stress")
    ]

    riv_goose_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("riv_goose_stress")
    ]
    
    # (unchanged) GHB WL files, if you ever wire them in
    ghb_wl_stress_files = [
        os.path.join(model_ws, f)
        for f in os.listdir(model_ws)
        if f.startswith("ghb_wl_stress")
    ]

    # ---------- DRN: enforce stage >= botm + 0.1 ft ----------

    # Static geographic drains
    for drn_file in drns_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)

    for drn_file in drnms_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)
    for drn_file in drnmn_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)

    for drn_file in drnn_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)
    # WL drains
    for drn_file in drn2_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)

    # AG drains
    for drn_file in drn_ag_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)

    # ---------- RIV: enforce stage/rbot vs botm ----------

    for riv_file in riv_turtle_stress_files:
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
        # case where stage is below rbot:
        df['diff'] = df['stage'] - df['rbot']
        df.loc[df['diff'] < 0,'stage'] = df['rbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(riv_file, sep=' ', index=False, header=False)

    for riv_file in riv_hazen_stress_files:
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
        # case where stage is below rbot:
        df['diff'] = df['stage'] - df['rbot']
        df.loc[df['diff'] < 0,'stage'] = df['rbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(riv_file, sep=' ', index=False, header=False)

    for riv_file in riv_goose_stress_files:
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
        # case where stage is below rbot:
        df['diff'] = df['stage'] - df['rbot']
        df.loc[df['diff'] < 0,'stage'] = df['rbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(riv_file, sep=' ', index=False, header=False)

def setup_pstpp(org_d,modnm,run_tag,template,flex_con=False,num_reals=96,
                high_dimensional=False,):
    
    assert os.path.exists(org_d)
    
    # make the 4-PP regional ghb field:
    #pp_ghb_files = init_regional_ghbs(d=org_d, mod_name='elk')
    
    temp_d = org_d + '_temp'
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(org_d,temp_d)

    # copy over head diff obs:
    shutil.copy2(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'),os.path.join(temp_d,'transient_well_targets_lookup_shrt.csv'))
    shutil.copy2(os.path.join('data', 'analyzed', 'transient_well_targets.csv'), os.path.join(temp_d, 'transient_well_targets.csv'))
    prep_deps(temp_d)

    if os.path.exists(template):
        shutil.rmtree(template)

    if not flex_con: # add water level obs
        hobs_fname = os.path.join(temp_d, f'{modnm}.obs')
        assert os.path.exists(hobs_fname)
        with open(hobs_fname, 'r') as f:
            lines = f.readlines()
        with open(hobs_fname,'w') as f:
            for line in lines:
                if ' head ' in line or ' HEAD ' in line and len(line.strip().split()) >= 5:
                    raw = line.strip().split()
                    raw[0] = raw[0] + '_k:{0}_i:{1}_j:{2}'.format(int(raw[2])-1,int(raw[3])-1,int(raw[4])-1)
                    line = ' '.join(raw) + '\n'
                f.write(line)

    pyemu.os_utils.run('mf6',cwd=temp_d)
    #pyemu.os_utils.run('zbud6',cwd=temp_d)

    # load flow model and model info:
    flow_dir = os.path.join(temp_d)
    sim = flopy.mf6.MFSimulation.load(sim_ws=flow_dir, exe_name='mf6')
    start_datetime = sim.tdis.start_date_time.array

    perlen = sim.tdis.perioddata.array['perlen']
    nper = len(perlen)
    dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(perlen),unit='d')

    m = sim.get_model(f'{modnm}')
    nlay = m.dis.nlay.data
    id = m.dis.idomain.array
    delr = m.dis.delr.array[0]
    nrow = m.dis.nrow.data
    ncol = m.dis.ncol.data

    pkg_lst = m.get_package_list()
    pkg_lst = [p.lower() for p in pkg_lst]

    # instantiate PEST container
    pf = pyemu.utils.PstFrom(original_d=temp_d, new_d=template,
                         remove_existing=True,
                         longnames=True, spatial_reference=m.modelgrid,
                         zero_based=False, start_datetime=start_datetime,
                         chunk_len=1000000)
    
    _ = build_flood_ineq_cells(ws=template, modnm=modnm, sample_stride=3, layer=0)
    # ------------------------------------    
    # load in geostats parms from run csv:
    # ------------------------------------
    pp_space = 8
    pp_vario = pyemu.geostats.ExpVario(contribution=1.0,a=5280*8,bearing=115.0)
    pp_geo = pyemu.geostats.GeoStruct(variograms=pp_vario, transform="log", name="k_pp_geo")

    gr_v = pyemu.geostats.ExpVario(contribution=1.0,a=5280,anisotropy=1.0,bearing=0.0)
    gr_gs  = pyemu.geostats.GeoStruct(variograms=gr_v, transform="log", name="k_grid_geo")

    temporal_v = pyemu.geostats.ExpVario(contribution=1.0, a=365.25*7, name='temporal_v')
    temporal_gs = pyemu.geostats.GeoStruct(variograms=temporal_v, transform='log', name='temporal_gs')
    temporal_gs.to_struct_file(os.path.join(template, 'temporal_gs.struct'))
    
    k_files = [f for f in os.listdir(template) if '_k_' in f and f.endswith('.txt')]
    k_files.sort()

    k33_files = [f for f in os.listdir(template) if "k33" in f]
    k33_files.sort()

    ss_files = [f for f in os.listdir(template) if 'sto_ss_' in f and f.endswith('.txt')]
    ss_files.sort()

    sy_files = [f for f in os.listdir(template) if 'sto_sy_' in f and f.endswith('.txt')]
    sy_files.sort()

    # load in par bounds:
    par = pd.read_csv(os.path.join('run_inputs',f'{modnm}',f'{modnm}_parm_controls.csv'))
    
    kcn = par.loc[par.parm=='k_cn']
    k_bounds_cn = {k:[kcn.lbound.values[0],kcn.ubound.values[0]] for k in range(nlay)}
    
    k33cn = par.loc[par.parm=='aniso_cn']
    k33_bounds_cn = {k:[k33cn.lbound.values[0],k33cn.ubound.values[0]] for k in range(nlay)}
    #k33_bounds_grd = {k:[0.2,5.0] for k in range(nlay)}
    
    sscn = par.loc[par.parm=='ss_cn']
    ss_bounds_cn = {k:[sscn.lbound.values[0],sscn.ubound.values[0]] for k in range(nlay)}
    
    sycn = par.loc[par.parm=='sy_cn']
    sy_bounds_cn = {k:[sycn.lbound.values[0],sycn.ubound.values[0]] for k in range(nlay)}
    
    # load ultimate (hard) bounds:
    # load them from pp parms because those types will likely always be used, but be sure to check this
    k_ubounds = {k:[kcn.ult_lbound.values[0],kcn.ult_ubound.values[0]] for k in range(nlay)}
    k33_ubounds = {k:[k33cn.ult_lbound.values[0],k33cn.ult_ubound.values[0]] for k in range(nlay)}
    ss_ubounds = {k:[sscn.ult_lbound.values[0],sscn.ult_ubound.values[0]] for k in range(nlay)}
    sy_ubounds = {k:[sycn.ult_lbound.values[0],sycn.ult_ubound.values[0]] for k in range(nlay)}


    # stacked_files = [k_files, k33_files, ss_files, sy_files]
    # stacked_ubnds = [k_ubounds, k33_ubounds, ss_ubounds, sy_ubounds]
    stacked_files = [k_files, ss_files, sy_files]
    stacked_ubnds = [k_ubounds, ss_ubounds, sy_ubounds]
    # stacked_files = [k_files, k33_files]
    # stacked_ubnds = [k_ubounds, k33_ubounds]
    
    for files, kubnds in zip(stacked_files,stacked_ubnds):
        assert len(files) > 0
        lays = [int(f.split('.')[0].split('_')[2].replace('layer',''))-1 for f in files]
        par_name_base = ''.join(files[0].split('_')[1]).replace("_","-") #.replace('.txt','').replace('layer','ly')
        par_name_base = par_name_base+"_k:"
        if flex_con and par_name_base != 'k_k:':
            continue
        # assert len(files) == nlay
        for k,f in zip(lays,files):
            ubnds = kubnds[k]
            if 'k33' in f:
                bnds_cn = k33_bounds_cn[k]

                # pf.add_parameters(f, par_type='constant', upper_bound=5, lower_bound=0.1,
                #         ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base+ str(k).zfill(3),
                #         pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k],initial_value=1.0)
                
                if high_dimensional:
                    pf.add_parameters(f, par_type='pilotpoints', upper_bound=5, lower_bound=0.1,
                                ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='pp-' + par_name_base+ str(k).zfill(3),
                                pargp='pp-' + par_name_base + str(k).zfill(3), pp_space=pp_space,geostruct=pp_geo)

            elif 'k' in f and 'k33' not in f:
                bnds_cn = k_bounds_cn[k]
                # if k==0:
                #     pf.add_parameters(f, par_type='constant', upper_bound=5.0, lower_bound=0.01,
                #             par_name_base='cn-' + par_name_base+ str(k).zfill(3),
                #             pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k])
                # else:
                #     pf.add_parameters(f, par_type='constant', upper_bound=10, lower_bound=0.1,
                #             par_name_base='cn-' + par_name_base+ str(k).zfill(3),
                #             pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k])                  
                if high_dimensional:
                    pf.add_parameters(f, par_type='pilotpoints', upper_bound=5, lower_bound=0.1,
                            par_name_base='pp-' + par_name_base,
                            pargp='pp-' + par_name_base + str(k).zfill(3), pp_space=pp_space,geostruct=pp_geo)
                    
            elif 'ss' in f:
                bnds_cn = ss_bounds_cn[k]
                if k > 0:
                    # pf.add_parameters(f, par_type='constant', upper_bound=10.0, lower_bound=0.01,
                    #         ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base+ str(k).zfill(3),
                    #         pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k])

                    if high_dimensional:
                        pf.add_parameters(f, par_type='pilotpoints', upper_bound=10.0, lower_bound=0.01,
                                                ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='pp-' + par_name_base+ str(k).zfill(3),
                                                pargp='pp-' + par_name_base + str(k).zfill(3),pp_space=pp_space,geostruct=pp_geo)

            elif 'sy' in f:
                bnds_cn = sy_bounds_cn[k]

                # pf.add_parameters(f, par_type='constant', upper_bound=1.5, lower_bound=0.33,
                #         ult_ubound=0.35, ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base+ str(k).zfill(3),
                #         pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k])

                if high_dimensional:
                    pf.add_parameters(f, par_type='pilotpoints', upper_bound=1.5, lower_bound=0.33,
                                            ult_ubound=0.35, ult_lbound=ubnds[0], par_name_base='pp-' + par_name_base+ str(k).zfill(3),
                                            pargp='pp-' + par_name_base + str(k).zfill(3),pp_space=pp_space,geostruct=pp_geo)
            if not ('ss' in f and k == 0):
                pf.add_observations(f,obsgp=par_name_base + str(k).zfill(3),prefix=par_name_base + str(k).zfill(3))             

    else:

        # recharge parmeterization:
        rch_pp_space = 24
        rch_pp_vario = pyemu.geostats.ExpVario(contribution=1.0, a=5280*7, bearing=0.0)
        rch_pp_geo = pyemu.geostats.GeoStruct(variograms=rch_pp_vario)

        rch_files = [f for f in os.listdir(template) if f.startswith("rch_") and f.endswith(".txt")]
        rch_files = sorted(rch_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        assert len(rch_files) == nper

        # --- NEW: only parameterize historical period (kper <= 323; sp <= 324) ---
        HIST_KPER_MAX = 323
        HIST_SP_MAX = HIST_KPER_MAX + 1  # filenames are 1-based SP numbers

        rch_files_hist = []
        for rch_file in rch_files:
            sp = int(rch_file.split(".")[0].split("_")[-1])  # 1-based
            if sp <= HIST_SP_MAX:
                rch_files_hist.append(rch_file)

        print(f"[rch] total files={len(rch_files)} | parameterized (historical)={len(rch_files_hist)} "
            f"| not parameterized (predictive)={len(rch_files) - len(rch_files_hist)}")

        # -------------------------------------------------------------------------
        # Pilot points (if you use them) — now only for historical SPs
        # -------------------------------------------------------------------------
        for rch_file in rch_files_hist:
            sp = int(rch_file.split(".")[0].split("_")[-1])
            kper = sp - 1
            dt = dts[kper]
            pf.add_parameters(
                rch_file,
                par_type="pilotpoints",
                upper_bound=5,
                lower_bound=0.1,
                par_name_base="rcht_{0:03d}".format(kper),
                pargp="rcht_pp",
                geostruct=rch_pp_geo,
                pp_space=rch_pp_space,
            )

        # -------------------------------------------------------------------------
        # Temporal constants — now only for historical SPs
        # (your pre/post 1999 logic stays, but predictive SPs are skipped entirely)
        # -------------------------------------------------------------------------
        for rch_file in rch_files_hist:
            sp = int(rch_file.split(".")[0].split("_")[-1])
            kper = sp - 1
            dt = dts[kper]

            if kper < 35:  # SP 36 is 1999 
                pf.add_parameters(
                    rch_file,
                    par_type="constant",
                    pargp="rcht",
                    par_name_base="rcht_{0:03d}".format(kper),
                    datetime=dt,
                    geostruct=temporal_gs,
                    upper_bound=5,
                    lower_bound=0.1,
                )
            else:
                pf.add_parameters(
                    rch_file,
                    par_type="constant",
                    pargp="rcht",
                    par_name_base="rcht_{0:03d}".format(kper),
                    datetime=dt,
                    geostruct=temporal_gs,
                    upper_bound=5,
                    lower_bound=0.1,
                )

        # --- riv paramterization ---
        
        # for pilot points we want to have the defined over a buffer around rivers rather than placed uniformly
        # across grid, this is a thought for now, but we will need to see how this works out in the end
        
        rivcond_cn = par.loc[par.parm=='rivcond_cn']
        rivstg_cn = par.loc[par.parm=='rivstg_cn']
        # riv_files = [f for f in os.listdir(template) if f.startswith('riv_') and f.endswith('.txt')]
        # riv_files = sorted(riv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # assert len(riv_files) == nper
        
        riv_turtle_files = [f for f in os.listdir(template) if f.startswith('riv_turtle') and f.endswith('.txt')]
        riv_turtle_files = sorted(riv_turtle_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(riv_turtle_files) == nper

        riv_hazen_files = [f for f in os.listdir(template) if f.startswith('riv_hazen') and f.endswith('.txt')]
        riv_hazen_files = sorted(riv_hazen_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(riv_hazen_files) == nper

        riv_goose_files = [f for f in os.listdir(template) if f.startswith('riv_goose') and f.endswith('.txt')]
        riv_goose_files = sorted(riv_goose_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(riv_goose_files) == nper

        # pf.add_parameters(riv_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[4], pargp='rivcond-cn',
        #                     par_name_base='rivcond-cn',
        #                     upper_bound=rivcond_cn['ubound'].values[0], #100
        #                     lower_bound=rivcond_cn['lbound'].values[0], #0.01
        #                     mfile_skip=0)

        # --- turtle ---
        pf.add_parameters(riv_turtle_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='rivstg-cn_turtle',
                            par_name_base='rivstg-cn_turtle', par_style="a",
                            upper_bound=2 , #rivstg_cn['ubound'].values[0],
                            lower_bound=-2, # rivstg_cn['lbound'].values[0],
                            initial_value=0.0,transform="none",
                            mfile_skip=0)
        
        pf.add_parameters(riv_turtle_files, par_type='grid',
            par_name_base='riv_grid_turtle', geostruct=gr_gs,
            pargp='riv_grid_turtle', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=2, lower_bound=0.001,mfile_skip=0,)

        # --- goose ---
        pf.add_parameters(riv_goose_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='rivstg-cn_goose',
                            par_name_base='rivstg-cn_goose', par_style="a",
                            upper_bound=5 , #rivstg_cn['ubound'].values[0],
                            lower_bound=-5, # rivstg_cn['lbound'].values[0],
                            initial_value=0.0,transform="none",
                            mfile_skip=0)
        
        pf.add_parameters(riv_goose_files, par_type='grid',
            par_name_base='riv_grid_goose', geostruct=gr_gs,
            pargp='riv_grid_goose', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=10, lower_bound=0.001,mfile_skip=0,)
        
        # --- hazen ---
        pf.add_parameters(riv_hazen_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='rivstg-cn_hazen',
                            par_name_base='rivstg-cn_hazen', par_style="a",
                            upper_bound=5 , #rivstg_cn['ubound'].values[0],
                            lower_bound=-5, # rivstg_cn['lbound'].values[0],
                            initial_value=0.0,transform="none",
                            mfile_skip=0)
        
        pf.add_parameters(riv_hazen_files, par_type='grid',
            par_name_base='riv_grid_hazen', geostruct=gr_gs,
            pargp='riv_grid_hazen', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=10, lower_bound=0.001,mfile_skip=0,)

        drncond_cn = par.loc[par.parm=='drncond_cn']
        drnstg_cn = par.loc[par.parm=='drnstg_cn']
        
        drn_s_files = [f for f in os.listdir(template) if f.startswith('drn_s_stress') and f.endswith('.txt')]
        drn_s_files = sorted(drn_s_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_s_files) == nper
  
        drn_ms_files = [f for f in os.listdir(template) if f.startswith('drn_ms_stress') and f.endswith('.txt')]
        drn_ms_files = sorted(drn_ms_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_ms_files) == nper      
        
        drn_mn_files = [f for f in os.listdir(template) if f.startswith('drn_mn_stress') and f.endswith('.txt')]
        drn_mn_files = sorted(drn_mn_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_mn_files) == nper    

        drn_n_files = [f for f in os.listdir(template) if f.startswith('drn_n_stress') and f.endswith('.txt')]
        drn_n_files = sorted(drn_n_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_n_files) == nper        
        
        # pf.add_parameters(filenames=drn_files, par_type='constant',
        #                     par_name_base='drncond-cn', 
        #                     pargp='drncond-cn', index_cols=[0, 1, 2], use_cols=[4],
        #                     upper_bound=drncond_cn['ubound'].values[0], lower_bound=drncond_cn['lbound'].values[0],
        #                     mfile_skip=0)

        # --- s ---  
        pf.add_parameters(drn_s_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnstg-cn-s',
                            par_name_base='drnstg-cn-s', par_style="a",
                            upper_bound=5,
                            lower_bound=-10,
                            mfile_skip=0,
                            transform="none")

        pf.add_parameters(drn_s_files, par_type='grid',
            par_name_base='drn_grid-s', geostruct=gr_gs,
            pargp='drn_grid-s', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=10, lower_bound=0.001,mfile_skip=0,)
 
         # --- ms ---  
        pf.add_parameters(drn_ms_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnstg-cn-ms',
                            par_name_base='drnstg-cn-ms', par_style="a",
                            upper_bound=5,
                            lower_bound=-5,
                            mfile_skip=0,
                            transform="none")

        pf.add_parameters(drn_ms_files, par_type='grid',
            par_name_base='drn_grid-ms', geostruct=gr_gs,
            pargp='drn_grid-ms', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=10, lower_bound=0.001,mfile_skip=0,)
 

         # --- mn ---  
        pf.add_parameters(drn_mn_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnstg-cn-mn',
                            par_name_base='drnstg-cn-mn', par_style="a",
                            upper_bound=5,
                            lower_bound=-5,
                            mfile_skip=0,
                            transform="none")

        pf.add_parameters(drn_mn_files, par_type='grid',
            par_name_base='drn_grid-mn', geostruct=gr_gs,
            pargp='drn_grid-mn', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=10, lower_bound=0.001,mfile_skip=0,)

         # --- n ---  
        pf.add_parameters(drn_n_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnstg-cn-n',
                            par_name_base='drnstg-cn-n', par_style="a",
                            upper_bound=5,
                            lower_bound=-7,
                            mfile_skip=0,
                            transform="none", initial_value=-5.0)

        pf.add_parameters(drn_n_files, par_type='grid',
            par_name_base='drn_grid-n', geostruct=gr_gs,
            pargp='drn_grid-n', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=10, lower_bound=0.001,mfile_skip=0) 

        drn_ag_files = [f for f in os.listdir(template) if f.startswith('drn_ag') and f.endswith('.txt')]   
        drn_ag_files = sorted(drn_ag_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_ag_files) == nper

        # pf.add_parameters(filenames=drn_ag_files, par_type="constant",
        #                   par_name_base='drnag_cn',                       
        #                   pargp='drnag_cn', index_cols=[0, 1, 2], use_cols=[4],
        #                   upper_bound=drncond_cn['ubound'].values[0], lower_bound=drncond_cn['lbound'].values[0],
        #                   mfile_skip=0)
        pf.add_parameters(drn_ag_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnagstg-cn',
                            par_name_base='drnagstg-cn', par_style="a",
                            upper_bound=2,
                            lower_bound=-2,
                            mfile_skip=0,
                            transform="none")
        pf.add_parameters(drn_ag_files, par_type='grid',
            par_name_base='drn_ag_grid', geostruct=gr_gs,
            pargp='drn_ag_grid', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=2, lower_bound=0.001,mfile_skip=0,)


        drn_wl_files = [f for f in os.listdir(template) if f.startswith('drn_wl') and f.endswith('.txt')]   
        drn_wl_files = sorted(drn_wl_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_wl_files) == nper

        # pf.add_parameters(filenames=drn_wl_files, par_type="constant",
        #                   par_name_base='drnwl_cn',                       
        #                   pargp='drnwl_cn', index_cols=[0, 1, 2], use_cols=[4],
        #                   upper_bound=drncond_cn['ubound'].values[0], lower_bound=drncond_cn['lbound'].values[0],
        #                   mfile_skip=0)
        pf.add_parameters(drn_wl_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnwlstg-cn',
                            par_name_base='drnwlstg-cn', par_style="a",
                            upper_bound=2,
                            lower_bound=-2,
                            mfile_skip=0,
                            transform="none")
        pf.add_parameters(drn_wl_files, par_type='grid',
            par_name_base='drn_wl_grid', geostruct=gr_gs,
            pargp='drn_wl_grid', index_cols=[0, 1, 2], use_cols=[4],
            upper_bound=5, lower_bound=0.001,mfile_skip=0,)

        prefixes = ["wel"]
        for prefix in prefixes:
            list_files = [f for f in os.listdir(template) if prefix in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
            print(list_files)
            if len(list_files) == 0:
                print("WARNING: not wel list files found for prefix '{0}'".format(prefix))
                continue
            list_files.sort()
            tag = list_files[0].split("_")[0]
            # pf.add_parameters(filenames=list_files, par_type='constant',
            #                       par_name_base='globalwel-mlt-{0}'.format(tag),
            #                       pargp='globalwel-mlt-{0}'.format(tag), index_cols=[0, 1, 2], use_cols=[3],
            #                       upper_bound=1.2, lower_bound=0.8, initial_value=1.0)
            if high_dimensional:
                for list_file in list_files:
                    sp = int(list_file.split('.')[0].split('_')[-1])
                    kper = int(sp) - 1
                    if kper >=4:
                        pf.add_parameters(filenames=list_file, par_type='constant',
                                        par_name_base='twel-mlt-{1}_kper:{0:04d}'.format(sp-1,tag),
                                        pargp='twel-mlt-{0}'.format(tag), index_cols=[0, 1, 2], use_cols=[3],
                                        upper_bound=1.3, lower_bound=0.8, initial_value=1.0, datetime=dts[kper], geostruct=temporal_gs)
                
                # grid scale mults correlated in space, constant in time
                pf.add_parameters(filenames=list_files, par_type='grid',
                        par_name_base='swel-mlt-{0}'.format(tag),
                        pargp='swel-mlt-{0}'.format(tag), index_cols=[0, 1, 2], use_cols=[3],
                       upper_bound=1.3, lower_bound=0.7, initial_value=1.0)
                    
       
        list_files = [f for f in os.listdir(template) if f.endswith('.txt') and "def_wel.wel_stress_period_data_" in f]
        print(list_files)
        include_def = False
        if len(list_files) == 0:
            print("WARNING: no deferred permit wel list files found")
        else:
            list_files.sort()
            tag = list_files[0].split("_")[0].replace("_","-")    
            # grid scale mults correlated in space, constant in time
            pf.add_parameters(filenames=list_files, par_type='grid',
                    par_name_base='defwel',
                    pargp='defwel', index_cols=[0, 1, 2], use_cols=[3],
                   upper_bound=1.0, lower_bound=0.0, initial_value=1.0,transform="none")
            include_def = True

       # ghb_files = [f for f in os.listdir(template) if f.startswith(f'ghb_') and f.endswith('.txt')]
        # ghb_files.sort()

        # assert len(ghb_files) == nper

        # # # Create arrays for ghbs edge
        # pf.add_parameters(filenames=ghb_files, par_type="constant",
        #                   par_name_base='ghb_stg_' + "cn",
        #                   pargp='ghb_stg_' + "gr", index_cols=[0, 1, 2],
        #                   upper_bound=10, lower_bound=-10,initial_value = 0.0,
        #                    use_cols=[3],par_style="a", mfile_skip = 1,
        #                   transform="none")

        # pf.add_parameters(filenames=ghb_files, par_type="constant",
        #                   par_name_base='ghb_cond'+ "cn",
        #                   pargp='ghb_cond' + "cn",
        #                   index_cols=[0, 1, 2], use_cols=[4],
        #                   upper_bound=300.0, lower_bound=0.01,
        #                   mfile_skip = 1,)

        # hard coded the df return order
        dfs = init_budget_process(template,modnm=modnm)
        pf.add_observations('budget.csv',index_cols=['datetime'],use_cols=dfs[0].columns.to_list(),obsgp='bud',ofile_sep=',',prefix='bud')

        # add layer zone budget obs:
        df_zb = init_zonbud_process(template,modnm=modnm)
        pf.add_observations('zbud.csv',index_cols=['datetime'],use_cols=df_zb[0].columns.to_list(),obsgp='zbudly',ofile_sep=',',prefix='zbudly')
        
        #adding riv obs
        #dfs = init_riv_process(template, modnm=modnm)
        #pf.add_observations("riv_flx_south.csv", index_cols=["datetime"], use_cols=dfs.columns.values.tolist(), ofile_sep=",",
        #                    obsgp=["greater_rivflow"]*len(dfs.columns.values.tolist()), prefix="g_rivflow")
                
        # pf.add_observations('elk.riv.obs.output.csv', index_cols=['datetime'], use_cols=dfs[2].columns.to_list(), obsgp='riv', ofile_sep=',',
        #                     prefix='riv')
        
        # pf.add_observations('elk.wel.obs.output.csv', index_cols=['datetime'], use_cols=dfs[3].columns.to_list(), obsgp='wel', ofile_sep=',',
        #                     prefix='wel')
        # Listbudget Obs
        # process model output
        flx, cum = init_listbudget_obs(template, modnm)
        
        pf.mod_sys_cmds.append('mf6')
        
        # add post process function to forward run script
        pf.add_py_function('elk03_setup_pst.py', 'process_listbudget_obs()', is_pre_cmd=None)
        pf.add_py_function('elk03_setup_pst.py', 'znbud_by_ly_process()', is_pre_cmd=None)
        # add call to processing script to pst forward run
        pf.post_py_cmds.append("process_listbudget_obs('{0}')".format(modnm))
        
        
        hdf = init_head_targets_process(template,modnm=modnm)
        pf.add_observations(f'{modnm}.ss_head.obs.output.csv', index_cols=['datetime'], use_cols=hdf[0].columns.to_list(), obsgp='sshds', ofile_sep=',',
                             prefix='sshds')

        pf.add_observations(f'{modnm}.trans_head.obs.output.csv', index_cols=['datetime'], use_cols=hdf[1].columns.to_list(), obsgp='transhds', ofile_sep=',',
                             prefix='transhds')
        
        dddf = init_dd_targets_process(template,modnm=modnm)
        pf.add_observations(f'{modnm}.trans_sim_dd_targs.output', index_cols=['datetime'], use_cols=dddf.columns.to_list(), obsgp='ddtrgs', ofile_sep=',',
                            prefix='ddtrgs')

        # pf.add_observations("auto_flow_reduce.csv",index_cols=["time","period","step","boundnumber","cellnumber"],
        #     use_cols=["rate-requested","rate-actual","wel-reduction"],prefix="afr",obsgp="afr",ofile_sep=",")
        # if include_def:
        #     pf.add_observations("defwel_auto_flow_reduce.csv",index_cols=["time","period","step","boundnumber","cellnumber"],
        #     use_cols=["rate-requested","rate-actual","wel-reduction"],prefix="dwafr",obsgp="afr",ofile_sep=",")
        
        df = process_autoreduce(model_ws=pf.new_d)
        if include_def:
            assert "deferred" in df.index.to_list()
            pf.add_py_function("elk03_setup_pst.py","round_defwel(model_ws='.')",is_pre_cmd=True)
            round_defwel(model_ws=pf.new_d)
            pf.add_observations("defwel_total_active.csv",index_cols=["col"],prefix="activedefwel",obsgp="activedefwel")
            
            

        pf.add_py_function("elk03_setup_pst.py","process_autoreduce(model_ws='.')",is_pre_cmd=False)
        pf.add_observations("autoreduce_summary.csv",index_cols="wel-pak",use_cols=df.columns.tolist(),prefix="autoreduce",obsgp="autoreduce",ofile_sep=',')




        pf.tmp_files.append(f"{modnm}.ss_head.obs.output")
        pf.tmp_files.append(f"{modnm}.trans_head.obs.output")
        pf.tmp_files.append(f"{modnm}.sspmp_head.obs.output")
        pf.tmp_files.append(f"{modnm}.trans_sim_dd_targs.output")

        # import flopy as part of the forward run process
        pf.extra_py_imports.append('flopy')    
        pf.extra_py_imports.append("shutil")  
        pf.extra_py_imports.append("pathlib")  
        
        
        #pf.mod_sys_cmds.append('mf6')
        #pf.add_py_function('elk03_setup_pst_nozon.py', 'interp_engine_ghb()', is_pre_cmd=True)
        pf.post_py_cmds.append(f'head_targets_process(modnm="{modnm}")')
        pf.post_py_cmds.append(f'dd_targets_process(modnm="{modnm}")')
        pf.post_py_cmds.append(f'budget_process(modnm="{modnm}")')
        pf.post_py_cmds.append(f'znbud_by_ly_process(modnm="{modnm}")')

        # ---------------------------------------------
        # Inequality obs: head - top in flood polygons
        # ---------------------------------------------
        # run once in template to generate initial CSV and get column names
        cwd = os.getcwd()
        os.chdir(template)
        try:
            ineq_df = flood_ineq_obs_process(modnm=modnm)
        finally:
            os.chdir(cwd)

        if not ineq_df.empty:
            pf.add_observations(
                "flood_ineq_obs.csv",
                index_cols=["datetime", "obsname"],  # time + cell id
                use_cols=["simval"],                 # single value column
                obsgp="hd_lt_top",
                ofile_sep=",",
                prefix="hdltp",
            )

            # ✅ register the function so the worker script can import it
            pf.add_py_function(
                "elk03_setup_pst.py",
                "flood_ineq_obs_process()",
                is_pre_cmd=None,
            )

            pf.post_py_cmds.append(f'flood_ineq_obs_process(modnm="{modnm}")')

        else:
            print("[flood_ineq] No inequality obs added (empty ineq_df).")

        
        #pf.post_py_cmds.append('riv_flux_process()')
        
        pf.add_py_function('elk03_setup_pst.py','budget_process()',is_pre_cmd=None)
        pf.add_py_function('elk03_setup_pst.py', 'head_targets_process()', is_pre_cmd=None)
        pf.add_py_function('elk03_setup_pst.py', 'dd_targets_process()', is_pre_cmd=None)
        pf.add_py_function('elk03_setup_pst.py', f"riv_drn_bot_chk(mnm='{modnm}')", is_pre_cmd=True)
        if include_def:
            sum_level_impacts(model_ws=pf.new_d)
            pf.add_py_function("elk03_setup_pst.py","sum_level_impacts(model_ws='.')",is_pre_cmd=False)
            pf.add_observations("sum_level_impacts.csv",index_cols=0,prefix="sum-level-impacts",
                obsgp="sum-level-impacts",ofile_sep=",")
        # add call to processing script to pst forward run
    
        # add obs via PstFrom
        ignore_cols = ['datetime', 'in-out', 'total', 'wel-in']
        cols = [c for c in flx.columns if c not in ignore_cols]
        pf.add_observations('listbudget_flx_obs.csv', insfile='listbudget_flx_obs.csv.ins',
                                   index_cols=['datetime'], use_cols=cols, prefix='flx')
        

    pf.parfile_relations.to_csv(os.path.join(pf.new_d, 'mult2model_info.csv'))
    curdir = os.getcwd()
    os.chdir(pf.new_d)
    df = pyemu.helpers.calc_array_par_summary_stats()
    os.chdir(curdir)
    pf.post_py_cmds.append('pyemu.helpers.calc_array_par_summary_stats()')
    pf.add_observations('arr_par_summary.csv', index_cols=['model_file'], use_cols=df.columns.tolist(),
                        obsgp='arrparsum', prefix='arrparsum',
                        ofile_sep=',')
                  
    # MODFLOW input value observations
    # summary statistic observations of modflow inputs resulting from multiplier application
    df = init_mfinput_obs(template, modnm)

    # add post process function to forward run script
    pf.add_py_function('elk03_setup_pst.py', 'process_mfinput_obs()', is_pre_cmd=None)
    # add call to processing script to pst forward run
    pf.post_py_cmds.append("process_mfinput_obs('{0}')".format(modnm))
    
    # add obs via PstFrom
    cols = ['upper_bound', 'lower_bound', 'min', 'qnt25', 'qnt50', 'qnt75', 'max', 'near_lbnd', 'near_ubnd']
    
    pf.add_observations('mfinput_obs.csv', insfile='mfinput_obs.csv.ins',
                                index_cols=['input'], use_cols=cols, prefix='mfin')
  
    # build pest control file
    pf.build_pst(version=None)


    pf.pst.control_data.noptmax = 0
    pf.pst.pestpp_options['additional_ins_delimiters'] = ','
    pf.pst.write(os.path.join(template,f'{modnm}.pst'),version=2)
    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst',cwd=template)
    pf.pst.set_res(os.path.join(template,f'{modnm}.base.rei'))
    print('phi',pf.pst.phi)
    if pf.pst.phi >= 1e-4:
        print('phi is greater than 1e-4, returning rei, investigate')
    rei = pf.pst.res
    # sort rei by residual magnitude:
    rei = rei.sort_values(by='residual')
    
    #if any deferred wels are present, set them to fixed and their parval1 to 0.0
    # before draw()
    if include_def:
        par = pf.pst.parameter_data
        dpar = par.loc[par.pname.str.contains("defwel"),:]
        assert len(dpar) > 0
        par.loc[dpar.parnme,"partrans"] = "fixed"
        par.loc[dpar.parnme,"parval1"] = 0.0


    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=True)
    pe.to_binary(os.path.join(template, 'prior.jcb'))
    pf.pst.pestpp_options['ies_par_en'] = 'prior.jcb'
    pf.pst.pestpp_options['save_binary'] = True

    # write the updated pest control file
    pf.pst.write(os.path.join(pf.new_d, f'{modnm}.pst'),version=2)

    shutil.copy(os.path.join(pf.new_d, f'{modnm}.obs_data.csv'),
                os.path.join(pf.new_d, f'{modnm}.obs_data_orig.csv'))

    for py_file in [f for f in os.listdir(".") if f.endswith(".py")]:
        shutil.copy2(py_file,os.path.join(pf.new_d,py_file+".bak"))

    shutil.copytree("run_inputs",os.path.join(pf.new_d,"run_inputs_bak"))


    return template # return the template directory name


def set_obsvals_and_weights(template_d, modnm="elk",
                            flow_weight_scheme='basic',
                            phi_factor_dict=None,):

    pst = pyemu.Pst(os.path.join(template_d, f'{modnm}.pst'))
    
    # load model dis:
    sim = flopy.mf6.MFSimulation.load(sim_ws=template_d, sim_name=modnm, load_only=['dis'])
    m = sim.get_model(modnm)
    nlay = m.dis.nlay.array
   
    # now set obsvals and weights
    obs = pst.observation_data
    obs.loc[:, 'weight'] = 0
    obs.loc[:, 'observed'] = False
    obs.loc[:, 'count'] = 0
    obs.loc[:, 'standard_deviation'] = 0
    obs.loc[:, 'obsval'] = 0.0  

    if flow_weight_scheme is not None:

        # ---------- steady-state heads ----------
        h_df = pd.read_csv(os.path.join('data', 'analyzed', 'elk_ss_targets.csv'),
                           index_col=['start_dt'], parse_dates=True)
        h_df.loc[:, "datetime"] = pd.to_datetime(h_df.index, format="%Y-%m-%d")
        h_df.loc[:, 'k'] = h_df.k.astype(int)
        h_df.loc[:, 'i'] = h_df.i.astype(int)
        h_df.loc[:, 'j'] = h_df.j.astype(int)
        h_df.loc[:, 'obsprefix'] = h_df.obsprefix.apply(lambda x: x.replace('.', '-'))
        h_df["otype"] = "head"

        hobspref = h_df.obsprefix.unique().tolist()
        all_obspref = list(set(hobspref))
        uprefixes = all_obspref.copy()
        uprefixes.sort()

        oname_obsval_dict = {}
        for prefix in uprefixes:
            uh_df = h_df.loc[h_df.obsprefix == prefix, :].copy()
            uk = uh_df.k.unique()
            uk.sort()
            
            prefix_shrt = prefix.split('_k:')[1]
            pobs = obs.loc[obs.obsnme.apply(lambda x: prefix_shrt in x), :].copy()
            pobs = pobs.loc[pobs.oname == "sshds", :]

            if pobs.shape[0] == 0:
                print('empty obs for prefix:{0}'.format(prefix))
                continue

            pobs.loc[:, 'k'] = uh_df.k.values[0].astype(int)
            pobs.loc[:, 'i'] = uh_df.i.values[0].astype(int)
            pobs.loc[:, 'j'] = uh_df.j.values[0].astype(int)

            for k in uk:
                kuh_df = uh_df.loc[uh_df.k == k, :].copy()
                if kuh_df.shape[0] == 0:
                    print('empty layer df for k:{0},prefix:{1}'.format(k, prefix))
                    continue
                ukobs = pobs.loc[pobs.k == k, :].copy()
                if ukobs.shape[0] == 0:
                    print('empty ukobs for k:{0},prefix:{1}'.format(k, prefix))
                    continue
                for head, dt in zip(kuh_df.loc[:, 'head_target'], kuh_df.datetime):
                    if dt < pd.to_datetime('1970-01-01'):
                        mn_oname = ukobs.iloc[0, :].obsnme
                        oname_obsval_dict.setdefault(mn_oname, []).append(head)
        
        # ---------- transient heads ----------
        t_df_loc = pd.read_csv(os.path.join('data', 'analyzed', 'transient_well_targets_lookup_shrt.csv'))
        t_df = pd.read_csv(os.path.join('data', 'analyzed', 'transient_well_targets.csv'))
        unq_prefix = t_df_loc['obsprefix'].unique()

        oname_obsval_dict_trans = {}
        for prefix in unq_prefix:
            if 'transh_grpid:' not in prefix:
                continue
       
            uh_df = t_df.loc[:, ['start_datetime', prefix]].copy()
            if uh_df[prefix].isnull().all():
                print('no data for prefix:{0}'.format(prefix))
                continue
            
            uh_df = uh_df.loc[uh_df[prefix].notnull(), :].copy()
        
            pobs = obs.loc[obs.obsnme.apply(lambda x: prefix in x), :].copy()
            dt_vals = uh_df['start_datetime'].unique()
            dt_vals.sort()
            pobs_w_meas = pobs.loc[pobs.obsnme.apply(lambda x: any(dt in x for dt in dt_vals)), :].copy()
      
            if pobs_w_meas.shape[0] == 0:
                print('empty obs for prefix:{0}'.format(prefix))
                continue
     
            for idx, row in pobs_w_meas.iterrows():
                oname = row.obsnme
                sim_date = oname.split(':')[-1]   # date at end of name
                val = uh_df.loc[uh_df['start_datetime'] == sim_date, prefix].values[0]
                if np.isnan(val):
                    assert False, 'nan value for {0}, something went wrong...qa needed'.format(oname)
                oname_obsval_dict_trans.setdefault(oname, []).append(val)

        # ---------- DD (SP-delta) targets ----------
        dd_obs = pd.read_csv(os.path.join(template_d, f'{modnm}.trans_obs_dd_targs.output'))
        idx_wells = dd_obs.columns[1:].values.tolist()   # first col is datetime

        oname_obsval_dict_dd = {}
        for prefix in idx_wells:
            if 'dd_grpid:' not in prefix:
                continue

            uh_df = dd_obs.loc[:, ['datetime', prefix]].copy()
            if uh_df[prefix].isnull().all():
                print('no data for prefix:{0}'.format(prefix))
                continue
            
            uh_df = uh_df.loc[uh_df[prefix].notnull(), :].copy()
        
            pobs = obs.loc[obs.obsnme.apply(lambda x: prefix in x), :].copy()
            dt_vals = uh_df['datetime'].unique()
            dt_vals.sort()
            pobs_w_meas = pobs.loc[pobs.obsnme.apply(lambda x: any(dt in x for dt in dt_vals)), :].copy()
      
            if pobs_w_meas.shape[0] == 0:
                print('empty obs for prefix:{0}'.format(prefix))
                continue
     
            for idx, row in pobs_w_meas.iterrows():
                oname = row.obsnme
                sim_date = oname.split(':')[-1]   # expecting yyyy-mm-dd at end
                val = uh_df.loc[uh_df['datetime'] == sim_date, prefix].values[0]
                if np.isnan(val):
                    assert False, 'nan value for {0}, something went wrong...qa needed'.format(oname)
                oname_obsval_dict_dd.setdefault(oname, []).append(val)

        
        print('\n\n\n  ---  found {0} ss gw level obs  ---  \n\n\n'.format(len(oname_obsval_dict)))
        assert len(oname_obsval_dict) > 0
        print('\n\n\n  ---  found {0} transient gw level obs  ---  \n\n\n'.format(len(oname_obsval_dict_trans)))
        assert len(oname_obsval_dict_trans) > 0
        print('\n\n\n  ---  found {0} dd gw level obs  ---  \n\n\n'.format(len(oname_obsval_dict_dd)))
        assert len(oname_obsval_dict_dd) > 0

        # ---------------------------------------------
        # Inequality obs: head - top <= 0  (flood zone)
        # ---------------------------------------------
        ineq_mask = obs.obgnme == "hd_lt_top"
        if ineq_mask.any():
            # target is 0.0 (no flooding); positive residual => head above top
            obs.loc[ineq_mask, "obsval"] = 0.0
            obs.loc[ineq_mask, "observed"] = True

            # choose a weight; you can tune this. Start modest, e.g. 0.1
            obs.loc[ineq_mask, "weight"] = 1.0
            obs.loc[ineq_mask, "standard_deviation"] = 0
            obs.loc[ineq_mask, "count"] = 1


        # -------------------- weight schemes --------------------
        if flow_weight_scheme == 'ss_only':
            for oname, vals in oname_obsval_dict.items():
                vals = np.array(vals)
                obs.loc[oname, 'obsval'] = vals.mean()
                obs.loc[oname, 'observed'] = True
                obs.loc[oname, 'standard_deviation'] = 5
                obs.loc[:, 'count'] = len(vals)
                obs.loc[oname, 'weight'] = 0.2
            obs.loc[(obs.obgnme == "greater_rivflow") &
                    (obs.obsnme.str.contains("1970-01-01")), 'weight'] = 1.0

        if flow_weight_scheme == 'all_wl_meas':
            # SS heads
            for oname, vals in oname_obsval_dict.items():
                vals = np.array(vals)
                obs.loc[oname, 'obsval'] = vals.mean()
                obs.loc[oname, 'observed'] = True
                obs.loc[oname, 'standard_deviation'] = 2.0
                obs.loc[:, 'count'] = len(vals)
                obs.loc[oname, 'weight'] = 0.1
                obs.loc[oname, 'obgnme'] = 'sshds'
                
            # transient heads
            for oname, vals in oname_obsval_dict_trans.items():
                vals = np.array(vals)
                year = int(oname.split(':')[8].split('-')[0])
                obs.loc[oname, 'obsval'] = vals
                obs.loc[oname, 'observed'] = True
                obs.loc[oname, 'standard_deviation'] = 2.0
                obs.loc[:, 'count'] = len(vals)
                obs.loc[oname, 'weight'] = 0.5
                if year >= 1990:
                    obs.loc[oname, 'obgnme'] = 'transhds'
                else:
                    obs.loc[oname, 'obgnme'] = 'transearlyhds'

            # DD (SP-delta) targets
            for oname, vals in oname_obsval_dict_dd.items():
                vals = np.array(vals)
                year = int(oname.split(':')[-1].split('-')[0])

                obs.loc[oname, 'obsval'] = vals
                obs.loc[oname, 'observed'] = True
                obs.loc[oname, 'standard_deviation'] = 2.0
                obs.loc[:, 'count'] = len(vals)
                obs.loc[oname, 'weight'] = 0.25  # tweak if you want DD stronger/weaker
                obs.loc[oname, 'obgnme'] = 'ddtrgs'
                        
        # bump SS settings a bit
        # obs.loc[obs.obgnme == 'sshds', 'standard_deviation'] = 3
        # obs.loc[obs.obgnme == 'sshds', 'weight'] = 0.333

    assert pst.nnz_obs > 0  

    nzobs = obs.loc[obs.weight > 0, :]
    vc = nzobs.obgnme.value_counts()
    for gname, c in zip(vc.index, vc.values):
        print('group ', gname, ' has ', c, ' nzobs')

    if phi_factor_dict is not None:
        with open(os.path.join(template_d, 'phi_facs.csv'), 'w') as f:
            keys = list(phi_factor_dict.keys())
            keys.sort()
            for key in keys:
                f.write('{0},{1}\n'.format(key, phi_factor_dict[key]))
        pst.pestpp_options['ies_phi_factor_file'] = 'phi_facs.csv'

    pst.write(os.path.join(template_d, f'{modnm}.pst'), version=2)
    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst', cwd=template_d)

    return obs



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


def run_ies(template_ws='template_d', modnm='elk', m_d=None, num_workers=12,niceness=False, noptmax=-1, num_reals=None,
              init_lam=None, drop_conflicts=False, local=True, hostname=None, port=4263,
               use_condor=False,bin_name="pestpp-ies",**kwargs):
    
    if m_d is None:
        m_d = template_ws.replace('template', 'master')

    pst = pyemu.Pst(os.path.join(template_ws, f'{modnm}.pst'))

    # Set control file options:
    pst.control_data.noptmax = noptmax
    pst.pestpp_options['ies_drop_conflicts'] = drop_conflicts
    pst.pestpp_options['overdue_giveup_fac'] = 10
    pst.pestpp_options['ies_bad_phi_sigma'] = 1.5
    pst.pestpp_options['ies_bad_phi'] = 1e+10
    #pst.pestpp_options["ies_n_iter_reinflate"] = [-2,999]
    pst.pestpp_options["ies_init_lam"] = -10
    pst.pestpp_options['panther_agent_freeze_on_fail'] = False

    pst.pestpp_options['save_binary'] = True
    if num_reals is not None:
        pst.pestpp_options['ies_num_reals'] = num_reals

    if init_lam is not None:
        pst.pestpp_options['ies_initial_lambda'] = init_lam
    pst.pestpp_options['ies_subset_size'] = -10
    for k,v in kwargs.items():
        pst.pestpp_options[k] = v
    # intit run log file:
    f = open(os.path.join(template_ws, 'elkpst_run.log'), 'w')
    f.close()

    # obs sainty check:
    pobs = pst.observation_data
    pobsmax = pobs.weight.max()
    if pobsmax <= 0:
        raise Exception('setting weighted obs failed!!!')
    pst.write(os.path.join(template_ws, f'{modnm}.pst'), version=2)

    prep_worker(template_ws, template_ws + '_clean')
    
    master_p = None

    if hostname is None:
        pyemu.os_utils.start_workers(template_ws, bin_name, f'{modnm}.pst',
                                 num_workers=num_workers, worker_root='.',
                                 master_dir=m_d, local=local,port=4269)

    elif use_condor:
        check_port_number(port)

        jobid = condor_submit(template_ws=template_ws + '_clean', pstfile=f'{modnm}.pst', conda_zip_pth='nddwrpy311.tar.gz',
                              subfile=f'{modnm}.sub',
                              workerfile='worker.sh', executables=['mf6', bin_name,'mp7'], request_memory=5000,
                              request_disk='15g', port=port,
                              num_workers=num_workers,niceness=niceness)

        # jwhite - commented this out so not starting local workers on the condor submit machine # no -ross
        pyemu.os_utils.start_workers(template_ws + '_clean', bin_name, f'{modnm}.pst', num_workers=0, worker_root='.',
                                     port=port, local=local, master_dir=m_d)

        if jobid is not None:
            # after run master is finished clean up condor by using condor_rm
            print(f'killing condor job {jobid}')
            os.system(f'condor_rm {jobid}')

    # if a master was spawned, wait for it to finish
    if master_p is not None:
        master_p.wait()


def prep_worker(org_d, new_d,run_flex_cond=False):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)
    exts = ['rei', 'hds', 'cbc', 'ucn', 'cbb', 'ftl', 'm3d', 'tso', 'ddn','log','rec','list','jcb']
    if run_flex_cond:
        files = [f for f in os.listdir(new_d) if f.lower().split('.')[-1] in exts]
        for f in files:
            if f != 'prior.jcb':  # need prior.jcb to run ies
                os.remove(os.path.join(new_d, f))
    else:
        files = [f for f in os.listdir(new_d) if f.lower().split('.')[-1] in exts]
        for f in files:
            if f != 'cond_post.jcb' and f != 'noise.jcb' and f != 'prior.jcb' and f != 'pmp_noise.jcb':  # need prior.jcb to run ies
                os.remove(os.path.join(new_d, f))
    mlt_dir = os.path.join(new_d, 'mult')
    for f in os.listdir(mlt_dir)[1:]:
        os.remove(os.path.join(mlt_dir, f))
    tpst = os.path.join(new_d, 'temp.pst')
    if os.path.exists(tpst):
        os.remove(tpst)


def condor_submit(template_ws, pstfile, conda_zip_pth='nddwrpy311.tar.gz', subfile='condor.sub', workerfile='worker.sh',
                  executables=[], request_memory=4000, request_disk='10g', port=4200, num_workers=71,niceness=False):
    '''
    :param template_ws: path to template_ws
    :param pstfile: name of pest control file
    :param conda_zip_pth: conda-pack zip file
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

    if not os.path.join(conda_zip_pth):
        str = f'conda-pack dir {conda_zip_pth} does not exist\n ' + f'consider running conda-pack while in your conda env\n'
        AssertionError(str)
    conda_base = conda_zip_pth.replace('.tar.gz', '')

    # should probably remove to remove tmp files to make copying faster...
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'temp')):
        shutil.rmtree(os.path.join(cwd, 'temp'))
    shutil.copytree(os.path.join(cwd, template_ws), 'temp')

    # zip template_ws
    os.system(f'tar cfvz temp.tar.gz temp')

    if not os.path.exists('log'):
        os.makedirs('log')

    # write worker file
    worker_f = open(os.path.join(cwd, workerfile), 'w')
    worker_lines = ['#!/bin/sh\n',
                    '\n',
                    '# make conda-pack dir\n',
                    f'mkdir {conda_base}\n',
                    f'tar -xf {conda_zip_pth} -C {conda_base}\n',
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
                f'transfer_input_files = temp.tar.gz, {conda_zip_pth}\n',
                '\n',
                ' # should I be nice?\n',
                f'nice_user = {niceness}\n',
                '# number of workers to start\n',
                f'queue {num_workers}']
    # sublines += ['\n',
    #             '# Set job priority (higher = higher priority, default is 0, max is 20)\n',
    #             'priority = -10\n',  # Change this value as needed
                
    #             '\n']
    sub_f.writelines(sublines)
    sub_f.close()

    os.system(f'condor_submit {subfile} > condor_jobID.txt')

    jobfn = open('condor_jobID.txt')
    lines = jobfn.readlines()
    jobfn.close()
    jobid = lines[1].split()[-1].replace('.', '')
    print(f'{num_workers} job(s) submitted to cluster {jobid}.')

    return int(jobid)


def draw_pmp_noise_reals(t_d, modnm):
    sim = flopy.mf6.MFSimulation.load(sim_ws=t_d, load_only=['dis'])
    mg = sim.get_model().modelgrid
    pst = pyemu.Pst(os.path.join(t_d, f'{modnm}.pst'))

    pr = pyemu.ParameterEnsemble.from_binary(
        pst=pst, filename=os.path.join(t_d, pst.pestpp_options['ies_par_en'])
    )
    pnames = [p for p in pr.columns if 'twel' in p]

    spd = pd.read_csv(os.path.join('tables', 'annual_stress_period_info.csv'))

    wpr = pr.loc[:, pnames].copy()

    num_reals = pr.shape[0]
   
    wellpkg_nms = np.unique([pn.split('_')[0].split('-')[-1] for pn in pnames])
    wellpkg_nms = ['cow']  # testing

    struct_dict = {}
    pinfo_parts = []

    for wp in wellpkg_nms:
        p = wpr.loc[:, wpr.columns.str.contains(wp)].copy().T.reset_index()
        p = p.rename(columns={'index': 'pname'})
        sp = p['pname'].str.split('_').str[1].str.replace('kper:', '').astype(int)
        p['kper'] = sp

        spd_shrt = spd[['stress_period', 'cum_days']].copy()
        spd_shrt['kper'] = spd_shrt.stress_period - 1
        spd_shrt = spd_shrt.loc[spd_shrt.kper > 0, :].copy()  # drop ss
        p = p.sort_values(by='kper').merge(spd_shrt, on='kper', how='left')

        # smoother, longer correlation
        v = pyemu.geostats.GauVario(a=20 * 365.0, contribution=1.0,anisotropy=1.0)
        gs = pyemu.geostats.GeoStruct(variograms=v, name=wp)

        struct_dict[gs] = p['pname'].tolist()

        df = p[['pname', 'cum_days']].rename(columns={'cum_days': 'time_distance_col'})
        pinfo_parts.append(df)

    pinfo_df = pd.concat(pinfo_parts, ignore_index=True).drop_duplicates('pname')

    def make_bounds_to_targets(
        pst,
        pinfo_df,
        time_distance_col="time_distance_col",
        years_to_target=20,
        target_lb=1.0,   # scalar (e.g., 0.90) or dict {pname: 0.90}
        target_ub=1.0,   # scalar (e.g., 1.10) or dict {pname: 1.10}
        time_units="days",  # "days" (default) or "years"
    ):
        """
        Build per-time bounds that move from PST (parlbnd, parubnd) toward
        target_lb and target_ub over the first `years_to_target`.

        Returns DataFrame with columns ['pname','lb','ub'].

        target_lb / target_ub can be scalars or dicts keyed by pname.
        """
        df = pinfo_df[["pname", time_distance_col]].copy()
        par = pst.parameter_data.set_index("parnme")
        lb0 = par["parlbnd"].to_dict()
        ub0 = par["parubnd"].to_dict()

        # unit handling
        if time_units == "days":
            horizon = float(years_to_target) * 365.0
            tvals = df[time_distance_col].astype(float).values
        elif time_units == "years":
            horizon = float(years_to_target)
            tvals = df[time_distance_col].astype(float).values
        else:
            raise ValueError("time_units must be 'days' or 'years'")

        def targ(val, name, default):
            if val is None:
                return default
            return val[name] if isinstance(val, dict) else float(val)

        lbs, ubs = [], []
        for name, t in zip(df["pname"].values, tvals):
            f = 0.0 if horizon <= 0 else max(0.0, min(1.0, t / horizon))  # progress 0→1
            # independent targets for lb and ub
            tlb = targ(target_lb, name, 1.0)
            tub = targ(target_ub, name, 1.0)

            lb = (1.0 - f) * lb0[name] + f * tlb
            ub = (1.0 - f) * ub0[name] + f * tub

            # guard: ensure lb <= ub
            if lb > ub:
                mid = 0.5 * (lb + ub)
                lb = mid
                ub = mid

            lbs.append(lb)
            ubs.append(ub)

        out = df[["pname"]].copy()
        out["lb"] = lbs
        out["ub"] = ubs
        return out

    bnds = make_bounds_to_targets(
        pst,
        pinfo_df,
        years_to_target=20,
        target_lb=0.9,
        target_ub=1.1)
    

    np.random.seed(1123556564)
    pe = autocorrelated_param_draw(
        pst,
        struct_dict,
        pinfo_df,
        time_distance_col='time_distance_col',
        num_reals=num_reals,
        verbose=True,
        enforce_bounds=True,
        bounds_df=bnds,
    )

    # quick look
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = pinfo_df.sort_values('time_distance_col')['pname'].tolist()
    x = pinfo_df.set_index('pname').loc[names, 'time_distance_col'].values
    for i in range(num_reals):
        ax.plot(x, pe.loc[i, names].values, alpha=0.35)

    # save pe to binary and replace prior.jcb
    pe.to_binary(os.path.join(t_d, 'pmp_noise.jcb'))
    pst.pestpp_options['ies_par_en'] = 'pmp_noise.jcb'
    pst.control_data.noptmax = -2
    pst.write(os.path.join(t_d,f"{modnm}.pst"),version=2)
    pyemu.os_utils.run(f"pestpp-ies {modnm}.pst",cwd=t_d)


def autocorrelated_param_draw(
    pst,
    struct_dict,
    pinfo_df,
    time_distance_col='distance',
    num_reals=100,
    verbose=True,
    enforce_bounds=True,
    bounds_df=None,
    draw_fixed=False
):
    """
    Construct an autocorrelated *parameter* ensemble from geostatistical structure(s).

    Args
    ----
    pst : pyemu.Pst
        Control file (or path). Parameter prior (variance) info is taken from pst.parameter_data.
    struct_dict : dict
        {GeoStruct_or_path: [param_name_1, param_name_2, ...], ...}
        IMPORTANT: The variogram structures used here should have sill=1.0. Scaling is
        done by each parameter's prior variance (from pst).
    pinfo_df : pd.DataFrame
        DataFrame with columns ["pname", time_distance_col] giving the distance (e.g., time)
        associated with each parameter listed in struct_dict.
    time_distance_col : str
        Column in pinfo_df that supplies the “distance” used in the covariance.
    num_reals : int
        Number of realizations.
    verbose : bool
        Print progress.
    enforce_bounds : bool
        If True, clamp ensemble values to [parlbnd, parubnd].
    draw_fixed : bool
        If False (default), fixed/tied parameters are held at parval1 in the ensemble.

    Returns
    -------
    pyemu.ParameterEnsemble
    """
    # --- Validate inputs
    if "pname" not in pinfo_df.columns or time_distance_col not in pinfo_df.columns:
        raise ValueError("pinfo_df must have columns ['pname', time_distance_col]")
    pinfo_df = pinfo_df.copy()
    pinfo_df.index = pinfo_df["pname"]

    par = pst.parameter_data.copy()
    par_names_all = par.parnme.tolist()

    # Flatten param name list from struct_dict and check existence
    passed_names = []
    for _, onames in struct_dict.items():
        passed_names.extend(onames)
    missing = list(set(passed_names) - set(par_names_all))
    if missing:
        raise Exception("Parameters in struct_dict not found in pst.parameter_data: " + str(missing))

    # Ensure we have distances for all passed names
    if not set(passed_names).issubset(set(pinfo_df.index)):
        need = list(set(passed_names) - set(pinfo_df.index))
        raise Exception(f"Missing distances in pinfo_df for: {need}")

   
    # --- Full diagonal covariance from parameter priors

    fcov = pyemu.Cov.from_parameter_data(pst)  # diagonal

    # cache std dev per parameter for scaling of geo-cov blocks
    std_dict = {name: np.sqrt(fcov.x[i]) for i, name in enumerate(fcov.names)}

    # --- Start with a diagonal draw (for all params), then replace blocks with correlated draws
    if verbose:
        print("--> drawing full parameter ensemble from diagonal prior")
    full_pe = pyemu.ParameterEnsemble.from_gaussian_draw(
        pst, fcov, num_reals=num_reals, by_groups=False
    )

    # --- For each GeoStruct block, build covariance and draw correlated columns
    keys = list(struct_dict.keys())
    # if any value is a path, load it
    keys_resolved = []
    for k in keys:
        print(k)
        if isinstance(k, str):
            keys_resolved.append(pyemu.geostats.GeoStruct.from_struct_file(k))
        else:
            keys_resolved.append(k)

    for gs in sorted(keys_resolved, key=lambda x: str(x)):
        onames = struct_dict.get(gs, [])
        if len(onames) == 0:
            continue
        if verbose:
            print(f"--> processing cov block with {len(onames)} params using {gs}")

        dvals = pinfo_df.loc[onames, time_distance_col].values
        # unit sill covariance; weights vector is ones (equal support)
        gcov = gs.covariance_matrix(dvals, np.ones(len(onames)), names=onames)

        # scale by each parameter's prior std to match the diagonal prior
        if verbose:
            print("...scaling covariance rows/cols by parameter std dev")
        for i, name in enumerate(gcov.names):
            s = std_dict.get(name, None)
            if s is None:
                raise Exception(f"Missing prior std for parameter '{name}'")
            gcov.x[:, i] *= s
            gcov.x[i, :] *= s

        # Draw only those columns, then splice into the full ensemble
        if verbose:
            print("...drawing correlated block")
        pe_block = pyemu.ParameterEnsemble.from_gaussian_draw(
            pst, gcov, num_reals=num_reals, fill=False, by_groups=False
        )
        pe_block = pe_block.loc[:, gcov.names]  # align

        full_pe.loc[:, gcov.names] = pe_block._df.values

    # --- Enforce bounds
    if enforce_bounds:
        if verbose:
            print("--> enforcing parameter bounds")

        par = pst.parameter_data.copy()
        # default to PST bounds
        lb_dict = par.set_index("parnme").parlbnd.to_dict()
        ub_dict = par.set_index("parnme").parubnd.to_dict()

        # If user supplied time-dependent bounds, override by pname
        if bounds_df is not None:
            if not {"pname", "lb", "ub"}.issubset(bounds_df.columns):
                raise ValueError("bounds_df must have columns: ['pname','lb','ub']")
            # prefer user bounds where provided
            bdf = bounds_df.drop_duplicates("pname").set_index("pname")[["lb","ub"]]
            for nm, lb, ub in bdf.itertuples(index=True, name=None):
                lb_dict[nm] = float(lb)
                ub_dict[nm] = float(ub)


        # vectorized clamp
        allvals = full_pe.values
        cols = full_pe.columns.tolist()
        for j, nm in enumerate(cols):
            lo = lb_dict.get(nm, -1e300)
            hi = ub_dict.get(nm,  1e300)
            v = allvals[:, j]
            v[v < lo] = lo
            v[v > hi] = hi
            allvals[:, j] = v
        full_pe._df.iloc[:, :] = allvals

    return full_pe


def draw_noise_reals(t_d,modnm):
    sim = flopy.mf6.MFSimulation.load(sim_ws=t_d,load_only=["dis"])
    m = sim.get_model()
    top = m.dis.top.array
    mg = sim.get_model().modelgrid
    X,Y = mg.xcellcenters,mg.ycellcenters
    pst = pyemu.Pst(os.path.join(t_d,f"{modnm}.pst"))
    obs = pst.observation_data
    obs["distance"] = np.nan
    
    pr = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,pst.pestpp_options["ies_par_en"]))
    num_reals = pr.shape[0]

    #first the tr head groups
    trobs = obs.loc[(obs.oname=="transhds") & (obs.weight>0),:]
    assert trobs.shape[0] > 0
    trobs["datetime"] = pd.to_datetime(trobs.datetime)
    trobs['id'] = trobs.obsnme.apply(lambda x: x.split('transh_')[1].split('_i')[0])
    trobs["iid"] = trobs.loc[:,"id"]
    print(trobs.columns)
    for col in ["k","i","j"]:
        trobs[col] = trobs[col].astype(int)
    trobs['x'] = trobs.apply(lambda x: X[x.i,x.j],axis=1)
    trobs['y'] = trobs.apply(lambda x: Y[x.i,x.j],axis=1)
    uids = trobs["iid"].unique()
    uids.sort()
    xvals,yvals = [],[]
    struct_dict = {}
    for uid in uids:
        uobs = trobs.loc[trobs["iid"]==uid,:].copy()
        print(uid,uobs.shape)
        xvals.append(uobs.x.iloc[0])
        yvals.append(uobs.y.iloc[0])
        d = (uobs.datetime - uobs.datetime.min()).dt.days
        obs.loc[uobs.obsnme,"distance"] = d
        v = pyemu.geostats.ExpVario(contribution=1.0,a=3650)
        gs = pyemu.geostats.GeoStruct(variograms=v,name=uid)
        struct_dict[gs] = uobs.obsnme.tolist()

    np.random.seed(1123556564)
    oe = pyemu.helpers.autocorrelated_draw(pst,struct_dict,num_reals=num_reals)

    trobs_top = pd.Series(
        [top[i, j] for i, j in zip(trobs.i.values, trobs.j.values)],
        index=trobs.obsnme.values,
        name="top_elev",
    )

    trobs['top_elev'] = trobs_top.values
    
    #some normal variates that are added to all obs
    #think of these like a spatial constant that shifts all timeseries obs together
    draws = np.random.normal(0,1.0,num_reals)
    
    #nzobs = obs.loc[(obs.weight>0) & (obs.obgnme.str.containts("hds")),"obsnme"]
    #assert len(nzobs) > 0
    for i,draw in enumerate(draws):
        oe.loc[i,trobs.obsnme] += (draw * 0.5) # only 0.5ft stdev for these noises...
        vals = oe.loc[i,trobs.obsnme].values
        temp = pd.DataFrame({"obsnme":trobs.obsnme,"vals":vals})
        temp['top_elev'] = trobs_top.loc[temp.obsnme].values
        temp['mask'] = temp.vals >= temp.top_elev
        if temp['mask'].any():
            temp.loc[temp['mask'],'vals'] = temp.loc[temp['mask'],'top_elev']
            oe.loc[i,temp.loc[temp['mask'],'obsnme']] = temp.loc[temp['mask'],'vals'].values
  
    #now for ss obs
    ss_groups = ["sshds"]
    ssobs = obs.loc[(obs.weight>0)&(obs.obgnme.str.contains("ss")),:]
    ssobs["standard_deviation"] = ssobs["standard_deviation"].astype(float)
    assert len(ssobs) > 0
    for group in ss_groups:
        sobs = ssobs.loc[ssobs.obgnme==group,:]
        print(group,sobs.shape)
        assert sobs.shape[0] > 0
        ovals = sobs.obsval
        std = sobs.standard_deviation.max()
        reals = np.array([ovals+(d*std) for d in draws])
        print(reals.shape)
        oe.loc[:,sobs.obsnme] = reals

    oe.to_binary(os.path.join(t_d,"noise.jcb"))
    pst.pestpp_options["ies_obs_en"] = "noise.jcb"
    pst.control_data.noptmax = -2
    pst.write(os.path.join(t_d,f"{modnm}.pst"),version=2)
    #pyemu.os_utils.run(f"pestpp-ies {modnm}.pst",cwd=t_d)

    with PdfPages(os.path.join(t_d,"noise_draws.pdf")) as pdf:

        for group in ss_groups:
            sobs = ssobs.loc[ssobs.obgnme==group,:]
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            ovals = sobs.obsval
            vals = oe.loc[:,sobs.obsnme].values
            [ax.plot(ovals,vals[i,:],'r-') for i in range(vals.shape[0])]
            mn = min(ax.get_xlim()[0],ax.get_ylim()[0])
            mx = max(ax.get_xlim()[1],ax.get_ylim()[1])
            ax.set_xlim(mn,mx)
            ax.set_ylim(mn,mx)
            ax.grid()
            ax.set_title(group,loc="left")
            pdf.savefig()
            plt.close(fig)
            

        for uid in uids:
            uobs = trobs.loc[trobs["iid"]==uid,:].copy()
            uobs.sort_values(by="datetime", inplace=True)
            dts = uobs.datetime.values

            # per-obs top_elev in the same order as uobs.obsnme
            tp_series = trobs_top.loc[uobs.obsnme].to_numpy()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.scatter(dts, uobs.obsval, marker="^", c="k", s=50, zorder=11)

            vals = oe.loc[:, uobs.obsnme].values  # (n_real, n_times)
            for r in range(vals.shape[0]):
                ax.plot(dts, vals[r, :], "r-", lw=0.5, alpha=0.5, zorder=10)

            # plot the truncation level actually used (top - 1.0) per time step
            ax.plot(dts, tp_series, "k--", lw=2.0, zorder=5, label="top elevation ft")

            ax.set_title(uobs.obgnme.iloc[0] + " " + uid+ ' top:' + str(tp_series[0]), loc="left")
            ax.grid()
            ax.grid(which="minor", color="k", alpha=0.1, linestyle=":")
            ax.set_xlim(trobs.datetime.min(), trobs.datetime.max())
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

def process_autoreduce(model_ws='.'):
    df = pd.read_csv(os.path.join(model_ws,"auto_flow_reduce.csv"))
    count = [len(df.cellnumber.unique())]
    reqsum = [np.abs(df["rate-requested"]).sum()]
    actsum = [np.abs(df["rate-actual"]).sum()]
    redsum = [np.abs(df["wel-reduction"]).sum()]
    defwel_fname = os.path.join(model_ws,"defwel_auto_flow_reduce.csv")
    if os.path.exists(defwel_fname):
        df = pd.read_csv(defwel_fname)
        count.append(len(df.cellnumber.unique()))
        reqsum.append(np.abs(df["rate-requested"]).sum())
        actsum.append(np.abs(df["rate-actual"]).sum())
        redsum.append(np.abs(df["wel-reduction"]).sum())
    df = pd.DataFrame(data={"auto-count":count,"request-sum":reqsum,"actual-sum":actsum,"reduction-sum":redsum},index=["existing","deferred"])
    df.index.name = "wel-pak"
    df.loc["total",:] = df.sum()
    df.to_csv(os.path.join(model_ws,"autoreduce_summary.csv"))
    return df


def defwel_respmat_run(t_d,use_ensemble=False,num_reals=None,run=False,
                       num_workers=15,e_m_d=None):
    pst_files = [f for f in os.listdir(t_d) if f.endswith(".pst")]
    assert len(pst_files) == 1
    pst = pyemu.Pst(os.path.join(t_d,pst_files[0]))
    en = None
    if e_m_d is not None:
        mpst_files = [f for f in os.listdir(e_m_d) if f.endswith(".pst")]
        assert len(mpst_files) == 1
        mpst = pyemu.Pst(os.path.join(e_m_d,mpst_files[0]))
        if use_ensemble:
            en = pst.ies.get("paren",ensemble_use_iter)
            if num_reals is not None:
                en = en.iloc[:num_reals,:]
        else:    
            diff = set(mpst.par_names).symmetric_difference(set(pst.par_names))
            print("parameter differences:",diff)
            base_vals = mpst.ies.get("paren",mpst.ies.phiactual.iteration.max()).loc["base",:]
            pst.parameter_data.loc[base_vals.index,"parval1"] = base_vals.values

    par = pst.parameter_data
    dwpar = par.loc[par.pname=="defwel",:]
    assert dwpar.shape[0] > 0
    par.loc[dwpar.parnme,"parval1"] = 0.0
    runs = {}
    runs["base"] = par.parval1.copy()
    if en is None:
        for pname in dwpar.parnme:
            pvals = par.parval1.astype(float).copy()
            pvals.loc[pname] = 0.1
            runs[pname] = pvals
    else:
        for real in en.index:
            for pname in dwpar.parnme:
                pvals = en.loc[real,:].copy()
                pvals.loc[pname] = 0.1
                runs["real:{0}_pname:{1}".format(real,pname)] = pvals
    df = pd.DataFrame(runs).T
    pyemu.Matrix.from_dataframe(df).to_dense(os.path.join(t_d,"sweep_in.bin"))
    pst.pestpp_options["sweep_parameter_file"] = "sweep_in.bin"
    pst.pestpp_options["save_dense"] = True
    pst.write(os.path.join(t_d,"pest.pst"),version=2)
    print("{0} resp matrix runs created".format(df.shape[0]))

    if run:
        m_d = t_d.replace("template","master")
        m_d += "_respmatsweep"
        assert t_d != m_d
        pyemu.os_utils.start_workers(t_d,"pestpp-swp","pest.pst",worker_root=".",master_dir=m_d,
            num_workers=num_workers)

def round_defwel(model_ws='.'):
    import numpy as np
    import os
    import pandas as pd
    mlt_file = os.path.join(model_ws,"mult","defwel_inst0_grid.csv")
    assert os.path.exists(mlt_file)
    df = pd.read_csv(mlt_file,index_col=0)
    df.iloc[:,-1] = np.rint(df.iloc[:,-1].values).astype(float)
    df.to_csv(mlt_file)
    tot = df.iloc[:,-1].sum()
    with open(os.path.join(model_ws,"defwel_total_active.csv"),'w') as f:
        f.write("col,total\n")
        f.write("num-active-defwel,{0}\n".format(tot))




def sum_level_impacts(model_ws="."):
    tol = 1.0
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws,load_only=["dis"])
    gwf = sim.get_model()
    botm = gwf.dis.botm.array
    obs_fname = os.path.join(model_ws,"elk_2lay.trans_head.obs.output.csv")

    assert os.path.exists(obs_fname)
    df = pd.read_csv(obs_fname,index_col=0,parse_dates=True)
    df.columns = df.columns.map(str.lower)
    edf = df.loc[:,df.columns.str.contains("exwel")]
    assert edf.shape[0] > 0
    ddf = df.loc[:,df.columns.str.contains("defwel")]
    assert ddf.shape[0] > 0

    dkijs = [(int(c.split("-")[1][1:]),int(c.split("-")[2][1:]),int(c.split("-")[3][1:])) for c in ddf.columns]
    bvals = np.array([botm[kij] for kij in dkijs])
    for t in edf.index:
        vals = ddf.loc[t,:].values
        #vals[::2] = bvals[::2] - 1
        vals[np.where(vals > bvals+tol)] = 0.0
        vals[vals!=0] = 1.0
        #print(t,np.cumsum(vals),vals.shape)
        ddf.loc[t,:] = vals

    ddftots = np.cumsum(ddf.sum(axis=1))
    #print(ddftots)

    ekijs = [(int(c.split("-")[1][1:]),int(c.split("-")[2][1:]),int(c.split("-")[3][1:])) for c in edf.columns]
    bvals = np.array([botm[kij] for kij in ekijs])
    for t in edf.index:
        vals = edf.loc[t,:].values
        #vals[::2] = bvals[::2] - 1
        vals[np.where(vals > bvals+tol)] = 0.0
        vals[vals!=0] = 1.0
        #print(t,np.cumsum(vals),vals.shape)
        edf.loc[t,:] = vals

    edftots = np.cumsum(edf.sum(axis=1))
    #print(edftots)

    df = pd.DataFrame({"defwel":ddftots,"exwel":edftots})
    df["total"] = df.sum(axis=1)
    df.to_csv(os.path.join(model_ws,"sum_level_impacts.csv"))
    return df
    

def extract_and_run_base_real(new_d,ies_master_d,t_d=None):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    pst_files = [f for f in os.listdir(ies_master_d) if f.endswith(".pst") and "elk" in f]
    print(pst_files)
    print(len(pst_files))
    assert len(pst_files) == 1
    mpst = pyemu.Pst(os.path.join(ies_master_d,pst_files[0]))
    pe = mpst.ies.get("paren",mpst.ies.phiactual.iteration.max())

    mpst.parameter_data.loc[pe.columns,"parval1"] = pe.loc["base",:].values
    mpst.control_data.noptmax = 0
    if t_d is not None:
        ies_master_d = t_d
    mpst.write(os.path.join(ies_master_d,"base.pst"))
    prep_deps(ies_master_d)
    
    pyemu.os_utils.run("pestpp-ies base.pst",cwd=ies_master_d)
    sim = flopy.mf6.MFSimulation.load(sim_ws=ies_master_d)
    gwf = sim.get_model()
    
    sim.set_sim_path(new_d)
    gwf.set_all_data_external(external_data_folder=".")
    sim.write_simulation()
    import process_deferred_permits
    process_deferred_permits.main(new_d)
    prep_deps(new_d)
    pyemu.os_utils.run("mf6",cwd=new_d)



def prep_defwel_mou(org_d,new_d,):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    list_files = [f for f in os.listdir(org_d) if f.endswith('.txt') and "def_wel.wel_stress_period_data_" in f]
    print(list_files)
    if len(list_files) == 0:
        print("adding deferred wells to model")
        t_d = org_d + "_temp"
        if os.path.exists(t_d):
            shutil.rmtree(t_d)
        shutil.copytree(org_d,t_d)
        import process_deferred_permits
        process_deferred_permits.main(model_ws=t_d)
        org_d = t_d

    sim = flopy.mf6.MFSimulation.load(sim_ws=org_d,load_only=["tdis"])
    gwf = sim.get_model()
    start_datetime = gwf.start_datetime
    pf = pyemu.utils.PstFrom(original_d=org_d, new_d=new_d,
                         remove_existing=True,
                         longnames=True,
                         zero_based=False, start_datetime=start_datetime,
                         chunk_len=1000000)
            
       
    list_files = [f for f in os.listdir(pf.new_d) if f.endswith('.txt') and "def_wel.wel_stress_period_data_" in f]
    print(list_files)
    assert len(list_files) > 0
    list_files.sort()
    tag = list_files[0].split("_")[0].replace("_","-")    
    # grid scale mults correlated in space, constant in time
    pf.add_parameters(filenames=list_files, par_type='grid',
            par_name_base='defwel',
            pargp='defwel', index_cols=[0, 1, 2], use_cols=[3],
           upper_bound=1.0, lower_bound=0.0, initial_value=0.0,transform="none")
       
 
    pf.mod_sys_cmds.append('mf6')
        
    df = process_autoreduce(model_ws=pf.new_d)
    assert "deferred" in df.index.to_list()
    pf.add_py_function("elk03_setup_pst.py","round_defwel(model_ws='.')",is_pre_cmd=True)
    round_defwel(model_ws=pf.new_d)
    pf.add_observations("defwel_total_active.csv",index_cols=["col"],prefix="activedefwel",obsgp="activedefwel")
                
    pf.add_py_function("elk03_setup_pst.py","process_autoreduce(model_ws='.')",is_pre_cmd=False)
    pf.add_observations("autoreduce_summary.csv",index_cols="wel-pak",use_cols=df.columns.tolist(),prefix="autoreduce",obsgp="autoreduce",ofile_sep=',')




    # import flopy as part of the forward run process
    pf.extra_py_imports.append('flopy')    
    pf.extra_py_imports.append("shutil")  
    pf.extra_py_imports.append("pathlib")  
         
    pst = pf.build_pst("pest.pst")

    #pst = pyemu.Pst(os.path.join(new_d,pst_files[0]))
    par = pst.parameter_data
    dpar = par.loc[par.pname=="defwel",:].copy()
    assert len(dpar) > 0

    par["partrans"] = "fixed"
    par.loc[dpar.parnme,"partrans"] = "none"
    par.loc[dpar.parnme,"parlnd"] = 0.0
    par.loc[dpar.parnme,"parund"] = 1.0
    par.loc[dpar.parnme,"pargp"] = "decvar"

    obs = pst.observation_data
    obs["weight"] = 0.0

    #obj2 = obs.loc[(obs.obsnme.str.contains("wel-pak:deferred")) & (obs.obsnme.str.contains("actual-sum")),"obsnme"]
    obj2 = obs.loc[obs.obsnme.str.contains("activedefwel"),"obsnme"]
    
    #obj2["datetime"] = pd.to_datetime(obj2.datetime)
    #obj2 = obj2.loc[obj2.datetime==obj2.datetime.max(),"obsnme"]
    print(obj2)
    print(len(obj2))
    
    assert len(obj2) == 1
    obj2 = obj2.values[0]
    obs.loc[obj2,"weight"] = 1.0
    obs.loc[obj2,"obgnme"] = "greater_than"
    

    #obj1 = obs.loc[(obs.obsnme.str.contains("wel-pak:total")) & (obs.obsnme.str.contains("auto-count")),"obsnme"]
    obj1 = obs.loc[(obs.obsnme.str.contains("wel-pak:total")) & (obs.obsnme.str.contains("reduction-sum")),"obsnme"]
    print(obj1)
    assert len(obj1) == 1
    obj1 = obj1.values[0]
    obs.loc[obj1,"weight"] = 1.0
    obs.loc[obj1,"obgnme"] = "less_than"

    dvpop = pyemu.ParameterEnsemble.from_uniform_draw(pst,num_reals=dpar.shape[0]*2,fill=False)
    print(dvpop.shape)
    for i,val in enumerate(np.arange(0,1.1,.1)):
        dvpop.iloc[i,:] = val
    dvpop.to_dense(os.path.join(new_d,"initial_dvpop.bin"))
    pst.pestpp_options["mou_save_population_every"] = 1
    pst.pestpp_options["save_dense"] = True
    pst.pestpp_options["mou_dv_population_file"] = "initial_dvpop.bin"
    pst.pestpp_options["mou_objectives"] = [obj1,obj2]
    pst.pestpp_options["mou_population_size"] = 1
    pst.control_data.noptmax = -1
    pst.write(os.path.join(new_d,"pest.pst"),version=2)
    prep_deps(new_d)
    pyemu.os_utils.run("pestpp-mou pest.pst",cwd=new_d)
    pst.pestpp_options["mou_population_size"] = dvpop.shape[0]
    pst.control_data.noptmax = 100
    pst.write(os.path.join(new_d,"pest.pst"),version=2)


    
def plot_mou_results(m_d="master_mou"):
    fig_d= os.path.join(m_d,"pareto_figs")
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    dv = pst.mou.dvpop
    arc = pst.mou.paretosum_archive
    arc = arc.loc[arc.nsga2_front==1,:]
    obs = pst.observation_data
    oobs = obs.loc[obs.weight>0,:]
    for oname,usecol in zip(oobs.obsnme,oobs.usecol):
        arc[usecol] = arc.pop(oname)
    usecols = oobs.usecol.unique()
    usecols.sort()
    #labels = ["cumulative flux rate for deferred permits","number of wells impacted (ie autoreduce activated)"]
    labels = ["autoreduced volume","number deferred active"]
    gens = arc.generation.unique()
    gens.sort()
    
    final = arc.loc[arc.generation==max(gens),:].copy()

    
    xlim = (arc[usecols[0]].min(),arc[usecols[0]].max())
    xlim = [-100,xlim[1]]
    ylim = (arc[usecols[1]].min(),arc[usecols[1]].max())
    if os.path.exists(fig_d):
        shutil.rmtree(fig_d)
    os.makedirs(fig_d)
    ifig = 1
    for gen in gens:
        garc = arc.loc[arc.generation==gen,:].copy()
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        vals1 = garc.loc[:,usecols[0]].values
        vals2 = garc.loc[:,usecols[1]].values
        ax.scatter(vals1,vals2,marker=".",s=10,c="b")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid()
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title("generation {0}".format(gen),loc="left")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_d,"pareto{0:04d}.png".format(ifig)),dpi=500)
        print(gen)
        ifig += 1
        
    fps = 15
    pyemu.os_utils.run("ffmpeg -i pareto{0:04d}.png -vf palettegen=256 palette.png".format(gen),cwd=fig_d)
    pyemu.os_utils.run("ffmpeg -r {0} -y -s 1920X1080 -i pareto%04d.png -i palette.png -filter_complex \"scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse\" temp.gif".format(fps),
            cwd=fig_d)
    pyemu.os_utils.run("ffmpeg -i temp.gif -c copy -final_delay 300 -loop 0 pareto.gif",cwd=fig_d)
    


   
def process_mou_well_counts(m_d):

    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    dv = pst.mou.dvpop
    arc = pst.mou.paretosum_archive
    arc = arc.loc[arc.nsga2_front==1,:]
    obs = pst.observation_data
    oobs = obs.loc[obs.weight>0,:]
    for oname,usecol in zip(oobs.obsnme,oobs.usecol):
        arc[usecol] = arc.pop(oname)
    usecols = oobs.usecol.unique()
    usecols.sort()
    gens = arc.generation.unique()
    gens.sort() 
    farc = arc.loc[arc.generation==max(gens),:].copy()
    print(farc.loc[:,usecols[0]].describe())
    #fif = farc.loc[:,usecols[0]].quantile(0.2)
    fif = 1e6
    print(fif)
    print(farc.shape)
    farc = farc.loc[farc[usecols[0]]<=fif,:]
    print(farc.shape)
    smem = set(farc.member.to_list())
    dv = pst.mou.dvpop
    fdv = dv.loc[dv.index.get_level_values(1).isin(smem),:]
    print(fdv.shape)
    #final = final.loc[]
    #print(usecols)
    fdv = np.rint(fdv)
    print(fdv.sum())
    fdv.to_csv(os.path.join(m_d,"min_impact.dvpop.csv"))
    fdv.sum().to_csv(os.path.join(m_d,"min_impact.count.csv"))     




def plot_mou_sidebyside():
    fig_d= os.path.join("pareto_figs")
    pst1 = pyemu.Pst(os.path.join("master_mou2","pest.pst"))
    pst2 = pyemu.Pst(os.path.join("master_mou2_mar","pest.pst"))
    
    arc1 = pst1.mou.paretosum_archive
    arc1 = arc1.loc[arc1.nsga2_front==1,:]
    arc2 = pst2.mou.paretosum_archive
    arc2 = arc2.loc[arc2.nsga2_front==1,:]
    obs = pst1.observation_data
    oobs = obs.loc[obs.weight>0,:]
    for oname,usecol in zip(oobs.obsnme,oobs.usecol):
        arc1[usecol] = arc1.pop(oname)
        arc2[usecol] = arc2.pop(oname)
    usecols = oobs.usecol.unique()
    usecols.sort()
    #labels = ["cumulative flux rate for deferred permits","number of wells impacted (ie autoreduce activated)"]
    labels = ["autoreduced volume","number of deferred permits active"]
    gens = arc1.generation.unique()
    gens.sort()
    
    farc1 = arc1.loc[arc1.generation==max(gens),:].copy()
    farc2 = arc2.loc[arc2.generation==max(gens),:].copy()
    smem1 = set(farc1.member.unique().tolist())
    smem2 = set(farc2.member.unique().tolist())

    dv = pst1.mou.dvpop
    fdv1 = dv.loc[dv.index.get_level_values(1).isin(smem1),:]
    vals = fdv1.values
    vals[vals<0.5] = 0
    vals[vals>=0.5] = 1
    fdv1.loc[:,:] = vals
    #fdv1 = pd.DataFrame(vals,index=fdv1.index.get_level_values(1),columns=fdv1.columns)


    dv = pst2.mou.dvpop
    fdv2 = dv.loc[dv.index.get_level_values(1).isin(smem2),:]
    vals = fdv2.values
    vals[vals<0.5] = 0
    vals[vals>=0.5] = 1
    fdv2.loc[:,:] = vals

    par = pst1.parameter_data
    fdv2.columns = fdv2.columns.map(lambda x: "k{0}-i{1}-j{2}".format(par.loc[x,"idx0"],par.loc[x,"idx1"],par.loc[x,"idx1"]))
    fdv1.columns = fdv1.columns.map(lambda x: "k{0}-i{1}-j{2}".format(par.loc[x,"idx0"],par.loc[x,"idx1"],par.loc[x,"idx1"]))

    sum1 = fdv1.sum().sort_index()
    sum1 /= sum1.max()
    sum1 *= 100
    sum2 = fdv2.sum().sort_index()
    sum2 /= sum2.max()
    sum2 *= 100
    print(sum1,sum2)

    xlim = (farc1[usecols[0]].min(),farc1[usecols[0]].max())
    xlim = [-100,xlim[1]]
    ylim = (farc1[usecols[1]].min(),farc1[usecols[1]].max())

    fig,axes = plt.subplots(2,1,figsize=(8.5,8))
    # vals1 = farc1.loc[:,usecols[0]].values
    # vals2 = farc1.loc[:,usecols[1]].values
    # ax = axes[0,0]
    # ax.scatter(vals1,vals2,marker=".",s=10,c="b")
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.grid()
    # ax.set_xlabel(labels[0])
    # ax.set_ylabel(labels[1])
    # ax.set_title("no MAR pareto front",loc="left")

    # vals1 = farc2.loc[:,usecols[0]].values
    # vals2 = farc2.loc[:,usecols[1]].values
    # ax = axes[0,1]
    # ax.scatter(vals1,vals2,marker=".",s=10,c="b")
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.grid()
    # ax.set_xlabel(labels[0])
    # ax.set_ylabel(labels[1])
    # ax.set_title("w/ MAR pareto front",loc="left")

    
    sum1.plot(kind="bar",ax=axes[0])
    sum2.plot(kind="bar",ax=axes[1])
    #(sum1 - sum2).plot(kind="bar",ax=axes[2])
    axes[0].set_xticklabels(sum1.index,rotation=90,fontsize=6)
    axes[0].set_ylabel("percent of solution present")
    axes[1].set_ylabel("percent of solution present")
    axes[0].set_title("Without MAR",loc="left")
    axes[1].set_title("With MAR",loc="left")
    
    plt.tight_layout()
    plt.savefig("parete_compare.pdf")
    plt.close(fig)
    


    
    xlim = (farc1[usecols[0]].min(),farc1[usecols[0]].max())
    xlim = [-10000,xlim[1]]
    ylim = (farc1[usecols[1]].min(),farc1[usecols[1]].max())
    if os.path.exists(fig_d):
        shutil.rmtree(fig_d)
    os.makedirs(fig_d)
    ifig = 1
    for gen in gens:
        garc1 = arc1.loc[arc1.generation==gen,:].copy()
        garc2 = arc2.loc[arc2.generation==gen,:].copy()
     
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        vals1 = garc1.loc[:,usecols[0]].values
        vals2 = garc1.loc[:,usecols[1]].values
        ax.scatter(vals1,vals2,marker=".",s=20,c="b",label="w/o MAR")
        
        vals1 = garc2.loc[:,usecols[0]].values
        vals2 = garc2.loc[:,usecols[1]].values
        ax.scatter(vals1,vals2,marker="^",s=30,c="m",label="w/ MAR")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid()
        ax.legend(loc="lower right")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title("generation {0}".format(gen),loc="left")
        

        plt.tight_layout()
        plt.savefig(os.path.join(fig_d,"pareto{0:04d}.png".format(ifig)),dpi=500)
        print(gen)
        ifig += 1
        
    fps = 15
    pyemu.os_utils.run("ffmpeg -i pareto{0:04d}.png -vf palettegen=256 palette.png".format(gen),cwd=fig_d)
    pyemu.os_utils.run("ffmpeg -r {0} -y -s 1920X1080 -i pareto%04d.png -i palette.png -filter_complex \"scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse\" temp.gif".format(fps),
            cwd=fig_d)
    pyemu.os_utils.run("ffmpeg -i temp.gif -c copy -final_delay 300 -loop 0 pareto.gif",cwd=fig_d)


    
def export_mou_member(m_d,mem_name,org_t_d="template_mou_mar",new_t_d="mou_member_run"):
    if os.path.exists(new_t_d):
        shutil.rmtree(new_t_d)
    shutil.copytree(org_t_d,new_t_d)

    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    dvpop = pst.mou.dvpop
    mem = dvpop.loc[dvpop.index.get_level_values(1)==mem_name,:]
    #print(mem.columns)
    #exit()
    print(mem.shape)
    assert mem.shape[0] == 1
    par = pst.parameter_data
    par.loc[mem.columns,"parval1"] = mem.values.flatten().copy()
    pst.control_data.noptmax = 0
    pst.write(os.path.join(new_t_d,"pest.pst"),version=2)
    pyemu.os_utils.run_sp("pestpp-ies pest.pst",cwd=new_t_d)
  
    



if __name__ == "__main__":

    #extract_and_run_base_real("base_temp","master_flow_08_highdim_restrict_bcs_flood_full_final_rch")
    #prep_defwel_mou("base_model_MAR_testing_extfix","template_mou_mar")
    
    # m_d = "master_mou2_mar"
    # num_workers = 12
    # pyemu.os_utils.start_workers("template_mou_mar", "pestpp-mou", "pest.pst",
    #                              num_workers=num_workers, worker_root='.',
    #                              master_dir=m_d, local=True,port=4269)
    # plot_mou_results(m_d)
    # process_mou_well_counts(m_d)
    #plot_mou_sidebyside()
    
    # #process_respmat("master_flow_08_highdim_restrict_bcs_flood_full_final_rch_respmatsweep")
    #exit()
    #sum_level_impacts("template_flow_08_highdim_restrict_bcs_flood_full_final_rch")
    #process_autoreduce("test")
    #round_defwel("template_mou")
    #export_mou_member("master_mou2_mar","gen=98_member=23253_pso")
    #exit()

    print("starting elk03_setup_pst.py")

    modnm = 'elk_2lay'
        
    run_tag ='_08'
    org_d = os.path.join('model_ws', modnm+'_clean')
        
    
    #dir locations
    m_d_flow = "master_flow"+run_tag
    t_d_flow = "template_flow"+run_tag
    
    template = t_d_flow

    # prep the underlying model 
    prep_model = False
    # include the deferred permit wel pak
    include_def = True
    # prep the flow template
    prep_flow = True

    prep_pmp_noise = False
    # run ies for the flow template
    run_flow = False
    # plot ies results
    plot_flow = False
    # run sensitivity
    run_sensitivity = False
    
    run_respmat = False
    # use condor
    use_condor = True
    print(f' use condor: {use_condor}')
    # use lots of par or not
    high_dimensional = True
    print(f' high dimensional: {high_dimensional}')
    # tie window parameters
    tie_window_pars = True

    phi_factor_dict = {
        'sshds'        : 0.05,
        'transhds'     : 0.35,
        'transearlyhds': 0.30,
        'ddtrgs'       : 0.25,  # NEW
        'hd_lt_top'    : 0.05,
    }


    def renorm_factors(d, tiny=1e-18):
        s = sum(v for v in d.values() if v > tiny)
        return {k: (v/s if v > tiny else v) for k, v in d.items()}
    phi_factor_dict = renorm_factors(phi_factor_dict)
    
    print('phi factors:')   
    for k,v in phi_factor_dict.items():
        print(f'   {k}: {v:.4f}')

    if high_dimensional:
        m_d_flow += '_highdim_restrict_bcs_flood_full_final_rch'
        t_d_flow += '_highdim_restrict_bcs_flood_full_final_rch'
    else:
        m_d_flow += '_lowdim'
        t_d_flow += '_lowdim'
    
    if use_condor:
        num_reals_flow = 480
        num_workers_flow = 192
        hostname = '10.99.10.30'
        pid = os.getpid()
        current_time = int(time.time())
        seed_value = (pid + current_time) % 65536
        np.random.seed(seed_value)
        port = random.randint(4001, 4999)
        print(f'port #: {port}')
    else:
        num_reals_flow = 120
        num_workers_flow = 30
        hostname = None
        port = None
    
  
    # how many iters to use
    noptmax_flow = 4

    local = True

    if prep_model:
        import elk02_model_build
        sim_ws = elk02_model_build.main()

        print('Running setup_pst.py')
        print('Env path order:')
        for path in sys.path:
            print(path)

        org_d = os.path.join('model_ws', modnm+'_monthly_clean')
        set_initial_array_vals(org_d, modnm=modnm, run_tag=run_tag)


    # assure run inputs exisit:
    if not os.path.exists(os.path.join('run_inputs',f'{modnm}')):
        raise FileNotFoundError(f'Run inputs for {modnm} do not exist, please create them')

    org_d = os.path.join('model_ws', modnm+'_monthly_clean'+run_tag)
    
    if include_def:
        import process_deferred_permits
        process_deferred_permits.main(org_d)
        import elk02_model_build
        elk02_model_build.add_wel_head_obs(org_d)
        
    if prep_flow:
        print('{0}\n\npreparing flow-IES\n\n{1}'.format('*'*17,'*'*17))
        
         
        temp_dir = setup_pstpp(org_d,modnm,run_tag,t_d_flow,flex_con=False,num_reals=num_reals_flow,
                               high_dimensional=high_dimensional)

        print(f'------- flow-ies has been setup in {t_d_flow} ----------')

        set_obsvals_and_weights(t_d_flow,modnm=modnm,flow_weight_scheme= 'all_wl_meas',
                                phi_factor_dict=phi_factor_dict) 
        #print(abab)
        draw_noise_reals(t_d_flow,modnm)

        if prep_pmp_noise:
            print('draw pmp noise reals')
            draw_pmp_noise_reals(t_d_flow,modnm)

    
        

    if run_flow:
       print('*** running flow-ies to get posterior ***')
       run_ies(t_d_flow, modnm=modnm, m_d=m_d_flow,num_workers=num_workers_flow,niceness=False,noptmax=noptmax_flow,
               init_lam=None, local=local,
               use_condor=use_condor, hostname=hostname,port=port)#,ies_n_iter_reinflate=[-7,999])
    if plot_flow:
       import elk04_process_plot_results
       obsdict = elk04_process_plot_results.get_ies_obs_dict(m_d=m_d_flow, pst=None, modnm=modnm)
       #elk04_process_plot_results.plot_simple_timeseries(m_d_flow,modnm=modnm,noptmax=noptmax_flow)
       #elk04_process_plot_results.plot_simple_1to1(m_d_flow, modnm=modnm)
       elk04_process_plot_results.plot_simple_par_histo(m_d_flow, modnm=modnm)
       elk04_process_plot_results.plot_phi_sequence(m_d_flow, modnm=modnm)
       elk04_process_plot_results.plot_parm_violins(m_d_flow, f'{modnm}.pst', noptmax_flow)
       elk04_process_plot_results.plot_fancy_obs_v_sim(m_d_flow,obsdict)
       elk04_process_plot_results.plot_fancy_obs_v_sim_base(m_d_flow,obsdict)
       elk04_process_plot_results.plot_layer_one2one_wdepth(m_d_flow,obsdict, modnm=modnm)
       #elk04_process_plot_results.plot_water_budget(m_d_flow, obsdict,pie_year=2022)
       #elk04_process_plot_results.plot_simple_par_maps(m_d_flow,modnm)
       elk04_process_plot_results.base_posterior_param_forward_run(m_d0=m_d_flow, noptmax=noptmax_flow)
       m_d_base = m_d_flow + '_forward_run_base'
       init_ws = os.path.join("model_ws", "elk_2lay_monthly")
       elk04_process_plot_results.model_packages_to_shp_joined(m_d_base) 
       elk04_process_plot_results.plot_hds(m_d_base,sim_name=modnm,kstpkper=(0, 317))
       elk04_process_plot_results.plot_rech_initial_vs_base_mel(
            model_ws_initial=init_ws,
            model_ws_base=m_d_base,
            sim_name=modnm,
            out_dir="fig_rech_compare",
            max_kper=None,   # or 323 if you only want historical
        )
       
       elk04_process_plot_results.plot_well_pumping_comparison_mel(
            init_ws=init_ws,
            post_ws=m_d_base,
            sim_name="elk_2lay",
            wel_pkg_name="wel",
            out_dir="fig_wel_pumping",
        )

       elk04_process_plot_results.plot_total_well_pumping_comparison_mel(
            init_ws=init_ws,
            post_ws=m_d_base,
            sim_name= "elk_2lay",
            wel_pkg_name= "wel",
            out_dir= "fig_wel_pumping"
        )

       elk04_process_plot_results.plot_rech_timeseries_initial_vs_base_mel(
            model_ws_initial=init_ws,
            model_ws_base=m_d_base,
            sim_name="elk_2lay",
            out_dir="fig_rech_compare_ts",
            dpi=300,
        )
       elk04_process_plot_results.analyze_drn_riv_fluxes_timeseries(
            sim_ws=m_d_base,
            model_name="elk_2lay",
            out_dir=os.path.join(m_d_base, "cbb_plots")
    )
         

    if run_respmat:

        defwel_respmat_run(t_d_flow,
        use_ensemble=False,num_reals=None,run=False,num_workers=5,
        e_m_d=m_d_flow)



    #if run_sensitivity:
        # update pst control file to tie parameters
        # prepare_sen(m_d=m_d_flow, pst_name='elk.pst', )

        # run_sen(m_d_flow, m_d=f'{m_d_flow}_sen', pst_name='elk_sen.pst',
        #         num_workers=num_workers_flow,local=local,
        #         use_condor=use_condor, hostname=hostname, port=port)

    print('All Done!, congrats we did a thing')  

