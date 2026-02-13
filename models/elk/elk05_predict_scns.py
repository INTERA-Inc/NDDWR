import os
import sys

from scipy.fft import dst
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
from master_flow_08_highdim_restrict_bcs_flood_full_final_forward_run_base.pyemu import pst
import pyemu
import flopy
import pypestutils
from pypestutils.pestutilslib import PestUtilsLib
from pypestutils import helpers as ppu
import platform
import pandas as pd
import geopandas as gpd
from shapely import Point
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import time 
import pathlib
import re
import random
from scipy import signal

import warnings            
warnings.filterwarnings('ignore')

# set fig formats:
import elk04_process_plot_results as wpp
wpp.set_graph_specifications()
wpp.set_map_specifications()


# ------------------------------------------------------- #
# Initialization Functions
# ------------------------------------------------------- #
def init_predict_ws(modnm, noptmax_flow, m_d_flow, pred_ws):

    # copy m_d_flow to a new dir
    keep_obs_jcb = f'{modnm}.{noptmax_flow}.obs.jcb'  # the one .jcb file to copy
    keep_par_jcb = f'{modnm}.{noptmax_flow}.par.jcb'  # the one .jcb file to copy
    
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
            # elif os.path.isfile(full_path) and f == 'mult':
            #     ignored.append(f)                # ignore .rec files
            elif f.endswith(".rec"):
                ignored.append(f)
            elif f.endswith(".rei"):
                ignored.append(f)
            # ignore .jcb files except the one we want to keep
            elif f.endswith(".jcb") and f != keep_obs_jcb and f != keep_par_jcb:
                ignored.append(f)
        return ignored
 
    # copytree with ignore function
    shutil.copytree(src, dst, ignore=ignore_large_files)

def init_clean_ws(m_d_flow, pred_ws): 
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

# ------------------------------------------------------- #
# universal across scenario modifiers
# ------------------------------------------------------- #
def modify_mult2mod(pred_ws='.',lst_mod_well_files=[]):
    mults = pd.read_csv(os.path.join(pred_ws, 'mult2model_info.csv'))
    print(len(mults))
    mults = mults.loc[mults['model_file'].isin(lst_mod_well_files) == False]
    print(len(mults))
    mults.to_csv(os.path.join(pred_ws, 'mult2model_info.csv'), index=False)
    print("modified mult2model_info.csv to remove modified well files.")


def _read_monthly_sp_info(sp_info_csv=os.path.join("tables", "monthly_stress_period_info.csv")):
    spd = pd.read_csv(sp_info_csv)
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])
    spd["stress_period"]  = spd["stress_period"].astype(int)
    if "year" not in spd.columns:
        spd["year"] = spd["end_datetime"].dt.year
    spd["month"] = spd["end_datetime"].dt.month
    return spd

def _replace_period_blocks_with_openclose(pkg_path, sp_to_fname):
    """
    Replace each BEGIN period <sp> ... END period <sp> block body with OPEN/CLOSE '<fname>'
    """
    with open(pkg_path, "r", newline="") as f:
        text = f.read()

    for sp, fname in sp_to_fname.items():
        pat = re.compile(
            rf"(?im)^(BEGIN\s+period\s+{sp}\s*\r?\n)(.*?)(^END\s+period\s+{sp}\s*$)",
            re.DOTALL,
        )

        def repl(m):
            return m.group(1) + f"    OPEN/CLOSE  '{fname}'\n" + m.group(3)

        text, _ = pat.subn(repl, text, count=1)

    with open(pkg_path, "w", newline="") as f:
        f.write(text)

def _rch_fname(sp): return f"rch_recharge_{sp}.txt"
def _wel_fname(sp): return f"wel_stress_period_data_{sp}.txt"

# ------------------------------------------------------- #
# Baseline Scenario Functions
# ------------------------------------------------------- #
def write_baseline_well_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]
    ep_end = spd.loc[spd.index[-1],'stress_period']
    
    lst_mod_well_files = []
    well_pkg_prefixes = ["cob","car","malt","minn","cow"]
    for prefix in well_pkg_prefixes:
        list_files = [f for f in os.listdir(pred_ws) if prefix in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
        print(list_files)
        well_sorted = sorted(list_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

        # drop list to well files 2020-2025:
        sp_rng = range(sp2025-5, sp2025)
        well_sorted = [f for f in well_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

        if len(well_sorted) == 0:
            continue
        
        # get median pumping rates from last 5 years of histo matching period:
        pmp = pd.DataFrame()
        for f in well_sorted:
            df = pd.read_csv(os.path.join(pred_ws, f), delim_whitespace=True, header=None)
            df.columns = ['layer', 'row', 'column', 'flux_cfd']
            df['sp'] = int(re.search(r'_(\d+)\.txt$', f).group(1))
            pmp = pd.concat([pmp, df], axis=0, ignore_index=True)   
        pmp = pmp.groupby(['layer','row','column']).agg({'flux_cfd':'median'}).reset_index()
        if prefix == 'cow':
            # add 561.5 acre-feet/year to COW pumping:
            add_pmp_acftyr = 561.5
            add_pmp_cfd = add_pmp_acftyr * -43560/365.25  # convert to cfd and negative for pumping
            # allocate 66% of 561.5 to well in layer, row, column (5,115, 47) and then the remaing 33% divide into remaining well sin layer 5:
            pmp['add_flux_cfd'] = 0.0
            pmp.loc[(pmp['layer'] == 5) & (pmp['row'] == 115) & (pmp['column'] == 47), 'add_flux_cfd'] = add_pmp_cfd * 0.66
            n_otherwbv = len(pmp.loc[(pmp['layer'] == 5) & ~((pmp['row'] == 115) & (pmp['column'] == 47)), 'add_flux_cfd'])
            pmp.loc[(pmp['layer'] == 5) & ~((pmp['row'] == 115) & (pmp['column'] == 47)), 'add_flux_cfd'] = (add_pmp_cfd*.33)/ n_otherwbv
            pmp['flux_cfd'] = pmp['flux_cfd'] + pmp['add_flux_cfd']
            pmp = pmp.drop(columns=['add_flux_cfd'])
            
        # write well files for prediction years:
        temp_well_files = []
        for sp in range(sp2025, ep_end+1):
            print(f"  writing {prefix} well file for stress period {sp}..")
            new_well_file = f'{prefix}_stress_period_data_{sp}.txt'
            pmp.to_csv(os.path.join(pred_ws, new_well_file), sep='\t', header=False, index=False)
            pmp.to_csv(os.path.join(pred_ws,'org', new_well_file), sep='\t', header=False, index=False)
            temp_well_files.append(new_well_file)

        lst_mod_well_files.extend(temp_well_files)

        # 1) map stress period -> filename from your list
        sp_to_file = {}
        for fn in temp_well_files:
            m = re.search(r'_(\d+)\.txt$', fn)
            if m:
                sp_to_file[int(m.group(1))] = fn
        # 2) read the .well file
        well_path = os.path.join(pred_ws, f"elk_2lay.{prefix}")
        with open(well_path, "r", newline="") as f:
            text = f.read()
        # 3) for each SP, replace its block body with exactly one OPEN/CLOSE line
        for sp, fname in sp_to_file.items():
            pat = re.compile(
                rf"(?im)^(BEGIN\s+period\s+{sp}\s*\r?\n)(.*?)(^END\s+period\s+{sp}\s*$)",
                re.DOTALL
            )

            def repl(m):
                begin = m.group(1)
                end   = m.group(3)
                body = f"    OPEN/CLOSE  '{fname}'\n"
                return begin + body + end

            text, nsub = pat.subn(repl, text, count=1)
        # 4) write back
        with open(well_path, "w", newline="") as f:
            f.write(text)

    return lst_mod_well_files

def nopt0chk_baseline(org_ws,pred_ws, modnm, noptmax_flow):
    pst = pyemu.Pst(os.path.join(pred_ws,'elk_2lay.pst'))
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_par_en'] = f'{modnm}.{noptmax_flow}.par.jcb'
    pst.pestpp_options["ies_obs_en"] = f'{modnm}.{noptmax_flow}.obs.jcb'
    pst.write(os.path.join(pred_ws,'elk_2lay.pst'),version=2)

    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst', cwd=pred_ws)

    well_pkg_prefixes = ["cob","car","malt","minn","cow"]

    fig, axs = plt.subplots(figsize=(10,8), nrows=len(well_pkg_prefixes)//2+1, ncols=2)
    axs[-1,-1].axis('off')
    for prefix in well_pkg_prefixes:
        df = pd.read_csv(os.path.join(pred_ws, f'{modnm}.base.obs.csv')) 
        old_df = pd.read_csv(os.path.join(org_ws, f'{modnm}.base.obs.csv'))

        well_cols = [col for col in df.columns if prefix in col]
        df_well = df[well_cols]
        bud_df = df_well.T.reset_index()
        bud_df.columns = ['obsnme', 'bud_out']
        bud_df = bud_df.loc[bud_df['obsnme'].str.contains('zn') == False]
        bud_df = bud_df.loc[bud_df['obsnme'].str.contains('out') == True]
        bud_df = bud_df.loc[bud_df['obsnme'].str.contains('flx') == False]
        bud_df = bud_df.loc[bud_df['obsnme'].str.contains('uin') == False]
        bud_df['datetime'] = pd.to_datetime(bud_df['obsnme'].str.split(':').str[-1], format='%Y-%m-%d')

        old_df_well = old_df[well_cols]
        bud_df_old = old_df_well.T.reset_index()
        bud_df_old.columns = ['obsnme', 'bud_out']
        bud_df_old = bud_df_old.loc[bud_df_old['obsnme'].str.contains('zn') == False]
        bud_df_old = bud_df_old.loc[bud_df_old['obsnme'].str.contains('out') == True]
        bud_df_old = bud_df_old.loc[bud_df_old['obsnme'].str.contains('flx') == False]
        bud_df_old = bud_df_old.loc[bud_df_old['obsnme'].str.contains('uin') == False]
        bud_df_old['datetime'] = pd.to_datetime(bud_df_old['obsnme'].str.split(':').str[-1], format='%Y-%m-%d')

        # plot pumping comparison:
        row_idx = well_pkg_prefixes.index(prefix)//2
        col_idx = well_pkg_prefixes.index(prefix)%2
        ax = axs[row_idx, col_idx]
        ax.plot(bud_df['datetime'], bud_df['bud_out']/43560*365.25, marker='o', linestyle='-')
        ax.plot(bud_df_old['datetime'], bud_df_old['bud_out']/43560*365.25, marker='x', linestyle='--')
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_ylabel(f'Annual {prefix.upper()} Pumping\n(acre-feet/year)')

    figoutput = os.path.join(pred_ws, 'input_figs','scn01_baseline')
    if not os.path.exists(figoutput):
        os.makedirs(figoutput)
    fig.savefig(os.path.join(figoutput, 'histo_match_vs_scn_pumping_baseline.png'), dpi=300)
    plt.close(fig)

def elk_write_rch_baseline_monthly_mean(pred_ws=".", pred_start="2025-01-01", lookback_years=5):
    """
    Baseline RCH:
      - Determine predictive SPs as those with start_datetime >= pred_start
      - Determine history window as [pred_start - lookback_years, pred_start)
      - Compute month-of-year mean recharge arrays (cell-by-cell) over history
      - Write to all predictive SP rch_recharge_{sp}.txt based on month-of-year
      - Also writes to pred_ws/org/...
    Returns: list of modified filenames
    """
    spd = _read_monthly_sp_info()
    pred_start = pd.Timestamp(pred_start)

    pred = spd.loc[spd["start_datetime"] >= pred_start].copy()
    pred_sps = pred["stress_period"].astype(int).tolist()
    if not pred_sps:
        raise ValueError(f"No predictive stress periods found starting at/after {pred_start.date()}")

    hist_start = pred_start - pd.DateOffset(years=lookback_years)
    hist = spd.loc[(spd["start_datetime"] >= hist_start) & (spd["start_datetime"] < pred_start)].copy()
    if hist.empty:
        raise ValueError(f"No historical SPs found in [{hist_start.date()}, {pred_start.date()})")

    # month -> mean array
    month_mean = {}
    for m in range(1, 13):
        sps_m = hist.loc[hist["month"] == m, "stress_period"].astype(int).tolist()
        if not sps_m:
            continue
        arrs = []
        for sp in sps_m:
            fp = os.path.join(pred_ws, _rch_fname(sp))
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing historical recharge file needed for baseline: {fp}")
            arrs.append(np.loadtxt(fp))
        month_mean[m] = np.nanmean(np.stack(arrs, axis=0), axis=0)

    modified = []
    for sp in pred_sps:
        m = int(spd.loc[spd["stress_period"] == sp, "month"].iloc[0])
        if m not in month_mean:
            raise ValueError(f"No baseline mean computed for month={m}")
        out = os.path.join(pred_ws, _rch_fname(sp))
        org = os.path.join(pred_ws, "org", _rch_fname(sp))
        np.savetxt(out, month_mean[m], fmt="%.8e", delimiter="\t")
        os.makedirs(os.path.dirname(org), exist_ok=True)
        np.savetxt(org, month_mean[m], fmt="%.8e", delimiter="\t")
        modified.append(_rch_fname(sp))

    return modified

def elk_write_wel_baseline_monthly_mean(modnm="elk_2lay", pred_ws=".", pred_start="2025-01-01", lookback_years=5):
    """
    Baseline WEL:
      - Compute month-of-year mean flux per (layer,row,col) over the lookback window.
      - Missing wells in any historic month are treated as 0 for that month.
      - Writes wel_stress_period_data_{sp}.txt for all predictive SPs based on month-of-year.
      - Updates the WEL package file {modnm}.wel so each predictive period OPEN/CLOSE's the new file.
      - Also writes files to pred_ws/org/.
    Returns: list of modified WEL filenames
    """
    spd = _read_monthly_sp_info()
    pred_start = pd.Timestamp(pred_start)

    pred = spd.loc[spd["start_datetime"] >= pred_start].copy()
    pred_sps = pred["stress_period"].astype(int).tolist()
    if not pred_sps:
        raise ValueError(f"No predictive SPs found starting at/after {pred_start.date()}")

    hist_start = pred_start - pd.DateOffset(years=lookback_years)
    hist = spd.loc[(spd["start_datetime"] >= hist_start) & (spd["start_datetime"] < pred_start)].copy()
    if hist.empty:
        raise ValueError(f"No historical SPs found in [{hist_start.date()}, {pred_start.date()})")

    def read_wel_series(sp):
        fp = os.path.join(pred_ws, _wel_fname(sp))
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing historical WEL file needed for baseline: {fp}")
        df = pd.read_csv(fp, delim_whitespace=True, header=None)
        if df.shape[1] < 4:
            raise ValueError(f"WEL file does not have 4 columns (k i j flux): {fp}")
        df = df.iloc[:, :4]
        df.columns = ["layer", "row", "column", "flux_cfd"]
        idx = pd.MultiIndex.from_frame(df[["layer", "row", "column"]].astype(int))
        return pd.Series(df["flux_cfd"].astype(float).values, index=idx)

    # month -> list of Series
    month_to_series = {m: [] for m in range(1, 13)}
    for sp, m in zip(hist["stress_period"].astype(int), hist["month"].astype(int)):
        month_to_series[m].append(read_wel_series(sp))

    # month -> mean Series
    month_mean = {}
    for m in range(1, 13):
        sers = month_to_series.get(m, [])
        if not sers:
            continue
        all_idx = sers[0].index
        for s in sers[1:]:
            all_idx = all_idx.union(s.index)
        mat = np.vstack([s.reindex(all_idx, fill_value=0.0).values for s in sers])
        month_mean[m] = pd.Series(np.nanmean(mat, axis=0), index=all_idx)

    # write predictive SP files + build mapping for package update
    sp_to_file = {}
    modified = []
    for sp in pred_sps:
        m = int(spd.loc[spd["stress_period"] == sp, "month"].iloc[0])
        if m not in month_mean:
            raise ValueError(f"No baseline mean computed for month={m} (WEL)")

        s = month_mean[m]
        out_fn = _wel_fname(sp)

        out_path = os.path.join(pred_ws, out_fn)
        org_path = os.path.join(pred_ws, "org", out_fn)

        idx_df = pd.DataFrame(list(s.index), columns=["layer", "row", "column"])
        out_df = idx_df.copy()
        out_df["flux_cfd"] = s.values

        out_df.to_csv(out_path, sep="\t", header=False, index=False)
        os.makedirs(os.path.dirname(org_path), exist_ok=True)
        out_df.to_csv(org_path, sep="\t", header=False, index=False)

        sp_to_file[int(sp)] = out_fn
        modified.append(out_fn)

    wel_pkg = os.path.join(pred_ws, f"{modnm}.wel")
    if not os.path.exists(wel_pkg):
        raise FileNotFoundError(f"Expected WEL package file not found: {wel_pkg}")

    _replace_period_blocks_with_openclose(wel_pkg, sp_to_file)
    return modified

# ------------------------------------------------------- #
# Full Permit Use Scenario Functions
# ------------------------------------------------------- #
def write_full_alloc_well_files(modnm, pred_ws):
    print("writing full allocation well files...")

    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]
    ep_end = spd.loc[spd.index[-1],'stress_period']

    # add COW well files for prediction years:
    cow_files = [f for f in os.listdir(pred_ws) if "cow" in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
    cow_sorted = sorted(cow_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # open cow well file for 2024
    curr_cow = f'cow_stress_period_data_{sp2025-1}.txt'
    cow_df = pd.read_csv(os.path.join(pred_ws, curr_cow), delim_whitespace=True,header=None)
    cow_df.columns = ['layer','row','column','flux_cfd']
    # cow_df['ratio'] = cow_df['flux_cfd'] / cow_df['flux_cfd'].sum()

    target_predict_pmp = 2130 # acre-feet/year
    target_predict_cfd = target_predict_pmp * -43560/365.25  # convert to cfd and negative for pumping
    # cow_df['flux_cfd'] = cow_df['ratio'] * target_predict_cfd
    # cow_df = cow_df.drop(columns=['ratio'])
    pmp['add_flux_cfd'] = 0.0
    pmp['flux_cfd'] = 0.0 # we are zeroing out existing pumping and adding in full allocation based on what state requested
    pmp.loc[(pmp['layer'] == 5) & (pmp['row'] == 115) & (pmp['column'] == 47), 'add_flux_cfd'] = target_predict_cfd * 0.66
    n_otherwbv = len(pmp.loc[(pmp['layer'] == 5) & ~((pmp['row'] == 115) & (pmp['column'] == 47)), 'add_flux_cfd'])
    pmp.loc[(pmp['layer'] == 5) & ~((pmp['row'] == 115) & (pmp['column'] == 47)), 'add_flux_cfd'] = (target_predict_cfd*.33)/ n_otherwbv
    pmp['flux_cfd'] = pmp['flux_cfd'] + pmp['add_flux_cfd']
    pmp = pmp.drop(columns=['add_flux_cfd'])

    lst_cow_well_files = []
    for sp in range(sp2025, ep_end+1):
        print(f"  writing COW well file for stress period {sp}..")
        new_cow_file = f'cow_stress_period_data_{sp}.txt'
        cow_df.to_csv(os.path.join(pred_ws, new_cow_file), sep='\t', header=False, index=False)
        cow_df.to_csv(os.path.join(pred_ws,'org', new_cow_file), sep='\t', header=False, index=False)
        lst_cow_well_files.append(new_cow_file)
    
    # make sure well files are called in .nam file:
    cow_path = os.path.join(pred_ws, "elk_2lay.cow")  # adjust if needed

    # 1) map stress period -> filename from your list
    sp_to_file = {}
    for fn in lst_cow_well_files:
        m = re.search(r'_(\d+)\.txt$', fn)
        if m:
            sp_to_file[int(m.group(1))] = fn

    # 2) read the .cow file
    with open(cow_path, "r", newline="") as f:
        text = f.read()

    # 3) for each SP, replace its block body with exactly one OPEN/CLOSE line
    for sp, fname in sp_to_file.items():
        # matches:
        #   BEGIN period <sp>
        #      ... (any content, including multiple OPEN/CLOSE lines)
        #   END period <sp>
        pat = re.compile(
            rf"(?im)^(BEGIN\s+period\s+{sp}\s*\r?\n)(.*?)(^END\s+period\s+{sp}\s*$)",
            re.DOTALL
        )

        def repl(m):
            begin = m.group(1)
            end   = m.group(3)
            # overwrite the entire body with exactly one OPEN/CLOSE line + a newline before END
            body = f"    OPEN/CLOSE  '{fname}'\n"
            return begin + body + end

        text, nsub = pat.subn(repl, text, count=1)
        # If nsub == 0, the period block wasn't found; we skip silently.

    # 4) write back
    with open(cow_path, "w", newline="") as f:
        f.write(text)


    # write Min-Dakk allocation well files for prediction years:
    # minn_files = [f for f in os.listdir(pred_ws) if "minn" in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
    # minn_sorted = sorted(minn_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # # open minn well file for 2024
    # curr_minn = f'minn_stress_period_data_{sp2025-1}.txt'
    # minn_df = pd.read_csv(os.path.join(pred_ws, curr_minn), delim_whitespace=True,header=None)
    # minn_df.columns = ['layer','row','column','flux_cfd']
    # minn_df['ratio'] = minn_df['flux_cfd'] / minn_df['flux_cfd'].sum()

    # target_predict_pmp = 1240  # acre-feet/year
    # target_predict_cfd = target_predict_pmp * -43560/365.25  # convert to cfd and negative for pumping
    # minn_df['flux_cfd'] = minn_df['ratio'] * target_predict_cfd
    # minn_df = minn_df.drop(columns=['ratio'])

    # # get ratio amongst Min-Dakk pumping wells:
    # lst_minn_well_files = []
    # for sp in range(sp2025, ep_end+1):
    #     print(f"  writing Min-Dakk well file for stress period {sp}..")
    #     new_minn_file = f'minn_stress_period_data_{sp}.txt'
    #     minn_df.to_csv(os.path.join(pred_ws, new_minn_file), sep='\t', header=False, index=False)
    #     minn_df.to_csv(os.path.join(pred_ws,'org', new_minn_file), sep='\t', header=False, index=False)
    #     lst_minn_well_files.append(new_minn_file)

    # # make sure well files are called in .nam file:
    # minn_path = os.path.join(pred_ws, "elk_2lay.minn")  
    # # 1) map stress period -> filename from your list
    # sp_to_file = {}
    # for fn in lst_minn_well_files:
    #     m = re.search(r'_(\d+)\.txt$', fn)
    #     if m:
    #         sp_to_file[int(m.group(1))] = fn
    # # 2) read the .minn file
    # with open(minn_path, "r", newline="") as f:
    #     text = f.read()
    # # 3) for each SP, replace its block body with exactly one OPEN/CLOSE line
    # for sp, fname in sp_to_file.items():
    #     pat = re.compile(
    #         rf"(?im)^(BEGIN\s+period\s+{sp}\s*\r?\n)(.*?)(^END\s+period\s+{sp}\s*$)",
    #         re.DOTALL
    #     )

    #     def repl(m):
    #         begin = m.group(1)
    #         end   = m.group(3)
    #         body = f"    OPEN/CLOSE  '{fname}'\n"
    #         return begin + body + end

    #     text, nsub = pat.subn(repl, text, count=1)
    # # 4) write back
    # with open(minn_path, "w", newline="") as f:
    #     f.write(text)
    
    # car_files = [f for f in os.listdir(pred_ws) if "car" in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
    # car_sorted = sorted(car_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # # open car well file for 2024
    # curr_car = f'car_stress_period_data_{sp2025-1}.txt'
    # car_df = pd.read_csv(os.path.join(pred_ws, curr_car), delim_whitespace=True,header=None)
    # car_df.columns = ['layer','row','column','flux_cfd']
    # if  car_df['flux_cfd'].sum() < 1e-4:
    #      car_df['flux_cfd'] = -1.0
    # car_df['ratio'] = car_df['flux_cfd'] / car_df['flux_cfd'].sum()

    # target_predict_pmp = 1240  # acre-feet/year
    # target_predict_cfd = target_predict_pmp * -43560/365.25  # convert to cfd and negative for pumping
    # car_df['flux_cfd'] = car_df['ratio'] * target_predict_cfd
    # car_df = car_df.drop(columns=['ratio'])

    # # get ratio amongst Cargill pumping wells:
    # lst_car_well_files = []
    # for sp in range(sp2025, ep_end+1):
    #     print(f"  writing Cargill well file for stress period {sp}..")
    #     new_car_file = f'car_stress_period_data_{sp}.txt'
    #     car_df.to_csv(os.path.join(pred_ws, new_car_file), sep='\t', header=False, index=False)
    #     car_df.to_csv(os.path.join(pred_ws,'org', new_car_file), sep='\t', header=False, index=False)
    #     lst_car_well_files.append(new_car_file)

    # # make sure well files are called in .nam file:
    # car_path = os.path.join(pred_ws, "elk_2lay.car")  
    # # 1) map stress period -> filename from your list
    # sp_to_file = {}
    # for fn in lst_car_well_files:
    #     m = re.search(r'_(\d+)\.txt$', fn)
    #     if m:
    #         sp_to_file[int(m.group(1))] = fn
    # # 2) read the .car file
    # with open(car_path, "r", newline="") as f:
    #     text = f.read()
    # # 3) for each SP, replace its block body with exactly one OPEN/CLOSE line
    # for sp, fname in sp_to_file.items():
    #     pat = re.compile(
    #         rf"(?im)^(BEGIN\s+period\s+{sp}\s*\r?\n)(.*?)(^END\s+period\s+{sp}\s*$)",
    #         re.DOTALL
    #     )

    #     def repl(m):
    #         begin = m.group(1)
    #         end   = m.group(3)
    #         body = f"    OPEN/CLOSE  '{fname}'\n"
    #         return begin + body + end

    #     text, nsub = pat.subn(repl, text, count=1)
    # # 4) write back
    # with open(car_path, "w", newline="") as f:
    #     f.write(text)

    lst_mod_well_files = lst_cow_well_files #+ lst_car_well_files

    return lst_mod_well_files

def nopt0chk_full_permit_use(org_ws,pred_ws, modnm, noptmax_flow):
    pst = pyemu.Pst(os.path.join(pred_ws,'elk_2lay.pst'))
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_par_en'] = f'{modnm}.{noptmax_flow}.par.jcb'
    pst.pestpp_options["ies_obs_en"] = f'{modnm}.{noptmax_flow}.obs.jcb'
    pst.write(os.path.join(pred_ws,'elk_2lay.pst'),version=2)
    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst', cwd=pred_ws)

    df = pd.read_csv(os.path.join(pred_ws, f'{modnm}.base.obs.csv')) 
    old_df = pd.read_csv(os.path.join(org_ws, f'{modnm}.base.obs.csv'))
    
    # get col names with cow in them:
    cow_cols = [col for col in df.columns if 'cow' in col]
    df = df[cow_cols]
    bud_df_cols = [col for col in df.columns if ('bud' in col) and ('out' in col)]
    bud_df = df[bud_df_cols].T.reset_index()
    bud_df.columns = ['obsnme', 'bud_out']
    # if 'zn in obsnme', drop:
    bud_df = bud_df.loc[bud_df['obsnme'].str.contains('zn') == False]
    bud_df['datetime'] = pd.to_datetime(bud_df['obsnme'].str.split(':').str[-1], format='%Y-%m-%d')
    
    old_df = old_df[bud_df_cols].T.reset_index()
    old_df.columns = ['obsnme', 'bud_out']
    old_df = old_df.loc[old_df['obsnme'].str.contains('zn') == False]
    old_df['datetime'] = pd.to_datetime(old_df['obsnme'].str.split(':').str[-1], format='%Y-%m-%d')

    cow_bud = bud_df.copy()
    cow_old = old_df.copy()

    # reload base files:
    df = pd.read_csv(os.path.join(pred_ws, f'{modnm}.base.obs.csv')) 
    old_df = pd.read_csv(os.path.join(org_ws, f'{modnm}.base.obs.csv'))

    minn_cols = [col for col in df.columns if 'minn' in col]
    df_minn = df[minn_cols]
    bud_df_minn = df_minn.T.reset_index()
    bud_df_minn.columns = ['obsnme', 'bud_out']
    bud_df_minn = bud_df_minn.loc[bud_df_minn['obsnme'].str.contains('zn') == False]
    bud_df_minn = bud_df_minn.loc[bud_df_minn['obsnme'].str.contains('flx') == False]
    bud_df_minn = bud_df_minn.loc[bud_df_minn['obsnme'].str.contains('out') == True]
    bud_df_minn['datetime'] = pd.to_datetime(bud_df_minn['obsnme'].str.split(':').str[-1], format='%Y-%m-%d')

    old_df_minn = old_df[minn_cols]
    bud_df_minn_old = old_df_minn.T.reset_index()
    bud_df_minn_old.columns = ['obsnme', 'bud_out']
    bud_df_minn_old = bud_df_minn_old.loc[bud_df_minn_old['obsnme'].str.contains('zn') == False]
    bud_df_minn_old = bud_df_minn_old.loc[bud_df_minn_old['obsnme'].str.contains('flx') == False]
    bud_df_minn_old = bud_df_minn_old.loc[bud_df_minn_old['obsnme'].str.contains('out') == True]
    bud_df_minn_old['datetime'] = pd.to_datetime(bud_df_minn_old['obsnme'].str.split(':').str[-1], format='%Y-%m-%d')

     # plot COW pumping comparison:
    fig, [ax1,ax2] = plt.subplots(figsize=(8,6), nrows=2, ncols=1, sharex=True)
    ax1.plot(cow_bud['datetime'], cow_bud['bud_out']/43560*365.25, marker='o', linestyle='-')
    ax1.plot(cow_old['datetime'], cow_old['bud_out']/43560*365.25, marker='x', linestyle='--')
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.set_ylabel('Annual COW Pumping\n(acre-feet/year)')
    ax1.set_title('Pumping Comparison: Posterior vs. Full Permit Use',fontsize=11)
    
    ax2.plot(bud_df_minn['datetime'], bud_df_minn['bud_out']/43560*365.25, marker='o', linestyle='-')
    ax2.plot(bud_df_minn_old['datetime'], bud_df_minn_old['bud_out']/43560*365.25, marker='x', linestyle='--')
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.set_ylabel('Annual Min-Dakk Pumping\n(acre-feet/year)')

    figoutput = os.path.join(pred_ws, 'input_figs','scn02_full_permit_use')
    if not os.path.exists(figoutput):
        os.makedirs(figoutput)
    fig.savefig(os.path.join(figoutput, 'histo_match_vs_scn_pumping_fullpermit.png'), dpi=300)
    plt.close(fig)

# ------------------------------------------------------- #
# Ultimate Drought Scenario Functions
# ------------------------------------------------------- #
def write_ghb_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]

    # read in ghb files:
    ghb_files = [f for f in os.listdir(pred_ws) if f"{modnm}_wbv.ghb" in f and f.endswith('.txt')]
    ghb_sorted = sorted(ghb_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # get three year drought at begnining of prediction period:
    sp_rng = range(sp2025, sp2025+3)
    ghb_sorted = [f for f in ghb_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    for ghb in ghb_sorted:
        print(f"  modifying ghb file {ghb}..")
        ghb_df = pd.read_csv(os.path.join(pred_ws, ghb), delim_whitespace=True,header=None)
        ghb_df.columns = ['layer','row','column','stage','cond']
        # set cond to small #:
        ghb_df['stage'] = ghb_df['stage'] - 4.0  # lower stage by 4 feet

        # write modified ghb file:
        ghb_df.to_csv(os.path.join(pred_ws, ghb), sep='\t', header=False, index=False)
        ghb_df.to_csv(os.path.join(pred_ws,'org', ghb), sep='\t', header=False, index=False)

    return ghb_sorted

def write_riv_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]

    # read in riv files:
    riv_files = [f for f in os.listdir(pred_ws) if f"otriv" in f and f.endswith('.txt')]
    riv_sorted = sorted(riv_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # get three year drought at begnining of prediction period:
    sp_rng = range(sp2025, sp2025+3)
    otriv_sorted = [f for f in riv_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    riv_files = [f for f in os.listdir(pred_ws) if f.endswith('.txt') and f.startswith('riv_s')]
    riv_sorted = sorted(riv_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))
    riv_sorted = [f for f in riv_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]
    
    riv_sorted = riv_sorted + otriv_sorted

    for riv in riv_sorted:
        print(f"  modifying riv file {riv}..")
        riv_df = pd.read_csv(os.path.join(pred_ws, riv), delim_whitespace=True,header=None)
        riv_df.columns = ['layer','row','column','stage','cond','rbot']
        riv_df['stage'] = riv_df['stage']-3.0

        # write modified riv file:
        riv_df.to_csv(os.path.join(pred_ws, riv), sep='\t', header=False, index=False)
        riv_df.to_csv(os.path.join(pred_ws,'org', riv), sep='\t', header=False, index=False)

    return riv_sorted

def write_rch_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]
    drought_yr = spd.loc[spd['year'] == 2021,'stress_period'].values[0]
    
    # read in rch files:
    rch_files = [f for f in os.listdir(pred_ws) if f.startswith('rch_') and f.endswith('.txt')]
    rch_sorted = sorted(rch_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # loa representative drought rch file:
    drought_rch_file = f'rch_recharge_{drought_yr}.txt'
    dry_df = pd.read_csv(os.path.join(pred_ws, drought_rch_file), delim_whitespace=True,header=None)

    # drop list to rch files before 2025:
    sp_rng = range(sp2025, sp2025+3)
    rch_sorted = [f for f in rch_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    for rch in rch_sorted:
        print(f"  modifying rch file {rch}..")
        # write drought recharge:
        dry_df.to_csv(os.path.join(pred_ws, rch), sep='\t', header=False, index=False)
        dry_df.to_csv(os.path.join(pred_ws,'org', rch), sep='\t', header=False, index=False)

    return rch_sorted, sp_rng

def write_cargill_drought_use_well_files(modnm, pred_ws):
    print("writing cargill drought use well files...")

    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]
    ep_end = spd.loc[spd.index[-1],'stress_period']

    car_files = [f for f in os.listdir(pred_ws) if "car" in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
    car_sorted = sorted(car_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # open car well file for 2024
    curr_car = f'car_stress_period_data_{sp2025-1}.txt'
    car_df = pd.read_csv(os.path.join(pred_ws, curr_car), delim_whitespace=True,header=None)
    car_df.columns = ['layer','row','column','flux_cfd']
    if  car_df['flux_cfd'].sum() < 1e-4:
         car_df['flux_cfd'] = -1.0
    car_df['ratio'] = car_df['flux_cfd'] / car_df['flux_cfd'].sum()

    target_predict_pmp = 1240/2  # acre-feet/year
    target_predict_cfd = target_predict_pmp * -43560/365.25  # convert to cfd and negative for pumping
    car_df['flux_cfd'] = car_df['ratio'] * target_predict_cfd
    car_df = car_df.drop(columns=['ratio'])

    # get ratio amongst Cargill pumping wells:
    lst_car_well_files = []
    for sp in range(sp2025, sp2025+3):
        print(f"  writing Cargill well file for stress period {sp}..")
        new_car_file = f'car_stress_period_data_{sp}.txt'
        car_df.to_csv(os.path.join(pred_ws, new_car_file), sep='\t', header=False, index=False)
        car_df.to_csv(os.path.join(pred_ws,'org', new_car_file), sep='\t', header=False, index=False)
        lst_car_well_files.append(new_car_file)

    # make sure well files are called in .nam file:
    car_path = os.path.join(pred_ws, "elk_2lay.car")  
    # 1) map stress period -> filename from your list
    sp_to_file = {}
    for fn in lst_car_well_files:
        m = re.search(r'_(\d+)\.txt$', fn)
        if m:
            sp_to_file[int(m.group(1))] = fn
    # 2) read the .car file
    with open(car_path, "r", newline="") as f:
        text = f.read()
    # 3) for each SP, replace its block body with exactly one OPEN/CLOSE line
    for sp, fname in sp_to_file.items():
        pat = re.compile(
            rf"(?im)^(BEGIN\s+period\s+{sp}\s*\r?\n)(.*?)(^END\s+period\s+{sp}\s*$)",
            re.DOTALL
        )

        def repl(m):
            begin = m.group(1)
            end   = m.group(3)
            body = f"    OPEN/CLOSE  '{fname}'\n"
            return begin + body + end

        text, nsub = pat.subn(repl, text, count=1)
    # 4) write back
    with open(car_path, "w", newline="") as f:
        f.write(text)

    lst_mod_well_files = lst_car_well_files

    return lst_mod_well_files

def elk_overwrite_rch_drought_only(
    pred_ws=".",
    pred_start="2025-01-01",
    drought_years=3,
    drought_source_year=2021,
):
    """
    Drought RCH overwrite ONLY (baseline RCH is already baked into the base model build):

    - Identify predictive SPs as those with start_datetime >= pred_start
    - Take the first drought_years*12 predictive months as the drought window
    - For each drought-month SP, overwrite rch_recharge_{sp}.txt with the array from drought_source_year
      matching month-of-year (Jan->Jan drought_source_year, etc.)
    - Writes both to pred_ws/ and pred_ws/org/

    Returns:
        modified_files (list[str]), drought_sps (list[int])
    """
    spd = _read_monthly_sp_info()
    pred_start = pd.Timestamp(pred_start)

    pred = spd.loc[spd["start_datetime"] >= pred_start, ["stress_period", "month"]].copy()
    pred = pred.sort_values("stress_period")
    if pred.empty:
        raise ValueError(f"No predictive SPs found starting at/after {pred_start.date()}")

    drought_n = int(drought_years) * 12
    drought_pred = pred.iloc[:drought_n].copy()
    drought_sps = drought_pred["stress_period"].astype(int).tolist()

    src = spd.loc[spd["year"] == int(drought_source_year), ["stress_period", "month"]].copy()
    if src.empty:
        raise ValueError(f"No SPs found for drought_source_year={drought_source_year}")

    # month -> first SP in drought_source_year for that month
    month_to_srcsp = {}
    for m in range(1, 13):
        rows = src.loc[src["month"] == m, "stress_period"].astype(int).tolist()
        if not rows:
            raise ValueError(f"No SP found for drought_source_year={drought_source_year}, month={m}")
        month_to_srcsp[m] = rows[0]

    modified = []
    for sp, m in zip(drought_pred["stress_period"].astype(int), drought_pred["month"].astype(int)):
        srcsp = month_to_srcsp[m]
        srcfile = os.path.join(pred_ws, _rch_fname(srcsp))
        if not os.path.exists(srcfile):
            raise FileNotFoundError(f"Missing drought source recharge file: {srcfile}")

        arr = np.loadtxt(srcfile)

        out = os.path.join(pred_ws, _rch_fname(sp))
        org = os.path.join(pred_ws, "org", _rch_fname(sp))
        np.savetxt(out, arr, fmt="%.8e", delimiter="\t")
        os.makedirs(os.path.dirname(org), exist_ok=True)
        np.savetxt(org, arr, fmt="%.8e", delimiter="\t")

        modified.append(_rch_fname(sp))

    return sorted(set(modified)), drought_sps

def elk_drop_riv_stage_during_drought(modnm="elk_2lay", pred_ws=".", drought_sps=None,
                                     stage_drop_ft=2.0,
                                     min_rbot_clear_ft=0.1,
                                     min_stage_above_bot_ft=3.0,
                                     min_stage_above_rbot_ft=0.1):
    """
    For each RIV external SP file in drought_sps:
      - subtract stage_drop_ft from stage and rbot
      - enforce:
          rbot >= botm + min_rbot_clear_ft
          stage >= max(rbot + min_stage_above_rbot_ft, botm + min_stage_above_bot_ft)

    Requires flopy to load DIS and get botm.
    Returns: list of modified riv filenames
    """
    if drought_sps is None:
        raise ValueError("drought_sps must be provided (list of int stress periods).")

    try:
        import flopy
    except Exception as e:
        raise ImportError("flopy is required for RIV safety corrections (reading botm).") from e

    sim = flopy.mf6.MFSimulation.load(sim_ws=pred_ws, load_only=["dis"])
    m = sim.get_model(modnm)
    botm = np.asarray(m.dis.botm.array)

    drought_set = set(int(s) for s in drought_sps)

    # Find candidate riv files: contains 'riv' and ends with _{sp}.txt
    txts = [f for f in os.listdir(pred_ws) if f.endswith(".txt")]
    riv_files = []
    for f in txts:
        msp = re.search(r"_(\d+)\.txt$", f)
        if not msp:
            continue
        sp = int(msp.group(1))
        if sp not in drought_set:
            continue
        if "riv" not in f.lower():
            continue
        # avoid rch/wel
        if f.lower().startswith("rch") or f.lower().startswith("wel"):
            continue
        riv_files.append(f)

    modified = []
    for fn in sorted(set(riv_files)):
        fp = os.path.join(pred_ws, fn)
        df = pd.read_csv(fp, delim_whitespace=True, header=None)
        if df.shape[1] < 6:
            continue  # skip anything not a 6-col RIV file
        df = df.iloc[:, :6].copy()
        df.columns = ["layer", "row", "column", "stage", "cond", "rbot"]

        df["stage"] = df["stage"].astype(float) - float(stage_drop_ft)
        df["rbot"]  = df["rbot"].astype(float)  - float(stage_drop_ft)

        stage = df["stage"].to_numpy(float)
        rbot  = df["rbot"].to_numpy(float)

        for i in range(len(df)):
            k0 = int(df.loc[i, "layer"]) - 1
            r0 = int(df.loc[i, "row"]) - 1
            c0 = int(df.loc[i, "column"]) - 1

            cellbot = float(botm[k0, r0, c0])

            # rbot constraint
            rbot[i] = max(rbot[i], cellbot + float(min_rbot_clear_ft))

            # stage constraint
            stage_min = max(rbot[i] + float(min_stage_above_rbot_ft),
                            cellbot + float(min_stage_above_bot_ft))
            stage[i] = max(stage[i], stage_min)

        df["rbot"] = rbot
        df["stage"] = stage

        df.to_csv(fp, sep="\t", header=False, index=False)
        org = os.path.join(pred_ws, "org", fn)
        os.makedirs(os.path.dirname(org), exist_ok=True)
        df.to_csv(org, sep="\t", header=False, index=False)

        modified.append(fn)

    return modified
# ------------------------------------------------------- #
# ASR Scenario Functions
# ------------------------------------------------------- #
def write_asr_well_files(modnm, pred_ws,rate_acftyr=500,k=5,i=83,j=38):
    print("writing ASR well files...")

    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]
    ep_end = spd.loc[spd.index[-1],'stress_period']

    sim = flopy.mf6.MFSimulation.load(sim_ws=pred_ws, sim_name=modnm)
    gwf = sim.get_model(modnm)

    # convert rate to cfd:
    rate_cfd = rate_acftyr * 43560 / 365.25  # positive for injection
    layer_k = int(k)      
    row_i   = int(i)     
    col_j   = int(j)      

    well_dict = {}
    for sp in range(1, ep_end + 1):
        sp_key = sp - 1  
        if sp_key < sp2025 - 1:
            record = ((layer_k, row_i, col_j), 0.0)
        else:
            record = ((layer_k, row_i, col_j), rate_cfd)  
        well_dict[sp_key] = [record]  

    well = flopy.mf6.ModflowGwfwel(gwf,
                                stress_period_data=well_dict,
                                pname='asr',
                                save_flows=True,
                                maxbound=1,
                                filename=f'{modnm}.asr')
    
    sim.write_simulation()

    # copy .nam file to org_mws folder:
    shutil.copy2(os.path.join(pred_ws, f'{modnm}.nam'), os.path.join(pred_ws,'org_mws', f'{modnm}.nam'))

def write_lst_asr_well_files(modnm, pred_ws,rate_acftyr=500,kij=[]):
    print("writing ASR well files...")

    # load in stress period info:
    spd = pd.read_csv(os.path.join("tables", "monthly_stress_period_info.csv"))
    spd["start_datetime"] = pd.to_datetime(spd["start_datetime"])
    spd["end_datetime"]   = pd.to_datetime(spd["end_datetime"])

    sp2025 = spd.loc[spd['year'] == 2025,'stress_period'].values[0]
    ep_end = spd.loc[spd.index[-1],'stress_period']

    sim = flopy.mf6.MFSimulation.load(sim_ws=pred_ws, sim_name=modnm)
    gwf = sim.get_model(modnm)

    # convert rate to cfd:
    rate_cfd = rate_acftyr * 43560 / 365.25  # positive for injection    

    well_dict = {}
    for sp in range(1, ep_end + 1):
        sp_key = sp - 1 
        record = [] 
        for kij_tuple in kij:
            layer_k = int(kij_tuple[0])      
            row_i   = int(kij_tuple[1])     
            col_j   = int(kij_tuple[2])
            if sp_key < sp2025 - 1:
                rec = ((layer_k, row_i, col_j), 0.0)
            else:
                rec = ((layer_k, row_i, col_j), rate_cfd) 
            record += [rec]
        well_dict[sp_key] = record  

    well = flopy.mf6.ModflowGwfwel(gwf,
                                stress_period_data=well_dict,
                                pname='asr',
                                save_flows=True,
                                maxbound=1,
                                filename=f'{modnm}.asr')
    
    sim.write_simulation()

    # copy .nam file to org_mws folder:
    shutil.copy2(os.path.join(pred_ws, f'{modnm}.nam'), os.path.join(pred_ws,'org_mws', f'{modnm}.nam'))

def znbud_by_ly_process(modnm='elk_2lay'):
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
    dtim = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    # add value from 'zone' column in df to each dtim val:
    dtim = [f"{d.strftime('%Y-%m-%d')}_zn-{int(z)}" for d,z in zip(dtim,df['zbly_zone'])]
 
    df.index = dtim
    df.index.name = "datetime"
    df.to_csv("zbud.csv")
    dfs = [df]
    return dfs

def init_zonbud_process(d,modnm='wahp'):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = znbud_by_ly_process(modnm=modnm)
    os.chdir(b_d)
    return dfs

def rewrite_zbud_ins_file(modnm='elk_2lay', pred_ws='.'):
    template = os.path.join(pred_ws,'temp_template')

    # if os.path.exists(template) remove it
    if os.path.exists(template):
        shutil.rmtree(template)

    init_clean_ws(pred_ws, template)
    pyemu.os_utils.run('mf6',cwd=template)
    df_zb = init_zonbud_process(template,modnm=modnm)

    # if 'org' folder in template, remove it
    org_folder = os.path.join(template, 'org')
    if os.path.exists(org_folder):
        shutil.rmtree(org_folder)
    mult_folder = os.path.join(template, 'mult')
    if os.path.exists(mult_folder):
        shutil.rmtree(mult_folder)
    org_mws_folder = os.path.join(template, 'org_mws')
    if os.path.exists(org_mws_folder):
        shutil.rmtree(org_mws_folder)

    temp = os.path.join(pred_ws,'temp_temp')
    pf = pyemu.utils.PstFrom(original_d=template, new_d=temp,
                        remove_existing=True,
                        longnames=True, 
                        zero_based=False)
    pf.add_observations('zbud.csv',index_cols=['datetime'],use_cols=df_zb[0].columns.to_list(),obsgp='zbudly',ofile_sep=',',prefix='zbudly')
    pst = pf.build_pst(version=None)

    # copy zbud.csv.ins to pred_ws:
    shutil.copy2(os.path.join(temp, 'zbud.csv.ins'), os.path.join(pred_ws, 'zbud.csv.ins'))

    pst.control_data.noptmax = 0
    pst.pestpp_options['additional_ins_delimiters'] = ','
    pst.write(os.path.join(temp,f'{modnm}.pst'),version=2)
    try:
        pyemu.os_utils.run(f'pestpp-ies {modnm}.pst',cwd=temp)
    except Exception as e:
        print("pestpp-ies failed, but it is suppose to all we want here is the obs_data.csv file")
    obs_df_asr = pd.read_csv(os.path.join(temp, f'{modnm}.obs_data.csv'))
    obs_df_asr['weight'] = 0.0
    obs_df = pd.read_csv(os.path.join(pred_ws, f'{modnm}.obs_data.csv'))

    new_rows = obs_df_asr.loc[~obs_df_asr['obsnme'].isin(obs_df['obsnme'])]
    obs_df = pd.concat([obs_df, new_rows], ignore_index=True)

    obs_df.to_csv(os.path.join(pred_ws, f'{modnm}.obs_data.csv'), index=False)

    shutil.rmtree(temp)
    shutil.rmtree(template)


# ------------------------------------------------------- #
# Condor/Run Functions
# ------------------------------------------------------- #

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

def run_ies(template_ws='template_d', modnm='wahp', m_d=None, num_workers=12,niceness=False, noptmax=-1, num_reals=None,
              init_lam=None, drop_conflicts=False, local=True, hostname=None, port=4263, par_post_nm='', obs_post_nm='',
               use_condor=False,**kwargs):
    
    if m_d is None:
        m_d = template_ws.replace('template', 'master')

    pst = pyemu.Pst(os.path.join(template_ws, f'{modnm}.pst'))

    # Set control file options:
    pst.control_data.noptmax = noptmax
    pst.pestpp_options['ies_drop_conflicts'] = drop_conflicts
    pst.pestpp_options['overdue_giveup_fac'] = 10
    pst.pestpp_options['ies_bad_phi_sigma'] = 1.5
    pst.pestpp_options['ies_bad_phi'] = 1e+20
    pst.pestpp_options["ies_multimodal_alpha"] = 0.99
    pst.pestpp_options['panther_agent_freeze_on_fail'] = True
    pst.pestpp_options['save_binary'] = True
    
    if num_reals is not None:
        pst.pestpp_options['ies_num_reals'] = num_reals

    if init_lam is not None:
        pst.pestpp_options['ies_initial_lambda'] = init_lam
    pst.pestpp_options['ies_subset_size'] = -10
    for k,v in kwargs.items():
        pst.pestpp_options[k] = v
    # intit run log file:
    f = open(os.path.join(template_ws, 'wahppst_run.log'), 'w')
    f.close()

    # obs sainty check:
    pobs = pst.observation_data
    pobsmax = pobs.weight.max()
    if pobsmax <= 0:
        raise Exception('setting weighted obs failed!!!')
    pst.write(os.path.join(template_ws, f'{modnm}.pst'), version=2)

    prep_worker(template_ws, template_ws + '_clean',par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)
    
    master_p = None

    if hostname is None:
        pyemu.os_utils.start_workers(template_ws, 'pestpp-ies', f'{modnm}.pst',
                                 num_workers=num_workers, worker_root='.',
                                 master_dir=m_d, local=local,port=4269)

    elif use_condor:
        check_port_number(port)

        jobid = condor_submit(template_ws=template_ws + '_clean', pstfile=f'{modnm}.pst', conda_zip_pth='nddwrpy311.tar.gz',
                              subfile=f'{modnm}.sub',
                              workerfile='worker.sh', executables=['mf6', 'pestpp-ies','mp7'], request_memory=5000,
                              request_disk='15g', port=port,
                              num_workers=num_workers,niceness=niceness)

        # jwhite - commented this out so not starting local workers on the condor submit machine # no -ross
        pyemu.os_utils.start_workers(template_ws + '_clean', 'pestpp-ies', f'{modnm}.pst', num_workers=0, worker_root='.',
                                     port=port, local=local, master_dir=m_d)

        if jobid is not None:
            # after run master is finished clean up condor by using condor_rm
            print(f'killing condor job {jobid}')
            os.system(f'condor_rm {jobid}')

    # if a master was spawned, wait for it to finish
    if master_p is not None:
        master_p.wait()

def prep_worker(org_d, new_d,run_flex_cond=False,par_post_nm='', obs_post_nm=''):
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
            if f != 'cond_post.jcb' and f != 'noise.jcb' and f != par_post_nm and f != obs_post_nm:  # need prior.jcb to run ies
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


# ------------------------------------------------------- #
# Plot results functions
# ------------------------------------------------------- #
def _col(df, *cands):
    """Return the first existing column name from candidates."""
    for c in cands:
        if c in df.columns:
            return c
    return None

def plot_inset(ax, wl_loc, cpts, aq_extent=None, drains=None, wls=None):
    """
    Robust inset plotter.
    - Accepts wl_loc as 'grpid:<group>_k:<k>' OR a location string.
    - Works with cpts columns like: 'grpid', 'grp_id', 'group number', 'group_number', 'location', 'loc_id'
    - Optional fallback via wls (expects columns: group_number, model_layer, location)
    """
    import numpy as np
    import pandas as pd

    if cpts is None or len(cpts) == 0:
        raise ValueError("plot_inset(): cpts is empty")

    # normalize columns for easier matching
    c = cpts.copy()
    c.columns = [str(x).strip().lower() for x in c.columns]

    # column candidates
    col_grpid = "grpid" if "grpid" in c.columns else ("grp_id" if "grp_id" in c.columns else None)
    col_grpnum = None
    for cand in ["group_number", "group number", "groupnum", "grpnum", "group"]:
        if cand in c.columns:
            col_grpnum = cand
            break

    col_loc = None
    for cand in ["location", "loc_id", "locid", "site", "station"]:
        if cand in c.columns:
            col_loc = cand
            break

    col_lay = None
    for cand in ["k", "layer", "model_layer", "lay"]:
        if cand in c.columns:
            col_lay = cand
            break

    # parse wl_loc
    grp_num = None
    k0 = None
    wl_loc_str = str(wl_loc)

    if wl_loc_str.startswith("grpid:") and "_k:" in wl_loc_str:
        try:
            grp_num = int(wl_loc_str.split("grpid:")[1].split("_k:")[0])
            k0 = int(wl_loc_str.split("_k:")[1].split("_")[0])
        except Exception:
            grp_num, k0 = None, None

    # 1) best: direct grpid match if possible
    sites = None
    if col_grpid is not None:
        sites = c.loc[c[col_grpid].astype(str) == wl_loc_str].copy()
        if sites.empty:
            # sometimes whitespace/case issues
            sites = c.loc[c[col_grpid].astype(str).str.strip().str.lower() == wl_loc_str.strip().lower()].copy()

    # 2) group number match
    if (sites is None or sites.empty) and (grp_num is not None) and (col_grpnum is not None):
        # group number columns are sometimes strings; coerce carefully
        try:
            sites = c.loc[pd.to_numeric(c[col_grpnum], errors="coerce").astype("Int64") == grp_num].copy()
        except Exception:
            sites = c.loc[c[col_grpnum].astype(str) == str(grp_num)].copy()

    # 3) location match
    if (sites is None or sites.empty) and (col_loc is not None):
        sites = c.loc[c[col_loc].astype(str) == wl_loc_str].copy()
        if sites.empty:
            sites = c.loc[c[col_loc].astype(str).str.contains(wl_loc_str, na=False)].copy()

    # 4) fallback via wls: group/layer -> location -> try again
    if (sites is None or sites.empty) and (wls is not None) and (grp_num is not None) and (k0 is not None):
        w = wls.copy()
        w.columns = [str(x).strip().lower() for x in w.columns]
        if "group_number" in w.columns and "model_layer" in w.columns and "location" in w.columns:
            w2 = w.loc[(w["group_number"].astype(int) == grp_num) & ((w["model_layer"].astype(int) - 1) == k0)]
            if not w2.empty and col_loc is not None:
                loc = str(w2["location"].iloc[0])
                sites = c.loc[c[col_loc].astype(str) == loc].copy()

    if sites is None or sites.empty:
        raise ValueError(f"plot_inset(): couldn't find site '{wl_loc}'")

    # layer filter if we know k0 and have a layer column
    if (k0 is not None) and (col_lay is not None):
        try:
            if col_lay == "model_layer":
                sites = sites.loc[pd.to_numeric(sites[col_lay], errors="coerce") == (k0 + 1)].copy()
            else:
                sites = sites.loc[pd.to_numeric(sites[col_lay], errors="coerce") == k0].copy()
        except Exception:
            pass

    # --- draw base layers ---
    if aq_extent is not None:
        try:
            aq_extent.boundary.plot(ax=ax, linewidth=1)
        except Exception:
            pass

    if drains is not None:
        try:
            drains.plot(ax=ax, markersize=2, alpha=0.5)
        except Exception:
            pass

    # --- highlight point(s) ---
    # cpts can be GeoDataFrame; if not, we still try to plot if geometry exists
    if "geometry" in sites.columns:
        try:
            sites.plot(ax=ax, markersize=30, color="red", zorder=10)
        except Exception:
            pass
    else:
        ax.text(0.5, 0.5, "No geometry in cpts row(s)", ha="center", va="center", transform=ax.transAxes)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Location inset", fontsize=10)



def plot_scn_hydrograpghs(pred_ws_list, modnm='elk_2lay',plot_quantiles=True,plt_base_only=False,zoom_predict=False):
    o_d = os.path.join( 'results', 'figures', 'scenario_results')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    # init info from just one of the scn workspaces:
    m_d = pred_ws_list[0]
    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('trans'), :].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)

    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])

    obs_adj = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'))
    obs_adj = obs_adj.loc[obs_adj.name.str.contains('grpid')]
    weighted = obs_adj.loc[obs_adj.weight > 0, :].copy()
    weighted['datetime'] = pd.to_datetime(weighted.name.apply(lambda x: x.split(':')[-1]))
    weighted['id'] = weighted.name.apply(lambda x: 'grpid:' + x.split('grpid:')[1].split('_k')[0] + '_k:' + x.split('_k:')[1].split('_')[0])
    unq = weighted.id.unique()
    # get gwobs with unq ids:
    gwobs = gwobs.loc[gwobs.id.isin(unq), :].copy()
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis'])
    m = sim.get_model(modnm)

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)
    top = m.dis.top.array
    botm = m.dis.botm.array
    usites = gwobs.id.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site': usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)

    # load shapefiles needed for plotting:
    cpts = pd.read_csv(os.path.join('data', 'analyzed', 'transient_well_targets_lookup.csv'))
    cpts['grpid'] = 'grpid:' + cpts['group number'].astype(str) + '_k:' + cpts['k'].astype(str)
    # make geodataframe from geometry column:
    cpts = gpd.GeoDataFrame(data=cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=2265)
    cpts = cpts.groupby(['grpid']).last().reset_index()
    cpts.columns = cpts.columns.str.lower()
    
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'wahp')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'wahp')
    aq_extent = gpd.read_file(os.path.join(g_d, 'wahp_outline_full.shp'))
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'wahp7ly_cell_size_660ft_epsg2265.grid.shp'))
    drains = gpd.read_file(os.path.join(g_d, 'flow_lines_all_clipped.shp'))
    aq_extent = aq_extent.to_crs(modelgrid.crs)

    wls = pd.read_csv(os.path.join('data', 'raw','water_lvl_targs_manual_ly_assign.csv'))
    wls['grp_id'] = 'grpid:' + wls['group number'].astype(str) + '_k:' + (wls['manually_corrected_lay']-1).astype(str)

    usites = usitedf['site'].values
    
    scn_results_dict ={}
    pst = pyemu.Pst(os.path.join(pred_ws_list[0],f'{modnm}.pst'))
    for ws in pred_ws_list:         
        jcbName = os.path.join(ws,f'{modnm}.0.obs.jcb')
        assert os.path.exists(jcbName), f'obs jcb {jcbName} not found! Scenario probably did not run have fun debugging!'
        jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
        scn_tag = ws.split('_')[2]
        scn_results_dict[scn_tag] = jcb
        # store jcd in dict:
        scn_results_dict[scn_tag] = jcb
    
    if scn_tag in ['scn04','scn05']:
        #HARD CODED!!!!!!!!!!!!!!
        asr = True
        asr_i=83
        asr_j=38
        asr_scn4 = modelgrid.loc[(modelgrid.row == asr_i) & (modelgrid.col == asr_j), :]
        asr_scn4.geometry = asr_scn4.geometry.centroid

        asr_i=114
        asr_j=45
        asr_scn5 = modelgrid.loc[(modelgrid.row == asr_i) & (modelgrid.col == asr_j), :]
        asr_scn5.geometry = asr_scn5.geometry.centroid
        
        asr_grd_loc = pd.concat([asr_scn4, asr_scn5], ignore_index=True)
        
    else:
        asr = False
        asr_grd_loc = None
        
    colors = [
        "#b7bbc0", # scn01 - grey (baseline)
        "#f1a924", # scn02 - orange (full permit use)
        "#9e1818", # scn03 - red (ultimate drought)
        "#5a812c", # scn04 - green (drought + asr at 500 acft/yr)
        "#49b6e9", # scn05 - yellow (drought + asr at 200 acft/yr)
        "#e723e7", # scn06 - new loc asr
        ]
    darker_colors = [
        "#7f8286", # scn01 - dark grey (baseline)
        "#ee6c15", # scn02 - dark orange (full permit use)
        "#e23232", # scn03 - dark red (ultimate drought)
        "#81e70c", # scn04 - dark green (drought + asr at 500 acft/yr)
        "#151792", # scn05 - dark yellow (drought + asr at 200 acft/yr)
        "#990e99", # scn06 - dark new loc asr
        ]
    
    asr_grd_loc['color'] = 'cyan'
    asr_grd_loc.loc[0, 'color'] = "#81e70c"
    asr_grd_loc.loc[1, 'color'] = "#990e99"
    
    years10 = mdates.YearLocator(10)
    years20 = mdates.YearLocator(20)
    years_fmt = mdates.DateFormatter('%Y')
    
    with PdfPages(os.path.join(o_d, 'scenario_hydrograpghs.pdf')) as pdf:
        for site in usites:
            fig = plt.figure(figsize=(11, 8))
            gs = gridspec.GridSpec(7, 6)
            ax1 = fig.add_subplot(gs[0:3, 0:2])
            ax2 = fig.add_subplot(gs[0:3, 2])
            ax3 = fig.add_subplot(gs[0:3, 4:])
            ax4 = fig.add_subplot(gs[3:6, :])
            ax5 = fig.add_subplot(gs[6:, :])
            # turn off ax5, to make room for legend:
            ax5.axis('off')

            # limit to just target wells:
            #if site not in wls['grp_id'].values:
            #    continue
            uobs = gwobs.loc[gwobs.id == site, :].copy()
            uobs.sort_values(by='datetime', inplace=True)
    
            k, i, j = uobs.k.values[0], uobs.i.values[0], uobs.j.values[0]
            oobs = uobs.loc[uobs.observed == True, :]
            wobs = oobs.loc[oobs.weight > 0, :]
            #if wobs.shape[0] == 0:
            #    continue
            dts = uobs.datetime.values

            ax3.set_xticklabels('')
            ax3.set_yticklabels('')
            ax3.tick_params(axis='both', which='both', direction='in')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.tick_params(axis='both', which='both', length=0)

            plot_inset(ax=ax1, wl_loc=site, cpts=cpts, aq_extent=aq_extent, drains=drains, asr_loc=asr_grd_loc)
            wpp.plot_vert_xsec(ax2, m_d, m, wl_loc=site, mwl=wobs.obsval.mean(), cpts=cpts)
            
            site_info = cpts.loc[cpts.grpid == site, :].copy()
            mod_top = site_info['lse_navd88'].values[0]
            top_screen = site_info['top_screen'].values[0]
            bot_screen = site_info['bot_screen'].values[0]
            top_scr_elev = mod_top - top_screen
            bot_scr_elev = mod_top - bot_screen
            
            stor_ymx = []
            stor_ymn = []
            cnt = 0
            lw = 1.8
            zo = 0
            for ws in pred_ws_list:
                scn_tag = ws.split('_')[2]
                scn_num = int(scn_tag.replace('scn',''))
                
                # load result from ws:
                vals = scn_results_dict[scn_tag].loc[:, uobs.obsnme].values
                if not plt_base_only:
                    if plot_quantiles:
                        # plot quantile envelopes:
                        q10 = np.percentile(vals, 10, axis=0)
                        q90 = np.percentile(vals, 90, axis=0)

                        ax4.fill_between(dts, q10, q90, color=colors[scn_num-1], alpha=0.2, zorder=1)
                        ax4.plot(dts, q10, color=colors[scn_num-1], linestyle='--', lw=1.0, zorder=5)
                        ax4.plot(dts, q90, color=colors[scn_num-1], linestyle='--', lw=1.0, zorder=5)
                        min_q90 = q90.min()
                        ult_mn = min_q90 -10
                    else:
                        [ax4.plot(dts, vals[i, :], color=colors[scn_num-1], alpha=0.5, lw=0.1,zorder=3) for i in range(vals.shape[0])]
                        min_post_vals = vals.min(axis=0)
                        ult_mn = min_post_vals.min() -10
                else:
                    ult_mn = vals.min() -10   
                # plot base:
                min_val = vals.min()
                max_val = vals.max()
                base_vals = scn_results_dict[scn_tag].loc[:, uobs.obsnme].loc['base', :].values
                ax4.plot(dts, base_vals, color=darker_colors[scn_num-1], lw=1.5, zorder=10)

                # plot warnign period:
                below = np.array(base_vals) < top_scr_elev
                ax4.fill_between(
                    dts,
                    0, 1,  # from bottom to top of axes (in axes coords)
                    where=below,
                    transform=ax4.get_xaxis_transform(),  # x in data coords, y in axes coords
                    alpha=0.2,
                    facecolor="orange",                   # or colors[scn_num-1] if you prefer
                    edgecolor=None,   # <-- edge color per scenario
                    zorder=0
                )
                ax4.fill_between(
                    dts,
                    0, 1,  # from bottom to top of axes (in axes coords)
                    where=below,
                    transform=ax4.get_xaxis_transform(),  
                    facecolor='none',                
                    edgecolor=darker_colors[scn_num-1],   # <-- edge color per scenario
                    linewidth=lw,
                    zorder=zo
                )
                
                lw = lw - 0.3
                zo+=1
          
                mx_obs = np.max(base_vals)
                mn_obs = np.min(base_vals)

                midpoint = np.median([mx_obs, mn_obs])
                ult_mx = midpoint + 20
                #ult_mn = midpoint - 30

                stor_ymx.append(ult_mx)
                stor_ymn.append(ult_mn)

                ax4.set_ylim(min(stor_ymn) - 20, max(stor_ymx) + 20)
                ax4.yaxis.set_major_locator(ticker.MultipleLocator(60))  # Major ticks every 50
                ax4.yaxis.set_minor_locator(ticker.MultipleLocator(20))  # Minor ticks every 10
                                # reset xlim to focus on predictive period:
                if zoom_predict:
                    ax4.set_xlim(pd.to_datetime('2020-01-01'), pd.to_datetime('2045-01-01'))
                else:
                    ax4.set_xlim(pd.to_datetime('1970-01-01'), pd.to_datetime('2045-01-01'))

                if cnt == len(pred_ws_list) - 1:
                    t = top[int(i), int(j)]
                    bslice = botm[:, int(i), int(j)]
                    xlim = ax4.get_xlim()
                    xg = np.linspace(xlim[0], xlim[1], 500)

                    # amplitude of "grass blades" 
                    amp = 0.9  # increase if you want taller blades
                    # freq (controls how many blades appear)
                    freq = 180 
                    yg = t + amp * signal.sawtooth(2 * np.pi * freq * (xg - xlim[0]) / (xlim[1] - xlim[0]), width=0.5)
                    # plot the "grass" line that is top of model
                    ax4.plot(xg, yg, color='green', lw=1.2)
                    #ax4.plot(xg, np.ones_like(xg)*t, color='green', lw=1.2)
                    ax4.fill_between(xg, min(yg), yg, where=yg>=min(yg), color='green', alpha=0.4)
                    
                    # plot bottom of model layers:
                    for b in bslice:
                        ax4.plot(xlim, [b, b], 'c--', lw=1.5, alpha=0.5)


                ax4.xaxis.set_major_locator(years10)
                ax4.xaxis.set_major_formatter(years_fmt)
                ax4.get_xaxis().set_tick_params(direction='in')
                ax4.tick_params(axis='both',direction='in', which='major', labelsize=11)
                ax4.set_ylabel('Water level (ft-ASL)', fontsize=12)
                ax4.tick_params(axis='both', which='major', labelsize=11)
                # comma formateed y-axis labels:
                ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                # add vertical line for predictive period:
                ax4.axvline(x=pd.to_datetime('2025-01-01'), color='grey', linestyle='-.', linewidth=1)
                if cnt == len(pred_ws_list) - 1:
                    ax4.text(pd.to_datetime('2025-01-01'), ax4.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                            ha='left', va='bottom', color='grey')

                # --- PROXY HANDLES (make sure styles match your plots) ---
                leg_s1  = mlines.Line2D([], [], color=darker_colors[0], lw=1.5, label='Scenario 1 - Baseline')
                leg_s2  = mlines.Line2D([], [], color=darker_colors[1], lw=1.5, label='Scenario 2 - Full permit use')
                leg_s3  = mlines.Line2D([], [], color=darker_colors[2], lw=1.5, label='Scenario 3 - Full permit use with drought')
                leg_s4  = mlines.Line2D([], [], color=darker_colors[3], lw=1.5, label='Scenario 4 - Drought with ASR (500 acft/yr)')
                #leg_s5  = mlines.Line2D([], [], color=darker_colors[4], lw=1.5, label='Scenario 5 - Drought with ASR (200 acft/yr)')
                leg_s6  = mlines.Line2D([], [], color=darker_colors[4], lw=1.5, label='Scenario 6 - COW centered ASR (500 acft/yr)')
                leg_warning = mpatches.Patch(facecolor='orange', edgecolor=None, alpha=0.4, label='Periods below top of screened interval')
                # grass/top-of-model line
                leg_top = mlines.Line2D([], [], color='green', lw=1.2, label='Model top (ground surface)')
                # layer contacts (match your plotted style: you used 'c--')
                leg_cnt = mlines.Line2D([], [], color='c', lw=1.5, linestyle='--', alpha=0.5, label='Layer contacts')
                # predictive period (optional: include it if you want)
                leg_pred = mlines.Line2D([], [], color='grey', lw=1.0, linestyle='-.', label='Predictive period')

                # --- SPLIT INTO TWO LEGENDS ---
                scenario_handles = [leg_s1, leg_s2, leg_s3, leg_s4, leg_s6, leg_warning]
                other_handles    = [leg_top, leg_cnt, leg_pred]
                
                # Legend A: scenarios (two columns), centered at bottom
                legA = fig.legend(
                    handles=scenario_handles,
                    loc='lower center',
                    bbox_to_anchor=(0.61, 0.11),  # x=center, y a bit above bottom edge
                    ncol=2,
                    frameon=True,
                    framealpha=0.6,
                    fontsize=9,
                )

                # Legend B: other items (single row), centered just above Legend A
                legB = fig.legend(
                    handles=other_handles,
                    loc='lower center',
                    bbox_to_anchor=(0.21, 0.11),  # slightly higher than legA
                    ncol=1,                      # lay these out in one row; adjust if you prefer two rows
                    frameon=True,
                    framealpha=0.6,
                    fontsize=9,
                )
                
                if cnt == len(pred_ws_list) - 1:
                    grp_num = int(site.split('_')[0].split(':')[1])
                    k = int(site.split('_')[1].split(':')[1])
                    sites_grp = wls[wls['group number'] == int(grp_num)]
                    grp_full = sites_grp.copy()
                    idx_well = grp_full['index_well_flag'] == 1
        
                    sites_grp = sites_grp[sites_grp['manually_corrected_lay']-1 == int(k)]
                    aq_key = {0: "Wahpeton Shallow Sands-1",1: "Wahpeton Shallow Sands-2",
                                        2: "Wahpeton Shallow Plain-3",3: "Confing Unit",
                                        4: "Wahpeton Buried Valley",5: "Deep Clay",6: "Wild Rice"}
                    current_aq = aq_key.get(k)

                    # sort grp_full by manually_corrected_lay
                    grp_full = grp_full.sort_values(by=['manually_corrected_lay'])
                    if idx_well.any():
                        ax3.text(0.5, 0.8,
                                f'{current_aq}\n Group: {grp_num}\n ***Index Well***\n Loc ID:{site_info.loc_id.values[0]}\n Layer: {k + 1}, \nRow: {int(i) + 1}, Column: {int(j) + 1}\nModel top: {t}\n\n',
                                fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
                    else:
                        ax3.text(0.5, 0.75,
                                f'{current_aq}\n Group: {grp_num}\n Loc ID:{site_info.loc_id.values[0]}\n Layer: {k + 1}, \nRow: {int(i) + 1}, Column: {int(j) + 1}\nModel top: {t}\n\n',
                                fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
                        
                    # add text that icludes grp_full [loc_id, assigned aquifer]:
                    ax3.text(0.5, 0.27, 'All wells in group:\n' + '\n'.join(
                        [f"{row['loc_id']} - {aq_key.get(row['manually_corrected_lay']-1)}" for idx, row in grp_full.iterrows()]),
                            fontsize=9, ha='center', va='center', color='black', transform=ax3.transAxes)
                cnt+=1
            pdf.savefig()
            plt.close(fig)
            print(f'  finished plotting hydrograph for site {site}...')

    pdf.close()

def plot_scn_hydrographs_elk(
    pred_ws_list,
    modnm="elk_2lay",
    plot_quantiles=True,
    plt_base_only=False,
    zoom_predict=False,
    qlo=10,
    qhi=90,
    out_pdf=None,
):
    """
    Elk scenario hydrographs (updated):

    - Y-axis limits include quantile envelopes (qlo/qhi) + base lines.
    - Always plots model top (tan dashed line).
    - Plots layer contacts (botm slices) as dashed lines (no special layer-1 bottom line).
    - Legend lives outside plot area (bottom strip axis), similar to your Wahpeton layout.
    - Vertical markers:
        * 2000-01-01: annual -> monthly transition
        * 2024-01-01: predictive period start
    """
    import os
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    import pyemu
    import flopy

    import elk04_process_plot_results as wpp

    # ---- scenario label mapping (edit if needed) ----
    SCN_NAME_MAP = {1: "Baseline", 2: "Drought"}

    # vertical markers
    DT_MONTHLY_START = pd.to_datetime("2000-01-01")
    DT_PRED_START = pd.to_datetime("2024-01-01")

    # colors (keep your palette)
    colors = ["#b7bbc0", "#f1a924", "#9e1818", "#5a812c", "#49b6e9", "#e723e7"]
    darker_colors = ["#7f8286", "#ee6c15", "#e23232", "#81e70c", "#151792", "#990e99"]

    years10 = mdates.YearLocator(10)
    years_fmt = mdates.DateFormatter("%Y")

    def _scenario_label(ws: str) -> str:
        low = ws.lower()
        if "drought" in low:
            return "Drought"
        scn_tag = ws.split("_")[2] if len(ws.split("_")) >= 3 else ws
        try:
            scn_num = int(str(scn_tag).replace("scn", ""))
        except Exception:
            scn_num = None
        return SCN_NAME_MAP.get(scn_num, scn_tag)

    def _scenario_num(ws: str) -> int:
        scn_tag = ws.split("_")[2] if len(ws.split("_")) >= 3 else ws
        try:
            return int(str(scn_tag).replace("scn", ""))
        except Exception:
            return 1

    def _lower_cols(df):
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    def _pick_col(df, candidates, required=True, label="column"):
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            c0 = cand.lower()
            if c0 in cols:
                return cols[c0]
        if required:
            raise KeyError(f"Missing {label}. Tried: {candidates}. Available: {list(df.columns)}")
        return None

    # output
    o_d = os.path.join("results", "figures", "scenario_results")
    os.makedirs(o_d, exist_ok=True)
    if out_pdf is None:
        out_pdf = os.path.join(o_d, "scenario_hydrographs.pdf")

    # ---- init from first scenario workspace ----
    m_d0 = pred_ws_list[0]
    pst0 = pyemu.Pst(os.path.join(m_d0, f"{modnm}.pst"))
    obs = pst0.observation_data.copy()

    # transient head obs to build site list + k/i/j/dts
    gwobs = obs.loc[obs.obgnme.astype(str).str.contains("trans"), :].copy()
    gwobs["datetime"] = pd.to_datetime(gwobs["datetime"], errors="coerce")
    gwobs.loc[gwobs.obgnme.astype(str).str.contains("trans"), "id"] = (
        gwobs.loc[gwobs.obgnme.astype(str).str.contains("trans"), "obsnme"]
        .astype(str)
        .apply(lambda x: x.split("transh_")[1].split("_i")[0])
    )

    # restrict to sites actually used (weight>0)
    obs_adj = pd.read_csv(os.path.join(m_d0, f"{modnm}.adjusted.obs_data.csv"))
    obs_adj = _lower_cols(obs_adj)
    name_col = _pick_col(obs_adj, ["name"], True, "adjusted obs name column")
    wgt_col = _pick_col(obs_adj, ["weight"], True, "adjusted obs weight column")

    weighted = obs_adj.loc[obs_adj[wgt_col].astype(float) > 0, :].copy()
    weighted = weighted.loc[weighted[name_col].astype(str).str.contains("grpid", na=False)].copy()
    weighted["id"] = weighted[name_col].astype(str).apply(
        lambda x: "grpid:" + x.split("grpid:")[1].split("_k")[0] + "_k:" + x.split("_k:")[1].split("_")[0]
    )
    gwobs = gwobs.loc[gwobs["id"].isin(weighted["id"].unique()), :].copy()

    # load model for top/botm
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d0, exe_name="mf6", load_only=["dis"])
    m = sim.get_model(modnm)
    for col in ["k", "i", "j"]:
        gwobs[col] = gwobs[col].astype(int)

    top = m.dis.top.array
    botm = m.dis.botm.array

    # order sites by total weight
    usites = np.array(sorted(gwobs["id"].unique()))
    usitedf = pd.DataFrame({"site": usites})
    usitedf["weight_sum"] = usitedf["site"].apply(lambda s: float(gwobs.loc[gwobs["id"] == s, "weight"].sum()))
    usitedf = usitedf.sort_values(["weight_sum", "site"], ascending=[False, True])
    usites = usitedf["site"].values

    # ---- load cpts for inset + xsec ----
    cpts = pd.read_csv(os.path.join("data", "analyzed", "transient_well_targets_lookup.csv"))
    cpts = _lower_cols(cpts)

    grpcol = _pick_col(cpts, ["group_number", "group number", "groupnum", "grpnum", "group"], True, "group column")
    laycol = _pick_col(cpts, ["k", "model_layer", "layer"], True, "layer column")
    xcol = _pick_col(cpts, ["x_2265", "x"], True, "x coord column")
    ycol = _pick_col(cpts, ["y_2265", "y"], True, "y coord column")

    layvals = pd.to_numeric(cpts[laycol], errors="coerce")
    if laycol.lower() == "model_layer" or (layvals.dropna().min() >= 1 and layvals.dropna().max() <= 10):
        k0 = (layvals.astype(int) - 1).astype(int)
    else:
        k0 = layvals.astype(int)

    cpts["grpid"] = "grpid:" + cpts[grpcol].astype(int).astype(str) + "_k:" + k0.astype(int).astype(str)
    cpts = gpd.GeoDataFrame(cpts, geometry=gpd.points_from_xy(cpts[xcol], cpts[ycol]), crs=2265)
    cpts = cpts.groupby(["grpid"]).last().reset_index()

    # fallback wls table for location IDs (optional text panel)
    wls = pd.read_csv(os.path.join("data", "analyzed", "transient_well_targets_lookup.csv"))
    wls = _lower_cols(wls)
    if "group_number" not in wls.columns and "group number" in wls.columns:
        wls = wls.rename(columns={"group number": "group_number"})

    # GIS layers
    g_d = os.path.join("..", "..", "gis", "input_shps", "elk")
    aq_extent = None
    drains = None
    try:
        aq_extent = gpd.read_file(os.path.join(g_d, "elk_boundary.shp"))
    except Exception:
        pass
    try:
        drains = gpd.read_file(os.path.join(g_d, "drn_raw.shp"))
    except Exception:
        pass

    # ---- load scenario ensembles ----
    scn_results = {}
    pst_for_jcb = pyemu.Pst(os.path.join(pred_ws_list[0], f"{modnm}.pst"))
    for ws in pred_ws_list:
        jcb_name = os.path.join(ws, f"{modnm}.0.obs.jcb")
        assert os.path.exists(jcb_name), f"obs jcb {jcb_name} not found!"
        scn_tag = ws.split("_")[2]
        scn_results[scn_tag] = pyemu.ObservationEnsemble.from_binary(pst=pst_for_jcb, filename=jcb_name)

    # ---- helper: compute y-lims from quantiles + base lines ----
    def _compute_ylim_from_quantiles_and_base(dts, series_list, pad=10.0):
        """
        series_list: list of dicts with keys:
          - qlo (array) optional
          - qhi (array) optional
          - base (array) required
        Returns (ymin, ymax)
        """
        vals = []
        for s in series_list:
            if "base" in s and s["base"] is not None:
                vals.append(np.asarray(s["base"], dtype=float))
            if "qlo" in s and s["qlo"] is not None:
                vals.append(np.asarray(s["qlo"], dtype=float))
            if "qhi" in s and s["qhi"] is not None:
                vals.append(np.asarray(s["qhi"], dtype=float))
        if not vals:
            return None, None
        y = np.concatenate([v[np.isfinite(v)] for v in vals if v is not None and np.isfinite(v).any()])
        if y.size == 0:
            return None, None
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        # add padding
        if ymax > ymin:
            pad2 = max(pad, 0.10 * (ymax - ymin))
        else:
            pad2 = pad
        return ymin - pad2, ymax + pad2

    # ---- plot ----
    with PdfPages(out_pdf) as pdf:
        for site in usites:
            uobs = gwobs.loc[gwobs["id"] == site, :].copy().sort_values("datetime")
            if uobs.empty:
                continue

            dts = pd.to_datetime(uobs["datetime"].values)
            if dts.size == 0:
                continue

            k = int(uobs["k"].iloc[0])
            i = int(uobs["i"].iloc[0])
            j = int(uobs["j"].iloc[0])

            fig = plt.figure(figsize=(11, 8))
            gs = gridspec.GridSpec(7, 6)
            ax1 = fig.add_subplot(gs[0:3, 0:2])   # inset
            ax2 = fig.add_subplot(gs[0:3, 2])     # xsec
            ax3 = fig.add_subplot(gs[0:3, 4:])    # text
            ax4 = fig.add_subplot(gs[3:6, :])     # hydrograph
            ax5 = fig.add_subplot(gs[6:, :])      # legend strip
            ax5.axis("off")

            # clean ax3
            ax3.set_xticklabels("")
            ax3.set_yticklabels("")
            ax3.tick_params(axis="both", which="both", direction="in", length=0)
            for sp in ["top", "right", "left", "bottom"]:
                ax3.spines[sp].set_visible(False)

            # inset + xsec
            try:
                plot_inset(ax=ax1, wl_loc=site, cpts=cpts, aq_extent=aq_extent, drains=drains, wls=wls)
            except Exception as e:
                ax1.text(0.5, 0.5, f"Inset skipped:\n{site}\n{e}",
                         ha="center", va="center", transform=ax1.transAxes, fontsize=8, color="red")
                ax1.set_xticks([]); ax1.set_yticks([])

            try:
                wpp.plot_vert_xsec(ax2, m_d0, m, wl_loc=site, mwl=np.nan, cpts=cpts)
            except Exception as e:
                ax2.text(0.5, 0.5, f"Xsec skipped:\n{e}",
                         ha="center", va="center", transform=ax2.transAxes, fontsize=8, color="red")
                ax2.set_xticks([]); ax2.set_yticks([])

            # plot scenarios (base + quantile envelope)
            series_for_ylim = []
            scenario_handles = []

            for ws in pred_ws_list:
                scn_tag = ws.split("_")[2]
                scn_num = _scenario_num(ws)
                scn_num_c = max(1, min(scn_num, len(colors)))
                label = _scenario_label(ws)

                vals = scn_results[scn_tag].loc[:, uobs["obsnme"]].values

                q_lo_arr = None
                q_hi_arr = None

                if (not plt_base_only) and plot_quantiles:
                    q_lo_arr = np.nanpercentile(vals, qlo, axis=0)
                    q_hi_arr = np.nanpercentile(vals, qhi, axis=0)
                    ax4.fill_between(dts, q_lo_arr, q_hi_arr, color=colors[scn_num_c - 1], alpha=0.2, zorder=1)
                    ax4.plot(dts, q_lo_arr, color=colors[scn_num_c - 1], linestyle="--", lw=1.0, zorder=4)
                    ax4.plot(dts, q_hi_arr, color=colors[scn_num_c - 1], linestyle="--", lw=1.0, zorder=4)

                base_vals = scn_results[scn_tag].loc[:, uobs["obsnme"]].loc["base", :].values
                ax4.plot(dts, base_vals, color=darker_colors[scn_num_c - 1], lw=1.8, zorder=10)

                series_for_ylim.append({"base": base_vals, "qlo": q_lo_arr, "qhi": q_hi_arr})

                scenario_handles.append(
                    mlines.Line2D([], [], color=darker_colors[scn_num_c - 1], lw=2.2, label=label)
                )

            # x-lims
            if zoom_predict:
                ax4.set_xlim(pd.to_datetime("2020-01-01"), pd.to_datetime("2045-01-01"))
            else:
                ax4.set_xlim(pd.to_datetime("1965-01-01"), pd.to_datetime("2045-01-01"))

            # y-lims from base + percentiles (this is the main fix you wanted)
            y0, y1 = _compute_ylim_from_quantiles_and_base(dts, series_for_ylim, pad=1.0)
            if y0 is not None:
                ax4.set_ylim(y0, y1)

            # vertical lines
            ax4.axvline(DT_MONTHLY_START, color="0.4", linestyle=":", lw=1.4, zorder=20)
            ax4.axvline(DT_PRED_START, color="0.4", linestyle="-.", lw=1.6, zorder=20)

            # draw model top + layer contacts AFTER ylims are set
            # (fixes “model top not plotting in the right place” issues)
            tcell = float(top[i, j])
            xlim = ax4.get_xlim()
            # model top always
            ax4.plot(xlim, [tcell, tcell], linestyle="--", color="tan", lw=1.5, zorder=15)

            # layer contacts (all botm slices at i,j)
            bslice = botm[:, i, j]
            for b in bslice:
                ax4.plot(xlim, [float(b), float(b)], linestyle="--", color="c", lw=1.5, alpha=0.5, zorder=14)

            ax4.set_xlim(xlim)

            # axes formatting
            ax4.xaxis.set_major_locator(years10)
            ax4.xaxis.set_major_formatter(years_fmt)
            ax4.tick_params(axis="both", direction="in", which="major", labelsize=11)
            ax4.set_ylabel("Water level (ft-ASL)", fontsize=12)
            ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

            # ---- legend outside plot (ax5) ----
            h_top = mlines.Line2D([], [], color="tan", linestyle="--", lw=1.5, label="Model top")
            h_cnt = mlines.Line2D([], [], color="c", linestyle="--", lw=1.5, alpha=0.5, label="Layer contacts")
            h_pred = mlines.Line2D([], [], color="0.4", linestyle="-.", lw=1.6, label="Predictive period (starts 1/1/2024)")
            h_month = mlines.Line2D([], [], color="0.4", linestyle=":", lw=1.4, label="Monthly period (starts 1/1/2000)")
            h_env = mpatches.Patch(facecolor="0.6", alpha=0.25, label=f"Uncertainty envelope (P{qlo}–P{qhi})")

            # de-duplicate scenario handles
            seen = set()
            scen_unique = []
            for h in scenario_handles:
                if h.get_label() not in seen:
                    scen_unique.append(h)
                    seen.add(h.get_label())

            # LEFT legend (reference items)
            leg_left = ax5.legend(
                handles=[h_top, h_cnt, h_pred, h_month],
                loc="center left",
                bbox_to_anchor=(0.01, 0.5),
                ncol=1,
                frameon=True,
                framealpha=0.6,
                fontsize=9,
            )
            ax5.add_artist(leg_left)

            # RIGHT legend (scenarios + envelope)
            leg_right = ax5.legend(
                handles=scen_unique + [h_env],
                loc="center right",
                bbox_to_anchor=(0.99, 0.5),
                ncol=1,
                frameon=True,
                framealpha=0.6,
                fontsize=9,
            )


            # text panel
            grp_num = int(site.split("_")[0].split(":")[1])
            lay0 = int(site.split("_")[1].split(":")[1])
            aq_key = {0: "Soils/Clay/Silt", 1: "Elk Valley Aquifer"}
            current_aq = aq_key.get(lay0, f"Layer {lay0+1}")

            loc_id_str = "NA"
            if ("group_number" in wls.columns) and ("model_layer" in wls.columns) and ("location" in wls.columns):
                rows = wls[(wls["group_number"].astype(int) == grp_num) & ((wls["model_layer"].astype(int) - 1) == lay0)]
                if not rows.empty:
                    loc_id_str = str(rows["location"].iloc[0])

            ax3.text(
                0.5, 0.75,
                f"{current_aq}\nGroup: {grp_num}\nLoc: {loc_id_str}\nLayer: {lay0 + 1}\nRow: {int(i) + 1}, Col: {int(j) + 1}\nModel top: {tcell:.2f}\n",
                fontsize=12, ha="center", va="center", color="blue", transform=ax3.transAxes
            )

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote: {out_pdf}")


def plot_rowbin_total_pumping_by_scenario_mimic_mel(
    pred_ws_list,
    modnm="elk_2lay",
    wel_pkg_name="wel",
    row_bin_size=50,
    cfd_to_gpm=1.0 / 192.5,
    convert_to_positive_withdrawal=False,  # keep MF6 sign by default (negative=pumping)
    out_pdf=None,
    dpi=300,
    progress_kper_step=25,
    zoom_predict=False,
    add_inset_map=True,
    add_recharge_annual=True,   # overlays ANNUAL-AVERAGE recharge on SAME axis
):
    """
    Row-bin pumping totals (WEL) with Baseline vs Drought + annual-average overlays,
    plus optional annual-average recharge overlay computed FAST by reading the
    full recharge array ONCE (works well with external arrays).

    Pumping styling:
      - Baseline: thin black solid (SP series) + thick semi-opaque annual avg
      - Drought : thin red dashed (SP series) + thick semi-opaque annual avg

    Recharge styling (same axis):
      - Baseline recharge annual avg: thick semi-opaque black dotted
      - Drought  recharge annual avg: thick semi-opaque red dotted

    Notes:
      - Dates are END of stress period, built from TDIS.
      - Pumping is summed from WEL SPD (gpm).
      - Recharge is summed volumetrically from RCH (ft/d * area -> ft^3/d -> gpm).
    """
    import os
    import re
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import flopy

    # -------------------------
    # helpers
    # -------------------------
    def _scenario_label(ws: str) -> str:
        low = ws.lower()
        if "drought" in low:
            return "Drought"
        if "baseline" in low:
            return "Baseline"
        return os.path.basename(ws)

    def _scenario_key(ws: str) -> str:
        lab = _scenario_label(ws).lower()
        if "drought" in lab:
            return "drought"
        if "baseline" in lab:
            return "baseline"
        return lab

    def _build_sp_end_dates(sim) -> np.ndarray:
        tdis = sim.get_package("tdis")
        raw = (
            tdis.start_date_time.get_data()
            if hasattr(tdis.start_date_time, "get_data")
            else tdis.start_date_time
        )
        m = re.search(r"\d{4}-\d{2}-\d{2}", str(raw))
        if not m:
            raise ValueError(f"Cannot parse start_date_time from TDIS: {raw!r}")
        start_date = pd.to_datetime(m.group(0))

        perdata = tdis.perioddata.array
        perlen = np.atleast_1d(perdata["perlen"]).astype(float)
        nper = perlen.size

        sp_dates = [start_date]
        for i in range(1, nper):
            sp_dates.append(sp_dates[-1] + pd.Timedelta(days=float(perlen[i - 1])))

        sp_end_dates = np.array(
            [d + pd.Timedelta(days=float(pl)) for d, pl in zip(sp_dates, perlen)],
            dtype="datetime64[ns]",
        )
        return sp_end_dates

    def _detect_rate_field(wel, nper):
        rate_field = None
        for kper in range(nper):
            try:
                arr = wel.stress_period_data.get_data(kper)
            except Exception:
                arr = None
            if arr is not None and len(arr) > 0 and hasattr(arr, "dtype") and arr.dtype.names:
                names = {n.lower(): n for n in arr.dtype.names}
                for cand in ("q", "rate", "flux"):
                    if cand in names:
                        rate_field = names[cand]
                        break
            if rate_field is not None:
                break
        if rate_field is None:
            raise ValueError("Cannot find q/rate/flux field in WEL SPD.")
        return rate_field

    def _annual_average(dates, values):
        s = pd.Series(values, index=pd.to_datetime(dates))
        annual = s.resample("YS").mean()
        return annual.index.to_numpy(), annual.values

    def _get_active_mask_from_dis(ws_clean: str):
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=ws_clean,
            exe_name="mf6",
            load_only=["dis"],
            verbosity_level=0,
        )
        gwf = sim.get_model(modnm)
        dis = gwf.dis
        nrow = int(dis.nrow.data)
        ncol = int(dis.ncol.data)

        active = None
        try:
            idomain = dis.idomain.array
            if idomain is not None:
                active = np.any(idomain > 0, axis=0) if np.asarray(idomain).ndim == 3 else (np.asarray(idomain) > 0)
        except Exception:
            active = None

        if active is None:
            active = np.ones((nrow, ncol), dtype=bool)

        return active.astype(bool), nrow, ncol

    def _draw_inset(ax, active_mask, r0, r1, nrow, ncol):
        axins = inset_axes(ax, width="22%", height="35%", loc="upper left", borderpad=1.0)
        img = np.zeros((nrow, ncol), dtype=float)
        img[active_mask] = 1.0
        axins.imshow(img, origin="upper", interpolation="nearest")

        rect = Rectangle(
            (-0.5, r0 - 0.5),
            ncol,
            (r1 - r0 + 1),
            fill=False,
            linewidth=1.5,
            edgecolor="red",
        )
        axins.add_patch(rect)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title(f"Rows {r0+1}-{r1+1}", fontsize=8)
        for sp in axins.spines.values():
            sp.set_linewidth(0.8)

    def _get_rch_3d_ftd(gwf) -> np.ndarray:
        if not hasattr(gwf, "rch"):
            raise ValueError("Model has no RCH package (gwf.rch not found).")

        rch = gwf.rch
        full = None
        if hasattr(rch, "recharge") and hasattr(rch.recharge, "array"):
            full = rch.recharge.array
        elif hasattr(rch, "rech") and hasattr(rch.rech, "array"):
            full = rch.rech.array
        else:
            raise ValueError("RCH package does not have recharge/rech arrays.")

        full = np.asarray(full, dtype=float)

        if full.ndim == 2:
            full = full[None, ...]
        elif full.ndim == 3:
            pass
        elif full.ndim == 4:
            full = full.sum(axis=1)
        else:
            raise ValueError(f"Unsupported RCH shape: {full.shape}")

        return full  # (nper,nrow,ncol) ft/day

    def _load_rowbin_totals(ws_clean: str):
        t0 = time.perf_counter()
        print(f"\n[INFO] Loading MF6: {ws_clean}")
        sim = flopy.mf6.MFSimulation.load(sim_ws=ws_clean, exe_name="mf6")
        gwf = sim.get_model(modnm)
        dis = gwf.dis

        nrow = int(dis.nrow.data)
        ncol = int(dis.ncol.data)

        delr = np.asarray(dis.delr.array, dtype=float)
        delc = np.asarray(dis.delc.array, dtype=float)
        area2d = delc[:, None] * delr[None, :]

        active2d = None
        try:
            idomain = dis.idomain.array
            if idomain is not None:
                idomain = np.asarray(idomain)
                if idomain.ndim == 3:
                    active2d = np.any(idomain > 0, axis=0)
                elif idomain.ndim == 2:
                    active2d = idomain > 0
        except Exception:
            active2d = None
        if active2d is None:
            active2d = np.ones((nrow, ncol), dtype=bool)

        sp_end_dates = _build_sp_end_dates(sim)
        nper = sp_end_dates.size

        nbins = int(np.ceil(nrow / row_bin_size))
        bins = [(b * row_bin_size, min(nrow - 1, (b + 1) * row_bin_size - 1)) for b in range(nbins)]
        pump = np.zeros((nbins, nper), dtype=float)

        wel = gwf.get_package(wel_pkg_name)
        if wel is None:
            raise ValueError(f"No WEL package named '{wel_pkg_name}' in workspace {ws_clean}")
        rate_field = _detect_rate_field(wel, nper)
        print(f"[INFO] WEL rate field: {rate_field}")

        rch_tot = None
        if add_recharge_annual:
            print("[INFO] Reading full RCH array once (fast path for external arrays)...")
            rch3d_ftd = _get_rch_3d_ftd(gwf)
            if rch3d_ftd.shape[0] != nper:
                raise ValueError(f"RCH nper mismatch: rch={rch3d_ftd.shape[0]} vs tdis={nper}")
            if rch3d_ftd.shape[1:] != (nrow, ncol):
                raise ValueError(f"RCH grid mismatch: rch={rch3d_ftd.shape[1:]} vs dis={(nrow,ncol)}")

            vol_cfd = rch3d_ftd * area2d[None, :, :]
            vol_cfd = np.where(active2d[None, :, :], vol_cfd, 0.0)
            vol_gpm = vol_cfd * cfd_to_gpm

            row_sum_gpm = np.nansum(vol_gpm, axis=2)  # (nper,nrow)
            starts = np.arange(0, nrow, row_bin_size, dtype=int)
            rch_binned = np.add.reduceat(row_sum_gpm, starts, axis=1)  # (nper,nbins)
            rch_tot = rch_binned.T  # (nbins,nper)

        print(f"[INFO] Accumulating pumping totals: nper={nper}, nbins={nbins}, bin={row_bin_size} rows")
        for kper in range(nper):
            if (kper % progress_kper_step) == 0:
                elapsed = time.perf_counter() - t0
                print(f"  kper {kper+1:>5}/{nper}  (elapsed {elapsed:,.1f}s)")

            try:
                arr = wel.stress_period_data.get_data(kper)
            except Exception:
                arr = None
            if arr is None or len(arr) == 0:
                continue

            if not (hasattr(arr, "dtype") and arr.dtype.names):
                raise TypeError(
                    f"WEL get_data(kper={kper}) returned {type(arr)} without dtype. "
                    f"Fields not available; cannot parse."
                )

            names = {n.lower(): n for n in arr.dtype.names}
            if "cellid" not in names:
                raise ValueError(f"No 'cellid' field found in WEL SPD. Fields={arr.dtype.names}")

            cellids = arr[names["cellid"]]
            q_cfd = arr[rate_field].astype(float)

            q_gpm = q_cfd * cfd_to_gpm
            if convert_to_positive_withdrawal:
                q_gpm = -q_gpm

            rows = np.fromiter((int(cid[1]) for cid in cellids), dtype=int, count=len(cellids))
            bidx = rows // row_bin_size
            pump[:, kper] = np.bincount(bidx, weights=q_gpm, minlength=nbins)

        print(f"[INFO] Done: {ws_clean}  (total elapsed {time.perf_counter()-t0:,.1f}s)")
        return sp_end_dates, bins, pump, rch_tot

    # -------------------------
    # output path
    # -------------------------
    o_d = os.path.join("results", "figures", "scenario_results")
    os.makedirs(o_d, exist_ok=True)
    if out_pdf is None:
        out_pdf = os.path.join(o_d, "elk_rowbin_pumping_baseline_vs_drought.pdf")

    clean_ws_list = []
    for ws in pred_ws_list:
        ws_clean = ws.replace("_ensemble", "_clean")
        if not os.path.exists(ws_clean):
            raise FileNotFoundError(f"Expected clean workspace not found: {ws_clean}")
        clean_ws_list.append(ws_clean)

    active_mask, nrow_map, ncol_map = (None, None, None)
    if add_inset_map:
        active_mask, nrow_map, ncol_map = _get_active_mask_from_dis(clean_ws_list[0])

    scen = {}
    for ws_clean in clean_ws_list:
        key = _scenario_key(ws_clean)
        lab = _scenario_label(ws_clean)
        print(f"\n=== Scenario: {lab} ({key}) ===")
        dates, bins, pump, rch_tot = _load_rowbin_totals(ws_clean)
        scen[key] = {"label": lab, "dates": dates, "bins": bins, "pump": pump, "rch": rch_tot}

    keys = list(scen.keys())
    if "baseline" in scen and "drought" in scen:
        k_base, k_dr = "baseline", "drought"
    elif len(keys) >= 2:
        k_base, k_dr = keys[0], keys[1]
    else:
        k_base, k_dr = keys[0], None

    # -------------------------
    # plot
    # -------------------------
    print(f"\n[INFO] Writing PDF: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        bins = scen[k_base]["bins"]
        nb = len(bins)

        for b, (r0, r1) in enumerate(bins):
            if (b % 5) == 0:
                print(f"  page {b+1}/{nb}")

            fig, ax = plt.subplots(figsize=(11, 6))

            if add_inset_map and active_mask is not None:
                _draw_inset(ax, active_mask, r0, r1, nrow_map, ncol_map)

            dates_b = scen[k_base]["dates"]
            pump_b = scen[k_base]["pump"][b, :]

            # baseline pumping (SP)
            ax.plot(dates_b, pump_b, color="black", linewidth=0.8, linestyle="-", label="Baseline pumping (SP)")
            # baseline pumping annual
            yr_b, avg_pump_b = _annual_average(dates_b, pump_b)
            ax.plot(yr_b, avg_pump_b, color="black", linewidth=2.5, alpha=0.6, linestyle="-",
                    label="Baseline pumping (annual avg)")

            if k_dr is not None:
                dates_d = scen[k_dr]["dates"]
                pump_d = scen[k_dr]["pump"][b, :]
                ax.plot(dates_d, pump_d, color="red", linewidth=0.8, linestyle="--", label="Drought pumping (SP)")
                yr_d, avg_pump_d = _annual_average(dates_d, pump_d)
                ax.plot(yr_d, avg_pump_d, color="red", linewidth=2.5, alpha=0.6, linestyle="--",
                        label="Drought pumping (annual avg)")

            # recharge annual averages ON SAME AXIS
            if add_recharge_annual and scen[k_base]["rch"] is not None:
                rch_b = scen[k_base]["rch"][b, :]
                yr_rb, avg_rch_b = _annual_average(dates_b, rch_b)
                ax.plot(
                    yr_rb,
                    avg_rch_b,
                    color="C0",
                    linewidth=2.0,
                    alpha=0.9,
                    linestyle=":",
                    label="Baseline recharge (annual avg)",
                )

                if k_dr is not None and scen[k_dr]["rch"] is not None:
                    rch_d = scen[k_dr]["rch"][b, :]
                    yr_rd, avg_rch_d = _annual_average(dates_d, rch_d)
                    ax.plot(
                        yr_rd,
                        avg_rch_d,
                        color="#f1a924",
                        linewidth=2.0,
                        alpha=0.9,
                        linestyle=":",
                        label="Drought recharge (annual avg)",
                    )

            if zoom_predict:
                ax.set_xlim(pd.to_datetime("2020-01-01"), pd.to_datetime("2045-01-01"))

            ax.set_title(f"Row-bin totals: rows {r0+1}–{r1+1} (bin {b+1}/{nb})", fontsize=13)
            ax.set_xlabel("Date (end of stress period)")
            ax.set_ylabel("Rate (gpm)", fontsize=12)
            ax.grid(alpha=0.25)

            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, loc: f"{v:,.0f}"))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.autofmt_xdate()

            ax.legend(loc="best", fontsize=8, framealpha=0.6)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    print(f"[DONE] Wrote: {out_pdf}")






    
def plot_scn_model_net_budget_annual_from_budobs_jcb_totals_nosto(
    pred_ws_list,
    modnm="elk_2lay",
    qlo=10,
    qhi=90,
    out_pdf=None,
    verbose=True,
):
    """
    Annual net budget (acre-ft/yr) from PEST obs ensemble (.0.obs.jcb),
    using *total_in/total_out* observations and subtracting storage terms.

    Per stress period (at each datetime):
        net_cfd = (total_in - sto-ss_in - sto-sy_in) - (total_out - sto-ss_out - sto-sy_out)
        net_af  = net_cfd * days_in_sp / 43560

    Annual:
        sum(net_af) by year

    Notes:
      - This avoids duplicate drn_out/drn_in name-collision issues, because totals already include them.
      - Adds a vertical line at predictive year = 2024.
      - Labels scn01 as "Baseline" and scn03 (or any folder containing 'drought') as "Drought" by default.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pyemu

    CUFT_PER_ACFT = 43560.0
    PREDICTIVE_YEAR = 2024

    if out_pdf is None:
        out_pdf = os.path.join(
            "results", "figures", "scenario_results", "model_net_budget_ANNUAL_noSTO.pdf"
        )
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    def _scenario_label(ws: str) -> str:
        """
        Map scenarios to requested names.
        - If folder name contains 'drought' -> Drought
        - Else if scn number == 1 -> Baseline
        - Else fallback: scnXX
        """
        low = ws.lower()
        if "drought" in low:
            return "Drought"
        scn_tag = ws.split("_")[2] if len(ws.split("_")) >= 3 else ws
        try:
            scn_num = int(str(scn_tag).replace("scn", ""))
        except Exception:
            scn_num = None
        if scn_num == 1:
            return "Baseline"
        return scn_tag

    def _parse_usecol_and_date(colname: str):
        """
        Expect obsnme format like:
          oname:bud_otype:lst_usecol:<USECOL>:<YYYY-MM-DD>
        Return (usecol_lower, dt_normalized) or (None, None).
        """
        s = str(colname)
        parts = s.split(":")
        if len(parts) < 5:
            return None, None
        usecol = parts[3].strip().lower()
        dt = pd.to_datetime(parts[-1], errors="coerce")
        if pd.isna(dt):
            return None, None
        return usecol, dt.normalize()

    # coefficients for required usecols
    # net = (total_in - sto_in) - (total_out - sto_out)
    REQ = {
        "total_in_datetime": +1.0,
        "total_out_datetime": -1.0,
        "sto-ss_in_datetime": -1.0,
        "sto-sy_in_datetime": -1.0,
        "sto-ss_out_datetime": +1.0,
        "sto-sy_out_datetime": +1.0,
    }

    # load pst once
    pst0 = pyemu.Pst(os.path.join(pred_ws_list[0], f"{modnm}.pst"))

    # Build meta (col->usecol, dt, coef) from first scenario ensemble
    first_ws = pred_ws_list[0]
    first_jcb = os.path.join(first_ws, f"{modnm}.0.obs.jcb")
    ens0 = pyemu.ObservationEnsemble.from_binary(pst=pst0, filename=first_jcb)

    rows = []
    for c in ens0.columns:
        usecol, dt = _parse_usecol_and_date(c)
        if usecol in REQ and dt is not None:
            rows.append((c, usecol, dt, REQ[usecol]))

    meta = pd.DataFrame(rows, columns=["col", "usecol", "dt", "coef"])
    if meta.empty:
        raise ValueError(
            "No required budget obs columns found in ensemble. "
            "Check obsnme format and REQ keys."
        )

    # build stress-period days from dt spacing
    dts = sorted(meta["dt"].unique())
    dts_ser = pd.Series(dts).sort_values().reset_index(drop=True)
    if len(dts_ser) == 1:
        days = pd.Series([365.25], index=dts_ser)
    else:
        dd = dts_ser.diff().dt.days.to_numpy(dtype=float)
        med = np.nanmedian(dd[1:]) if len(dd) > 1 else 365.25
        dd[0] = med if np.isfinite(med) else 365.25
        dd = np.clip(dd, 1, 366)
        days = pd.Series(dd, index=dts_ser)

    meta["days"] = meta["dt"].map(days).astype(float)
    meta["year"] = meta["dt"].dt.year.astype(int)
    years_sorted = sorted(meta["year"].unique())

    if verbose:
        print("Using jcb totals-minus-storage method.")
        print("Counts by usecol:\n", meta["usecol"].value_counts())

    # precompute per-column weights (ac-ft) per observation
    meta["w_acft"] = meta["coef"] * (meta["days"] / CUFT_PER_ACFT)

    # precompute column lists and weights by year
    cols_by_year = {}
    w_by_year = {}
    meta_idx = meta.set_index("col")
    for yr in years_sorted:
        cols_yr = meta.loc[meta["year"] == yr, "col"].tolist()
        cols_by_year[yr] = cols_yr
        w_by_year[yr] = meta_idx.loc[cols_yr, "w_acft"].to_numpy(dtype=float)

    scen_annual = {}  # label -> DataFrame(index=reals, columns=years)

    for ws in pred_ws_list:
        label = _scenario_label(ws)
        jcb = os.path.join(ws, f"{modnm}.0.obs.jcb")
        ens = pyemu.ObservationEnsemble.from_binary(pst=pst0, filename=jcb)

        # verify required columns exist
        missing_cols = [c for c in meta["col"].unique() if c not in ens.columns]
        if missing_cols:
            raise KeyError(
                f"{label}: ensemble missing {len(missing_cols)} required cols "
                f"(first 5): {missing_cols[:5]}"
            )

        out = pd.DataFrame(index=ens.index, columns=years_sorted, dtype=float)
        for yr in years_sorted:
            cols_yr = cols_by_year[yr]
            w = w_by_year[yr]
            X = ens.loc[:, cols_yr].to_numpy(dtype=float)
            out[yr] = X @ w

        scen_annual[label] = out

        if verbose and "base" in out.index:
            base = out.loc["base", :]
            print(f"[{label}] base annual AF min/max: {base.min():,.0f} / {base.max():,.0f}")

    # Plot
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

        # stable order: Baseline then Drought then others
        order = []
        for key in ["Baseline", "Drought"]:
            if key in scen_annual:
                order.append(key)
        for key in scen_annual:
            if key not in order:
                order.append(key)

        for label in order:
            df = scen_annual[label]
            vals = df[years_sorted].to_numpy(dtype=float)

            # uncertainty envelope
            qA = np.nanpercentile(vals, qlo, axis=0)
            qB = np.nanpercentile(vals, qhi, axis=0)
            ax.fill_between(years_sorted, qA, qB, alpha=0.25)

            # base realization (preferred)
            if "base" in df.index:
                ax.plot(years_sorted, df.loc["base", years_sorted].values, lw=2.0, label=label)
            else:
                ax.plot(years_sorted, np.nanmedian(vals, axis=0), lw=2.0, label=label)

        # zero line
        ax.axhline(0.0, color="0.3", lw=1.0)

        # predictive period line at year 2024
        ax.axvline(PREDICTIVE_YEAR, color="0.4", linestyle="-.", lw=1.5, zorder=10)
        ax.text(
            PREDICTIVE_YEAR,
            ax.get_ylim()[1] * 0.95,
            "Predictive period",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color="0.4",
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Annual net budget (acre-ft/yr)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.6)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote: {out_pdf}")


    
def elk_overwrite_wel_drought_only(
    pred_ws=".",
    pred_start="2025-01-01",
    drought_years=3,
    drought_source_year=2021,
):
    """
    Drought WEL overwrite ONLY (baseline pumping already baked into the base model build):

    - Identify predictive SPs as those with start_datetime >= pred_start
    - Take the first drought_years*12 predictive months as the drought window
    - For each drought-month SP, overwrite WEL stress files:
          <prefix>_stress_period_data_<sp>.txt
      using the file from drought_source_year matching month-of-year (Jan->Jan, etc.)
    - Writes both to pred_ws/ and pred_ws/org/

    Returns:
        modified_files (list[str]), drought_sps (list[int])
    """
    import os
    import re
    import shutil
    import numpy as np
    import pandas as pd

    spd = _read_monthly_sp_info()
    pred_start = pd.Timestamp(pred_start)

    pred = spd.loc[spd["start_datetime"] >= pred_start, ["stress_period", "month"]].copy()
    pred = pred.sort_values("stress_period")
    if pred.empty:
        raise ValueError(f"No predictive SPs found starting at/after {pred_start.date()}")

    drought_n = int(drought_years) * 12
    drought_pred = pred.iloc[:drought_n].copy()
    drought_sps = drought_pred["stress_period"].astype(int).tolist()

    src = spd.loc[spd["year"] == int(drought_source_year), ["stress_period", "month"]].copy()
    if src.empty:
        raise ValueError(f"No SPs found for drought_source_year={drought_source_year}")

    # month -> first SP in drought_source_year for that month
    month_to_srcsp = {}
    for m in range(1, 13):
        rows = src.loc[src["month"] == m, "stress_period"].astype(int).tolist()
        if not rows:
            raise ValueError(f"No SP found for drought_source_year={drought_source_year}, month={m}")
        month_to_srcsp[m] = rows[0]

    # --- discover wel stress files in pred_ws ---
    # Matches:
    #   wel_stress_period_data_157.txt
    #   wel_ag_stress_period_data_157.txt
    #   wel_muni_stress_period_data_157.txt
    pat = re.compile(r"^(?P<pfx>.+)_stress_period_data_(?P<sp>\d+)\.txt$", re.IGNORECASE)

    files = os.listdir(pred_ws)
    wel_map = {}  # prefix -> {sp:int -> filename}
    for fn in files:
        m = pat.match(fn)
        if not m:
            continue
        pfx = m.group("pfx")
        sp = int(m.group("sp"))
        wel_map.setdefault(pfx, {})[sp] = fn

    if not wel_map:
        # include a helpful hint by showing a couple filenames that start with 'wel'
        wel_like = [f for f in files if f.lower().startswith("wel")][:20]
        raise FileNotFoundError(
            f"No WEL stress files found in {pred_ws} matching '*_stress_period_data_###.txt'. "
            f"Examples in folder starting with 'wel': {wel_like}"
        )

    modified = []

    # --- overwrite drought SPs ---
    for sp, mon in zip(drought_pred["stress_period"].astype(int), drought_pred["month"].astype(int)):
        srcsp = month_to_srcsp[mon]

        for pfx, sp_to_fn in wel_map.items():
            srcfn = sp_to_fn.get(srcsp)
            if srcfn is None:
                raise FileNotFoundError(
                    f"Missing drought source WEL file for prefix '{pfx}', source SP={srcsp} (month={mon})."
                )

            # destination filename: keep same prefix pattern, swap SP
            dstfn = sp_to_fn.get(sp)
            if dstfn is None:
                dstfn = f"{pfx}_stress_period_data_{sp}.txt"

            srcpath = os.path.join(pred_ws, srcfn)
            outpath = os.path.join(pred_ws, dstfn)
            orgpath = os.path.join(pred_ws, "org", dstfn)
            os.makedirs(os.path.dirname(orgpath), exist_ok=True)

            shutil.copyfile(srcpath, outpath)
            shutil.copyfile(srcpath, orgpath)

            modified.append(dstfn)

    return sorted(set(modified)), drought_sps




def plot_zone_bud_ly_budget(pred_ws_list):
    fdir = os.path.join('results', 'figures', 'scenario_results')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    scn_results_dict ={}
    pst = pyemu.Pst(os.path.join(pred_ws_list[0],f'{modnm}.pst'))
    for ws in pred_ws_list:         
        jcbName = os.path.join(ws,f'{modnm}.0.obs.jcb')
        assert os.path.exists(jcbName), f'obs jcb {jcbName} not found! Scenario probably did not run have fun debugging!'
        jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
        scn_tag = ws.split('_')[2]
        scn_results_dict[scn_tag] = jcb
        # store jcd in dict:
        scn_results_dict[scn_tag] = jcb

    for scn_tag, obsdict in scn_results_dict.items():
        m_d = [ws for ws in pred_ws_list if scn_tag in ws]
        pst = pyemu.Pst(os.path.join(m_d[0], f'{modnm}.pst'))
        obs = pst.observation_data
        bobs = obs.loc[obs.obsnme.str.contains('zbud'), :]
        bobs.loc[:, 'datetime'] = pd.to_datetime(bobs.datetime)

        usecols = bobs['obsnme'].str.split(':').str[3].unique()

        pdf = PdfPages(os.path.join(fdir, f'budget_znbyly_{scn_tag}.pdf'))
        bud_cats = bobs.usecol.unique()
        bud_cats = set([s.split(':')[3] for s in bobs.obsnme])

        inobs = bobs.loc[bobs.obsnme.apply(lambda x: '-in' in x or 'from' in x and 'bud' in x), :].copy()
        outobs = bobs.loc[bobs.obsnme.apply(lambda x: '-out' in x or 'to' in x and 'bud' in x), :].copy()
        inobs['k'] = inobs.obsnme.str.split('-').str[-1].astype(int)
        outobs['k'] = outobs.obsnme.str.split('-').str[-1].astype(int)
        lays = sorted(inobs.k.unique())

        IN_COLORS = [
                '#8c510a',  # brown
                '#35978f',  # green-blue teal
                '#01665e',  # darker teal/blue
                '#3288bd',  # blue
                '#7b3294',  # purple
                "#c14bd6",  # darker purple
                "#64550F",  # deep purple
                "#5067e7",  # almost black-purple
            ]
        OUT_COLORS = [
                '#d9a900',  # gold
                '#f46d43',  # orange
                '#d73027',  # bright red
                '#a50026',  # dark red
                '#e7298a',  # pink-purple
                '#ce1256',  # magenta-purple
                '#67001f',  # very dark red-purple
                '#3f007d',  # dark purple
                '#382001',  # brown
                '#0e0d0d',  # black
            ]

        mpl.rcParams.update({'font.size': 12})
        
        for lay in lays:
            fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 8))
            ins = inobs.loc[inobs.obsnme.apply(lambda x: f'zn-{lay}' in x), :].copy()
            outs = outobs.loc[outobs.obsnme.apply(lambda x: f'zn-{lay}' in x), :].copy()

            pt_ins = obsdict._df.loc[:, ins.obsnme]
            pt_outs = obsdict._df.loc[:, outs.obsnme]

            pt_ins = pt_ins.loc[:, (pt_ins != 0).any(axis=0)]
            pt_outs = pt_outs.loc[:, (pt_outs != 0).any(axis=0)]

            in_types = pt_ins.columns.str.split(':').str[3].unique().str.replace('_datetime', '', regex=True).str.replace('zbly_', '', regex=True).tolist()
            out_types = pt_outs.columns.str.split(':').str[3].unique().str.replace('_datetime', '', regex=True).str.replace('zbly_', '', regex=True).tolist()

            factor = 0.00002296 * 365.25 # cf to acre-ft per year
            for i, in_type in enumerate(in_types):
                print(f'plotting {in_type} for layer {lay}...')
                mask = pt_ins.columns.str.contains(in_type)
                cols = pt_ins.columns[mask]
                ins_map = ins.set_index('obsnme')
                dates = ins_map.loc[cols, 'datetime'].to_numpy()            

                # base line
                ax3.plot(dates, pt_ins.loc['base', mask] * factor, c=IN_COLORS[i], lw=2, label=in_type)

                # only plot members whose time series sum != 0
                # for j in pt_ins.index[pt_ins.index != 'base']:
                #     s = pt_ins.loc[str(j), mask].to_numpy()
                #     if np.isclose(np.nansum(s), 0.0):  # skip if sums to zero (treats NaNs as 0)
                #         continue
                #     ax3.plot(dates, s * factor, c=IN_COLORS[i], lw=0.05, alpha=0.15)
                ax3.legend(loc='upper right', fontsize=8)
                ax3.semilogy()
                ax3.grid()
                ax3.set_ylabel('acre-ft/yr')
                ax3.set_ylim(1, 10**6)
                ax3.set_title('Inflows', fontsize=16)
            print('--'*10)
            for i, out_type in enumerate(out_types):
                mask = pt_outs.columns.str.contains(out_type)
                cols = pt_outs.columns[mask]
                outs_map = outs.set_index('obsnme')
                dates = outs_map.loc[cols, 'datetime'].to_numpy()            

                # base line
                ax4.plot(dates, pt_outs.loc['base', mask] * factor, c=OUT_COLORS[i], lw=2, label=out_type)

                # only plot members whose time series sum != 0
                # for j in pt_outs.index[pt_outs.index != 'base']:
                #     s = pt_outs.loc[str(j), mask].to_numpy()
                #     if np.isclose(np.nansum(s), 0.0):  # skip if sums to zero (treats NaNs as 0)
                #         continue
                #     ax4.plot(dates, s * factor, c=OUT_COLORS[i], lw=0.05, alpha=0.15)
                ax4.legend(loc='upper right', fontsize=8)
                ax4.semilogy()
                ax4.grid()  
                ax4.set_ylim(1, 10**6)
                ax4.set_title('Outflows', fontsize=16)

            pdf.savefig(fig)
            plt.close(fig)

        pdf.close()


def plot_scn_maxdd(pred_ws_list, modnm='elk_2lay',plot_quantiles=True,plt_base_only=False):
    o_d = os.path.join( 'results', 'figures', 'scenario_results')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    # init info from just one of the scn workspaces:
    m_d = pred_ws_list[0]
    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('trans'), :].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)

    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])

    obs_adj = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'))
    obs_adj = obs_adj.loc[obs_adj.name.str.contains('grpid')]
    weighted = obs_adj.loc[obs_adj.weight > 0, :].copy()
    weighted['datetime'] = pd.to_datetime(weighted.name.apply(lambda x: x.split(':')[-1]))
    weighted['id'] = weighted.name.apply(lambda x: 'grpid:' + x.split('grpid:')[1].split('_k')[0] + '_k:' + x.split('_k:')[1].split('_')[0])
    unq = weighted.id.unique()
    # get gwobs with unq ids:
    gwobs = gwobs.loc[gwobs.id.isin(unq), :].copy()
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis'])
    m = sim.get_model(modnm)

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)
    top = m.dis.top.array
    botm = m.dis.botm.array
    usites = gwobs.id.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site': usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)

    # load shapefiles needed for plotting:
    cpts = pd.read_csv(os.path.join('data', 'analyzed', 'transient_well_targets_lookup.csv'))
    cpts['grpid'] = 'grpid:' + cpts['group number'].astype(str) + '_k:' + cpts['k'].astype(str)
    # make geodataframe from geometry column:
    cpts = gpd.GeoDataFrame(data=cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=2265)
    cpts = cpts.groupby(['grpid']).last().reset_index()
    cpts.columns = cpts.columns.str.lower()
    
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'wahp')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'wahp')
    aq_extent = gpd.read_file(os.path.join(g_d, 'wahp_outline_full.shp'))
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'wahp7ly_cell_size_660ft_epsg2265.grid.shp'))
    drains = gpd.read_file(os.path.join(g_d, 'flow_lines_all_clipped.shp'))
    aq_extent = aq_extent.to_crs(modelgrid.crs)

    wls = pd.read_csv(os.path.join('data', 'raw','water_lvl_targs_manual_ly_assign.csv'))
    wls['grp_id'] = 'grpid:' + wls['group number'].astype(str) + '_k:' + (wls['manually_corrected_lay']-1).astype(str)

    usites = usitedf['site'].values
    
    scn_results_dict ={}
    pst = pyemu.Pst(os.path.join(pred_ws_list[0],f'{modnm}.pst'))
    d_scns = sorted(pred_ws_list, key=lambda x: int(x.split('scn')[1].split('_')[0]))
    for ws in d_scns:         
        jcbName = os.path.join(ws,f'{modnm}.0.obs.jcb')
        assert os.path.exists(jcbName), f'obs jcb {jcbName} not found! Scenario probably did not run have fun debugging!'
        jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
        scn_tag = ws.split('_')[2]
        scn_results_dict[scn_tag] = jcb
        # store jcd in dict:
        scn_results_dict[scn_tag] = jcb

    
    if scn_tag in ['scn04','scn05']:
        #HARD CODED!!!!!!!!!!!!!!
        asr = True
        asr_i=108
        asr_j=40
        asr_scn4 = modelgrid.loc[(modelgrid.row == asr_i) & (modelgrid.col == asr_j), :]
        asr_scn4.geometry = asr_scn4.geometry.centroid

        # asr_i=114
        # asr_j=45
        # asr_scn5 = modelgrid.loc[(modelgrid.row == asr_i) & (modelgrid.col == asr_j), :]
        # asr_scn5.geometry = asr_scn5.geometry.centroid
        
        kij = [(5,108,40),(5,108,46),(5,108,52),(5,114,40),(5,114,52),(5,120,40),(5,120,46),(5,120,52),(5,115,44),(5,111,49)]
        for k,i,j in kij:
            new_row = pd.DataFrame({'layer': [k], 'row': [i], 'col': [j]}, index=[f'asr_newloc_k:{k}_i:{i}_j:{j}'])
            new_row['geometry'] = modelgrid.loc[(modelgrid.row == i) & (modelgrid.col == j), :].geometry.values[0]
            asr_scn4 = pd.concat([asr_scn4, new_row], ignore_index=False)
        
        asr_grd_loc = asr_scn4
        
    else:
        asr = False
        asr_grd_loc = None
        
    colors = [
        "#b7bbc0", # scn01 - grey (baseline)
        "#f1a924", # scn02 - orange (full permit use)
        "#9e1818", # scn03 - red (ultimate drought)
        "#5a812c", # scn04 - green (drought + asr at 500 acft/yr)
        "#49b6e9", # scn05 - yellow (drought + asr at 200 acft/yr)
        "#e723e7", # scn06 - new loc asr
        ]
    darker_colors = [
        "#7f8286", # scn01 - dark grey (baseline)
        "#ee6c15", # scn02 - dark orange (full permit use)
        "#e23232", # scn03 - dark red (ultimate drought)
        "#81e70c", # scn04 - dark green (drought + asr at 500 acft/yr)
        "#151792", # scn05 - dark yellow (drought + asr at 200 acft/yr)
        "#990e99", # scn06 - dark new loc asr
        ]
    
    asr_grd_loc['color'] = 'cyan'
    asr_grd_loc.loc[0, 'color'] = "#81e70c"
    asr_grd_loc.loc[1, 'color'] = "#990e99"

    
    with PdfPages(os.path.join(o_d, 'scenario_hydrograpghs.pdf')) as pdf:
        for site in usites:
            fig = plt.figure(figsize=(11, 8))
            gs = gridspec.GridSpec(7, 6)
            ax1 = fig.add_subplot(gs[0:3, 0:2])
            ax2 = fig.add_subplot(gs[0:3, 2])
            ax3 = fig.add_subplot(gs[0:3, 4:])
            #ax4 = fig.add_subplot(gs[3:6, 1:-1])
            ax5 = fig.add_subplot(gs[6:, :])
            # turn off ax5, to make room for legend:
            ax5.axis('off')
            
            # limit to just target wells:
            #if site not in wls['grp_id'].values:
            #    continue
            uobs = gwobs.loc[gwobs.id == site, :].copy()
            uobs.sort_values(by='datetime', inplace=True)
    
            k, i, j = uobs.k.values[0], uobs.i.values[0], uobs.j.values[0]
            oobs = uobs.loc[uobs.observed == True, :]
            wobs = oobs.loc[oobs.weight > 0, :]
            #if wobs.shape[0] == 0:
            #    continue
            dts = uobs.datetime.values

            ax3.set_xticklabels('')
            ax3.set_yticklabels('')
            ax3.tick_params(axis='both', which='both', direction='in')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.tick_params(axis='both', which='both', length=0)

            plot_inset(ax=ax1, wl_loc=site, cpts=cpts, aq_extent=aq_extent, drains=drains,asr_loc=asr_grd_loc)
            wpp.plot_vert_xsec(ax2, m_d, m, wl_loc=site, mwl=wobs.obsval.mean(), cpts=cpts)
            
            stor_xmx = 0
            stor_xmn = 0
            for ws in d_scns:
                scn_tag = ws.split('_')[2]
                scn_num = int(scn_tag.replace('scn',''))
                # load result from ws:
                vals = scn_results_dict[scn_tag].loc[:, uobs.obsnme].values
                if np.shape(vals)[1] < 54:
                    continue
                # end time vals:
                end_dd = vals[:,-1]
                end_hm = vals[:, 55] # water levels at end of histo match
                dd = end_dd - end_hm
                min_val = dd.min()
                max_val = dd.max()
                if stor_xmx < max_val:
                    stor_xmx = max_val
                if stor_xmn > min_val or min_val!=0:
                    stor_xmn = min_val


            cnt = 0
            for ws in d_scns:
                scn_tag = ws.split('_')[2]
                scn_num = int(scn_tag.replace('scn',''))
                
                ax = fig.add_subplot(gs[3:6,cnt])

                # load result from ws:
                vals = scn_results_dict[scn_tag].loc[:, uobs.obsnme].values
                if np.shape(vals)[1] < 54:
                    continue
                # end time vals:
                end_dd = vals[:,-1]
                 # get vals at 2024-01-01:
                end_hm = vals[:, 55] # water levels at end of histo match
                temp_df = pd.DataFrame({'end_dd': end_dd, 'end_hm': end_hm}) 
                dd = end_dd - end_hm
                
                ax.hist(dd, bins=30, color=colors[scn_num-1], alpha=0.5, edgecolor=None, label=f'Scenario {scn_num}', zorder=3)
                min_val = dd.min()
                max_val = dd.max()
                if stor_xmx < max_val:
                    stor_xmx = max_val
                    print(f'site {site} scn {scn_num} new stor_xmx: {stor_xmx}')
                if stor_xmn > min_val:
                    stor_xmn = min_val
                    print(f'site {site} scn {scn_num} new stor_xmn: {stor_xmn}')

                base_vals = scn_results_dict[scn_tag].loc[:, uobs.obsnme].loc['base', :].values[-1]
                base_hm = scn_results_dict[scn_tag].loc[:, uobs.obsnme].loc['base', :].values[55]
                base_vals = base_vals - base_hm
                # plot vetical line for base:
                ax.axvline(x=base_vals, color=darker_colors[scn_num-1], lw=2.0, zorder=10,linestyle='--')

                # --- PROXY HANDLES (make sure styles match your plots) ---
                leg_s1  = mlines.Line2D([], [], color=darker_colors[0], lw=1.5, label='Scenario 1 - Baseline')
                leg_s2  = mlines.Line2D([], [], color=darker_colors[1], lw=1.5, label='Scenario 2 - Full permit use')
                leg_s3  = mlines.Line2D([], [], color=darker_colors[2], lw=1.5, label='Scenario 3 - Drought')
                leg_s4  = mlines.Line2D([], [], color=darker_colors[3], lw=1.5, label='Scenario 4 - ASR 10 wells (3,465 acft/yr)')
                #leg_s5  = mlines.Line2D([], [], color=darker_colors[4], lw=1.5, label='Scenario 5 - Drought with ASR (200 acft/yr)')
                #leg_s6  = mlines.Line2D([], [], color=darker_colors[5], lw=1.5, label='Scenario 6 - COW centered ASR (500 acft/yr)')

                #ax4.set_xlabel('Maximum Drawdown at 2045-01-01 (ft)', fontsize=12)
                ax.set_ylim(0,35)
                
                ax.set_xlim(stor_xmn -5, stor_xmx +5)
                if cnt==3:
                    ax.set_xlabel('Drawdown at end of simulation (ft)', fontsize=12)
                if cnt!=0:
                    ax.set_yticklabels('')
                if cnt == 0:
                    ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'Scenario {scn_num}', fontsize=11)
                ax.grid(True)
                scenario_handles = [leg_s1, leg_s2, leg_s3, leg_s4]#, leg_s6]
    
                legA = fig.legend(
                    handles=scenario_handles,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0.11),  # x=center, y a bit above bottom edge
                    ncol=2,
                    frameon=True,
                    framealpha=0.6,
                    fontsize=9,
                )

                if cnt == len(pred_ws_list) - 1:
                    grp_num = int(site.split('_')[0].split(':')[1])
                    k = int(site.split('_')[1].split(':')[1])
                    sites_grp = wls[wls['group number'] == int(grp_num)]
                    grp_full = sites_grp.copy()
                    idx_well = grp_full['index_well_flag'] == 1
        
                    sites_grp = sites_grp[sites_grp['manually_corrected_lay']-1 == int(k)]
                    aq_key = {0: "Wahpeton Shallow Sands-1",1: "Wahpeton Shallow Sands-2",
                                        2: "Wahpeton Shallow Plain-3",3: "Confing Unit",
                                        4: "Wahpeton Buried Valley",5: "Deep Clay",6: "Wild Rice"}
                    current_aq = aq_key.get(k)

                    # sort grp_full by manually_corrected_lay
                    grp_full = grp_full.sort_values(by=['manually_corrected_lay'])
                    if idx_well.any():
                        ax3.text(0.5, 0.8,
                                f'{current_aq}\n Group: {grp_num}\n ***Index Well***\n Layer: {k + 1}, \nRow: {int(i) + 1}, Column: {int(j) + 1}\n\n',
                                fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
                    else:
                        ax3.text(0.5, 0.75,
                                f'{current_aq}\n Group: {grp_num}\n Layer: {k + 1}, \nRow: {int(i) + 1}, Column: {int(j) + 1}\n\n\n',
                                fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
                        
                    # add text that icludes grp_full [loc_id, assigned aquifer]:
                    ax3.text(0.5, 0.27, 'All wells in group:\n' + '\n'.join(
                        [f"{row['loc_id']} - {aq_key.get(row['manually_corrected_lay']-1)}" for idx, row in grp_full.iterrows()]),
                            fontsize=9, ha='center', va='center', color='black', transform=ax3.transAxes)
                cnt+=1 
            
            
            pdf.savefig()
            plt.close(fig)
            print(f'  finished plotting hydrograph for site {site}...')

    pdf.close()


def plot_zone_bud_terms_by_scenario_like_base(
    pred_ws_list,
    modnm,
    layer_k=5,                 # focus on this layer via "zn-{layer_k}" (set None for all layers)
    x_start="2020-01-01",      # start date for x-axis and data trimming; set None to keep all
    max_plots_per_page=6,
    ylim=(1, 10**6),           # semilogy y-limits
    tol=1e-12
):
    """
    Build two PDFs in results/figures/scenario_results:
      - budget_terms_IN.pdf
      - budget_terms_OUT.pdf

    Each page: up to 6 subplots (3x2). Each subplot is ONE budget term; each line is ONE scenario.
    Logic mirrors the original plotting code:
      * Use each scenario's own PST to get dates and its own JCB columns
      * Filter by 'zbud' and by direction (IN: '-in' or 'from'; OUT: '-out' or 'to')
      * Filter layer by substring 'zn-{layer_k}' (like original)
      * Pull values from the 'base' row only
      * No aggregation; just the time-ordered series per term (as in your original)
    Terms that are all-zero (or missing) across all scenarios are skipped (no empty panels).
    """

    # -------- setup --------
    fdir = os.path.join('results', 'figures', 'scenario_results')
    os.makedirs(fdir, exist_ok=True)
    mpl.rcParams.update({'font.size': 11})

    # conversion factor (cf -> acre-ft/yr) same as your script
    factor = 0.00002296 * 365.25
    x_start = pd.to_datetime(x_start) if x_start is not None else None

    # dicts to collect data per term across scenarios
    # term -> { scn_tag: (dates_array, values_array) }
    in_term_data  = {}
    out_term_data = {}
    # master term sets (union across scenarios)
    in_terms_set, out_terms_set = set(), set()

    # stable scenario colors
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7'])

    # -------- helper (matches your original “type” token extraction) --------
    def term_token(name: str) -> str:
        parts = name.split(':')
        token = parts[3] if len(parts) > 3 else parts[-1]
        token = re.sub(r'_datetime', '', token, flags=re.IGNORECASE)
        token = re.sub(r'^zbly_', '', token, flags=re.IGNORECASE)
        return token

    # -------- pass through scenarios, mirroring your per-scenario processing --------
    for ws in pred_ws_list:
        # scenario tag like your code
        parts = os.path.basename(ws).split('_')
        scn_tag = parts[2] if len(parts) > 2 else os.path.basename(ws)

        # load pst/obs for THIS scenario (like your loop)
        pst = pyemu.Pst(os.path.join(ws, f'{modnm}.pst'))
        obs = pst.observation_data.copy()
        bobs = obs.loc[obs.obsnme.str.contains('zbud'), :].copy()
        bobs.loc[:, 'datetime'] = pd.to_datetime(bobs['datetime'])

        # IN/OUT selection exactly like your lambda (same precedence)
        inobs  = bobs.loc[bobs.obsnme.apply(lambda x: ('-in'  in x) or (('from' in x) and ('bud' in x))), :].copy()
        outobs = bobs.loc[bobs.obsnme.apply(lambda x: ('-out' in x) or (('to'   in x) and ('bud' in x))), :].copy()

        # layer filter via 'zn-{layer_k}' (or keep all if layer_k is None)
        if layer_k is not None:
            key = f'zn-{layer_k}'
            inobs  = inobs.loc[inobs.obsnme.str.contains(key), :].copy()
            outobs = outobs.loc[outobs.obsnme.str.contains(key), :].copy()

        # load this scenario's JCB base row
        jcb_path = os.path.join(ws, f'{modnm}.0.obs.jcb')
        assert os.path.exists(jcb_path), f'obs jcb {jcb_path} not found!'
        jcb = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=jcb_path)

        # build pt_ins / pt_outs like your code, then drop all-zero columns (any row nonzero)
        if not inobs.empty:
            pt_ins = jcb._df.loc[:, inobs.obsnme]
            pt_ins = pt_ins.loc[:, (pt_ins != 0).any(axis=0)]
        else:
            pt_ins = None

        if not outobs.empty:
            pt_outs = jcb._df.loc[:, outobs.obsnme]
            pt_outs = pt_outs.loc[:, (pt_outs != 0).any(axis=0)]
        else:
            pt_outs = None

        # extract term lists from actual column names (exactly like your code)
        if pt_ins is not None and pt_ins.shape[1] > 0:
            in_types = (pt_ins.columns
                        .str.split(':').str[3]
                        .str.replace('_datetime', '', regex=True)
                        .str.replace('zbly_', '', regex=True)
                        .unique().tolist())
        else:
            in_types = []

        if pt_outs is not None and pt_outs.shape[1] > 0:
            out_types = (pt_outs.columns
                         .str.split(':').str[3]
                         .str.replace('_datetime', '', regex=True)
                         .str.replace('zbly_', '', regex=True)
                         .unique().tolist())
        else:
            out_types = []

        # Collect per-term (dates, values) for this scenario, like your plotting lines
        # IN terms
        if in_types:
            ins_map = inobs.set_index('obsnme')
            for in_type in in_types:
                mask = pt_ins.columns.str.contains(in_type, regex=False)
                cols = pt_ins.columns[mask]
                if len(cols) == 0:
                    continue
                # dates from THIS scenario's PST (keeps exact alignment like your code)
                dates = ins_map.loc[cols, 'datetime'].to_numpy()
                vals  = pt_ins.loc['base', mask].to_numpy(dtype=float) * factor

                # trim to x_start if requested
                if x_start is not None:
                    sel = pd.to_datetime(dates) >= x_start
                    dates = dates[sel]
                    vals  = vals[sel]
                    if dates.size == 0:
                        continue

                # skip if effectively all-zero
                if np.allclose(vals, 0.0, atol=tol):
                    continue

                in_terms_set.add(in_type)
                in_term_data.setdefault(in_type, {})[scn_tag] = (pd.to_datetime(dates), vals)

        # OUT terms
        if out_types:
            outs_map = outobs.set_index('obsnme')
            for out_type in out_types:
                mask = pt_outs.columns.str.contains(out_type, regex=False)
                cols = pt_outs.columns[mask]
                if len(cols) == 0:
                    continue
                dates = outs_map.loc[cols, 'datetime'].to_numpy()
                vals  = pt_outs.loc['base', mask].to_numpy(dtype=float) * factor

                if x_start is not None:
                    sel = pd.to_datetime(dates) >= x_start
                    dates = dates[sel]
                    vals  = vals[sel]
                    if dates.size == 0:
                        continue

                if np.allclose(vals, 0.0, atol=tol):
                    continue

                out_terms_set.add(out_type)
                out_term_data.setdefault(out_type, {})[scn_tag] = (pd.to_datetime(dates), vals)

    # prune terms that ended up empty across all scenarios
    in_terms  = sorted([t for t in in_terms_set  if t in in_term_data and len(in_term_data[t])  > 0])
    out_terms = sorted([t for t in out_terms_set if t in out_term_data and len(out_term_data[t]) > 0])

    # -------- plotting (paginate 6 per page) --------
    def write_pdf(terms, term_dict, title_suffix, pdf_name):
        outpdf = os.path.join(fdir, pdf_name)
        if len(terms) == 0:
            with PdfPages(outpdf) as pdf:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.axis('off')
                desc = f"zn-{layer_k}" if layer_k is not None else "all layers"
                sdt  = f" | start≥{x_start.date()}" if x_start is not None else ""
                ax.set_title(f"No {title_suffix} terms with data ({desc}){sdt}")
                pdf.savefig(fig); plt.close(fig)
            print(f"Wrote: {outpdf} (no terms)")
            return

        per_page = max(1, min(6, max_plots_per_page))
        nrows, ncols = (3, 2) if per_page >= 6 else (int(np.ceil(per_page/2)), 2)

        with PdfPages(outpdf) as pdf:
            for start in range(0, len(terms), per_page):
                chunk = terms[start:start+per_page]
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
                axes = np.atleast_1d(axes).ravel()

                # turn off unused panels
                for ax in axes[len(chunk):]:
                    ax.axis('off')

                # plot each term on its own panel
                for ax, term in zip(axes, chunk):
                    suffix = f" (zn-{layer_k})" if layer_k is not None else " (all layers)"
                    if x_start is not None:
                        suffix += f", start≥{x_start.date()}"
                    ax.set_title(term + suffix)

                    scn_map = term_dict.get(term, {})
                    if not scn_map:
                        ax.axis('off')
                        continue

                    drawn = False
                    for ci, (scn_tag, (dates, vals)) in enumerate(sorted(scn_map.items())):
                        if len(dates) == 0 or np.allclose(vals, 0.0, atol=tol):
                            continue
                        ax.plot(dates, vals, lw=1.8, color=color_cycle[ci % len(color_cycle)], label=scn_tag)
                        drawn = True

                    if not drawn:
                        ax.axis('off')
                        continue

                    ax.semilogy()
                    ax.grid(alpha=0.3)
                    ax.set_ylim(*ylim)
                    ax.set_ylabel('acre-ft/yr')
                    ax.legend(fontsize=8, ncol=1, loc='upper right', framealpha=0.7)
                    if x_start is not None:
                        ax.set_xlim(left=x_start)

                ttl = f"Zone Budget {title_suffix} Terms"
                ttl += f" (zn-{layer_k})" if layer_k is not None else " (all layers)"
                if x_start is not None:
                    ttl += f" | start≥{x_start.date()}"
                fig.suptitle(ttl, fontsize=14, y=0.98)
                fig.tight_layout(rect=[0, 0.02, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Wrote: {outpdf}")

    # write the two PDFs
    write_pdf(in_terms,  in_term_data,  "Inflows",  "budget_terms_IN.pdf")
    write_pdf(out_terms, out_term_data, "Outflows", "budget_terms_OUT.pdf")


if __name__ == "__main__":
    print('*** running predictions for flow-ies ***')
    modnm = 'elk_2lay'
    post_ws = 'master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base'

    noptmax_flow = 4
    par_post_nm = f'{modnm}.{noptmax_flow}.par.jcb'
    obs_post_nm = f'{modnm}.{noptmax_flow}.obs.jcb'

    # run controls:
    scenario = 'drought'
    prep_predict = False
    prep_scn = False
    run_en = False
    run_all = False
    use_condor = True
    
    # Elk scenario settings
    pred_start = "2024-01-01"   # prediction start date (first predictive SP)
    lookback_years = 5          # baseline monthly means computed over last 5 yrs
    drought_years = 3           # drought lasts 3 years
    drought_source_year = 2021  # use 2021 monthly recharge during drought
    riv_stage_drop_ft = 2.0     # drop stage/rbot by 2 ft during drought months

    plot_predict = True

    if use_condor:
        num_reals_flow = 280
        num_workers_flow = 140
        hostname = '10.99.10.30'
        pid = os.getpid()
        current_time = int(time.time())
        seed_value = (pid + current_time) % 65536
        np.random.seed(seed_value)
        port = random.randint(4001, 4999)
        print(f'port #: {port}')
    else:
        num_reals_flow = 6
        num_workers_flow = 6
        hostname = None
        port = None
    
    local = True

    all_scenarios = ['drought']

    if run_all:
        for scenario in all_scenarios:
            print(f'--- running scenario: {scenario} ---')
            if scenario == 'full_allocation':
                # Scenario 2: Full allocation for COW
                pred_ws = os.path.join('post_ies_scn02_full_alloc')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    base_well_files = write_baseline_well_files(modnm, pred_ws)
                    lst_mod_well_files = write_full_alloc_well_files(modnm, pred_ws)
                    all_well_files = base_well_files + lst_mod_well_files
                    all_well_files = list(set(all_well_files))
                    modify_mult2mod(pred_ws, all_well_files)
                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None, 
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)
                if plot_predict:
                    print("plotting prediction results...")
                    m_d=pred_ws+'_ensemble'
                    obsdict = wpp.get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
                    itrmx = max(obsdict)
                    wpp.plot_fancy_obs_v_sim(m_d,obsdict)

            elif scenario == 'baseline':
                print('--- preparing baseline scenario (ELK monthly climatology) ---')
                pred_ws = os.path.join('post_ies_scn01_baseline')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)

                if prep_scn:
                    nopt0chk_baseline(post_ws, pred_ws, modnm, noptmax_flow)

                if run_en:
                    run_ies(
                        pred_ws,
                        modnm=modnm,
                        m_d=pred_ws + '_ensemble',
                        num_workers=num_workers_flow,
                        num_reals=num_reals_flow,
                        niceness=False,
                        noptmax=-1,
                        init_lam=None,
                        local=local,
                        use_condor=use_condor,
                        hostname=hostname,
                        port=port,
                        par_post_nm=par_post_nm,
                        obs_post_nm=obs_post_nm
                    )

                if plot_predict:
                    print("plotting prediction results...")
                    m_d = pred_ws + '_ensemble'
                    obsdict = wpp.get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
                    itrmx = max(obsdict)
                    wpp.plot_fancy_obs_v_sim(m_d, obsdict)


            elif scenario == 'drought':
                print('--- preparing drought scenario (ELK: 2021 RCH + -2ft RIV for 3 years) ---')
                pred_ws = os.path.join('post_ies_scn03_drought')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)

                if prep_scn:
                    # Drought scenario under the updated workflow:
                    #   - Do NOT change predictive pumping (preserve IES posterior pumping tweaks)
                    #   - Predictive recharge is NOT parameterized, and baseline RCH is already baked in
                    #     -> only overwrite drought-window months.

                    # 1) Overwrite first drought_years*12 predictive months with drought_source_year recharge arrays
                    rch_files, drought_sps = elk_overwrite_rch_drought_only(
                        pred_ws=pred_ws,
                        pred_start=pred_start,
                        drought_years=drought_years,
                        drought_source_year=drought_source_year
                    )

                    # 2) Drop RIV stage/rbot during drought months (with safety vs botm)
                    riv_files = elk_drop_riv_stage_during_drought(
                        modnm=modnm,
                        pred_ws=pred_ws,
                        drought_sps=drought_sps,
                        stage_drop_ft=riv_stage_drop_ft
                    )
                    
                    wel_files, drought_wel_sps = elk_overwrite_wel_drought_only(
                        pred_ws=pred_ws,
                        pred_start=pred_start,          # "2024-01-01"
                        drought_years=3,
                        drought_source_year=2021,        
                    )

                    lst_mod_model_files = sorted(set((rch_files or []) + (riv_files or []) + (wel_files or [])))

                    # optional debug: list what we edited
                    try:
                        with open(os.path.join(pred_ws, "rm_list.txt"), "w", newline="") as f:
                            for item in lst_mod_model_files:
                                f.write(f"{item}\n")
                    except Exception:
                        pass

                    if lst_mod_model_files:
                        modify_mult2mod(pred_ws, lst_mod_model_files)

                    # 4) keep your existing nopt=0 check
                    nopt0chk_baseline(post_ws, pred_ws, modnm, noptmax_flow)

                if run_en:
                    run_ies(
                        pred_ws,
                        modnm=modnm,
                        m_d=pred_ws + '_ensemble',
                        num_workers=num_workers_flow,
                        num_reals=num_reals_flow,
                        niceness=False,
                        noptmax=-1,
                        init_lam=None,
                        local=local,
                        use_condor=use_condor,
                        hostname=hostname,
                        port=port,
                        par_post_nm=par_post_nm,
                        obs_post_nm=obs_post_nm
                    )

                if plot_predict:
                    print("plotting prediction results...")
                    m_d = pred_ws + '_ensemble'
                    obsdict = wpp.get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
                    itrmx = max(obsdict)
                    wpp.plot_fancy_obs_v_sim(m_d, obsdict)


            elif scenario == 'asr_10_well_at225':
                pred_ws = os.path.join('post_ies_scn04_asr_10_well_at225')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    base_well_files = write_baseline_well_files(modnm, pred_ws)
                    modify_mult2mod(pred_ws, base_well_files)

                    # write well package files to do ASR:
                    kij = [(5,108,40),(5,108,46),(5,108,52),(5,114,40),(5,114,52),(5,120,40),(5,120,46),(5,120,52),(5,115,44),(5,111,49)]
                    write_lst_asr_well_files(modnm, pred_ws,rate_acftyr=346.5,kij=kij)

                    rewrite_zbud_ins_file(modnm='elk_2lay', pred_ws=pred_ws)

                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None, 
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)
                if plot_predict:
                    print("plotting prediction results...")
                    m_d=pred_ws+'_ensemble'
                    obsdict = wpp.get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
                    itrmx = max(obsdict)
                    wpp.plot_fancy_obs_v_sim(m_d,obsdict)
            
            elif scenario == 'drought+asr200':
                pred_ws = os.path.join('post_ies_scn05_drought_asr200')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    base_well_files = write_baseline_well_files(modnm, pred_ws)
                    lst_mod_well_files = write_full_alloc_well_files(modnm, pred_ws)
                    all_well_files = base_well_files + lst_mod_well_files
                    all_well_files = list(dict.fromkeys(all_well_files))
                    
                    # modify predicitve ghb conductance to very very small values to simulate drought:
                    ghb_file = write_ghb_files(modnm, pred_ws)
                    lst_mod_well_files = all_well_files + ghb_file
                    modify_mult2mod(pred_ws, lst_mod_well_files)

                    # write well package files to do ASR:
                    write_asr_well_files(modnm, pred_ws,rate_acftyr=200,k=5,i=83,j=38)

                    rewrite_zbud_ins_file(modnm='elk_2lay', pred_ws=pred_ws)

                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None, 
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)
                if plot_predict:
                    print("plotting prediction results...")
                    m_d=pred_ws+'_ensemble'
                    obsdict = wpp.get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
                    itrmx = max(obsdict)
                    wpp.plot_fancy_obs_v_sim(m_d,obsdict)    
            
            elif scenario == 'drought+asr500cow':
                pred_ws = os.path.join('post_ies_scn05_drought_asr500cow')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    base_well_files = write_baseline_well_files(modnm, pred_ws)
                    lst_mod_well_files = write_full_alloc_well_files(modnm, pred_ws)
                    all_well_files = base_well_files + lst_mod_well_files
                    all_well_files = list(dict.fromkeys(all_well_files))
                    
                    # modify predicitve ghb conductance to very very small values to simulate drought:
                    ghb_file = write_ghb_files(modnm, pred_ws)
                    lst_mod_well_files = all_well_files + ghb_file
                    riv_files = write_riv_files(modnm, pred_ws)
                    lst_mod_well_files = lst_mod_well_files + riv_files
                    
                    modify_mult2mod(pred_ws, lst_mod_well_files)

                    # write well package files to do ASR:
                    write_asr_well_files(modnm, pred_ws,rate_acftyr=500,k=5,i=114,j=45)

                    rewrite_zbud_ins_file(modnm='elk_2lay', pred_ws=pred_ws)

                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None, 
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)
                if plot_predict:
                    print("plotting prediction results...")
                    m_d=pred_ws+'_ensemble'
                    obsdict = wpp.get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
                    itrmx = max(obsdict)
                    wpp.plot_fancy_obs_v_sim(m_d,obsdict)    

    if plot_predict:
        # get dirs that end in '_ensemble'
        pred_ws_list = [d for d in os.listdir('.') if d.startswith('post_ies_scn') and d.endswith('_ensemble')]
       #pred_ws_list = [d for d in os.listdir('.') if d.startswith('post_ies_scn') and d.endswith('_clean')]
        #plot_scn_hydrographs_elk(pred_ws_list, modnm='elk_2lay',plot_quantiles=True,plt_base_only=False,zoom_predict=False)

        # plot_scn_model_net_budget_annual_from_budobs_jcb_totals_nosto(
        #         pred_ws_list,
        #         modnm="elk_2lay",
        #         qlo=10,
        #         qhi=90,
        #     )

        # run row-bin pumping plots using the *_clean baselines
        plot_rowbin_total_pumping_by_scenario_mimic_mel(
            pred_ws_list=pred_ws_list,
            modnm="elk_2lay",
            wel_pkg_name="wel",      # change to "wel_0" if needed
            row_bin_size=50,         # group rows in blocks of 50
            convert_to_positive_withdrawal=True,  # keep MF6 sign convention
            out_pdf=os.path.join(
                "results",
                "figures",
                "scenario_results",
                "elk_rowbin_pumping_baseline_vs_drought.pdf",
            ),
            dpi=300,
            progress_kper_step=25,   # print progress every 25 stress periods
        )

        #plot_scn_maxdd(pred_ws_list, modnm='elk_2lay')



