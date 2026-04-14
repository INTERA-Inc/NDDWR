import os
import sys

# from scipy.fft import dst
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import pyemu
import flopy
# import pypestutils
# from pypestutils.pestutilslib import PestUtilsLib
# from pypestutils import helpers as ppu
import platform
import math
import calendar
import pandas as pd
import geopandas as gpd
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
import re
import random
import matplotlib.colors as mcolors
from scipy import signal
import contextily as cx
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


# set fig formats:
import swww04_process_plot_results_Spence as swww_plot
swww_plot.set_graph_specifications()
swww_plot.set_map_specifications()


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
def modify_mult2mod(pred_ws='.',lst_mod_files=[]):
    mults = pd.read_csv(os.path.join(pred_ws, 'mult2model_info.csv'))
    print(len(mults))
    mults = mults.loc[mults['model_file'].isin(lst_mod_files) == False]
    print(len(mults))
    mults.to_csv(os.path.join(pred_ws, 'mult2model_info.csv'), index=False)
    print("modified mult2model_info.csv to remove modified well files.")


# ------------------------------------------------------- #
# Baseline Scenario Functions
# ------------------------------------------------------- #
def write_baseline_well_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01",'stress_period'].values[0]
    sp_end = spd['stress_period'].iloc[-1]

    # Find all well files
    lst_mod_well_files = []
    list_files = [f for f in os.listdir(pred_ws) if 'wel' in f.split("_")[0] and f.endswith('.txt') and "_stress_period_data_" in f]
    well_sorted = sorted(list_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # drop list to well files from past five years (60 months)
    sp_rng = range(sp2024 - 60, sp2024+1)
    well_sorted = [f for f in well_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    # get median pumping rates from last 5 years of histo matching period per month
    pmp = pd.DataFrame()
    for f in well_sorted:
        df = pd.read_csv(os.path.join(pred_ws, f), delim_whitespace=True, header=None)
        df.columns = ['layer', 'row', 'column', 'flux_cfd']
        df['sp'] = int(re.search(r'_(\d+)\.txt$', f).group(1))
        pmp = pd.concat([pmp, df], axis=0, ignore_index=True)

    # Merge with spd to pull out month information
    pmp = pd.merge(pmp,spd[['stress_period','start_datetime']],left_on='sp',right_on='stress_period')
    pmp['month'] = pd.to_datetime(pmp['start_datetime']).dt.month

    # Drop columns used for the datetime merge
    pmp = pmp.drop(columns=['sp','stress_period','start_datetime'])
    pmp = pmp.groupby(['month','layer','row','column']).agg({'flux_cfd':'median'}).reset_index()
    
    # Save info as csv
    print('Saving baseline pumping info')
    pmp.to_csv('baseline_pumping_records.csv')
    
    # write well files for prediction years:
    temp_well_files = []
    for sp in range(sp2024+1, sp_end+1):
        # grab data for month corresponding to this stress period
        month = pd.Timestamp(spd.loc[spd['stress_period'] == sp,'start_datetime'].values[0]).month
        month_pmp = pmp.loc[pmp['month']==month]
        # Drop month column
        month_pmp = month_pmp.drop(columns='month')
        new_well_file = f'swww.wel_stress_period_data_{sp}.txt'
        month_pmp.to_csv(os.path.join(pred_ws, new_well_file), sep='\t', header=False, index=False)
        month_pmp.to_csv(os.path.join(pred_ws,'org', new_well_file), sep='\t', header=False, index=False)
        temp_well_files.append(new_well_file)

    lst_mod_well_files.extend(temp_well_files)

    return lst_mod_well_files

# --------------------------
# --- Baseline GHB
# --------------------------
def write_baseline_ghb_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01",'stress_period'].values[0]
    sp_end = spd['stress_period'].iloc[-1]

    # read in ghb files:
    ghb_files = [f for f in os.listdir(pred_ws) if ".ghb" in f and f.endswith('.txt')]
    ghb_sorted = sorted(ghb_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # drop list to ghb files from past five years (60 months)
    sp_rng = range(sp2024 - 60, sp2024+1)
    ghb_sorted = [f for f in ghb_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    ghb_main = pd.DataFrame()
    for ghb in ghb_sorted:
        ghb_df = pd.read_csv(os.path.join(pred_ws, ghb), delim_whitespace=True,header=None)
        ghb_df.columns = ['layer','row','column','stage','cond']
        ghb_df['sp'] = int(re.search(r'_(\d+)\.txt$', ghb).group(1))
        ghb_main = pd.concat([ghb_main, ghb_df], axis=0, ignore_index=True)

    ghb_main = pd.merge(ghb_main,spd[['stress_period','start_datetime']],left_on='sp',right_on='stress_period')
    ghb_main['month'] = pd.to_datetime(ghb_main['start_datetime']).dt.month
    ghb_main = ghb_main.drop(columns=['sp','stress_period','start_datetime'])
    ghb_main = ghb_main.groupby(['month','layer','row','column']).agg({'stage':'median',
                                                                       'cond':'median'}).reset_index()

    # write well files for prediction years:
    temp_ghb_files = []
    for sp in range(sp2024+1, sp_end+1):
        # grab data for month corresponding to this stress period
        month = pd.Timestamp(spd.loc[spd['stress_period'] == sp,'start_datetime'].values[0]).month
        month_ghb = ghb_main.loc[ghb_main['month']==month]
        month_ghb = month_ghb.drop(columns='month')
        new_ghb_file = f'swww.ghb_stress_period_data_{sp}.txt'
        month_ghb.to_csv(os.path.join(pred_ws, new_ghb_file), sep='\t', header=False, index=False)
        month_ghb.to_csv(os.path.join(pred_ws,'org', new_ghb_file), sep='\t', header=False, index=False)
        temp_ghb_files.append(new_ghb_file)

    lst_mod_ghb_files = []
    lst_mod_ghb_files.extend(temp_ghb_files)

    return lst_mod_ghb_files

# ----------------------
# --- GHB low K scenario
# ----------------------
def write_low_k_ghb_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01",'stress_period'].values[0]
    ep_end = spd['stress_period'].iloc[-1]

    # read in ghb files:
    ghb_files = [f for f in os.listdir(pred_ws) if ".ghb" in f and f.endswith('.txt')]
    ghb_sorted = sorted(ghb_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # drop list to ghb files from past five years (60 months)
    sp_rng = range(sp2024 - 60, sp2024+1)
    ghb_sorted = [f for f in ghb_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    ghb_main = pd.DataFrame()
    for ghb in ghb_sorted:
        ghb_df = pd.read_csv(os.path.join(pred_ws, ghb), delim_whitespace=True,header=None)
        ghb_df.columns = ['layer','row','column','stage','cond']
        ghb_df['sp'] = int(re.search(r'_(\d+)\.txt$', ghb).group(1))
        ghb_main = pd.concat([ghb_main, ghb_df], axis=0, ignore_index=True)

    ghb_main = pd.merge(ghb_main,spd[['stress_period','start_datetime']],left_on='sp',right_on='stress_period')
    ghb_main['month'] = pd.to_datetime(ghb_main['start_datetime']).dt.month
    ghb_main = ghb_main.drop(columns=['sp','stress_period','start_datetime'])
    ghb_main = ghb_main.groupby(['month','layer','row','column']).agg({'stage':'median',
                                                                       'cond':'median'}).reset_index()

    # Lower the conductance by a factor of 1/10
    ghb_main['cond'] /= 10

    # write well files for prediction years:
    temp_ghb_files = []
    for sp in range(sp2024+1, ep_end+1):
        # grab data for month corresponding to this stress period
        month = pd.Timestamp(spd.loc[spd['stress_period'] == sp,'start_datetime'].values[0]).month
        month_ghb = ghb_main.loc[ghb_main['month']==month]
        month_ghb = month_ghb.drop(columns='month')
        new_ghb_file = f'swww.ghb_stress_period_data_{sp}.txt'
        month_ghb.to_csv(os.path.join(pred_ws, new_ghb_file), sep='\t', header=False, index=False)
        month_ghb.to_csv(os.path.join(pred_ws,'org', new_ghb_file), sep='\t', header=False, index=False)
        temp_ghb_files.append(new_ghb_file)

    lst_mod_ghb_files = []
    lst_mod_ghb_files.extend(temp_ghb_files)

    return lst_mod_ghb_files


# ----------------------------
# --- GHB "drought" scenario
# ---------------------------
def write_drought_ghb_files(modnm, pred_ws):
    """
    Calculate the average rate of decline of stage at the GHB over the past 20
    years, and apply that to the median stage values from the past 5-years
    when projecting the GHB forward in time
    """
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01",'stress_period'].values[0]
    ep_end = spd['stress_period'].iloc[-1]

    # read in ghb files:
    ghb_files = [f for f in os.listdir(pred_ws) if ".ghb" in f and f.endswith('.txt')]
    ghb_sorted = sorted(ghb_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # drop list to ghb files from past 20 years (240 months)
    sp_rng = range(sp2024 - 240, sp2024+1)
    ghb_sorted = [f for f in ghb_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    ghb_main = pd.DataFrame()
    for ghb in ghb_sorted:
        ghb_df = pd.read_csv(os.path.join(pred_ws, ghb), delim_whitespace=True,header=None)
        ghb_df.columns = ['layer','row','column','stage','cond']
        ghb_df['sp'] = int(re.search(r'_(\d+)\.txt$', ghb).group(1))
        ghb_main = pd.concat([ghb_main, ghb_df], axis=0, ignore_index=True)

    ghb_main = pd.merge(ghb_main,spd[['stress_period','start_datetime']],left_on='sp',right_on='stress_period')
    ghb_main['month'] = pd.to_datetime(ghb_main['start_datetime']).dt.month
    ghb_main = ghb_main.groupby('start_datetime').mean()

    # --- Calculate average decline rate over past 20 years (ft decline per year)
    # !!!TODO This is hardcoded and might break if stress periods/dates are edited
    decline_rate = (ghb_main.loc['2003-12-01']['stage'] - ghb_main.loc['2023-12-01']['stage']) / 20

    # --- Create single input for projection based on past five years
    # read in ghb files:
    ghb_files = [f for f in os.listdir(pred_ws) if ".ghb" in f and f.endswith('.txt')]
    ghb_sorted = sorted(ghb_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # drop list to ghb files from past five years (60 months)
    sp_rng = range(sp2024 - 60, sp2024+1)
    ghb_sorted = [f for f in ghb_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    ghb_main = pd.DataFrame()
    for ghb in ghb_sorted:
        ghb_df = pd.read_csv(os.path.join(pred_ws, ghb), delim_whitespace=True,header=None)
        ghb_df.columns = ['layer','row','column','stage','cond']
        ghb_df['sp'] = int(re.search(r'_(\d+)\.txt$', ghb).group(1))
        ghb_main = pd.concat([ghb_main, ghb_df], axis=0, ignore_index=True)

    ghb_main = pd.merge(ghb_main,spd[['stress_period','start_datetime']],left_on='sp',right_on='stress_period')
    ghb_main['month'] = pd.to_datetime(ghb_main['start_datetime']).dt.month
    ghb_main = ghb_main.drop(columns=['sp','stress_period','start_datetime'])
    ghb_main = ghb_main.groupby(['month','layer','row','column']).agg({'stage':'median',
                                                                       'cond':'median'}).reset_index()
    # write well files for prediction years with added decline rate
    temp_ghb_files = []
    stage = []
    for sp in range(sp2024+1, ep_end+1):
        # grab data for month corresponding to this stress period
        year = pd.Timestamp(spd.loc[spd['stress_period'] == sp,'start_datetime'].values[0]).year - 2023
        month = pd.Timestamp(spd.loc[spd['stress_period'] == sp,'start_datetime'].values[0]).month
        month_ghb = ghb_main.loc[ghb_main['month']==month]
        month_ghb = month_ghb.drop(columns='month')
        month_ghb['stage'] -= year * decline_rate
        new_ghb_file = f'swww.ghb_stress_period_data_{sp}.txt'
        month_ghb.to_csv(os.path.join(pred_ws, new_ghb_file), sep='\t', header=False, index=False)
        month_ghb.to_csv(os.path.join(pred_ws,'org', new_ghb_file), sep='\t', header=False, index=False)
        temp_ghb_files.append(new_ghb_file)
        stage = stage + [month_ghb['stage'].values[0]]

    lst_mod_ghb_files = []
    lst_mod_ghb_files.extend(temp_ghb_files)

    return lst_mod_ghb_files


def write_baseline_rch_files(modnm, pred_ws):
    """
    For predictive stress periods, overwrite all RCH array files with the
    monthly median recharge surface computed from the previous ~5 years
    (stress periods sp2024-60 through sp2024, matched by month).
    """

    # Load stress-period info
    spd = pd.read_csv(os.path.join('tables', 'monthly_stress_period_info.csv'))

    # Last historical period
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01", 'stress_period'].values[0]

    # Last stress period in the sim
    sp_end = spd['stress_period'].iloc[-1]

    # Historical SP range for last 5 years
    # sp_rng = sp2024 - 60 ... sp2024 + 1 (inclusive)
    sp_rng = range(sp2024 - 60, sp2024 + 1)
    sp_rng_set = set(sp_rng)

    # --- Find all RCH files and sort by stress period
    rch_files = [f for f in os.listdir(pred_ws)
                 if ".rch" in f and f.endswith('.txt')]

    def _get_sp(fname):
        m = re.search(r'_(\d+)\.txt$', fname)
        if m is None:
            raise ValueError(f"Could not parse stress period from RCH file name: {fname}")
        return int(m.group(1))

    rch_sorted = sorted(rch_files, key=_get_sp)

    # Historical subset: last 5 years
    hist_rch_files = [
                f for f in rch_sorted
                if _get_sp(f) in sp_rng_set
                ]

    # Predictive subset: everything after sp2024
    pred_rch_files = [
                f for f in rch_sorted
                if sp2024 < _get_sp(f) <= sp_end
                ]

    # --- Build monthly stacks of 2D recharge arrays
    # month_arrays[month] -> list of 2D numpy arrays for that month
    month_arrays = {m: [] for m in range(1, 13)}
    base_shape = None

    for fname in hist_rch_files:
        sp = _get_sp(fname)

        # Look up month for this stress period
        start_dt = spd.loc[spd['stress_period'] == sp, 'start_datetime'].values[0]
        month = pd.Timestamp(start_dt).month

        path = os.path.join(pred_ws, fname)
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        arr = df.values

        # Sanity check: ensure consistent array shape
        if base_shape is None:
            base_shape = arr.shape
        else:
            if arr.shape != base_shape:
                raise RuntimeError(
                    f"Inconsistent RCH array shape: {fname} has shape {arr.shape}, "
                    f"expected {base_shape}."
                )

        month_arrays[month].append(arr)

    # --- Compute monthly median 2D surfaces
    month_median_df = {}
    for m in range(1, 13):
        arr_list = month_arrays[m]
        if len(arr_list) == 0:
            # If for some reason we have no data for a month, fail
            raise RuntimeError(
                f"No historical RCH arrays found for month {m} in the 5-year window; "
                "cannot compute monthly median recharge."
            )
        stack = np.stack(arr_list, axis=0)  # (n_sp_m, nrow, ncol)
        # !!! Swapping to mean
        median_arr = np.mean(stack, axis=0)
        month_median_df[m] = pd.DataFrame(median_arr)

    # Ensure 'org' backup folder exists
    org_dir = os.path.join(pred_ws, 'org')
    os.makedirs(org_dir, exist_ok=True)

    modified_rch_files = []

    # --- Overwrite predictive RCH files using monthly mean surfaces
    for rch_file in pred_rch_files:
        sp_pred = _get_sp(rch_file)

        # Find the month for this predictive stress period
        start_dt = spd.loc[spd['stress_period'] == sp_pred, 'start_datetime'].values[0]
        month = pd.Timestamp(start_dt).month

        median_df = month_median_df[month]

        pred_path = os.path.join(pred_ws, rch_file)
        org_path = os.path.join(org_dir, rch_file)

        # Write median monthly recharge surface into predictive file and backup
        median_df.to_csv(pred_path, sep='\t', header=False, index=False, float_format="%.6E")
        median_df.to_csv(org_path, sep='\t', header=False, index=False, float_format="%.6E")

        modified_rch_files.append(rch_file)

    # -- Edit the arr_par_summary.csv file to add essentially zero bounds to RCH
    # Updating for new recharge setup, edit both rch-pps and rch-cns
    par_data = pd.read_csv(os.path.join(pred_ws, 'swww.par_data.csv'))
    for par in ['rcht_cn']:
        par_data.loc[(par_data['pargp'] == par) &
                     (par_data['kper'] > 319), 'parlbnd'] = 0.99

        par_data.loc[(par_data['pargp'] == par) &
                     (par_data['kper'] > 319), 'parubnd'] = 1.01

        par_data.loc[(par_data['pargp'] == par) &
                     (par_data['kper'] > 319), 'parval1'] = 1.0

    par_data.to_csv(os.path.join(pred_ws, 'swww.par_data.csv'),
                    index=False)

    return modified_rch_files



def nopt0chk_baseline(org_ws,pred_ws, modnm, noptmax_flow):
    pst = pyemu.Pst(os.path.join(pred_ws,'swww.pst'))
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_par_en'] = f'{modnm}.{noptmax_flow}.par.jcb'
    pst.pestpp_options["ies_obs_en"] = f'{modnm}.{noptmax_flow}.obs.jcb'
    pst.write(os.path.join(pred_ws,'swww.pst'),version=2)
    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst', cwd=pred_ws)

    fig, ax = plt.subplots(1,1,figsize=(10,8))

    # Check wel package
    df = pd.read_csv(os.path.join(pred_ws, f'{modnm}.base.obs.csv'))
    old_df = pd.read_csv(os.path.join(org_ws, f'{modnm}.base.obs.csv'))

    well_cols = [col for col in df.columns if 'wel' in col]
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
    ax.plot(bud_df['datetime'], bud_df['bud_out']/43560*365.25, marker='o', linestyle='-')
    ax.plot(bud_df_old['datetime'], bud_df_old['bud_out']/43560*365.25, marker='x', linestyle='--')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylabel('Annual Pumping\n(acre-feet/year)')

    figoutput = os.path.join(pred_ws, 'input_figs','scn01_baseline')
    if not os.path.exists(figoutput):
        os.makedirs(figoutput)
    fig.savefig(os.path.join(figoutput, 'histo_match_vs_scn_pumping_baseline.png'), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------- #
# Deferred + Pending wel permits on top of previous 5-year median
# --------------------------------------------------------------- #
# Helper to distribute new allocations based on previously used bell curve
def expand_annual_to_growing_season(df,
                                    grow_months=(5, 6, 7, 8, 9),
                                    peak_month=7,
                                    sigma_mo=1.2):
    # Conversion
    CUFT_PER_ACFT = 43_560
    # Drop years with no pumping
    df = df.dropna(subset=["Year"]).copy()
    # Build normalized month weights once
    weights_raw = [math.exp(-0.5 * ((m - peak_month) / sigma_mo) ** 2) for m in grow_months]
    w_sum = sum(weights_raw)
    w_map = {m: (w / w_sum) for m, w in zip(grow_months, weights_raw)}
    rows = []
    for _, r in df.iterrows():
        # Check well use type
        use_type = r['use_type']

        # Only redistribute irrigation wells
        if use_type == 'Irrigation':
            ann_acft = float(r["use_acft"])
            for m in grow_months:
                m_acft = ann_acft * w_map[m]
                days   = calendar.monthrange(int(r["Year"]), m)[1]
                cfd    = m_acft * CUFT_PER_ACFT / days
                rows.append({
                    **r,
                    "Month": m,
                    "use_acft": m_acft,
                    "days": days,
                    "cfd": cfd
                })
        # Keep other well types intact
        else:
            annual_share = r["use_acft"]
            monthly_share = annual_share / 12.0
            for m in range(1, 13):
                days = calendar.monthrange(int(r["Year"]), m)[1]
                # now monthly_share really is the ac-ft for that month
                cfd = monthly_share * CUFT_PER_ACFT / days
                rows.append({
                    **r,
                    "Month": m,
                    "use_acft": monthly_share,
                    "days": days,
                    "cfd": cfd
                })
    return pd.DataFrame(rows)

def write_def_pend_well_files(modnm, pred_ws):
    print("writing full allocation well files...")
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01", 'stress_period'].values[0]
    ep_end = spd['stress_period'].iloc[-1]

    # Find all well files
    lst_mod_well_files = []
    list_files = [
        f for f in os.listdir(pred_ws)
        if 'wel' in f.split("_")[0]
        and f.endswith('.txt')
        and "_stress_period_data_" in f
        ]
    well_sorted = sorted(list_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # -------------------------------
    # Historical pumping (last 5 yrs)
    # -------------------------------
    # drop list to well files from past five years (60 months)
    sp_rng = range(sp2024 - 60, sp2024 + 1)
    well_sorted = [
        f for f in well_sorted
        if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng
        ]

    # get median pumping rates from last 5 years of histo matching period per month
    pmp = pd.DataFrame()
    for f in well_sorted:
        df = pd.read_csv(os.path.join(pred_ws, f), delim_whitespace=True, header=None)
        df.columns = ['layer', 'row', 'column', 'flux_cfd']
        df['sp'] = int(re.search(r'_(\d+)\.txt$', f).group(1))
        pmp = pd.concat([pmp, df], axis=0, ignore_index=True)

    # Merge with spd to pull out month information
    pmp = pd.merge(
        pmp,
        spd[['stress_period', 'start_datetime']],
        left_on='sp',
        right_on='stress_period'
        )
    pmp['month'] = pd.to_datetime(pmp['start_datetime']).dt.month

    # Drop columns used for the datetime merge
    pmp = pmp.drop(columns=['sp', 'stress_period', 'start_datetime'])
    pmp = (
        pmp
        .groupby(['month', 'layer', 'row', 'column'])
        .agg({'flux_cfd': 'median'})
        .reset_index()
        )

    # -------------------------------------------
    # Add in additional pumping (new permit PODs)
    # -------------------------------------------
    pods = gpd.read_file(
        os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww',
                     'Spiritwood_Warwick_aquifer_PermitPOD_withDeferred.shp')
        )
    # Filter to desired statuses
    pods = pods.loc[pods['status'].isin(['Pending Review', 'Deferred'])]
    pods = pods[['permit_num', 'req_acft', 'aquifer', 'geometry']]

    # Get model layer index k
    pods['k'] = 0
    pods.loc[pods['aquifer'] == 'Spiritwood', 'k'] = 2
    pods = pods.drop('aquifer', axis=1)

    # Model grid
    mg = gpd.read_file(
        os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww', 'sw_ww_modelgrid.shp')
        )
    mg['i'] = mg['row'] - 1
    mg['j'] = mg['column'] - 1

    # Spatial join to get i, j
    mg = mg.set_crs(2265)
    pods = pods.to_crs(2265)
    pods.geometry = pods.centroid
    pods = gpd.sjoin(pods, mg[['geometry', 'i', 'j']])
    pods.to_file(os.path.join('..','..','gis','input_shps','sw_ww','def_pend_pods.shp'))
    # -----------------------------------------
    # Build monthly pumping for new permits
    # using expand_annual_to_growing_season()
    # -----------------------------------------
    # Predictive stress periods and their years/months
    spd_dt = spd.copy()
    spd_dt['start_datetime'] = pd.to_datetime(spd_dt['start_datetime'])
    spd_dt['Year'] = spd_dt['start_datetime'].dt.year
    spd_dt['Month'] = spd_dt['start_datetime'].dt.month

    # Years in the predictive period (sp > sp2024)
    print(spd_dt)
    print(sp2024)
    pred_spd = spd_dt.loc[spd_dt['stress_period'] > sp2024].copy()
    pred_years = sorted(pred_spd['Year'].unique())

    # Base table: one row per (permit, year)
    print(pods)
    print(pred_years)
    base_rows = []
    for _, pod in pods.iterrows():
        for yr in pred_years:
            base_rows.append({
                'permit_num': pod['permit_num'],
                # !!! Here, mult will be applied to reduce annual acre-ft
                # for the 'realistic use' scenario
                'use_acft'  : pod['req_acft'],   # annual requested AF
                'use_type'  : 'Irrigation',
                'Year'      : yr,
                'k'         : pod['k'],
                'i'         : pod['i'],
                'j'         : pod['j'],
            })
    pods_base = pd.DataFrame(base_rows)
    # Use the existing helper to spread AF over growing season
    pods_monthly = expand_annual_to_growing_season(pods_base)
    # pods_monthly has columns including Year, Month, cfd

    # Link monthly rows to stress_period via Year + Month
    pods_monthly = pods_monthly.merge(
        spd_dt[['stress_period', 'Year', 'Month']],
        how='left',
        on=['Year', 'Month']
    )

    # Only keep predictive stress periods
    pods_monthly = pods_monthly.loc[pods_monthly['stress_period'] > sp2024].copy()

    # Make sure int columns are ints and convert back to 1-based for modflow input
    pods_monthly['i'] = pods_monthly['i'].astype(int) + 1
    pods_monthly['j'] = pods_monthly['j'].astype(int) + 1
    pods_monthly['k'] = pods_monthly['k'].astype(int) + 1

    # ----------------------------------------------------
    # Write well files for prediction years (sp > sp2024)
    # ----------------------------------------------------
    org_dir = os.path.join(pred_ws, 'org')
    os.makedirs(org_dir, exist_ok=True)

    temp_well_files = []
    for sp in range(sp2024 + 1, ep_end+1):
        # Historical part: grab data for month corresponding to this stress period
        month = pd.Timestamp(
            spd.loc[spd['stress_period'] == sp, 'start_datetime'].values[0]
        ).month
        month_pmp = pmp.loc[pmp['month'] == month].copy()
        # Drop month column
        month_pmp = month_pmp.drop(columns='month')

        # New permits part: pumping for this stress period
        pods_sp = pods_monthly.loc[pods_monthly['stress_period'] == sp].copy()
        if not pods_sp.empty:
            # Rename / map to layer,row,column,flux_cfd
            pods_sp = pods_sp.rename(columns={'k': 'layer', 'i': 'row', 'j': 'column'})
            # cfd from helper is positive; make it negative for pumping
            pods_sp['flux_cfd'] = -pods_sp['cfd'].astype(float).abs()
            pods_sp = pods_sp[['layer', 'row', 'column', 'flux_cfd']]

            # Combine historical + new permits and aggregate
            month_pmp = pd.concat([month_pmp, pods_sp], ignore_index=True)
            # Not sure if agg is necessary since all layer/row/col should be unique
            # and have only one entry per well
            month_pmp = (
                month_pmp
                .groupby(['layer', 'row', 'column'], as_index=False)
                .agg({'flux_cfd': 'sum'})
            )

        # Make sure ints
        month_pmp['layer'] = month_pmp['layer'].astype(int)
        month_pmp['row'] = month_pmp['row'].astype(int)
        month_pmp['column'] = month_pmp['column'].astype(int)

        new_well_file = f'swww.wel_stress_period_data_{sp}.txt'
        month_pmp.to_csv(os.path.join(pred_ws, new_well_file),
                         sep='\t', header=False, index=False)
        month_pmp.to_csv(os.path.join(org_dir, new_well_file),
                         sep='\t', header=False, index=False)
        temp_well_files.append(new_well_file)

    lst_mod_well_files.extend(temp_well_files)

    return lst_mod_well_files


def nopt0chk_full_permit_use(org_ws, pred_ws, modnm, noptmax_flow):
    pst = pyemu.Pst(os.path.join(pred_ws,'swww.pst'))
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_par_en'] = f'{modnm}.{noptmax_flow}.par.jcb'
    pst.pestpp_options["ies_obs_en"] = f'{modnm}.{noptmax_flow}.obs.jcb'
    pst.write(os.path.join(pred_ws,'swww.pst'),version=2)
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


# ---------------------------------------------- #
# Full allocation of rural and municipal permits
# ---------------------------------------------- #
def write_full_alloc_well_files(modnm, pred_ws):
    print("writing full allocation well files...")
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01", 'stress_period'].values[0]
    ep_end = spd['stress_period'].iloc[-1]

    # Find all well files
    lst_mod_well_files = []
    list_files = [
        f for f in os.listdir(pred_ws)
        if 'wel' in f.split("_")[0]
        and f.endswith('.txt')
        and "_stress_period_data_" in f
        ]
    well_sorted = sorted(list_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # --- Historical pumping (last 5 yrs)
    # drop list to well files from past five years (60 months)
    sp_rng = range(sp2024 - 60, sp2024 + 1)
    well_sorted = [
        f for f in well_sorted
        if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng
        ]

    # get median pumping rates from last 5 years of histo matching period per month
    pmp = pd.DataFrame()
    for f in well_sorted:
        df = pd.read_csv(os.path.join(pred_ws, f), delim_whitespace=True, header=None)
        df.columns = ['layer', 'row', 'column', 'flux_cfd']
        df['sp'] = int(re.search(r'_(\d+)\.txt$', f).group(1))
        pmp = pd.concat([pmp, df], axis=0, ignore_index=True)

    # Merge with spd to pull out month information
    pmp = pd.merge(
        pmp,
        spd[['stress_period', 'start_datetime']],
        left_on='sp',
        right_on='stress_period'
        )
    pmp['month'] = pd.to_datetime(pmp['start_datetime']).dt.month

    # Drop columns used for the datetime merge
    pmp = pmp.drop(columns=['sp', 'stress_period', 'start_datetime'])
    pmp = (
        pmp
        .groupby(['month', 'layer', 'row', 'column'])
        .agg({'flux_cfd': 'median'})
        .reset_index()
        )

    # --- Add in additional pumping (new permit PODs)
    # Fully allocated permit input
    full_use = gpd.read_file(
        os.path.join('data', 'analyzed',
                     'full_use_muni_rural_cfd_by_well.csv')
        )
    # Drop estimated well 1
    full_use = full_use.loc[full_use['Well'] != 'Estimated_Well_1']

    # Processed shapefie of pumping wells with spatial loc
    # -> Row/col are 1-indexed, K is zero indexed layer
    pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','wells_avg_pump.shp'))

    # Merge permit use with existing well spatial data
    full_use = pd.merge(full_use,pods,on='Well')

    # Filter/rename columns we need
    full_use['i'] = full_use['row'] - 1
    full_use['j'] = full_use['column'] - 1
    full_use['layer'] = full_use['k'].astype(int) + 1
    full_use = full_use[['row','column','layer','cfd_x']]

    # Distribute pumping evenely throughout the months
    full_use['flux_cfd'] = full_use['cfd_x'].astype(float) * -1

    # --- Build monthly pumping for new permits
    # --- using expand_annual_to_growing_season()
    # --- Predictive stress periods and their years/months
    spd_dt = spd.copy()
    spd_dt['start_datetime'] = pd.to_datetime(spd_dt['start_datetime'])
    spd_dt['Year'] = spd_dt['start_datetime'].dt.year
    spd_dt['Month'] = spd_dt['start_datetime'].dt.month

    # --- Write well files for prediction years (sp > sp2024)
    org_dir = os.path.join(pred_ws, 'org')
    os.makedirs(org_dir, exist_ok=True)

    temp_well_files = []
    for sp in range(sp2024 + 1, ep_end+1):
        # Historical part: grab data for month corresponding to this stress period
        month = pd.Timestamp(
            spd.loc[spd['stress_period'] == sp, 'start_datetime'].values[0]
            ).month
        month_pmp = pmp.loc[pmp['month'] == month].copy()
        # Drop month column
        month_pmp = month_pmp.drop(columns='month')

        # Full allocation part: pumping for this stress period (same for all sps)
        full_aloc = full_use.copy()

        # concat with existing pumping
        month_pmp = pd.concat([full_aloc,month_pmp])
        # New pumping is concatted first, so should work with keep = 'first'
        month_pmp = month_pmp.drop_duplicates(subset=['row','column','layer'],
                                              keep='first')

        # Not sure if agg is necessary since all layer/row/col should be unique
        # and have only one entry per well
        month_pmp = (
            month_pmp
            .groupby(['layer', 'row', 'column'], as_index=False)
            .agg({'flux_cfd': 'sum'})
        )

        # Make sure ints
        month_pmp['layer'] = month_pmp['layer'].astype(int)
        month_pmp['row'] = month_pmp['row'].astype(int)
        month_pmp['column'] = month_pmp['column'].astype(int)

        new_well_file = f'swww.wel_stress_period_data_{sp}.txt'
        month_pmp.to_csv(os.path.join(pred_ws, new_well_file),
                         sep='\t', header=False, index=False)
        month_pmp.to_csv(os.path.join(org_dir, new_well_file),
                         sep='\t', header=False, index=False)
        temp_well_files.append(new_well_file)

    lst_mod_well_files.extend(temp_well_files)

    return lst_mod_well_files


# ----------------------------------------------------------
# --- Full allocation + pending and deferred permit scenario
# ----------------------------------------------------------
def write_full_alloc_plus_def_pend_well_files(modnm, pred_ws, mult=1.0):
    """
    Scenario: max authorized + deferred/pending permits
    built on top of 5-year median monthly baseline pumping.

    - Baseline monthly pumping: median of last 5 years by calendar month
    - Full allocation: overrides baseline wells with full-use pumping
    - Deferred/Pending PODs: added on top for the stress period

    Writes predictive-period WEL files (sp > sp2024) into pred_ws and pred_ws/org.
    """
    # ------------------------
    # Stress period info
    # ------------------------
    spd = pd.read_csv(os.path.join('tables', 'monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01", 'stress_period'].values[0]
    ep_end = int(spd['stress_period'].iloc[-1])

    # ------------------------
    # Find all WEL files
    # ------------------------
    lst_mod_well_files = []
    list_files = [
        f for f in os.listdir(pred_ws)
        if 'wel' in f.split("_")[0]
        and f.endswith('.txt')
        and "_stress_period_data_" in f
    ]
    well_sorted = sorted(list_files, key=lambda f: int(re.search(r'_(\d+)\.txt$', f).group(1)))

    # -------------------------------
    # Historical pumping (last 5 yrs)
    # -------------------------------
    sp_rng = range(sp2024 - 60, sp2024 + 1)
    well_sorted = [f for f in well_sorted if int(re.search(r'_(\d+)\.txt$', f).group(1)) in sp_rng]

    pmp = pd.DataFrame()
    for f in well_sorted:
        df = pd.read_csv(os.path.join(pred_ws, f), delim_whitespace=True, header=None)
        df.columns = ['layer', 'row', 'column', 'flux_cfd']
        df['sp'] = int(re.search(r'_(\d+)\.txt$', f).group(1))
        pmp = pd.concat([pmp, df], axis=0, ignore_index=True)

    pmp = pd.merge(
        pmp,
        spd[['stress_period', 'start_datetime']],
        left_on='sp',
        right_on='stress_period'
    )
    pmp['month'] = pd.to_datetime(pmp['start_datetime']).dt.month
    pmp = pmp.drop(columns=['sp', 'stress_period', 'start_datetime'])
    pmp = (
        pmp
        .groupby(['month', 'layer', 'row', 'column'])
        .agg({'flux_cfd': 'median'})
        .reset_index()
    )

    # -------------------------------
    # Full allocation pumping overlay
    # -------------------------------
    full_use = pd.read_csv(
        os.path.join('data', 'analyzed', 'full_use_muni_rural_cfd_by_well.csv')
        )
    full_use = full_use.loc[full_use['Well'] != 'Estimated_Well_1']

    pods_existing = gpd.read_file(
        os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww', 'wells_avg_pump.shp')
        )

    full_use = pd.merge(full_use, pods_existing, on='Well')

    # Row/column are 1-indexed in that shapefile; k is 0-indexed layer
    full_use['row'] = full_use['row'].astype(int)        # already 1-based
    full_use['column'] = full_use['column'].astype(int)  # already 1-based
    full_use['layer'] = full_use['k'].astype(int) + 1
    full_use['flux_cfd'] = full_use['cfd_x'].astype(float) * -1.0
    full_use = full_use[['layer', 'row', 'column', 'flux_cfd']].copy()

    # -------------------------------------------
    # Deferred/Pending permits pumping (additive)
    # -------------------------------------------
    pods = gpd.read_file(
        os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww',
                     'Spiritwood_Warwick_aquifer_PermitPOD_withDeferred.shp')
        )
    pods = pods.loc[pods['status'].isin(['Pending Review', 'Deferred'])]
    pods = pods[['permit_num', 'req_acft', 'aquifer', 'geometry']]

    pods['k'] = 0
    pods.loc[pods['aquifer'] == 'Spiritwood', 'k'] = 2
    pods = pods.drop('aquifer', axis=1)

    mg = gpd.read_file(
        os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww', 'sw_ww_modelgrid.shp')
        )
    mg['i'] = mg['row'] - 1
    mg['j'] = mg['column'] - 1

    mg = mg.set_crs(2265)
    pods = pods.to_crs(2265)
    pods.geometry = pods.centroid
    pods = gpd.sjoin(pods, mg[['geometry', 'i', 'j']])

    # Build predictive years/months table
    spd_dt = spd.copy()
    spd_dt['start_datetime'] = pd.to_datetime(spd_dt['start_datetime'])
    spd_dt['Year'] = spd_dt['start_datetime'].dt.year
    spd_dt['Month'] = spd_dt['start_datetime'].dt.month

    pred_spd = spd_dt.loc[spd_dt['stress_period'] > sp2024].copy()
    pred_years = sorted(pred_spd['Year'].unique())

    # One row per (permit, year) to feed expand_annual_to_growing_season()
    # All permits are itrigation type so hard setting that here
    base_rows = []
    for _, pod in pods.iterrows():
        for yr in pred_years:
            base_rows.append({
                'permit_num': pod['permit_num'],
                'use_acft'  : pod['req_acft'] * mult,
                'use_type'  : 'Irrigation',
                'Year'      : yr,
                'k'         : pod['k'],
                'i'         : pod['i'],
                'j'         : pod['j'],
            })
    pods_base = pd.DataFrame(base_rows)

    pods_monthly = expand_annual_to_growing_season(pods_base)

    pods_monthly = pods_monthly.merge(
        spd_dt[['stress_period', 'Year', 'Month']],
        how='left',
        on=['Year', 'Month']
        )
    pods_monthly = pods_monthly.loc[pods_monthly['stress_period'] > sp2024].copy()

    # Convert to MODFLOW 1-based indices + make pumping negative
    pods_monthly['row'] = pods_monthly['i'].astype(int) + 1
    pods_monthly['column'] = pods_monthly['j'].astype(int) + 1
    pods_monthly['layer'] = pods_monthly['k'].astype(int) + 1
    pods_monthly['flux_cfd'] = -pods_monthly['cfd'].astype(float).abs()
    pods_monthly = pods_monthly[['stress_period', 'layer', 'row', 'column', 'flux_cfd']]

    # -------------------------------
    # Write predictive-period files
    # -------------------------------
    org_dir = os.path.join(pred_ws, 'org')
    os.makedirs(org_dir, exist_ok=True)

    temp_well_files = []
    for sp in range(sp2024 + 1, ep_end + 1):
        # baseline median pumping for this month
        month = pd.Timestamp(spd.loc[spd['stress_period'] == sp, 'start_datetime'].values[0]).month
        month_pmp = pmp.loc[pmp['month'] == month].copy().drop(columns='month')

        # 1) Full allocation overrides baseline where overlapping
        # concat full_use first so keep='first' keeps full allocation
        month_pmp = pd.concat([full_use.copy(), month_pmp], ignore_index=True)
        month_pmp = month_pmp.drop_duplicates(subset=['layer', 'row', 'column'], keep='first')

        # 2) Deferred/pending adds on top (sum if overlaps)
        pods_sp = pods_monthly.loc[pods_monthly['stress_period'] == sp, ['layer','row','column','flux_cfd']].copy()
        if not pods_sp.empty:
            month_pmp = pd.concat([month_pmp, pods_sp], ignore_index=True)
            month_pmp = (
                month_pmp
                .groupby(['layer', 'row', 'column'], as_index=False)
                .agg({'flux_cfd': 'sum'})
            )

        # Ensure ints
        month_pmp['layer'] = month_pmp['layer'].astype(int)
        month_pmp['row'] = month_pmp['row'].astype(int)
        month_pmp['column'] = month_pmp['column'].astype(int)

        # Write new files
        new_well_file = f'swww.wel_stress_period_data_{sp}.txt'
        month_pmp.to_csv(os.path.join(pred_ws, new_well_file), sep='\t', header=False, index=False)
        month_pmp.to_csv(os.path.join(org_dir, new_well_file), sep='\t', header=False, index=False)
        temp_well_files.append(new_well_file)

    lst_mod_well_files.extend(temp_well_files)
    return lst_mod_well_files


# ------------------------------
# --- ASR scenario
# ------------------------------
def write_full_alloc_def_pend_plus_asr_well_files(modnm, pred_ws, mult):
    """
    Scenario: full allocation + deferred/pending permits + ASR injection

    Assumes you already want the exact same logic as:
      - baseline 5-yr median monthly
      - full allocation overrides baseline
      - def/pend adds on top
    AND THEN:
      - ASR wells add positive injection for every predictive stress period

    Writes predictive WEL files (sp > sp2024) into pred_ws and pred_ws/org.
    """
    ASR_INJECTION = 2580.0

    # --- Build the "full alloc + def/pend" scenario first w/ only mult% expected p/d use
    # --> mult is 50%
    lst_mod_well_files = write_full_alloc_plus_def_pend_well_files(modnm, pred_ws, mult=mult)

    # --- Build ASR injection table (constant cfd each month)
    asr_wells = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','asr_locations_13.shp'))

    mg = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww','sw_ww_modelgrid.shp'))
    mg['i'] = mg['row'] - 1
    mg['j'] = mg['column'] - 1

    mg = mg.set_crs(2265)
    if asr_wells.crs != 2265:
        asr_wells = asr_wells.to_crs(2265)

    asr_wells = gpd.sjoin(asr_wells, mg[['geometry', 'i', 'j']], how='left')
    if asr_wells['i'].isna().any() or asr_wells['j'].isna().any():
        bad = asr_wells.loc[asr_wells['i'].isna() | asr_wells['j'].isna()]
        raise ValueError(f"{len(bad)} ASR locations did not join to model grid (check CRS/extent).")

    # cleanup
    for c in ['index_right', 'id']:
        if c in asr_wells.columns:
            asr_wells = asr_wells.drop(columns=c)

    # layer k=2 (0-based) -> layer 3 (1-based) in file
    asr_wells['k'] = 2  # All ASR in Spiritwood
    asr_wells['layer'] = asr_wells['k'] + 1
    asr_wells['row'] = asr_wells['i'].astype(int) + 1
    asr_wells['column'] = asr_wells['j'].astype(int) + 1

    # Total injection volume (ac-ft/yr), split evenly among wells
    total_acft_yr = ASR_INJECTION
    asr_wells['acre_ft_yr'] = total_acft_yr / len(asr_wells)

    # Convert to cfd, uniform through year
    # NOTE: positive for injection
    asr_wells['flux_cfd'] = asr_wells['acre_ft_yr'] * 43560.0 / 365.25

    # ASR data to add to each stress period in predictive period
    asr_add = asr_wells[['layer','row','column','flux_cfd']].copy()

    # --- Apply ASR injection to every predictive stress period file
    spd = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    sp2024 = spd.loc[spd['start_datetime'] == "2023-12-01",'stress_period'].values[0]
    ep_end = int(spd['stress_period'].iloc[-1])

    org_dir = os.path.join(pred_ws, 'org')
    os.makedirs(org_dir, exist_ok=True)

    for sp in range(sp2024 + 1, ep_end + 1):
        wel_file = os.path.join(pred_ws, f'swww.wel_stress_period_data_{sp}.txt')
        if not os.path.exists(wel_file):
            raise FileNotFoundError(wel_file)

        df = pd.read_csv(wel_file, delim_whitespace=True, header=None,
                         names=['layer','row','column','flux_cfd'])

        # Add ASR (positive) and sum if overlaps
        df = pd.concat([df, asr_add], ignore_index=True)
        df = (df.groupby(['layer','row','column'], as_index=False)
                .agg({'flux_cfd':'sum'}))

        # Write back (and also update org copy)
        df.to_csv(wel_file, sep='\t', header=False, index=False)
        df.to_csv(os.path.join(org_dir, f'swww.wel_stress_period_data_{sp}.txt'),
                  sep='\t', header=False, index=False)

    return lst_mod_well_files


def znbud_by_ly_process(modnm='swww'):
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


def init_zonbud_process(d,modnm='swww'):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = znbud_by_ly_process(modnm=modnm)
    os.chdir(b_d)
    return dfs


def rewrite_zbud_ins_file(modnm='swww', pred_ws='.'):
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
def plot_scn_hydrograpghs(pred_ws_list, modnm='swww',plot_quantiles=True,plt_base_only=False):
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
    cpts['grpid'] = 'grpid:' + cpts['group_number'].astype(str) + '_k:' + cpts['k'].astype(str)
    # make geodataframe from geometry column:
    cpts = gpd.GeoDataFrame(data=cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=2265)
    cpts = cpts.groupby(['grpid']).last().reset_index()
    cpts.columns = cpts.columns.str.lower()

    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww')
    sw_extent = gpd.read_file(os.path.join(g_d,'sw_extent_SJ.shp')).to_crs(2265)
    ww_extent = gpd.read_file(os.path.join(g_d,'warwick_boundary_model.shp')).to_crs(2265)
    k_barrier = gpd.read_file(os.path.join(g_d,'HFB_V7.shp')).to_crs(2265)
    rch_windows = gpd.read_file(os.path.join(g_d,'sw_recharge_window_large.shp')).to_crs(2265)
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'sw_ww_modelgrid.shp'))
    modelgrid = modelgrid.set_crs(2265)
    drains = gpd.read_file(os.path.join(g_d, 'RIV_lines.shp'))

    # Load the processed the PODs
    full_aloc_wells = ['15006106BBC',
                       '15006201DDA8',
                       '15006201DDB2',
                       '15006201DDC2',
                       '15006201DDDD2',
                       '15106131CBC',
                       '15106131CCC',
                       '15106334CAA',
                       'Estimated_Well_1',
                       ]
    full_aloc_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','wells_avg_pump.shp')).set_crs(2265)
    full_aloc_pods = full_aloc_pods.loc[full_aloc_pods['Well'].isin(full_aloc_wells)]
    def_pend_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','def_pend_pods.shp'))

    # Load WLs
    wls = pd.read_csv(os.path.join('data', 'raw','swww_sites_final.csv'))
    wls = wls.loc[wls['manually_corrected_lay']!='na']
    wls['grp_id'] = 'grpid:' + wls['group number'].astype(str) + '_k:' + (wls['manually_corrected_lay'].astype(int)-1).astype(str)

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

    colors = [
        "#b7bbc0",  # scn01 - grey (baseline)
        "#f1a924",  # scn02 - orange (full permit use)
        "#9e1818",  # scn03 - red (ultimate drought)
        "#5a812c",  # scn04 - green (drought + asr at 500 acft/yr)
        "#49b6e9",  # scn05 - light blue (drought + asr at 200 acft/yr)
        "#5a812c",  # scn06 - purple (full + def/pend + ASR)
        "#d64fa1",  # scn07 - magenta / pink
        ]

    darker_colors = [
        "#7f8286",  # scn01 - dark grey
        "#d16217",  # scn02 - dark orange
        "#6e1212",  # scn03 - dark red
        "#3e5a1f",  # scn04 - dark green
        "#1f6f9c",  # scn05 - dark blue
        "#3e5a1f",  # scn06 - dark purple
        "#8f2f6b",  # scn07 - dark magenta
        ]

    years10 = mdates.YearLocator(10)
    years_fmt = mdates.DateFormatter('%Y')

    with PdfPages(os.path.join(o_d, 'scenario_hydrograpghs.pdf')) as pdf:
        # ---- Head difference/depletion plots for each scenario
        # Load basemap tiles
        west, south, east, north = modelgrid.to_crs(3857).total_bounds
        # img, ext = cx.bounds2img(west,
        #                          south,
        #                          east,
        #                          north,
        #                          # source=cx.providers.USGS.USTopo
        #                          )
        # img, ext = cx.warp_tiles(img,ext,t_crs=2265)

        # Base run heads at 2023-12-01
        hds_base = flopy.utils.HeadFile(os.path.join('master_flow_Dev9_weightFlooding_forward_run_base',"swww.hds")).get_data(kstpkper=(0,318))
        modelgrid['base_heads_ww'] = hds_base[0,:,:].flatten()
        modelgrid['base_heads_sw'] = hds_base[2,:,:].flatten()


        for ws_idx, ws in enumerate(pred_ws_list):

            # Skip these scenarios in the plots for now
            ws = '_'.join(ws.split('_')[0:-1])
            fig,axes = plt.subplots(1,2,figsize=(12,8))
            # Scenario heads at end of scenario
            hds_scen = flopy.utils.HeadFile(os.path.join(ws,"swww.hds")).get_data(kstpkper=(0,558))

            # Visualize depletion
            # Scenario - Base, so negative values represent WL drops, postive is WL increase
            modelgrid['dep_ww'] = hds_scen[0,:,:].flatten() - modelgrid['base_heads_ww']
            modelgrid['dep_ww'] = np.where(modelgrid['dep_ww'] == 0, np.nan, modelgrid['dep_ww'])
            modelgrid['dep_sw'] = hds_scen[2,:,:].flatten() - modelgrid['base_heads_sw']
            modelgrid['dep_sw'] = np.where(modelgrid['dep_sw'] == 0, np.nan, modelgrid['dep_sw'])
            mean_ww = round(np.nanmean(modelgrid['dep_ww']),1)
            mean_sw = round(np.nanmean(modelgrid['dep_sw']),1)

            vmin1 = np.nanmin(modelgrid['dep_ww'])
            vmax1 = np.nanmax(modelgrid['dep_ww'])
            vmin2 = np.nanmin(modelgrid['dep_sw'])
            vmax2 = np.nanmax(modelgrid['dep_sw'])
            if vmin1 < vmin2:
                vmin = vmin1
            else:
                vmin = vmin2
            if vmax1 > vmax2:
                vmax = vmax1
            else:
                vmax = vmax2

            # Normalize around zero
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

            # Warwick
            modelgrid.plot(column='dep_ww',
                           ax=axes[0],
                           cmap='coolwarm_r',
                           legend=True,
                           legend_kwds={'shrink':0.6,
                                        'label':'Head Difference'
                                        },
                           vmin=vmin,
                           vmax=vmax,
                           norm=norm
                           )
            axes[0].set_title(f"Warwick: Mean = {mean_ww}")

            # Spiritwood
            modelgrid.plot(column='dep_sw',
                           ax=axes[1],
                           cmap='coolwarm_r',
                           legend=True,
                           legend_kwds={'shrink':0.6,
                                        'label':'Head Difference'
                                        },
                           vmin=vmin,
                           vmax=vmax,
                           norm=norm
                           )
            axes[1].set_title(f"Spiritwood: Mean = {mean_sw}")


            # -- Warwick
            # Plot the wells by type for scenario clarity -> Rural/Municipal
            full_aloc_pods.loc[full_aloc_pods['k']==0].plot(ax=axes[0],
                                                            color='#f1a924',
                                                            label='Municipal/Rural',
                                                            markersize=15,
                                                            edgecolor='k'
                                                            )
            # Plot the wells by type for scenario clarity -> Deferred/pending
            def_pend_pods.loc[def_pend_pods['k']==0].plot(ax=axes[0],
                                                          color='#9e1818',
                                                          label='Deferred/Pending',
                                                          markersize=15,
                                                          edgecolor='k'
                                                          )
            # -- Spiritwood
            # Plot the wells by type for scenario clarity -> Rural/Municipal
            full_aloc_pods.loc[full_aloc_pods['k']==2].plot(ax=axes[1],
                                                            color='#f1a924',
                                                            label='Municipal/Rural',
                                                            markersize=15,
                                                            edgecolor='k'
                                                            )
            # Plot the wells by type for scenario clarity -> Deferred/pending
            def_pend_pods.loc[def_pend_pods['k']==2].plot(ax=axes[1],
                                                          color='#9e1818',
                                                          label='Deferred/Pending',
                                                          markersize=15,
                                                          edgecolor='k'
                                                          )

            # Add basemap but preserve lims
            ylims = axes[1].get_ylim()
            xlims = axes[1].get_xlim()
            for ax in axes:
                # ax.imshow(img,
                #           extent=ext,
                #           origin="upper",
                #           zorder=0,
                #           alpha=0.6)
                ax.set_ylim(ylims)
                ax.set_xlim(xlims)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend()

            # Need to fix this lol
            y = 0.95
            if 'baseline' in ws:
                fig.suptitle("Baseline Head Difference\nNegative Values Imply Depletion",
                             y=y)
            elif 'pending' in ws:
                fig.suptitle("Deferred + Pending Head Difference\nNegative Values Imply Depletion",
                             y=y)
            elif 'full_alloc_def_pend_realUse' in ws:
                fig.suptitle("Full Allocation + Deferred (realistic use) + Pending Head Difference\nNegative Values Imply Depletion",
                             y=y)
            elif 'full_alloc_def_pend' in ws:
                fig.suptitle("Full Allocation + Deferred + Pending Head Difference\nNegative Values Imply Depletion",
                             y=y)
            elif 'ghb_low_stage' in ws:
                fig.suptitle("Full Allocation + GHB Declining Stage Head Difference\nNegative Values Imply Depletion",
                             y=y)
            elif 'full_alloc_asr_def_pend' in ws:
                fig.suptitle("Full Allocation + Deferred + Pending with ASR Head Difference\nNegative Values Imply Depletion",
                             y=y)
            else:
                fig.suptitle("Full Allocation Head Difference\nNegative Values Imply Depletion",
                             y=y)
            fig.tight_layout()

            # Save fig or not
            # pdf.savefig()
            plt.close(fig)

        # ---- Hydrographs
        scenario_base_records = []

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

            swww_plot.plot_inset(ax=ax1,
                                 wl_loc=site,
                                 cpts=cpts,
                                 sw_extent=sw_extent,
                                 ww_extent=ww_extent,
                                 k_barrier=k_barrier,
                                 rch_windows=rch_windows,
                                 drains=drains)
            swww_plot.plot_vert_xsec(ax2,
                                     m_d,
                                     m,
                                     wl_loc=site,
                                     mwl=wobs.obsval.mean(),
                                     cpts=cpts)

            # Plot the wells by type for scenario clarity -> Rural/Municipal
            #full_aloc_pods.plot(ax=ax1,
            #                    color='#f1a924'
            #                    )
            # Plot the wells by type for scenario clarity -> Deferred/pending
            #def_pend_pods.plot(ax=ax1,
            #                   color='#9e1818')

            stor_ymx = []
            stor_ymn = []
            cnt = 0
            for ws_idx, ws in enumerate(pred_ws_list):

                scn_tag = ws.split('_')[2]
                scn_num = int(scn_tag.replace('scn',''))

                #!!!
                if scn_num in [3,4,5]:
                    cnt += 1
                    continue

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

                # plot base:
                min_val = vals.min()
                max_val = vals.max()
                base_vals = scn_results_dict[scn_tag].loc[:, uobs.obsnme].loc['base', :].values
                ax4.plot(dts, base_vals, color=darker_colors[scn_num-1], lw=1.5, zorder=10)

                for dt, bval in zip(pd.to_datetime(dts), base_vals):
                    scenario_base_records.append({
                        "scenario": scn_tag,
                        "id": site,
                        "k": int(k),
                        "i": int(i),
                        "j": int(j),
                        "datetime": dt,
                        "base_real": float(bval),
                    })

                # Add a horizontal line showing base value at last historical SP
                if scn_num == 1:
                    ax4.axhline(base_vals[319],
                                color='k',
                                ls='--',
                                lw=1,
                                )

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
                ax4.set_xlim(pd.to_datetime('2020-01-01'), pd.to_datetime('2045-01-01'))

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
                ax4.axvline(x=pd.to_datetime('2024-01-01'), color='grey', linestyle='-.', linewidth=1)
                if cnt == len(pred_ws_list) - 1:
                    ax4.text(pd.to_datetime('2025-01-01'), ax4.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                            ha='left', va='bottom', color='grey')

                # --- PROXY HANDLES (make sure styles match your plots) ---
                leg_s1  = mlines.Line2D([], [], color=colors[0], lw=1.5, label='Scenario 0 - Baseline')
                leg_s2  = mlines.Line2D([], [], color=colors[1], lw=1.5, label='Scenario 1 - Full Authorized Use')
                leg_s4  = mlines.Line2D([], [], color=colors[6], lw=1.5, label='Scenario 2 - Pending & Deferred Permits')
                leg_s5  = mlines.Line2D([], [], color=colors[5], lw=1.5, label='Scenario 3 - ASR')

                # grass/top-of-model line
                leg_top = mlines.Line2D([], [], color='green', lw=1.2, label='Model top (ground surface)')
                # layer contacts (match your plotted style: you used 'c--')
                leg_cnt = mlines.Line2D([], [], color='c', lw=1.5, linestyle='--', alpha=0.5, label='Layer contacts')
                # predictive period (optional: include it if you want)
                leg_pred = mlines.Line2D([], [], color='grey', lw=1.0, linestyle='-.', label='Predictive period')
                # 2024 base value
                leg_24 = mlines.Line2D([], [], color='k', lw=1, linestyle='--', label='2024 Water Level')

                # --- SPLIT INTO TWO LEGENDS ---
                scenario_handles = [leg_s1, leg_s2, leg_s4, leg_s5]
                other_handles = [leg_top, leg_cnt, leg_pred, leg_24]

                # Legend A: scenarios (two columns), centered at bottom
                legA = fig.legend(
                    handles=scenario_handles,
                    loc='lower center',
                    bbox_to_anchor=(0.62, 0.11),  # x=center, y a bit above bottom edge
                    ncol=2,
                    frameon=True,
                    framealpha=0.6,
                    fontsize=9,
                )

                # Legend B: other items (single row), centered just above Legend A
                legB = fig.legend(
                    handles=other_handles,
                    loc='lower center',
                    bbox_to_anchor=(0.17, 0.08),  # slightly higher than legA
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

                    sites_grp = sites_grp[sites_grp['manually_corrected_lay'].astype(int)-1 == int(k)]
                    aq_key = {0: "Warwick-1",
                              1: "Confining Till-2",
                              2: "Spiritwood-3",
                              }
                    current_aq = aq_key.get(k)

                    # sort grp_full by manually_corrected_lay
                    grp_full = grp_full.sort_values(by=['manually_corrected_lay'])
                    if idx_well.any():
                        ax3.text(0.5, 0.8,
                                f'{current_aq}\n Group: {grp_num}\n ***Index Well***\n Layer: {k + 1}, \nRow: {int(i) + 1}, Column: {int(j) + 1}\nModel top: {t}\n\n',
                                fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
                    else:
                        ax3.text(0.5, 0.75,
                                f'{current_aq}\n Group: {grp_num}\n Layer: {k + 1}, \nRow: {int(i) + 1}, Column: {int(j) + 1}\nModel top: {t}\n\n',
                                fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)

                    # add text that icludes grp_full [loc_id, assigned aquifer]:
                    ax3.text(0.5, 0.27, 'All wells in group:\n' + '\n'.join(
                        [f"{row['loc_id']} - {aq_key.get(int(row['manually_corrected_lay'])-1)}" for idx, row in grp_full.iterrows()]),
                            fontsize=9, ha='center', va='center', color='black', transform=ax3.transAxes)
                cnt+=1
            pdf.savefig()
            plt.close(fig)
            print(f'  finished plotting hydrograph for site {site}...')
    pdf.close()

    # Save the scenario results
    if len(scenario_base_records) > 0:
        df_scn_base = pd.DataFrame(scenario_base_records)
        df_scn_base = df_scn_base.sort_values(["scenario", "id", "datetime"]).reset_index(drop=True)

        out_csv = os.path.join(o_d, "scenario_targets_base_longform.csv")
        df_scn_base.to_csv(out_csv, index=False)
        print(f"...Wrote scenario base export: {out_csv}")
    else:
        print("...No scenario base records collected; CSV not written.")


def plot_scn_net_water_budgets(pred_ws_list, modnm='swww',plot_quantiles=True,
                              lay=None):

    fdir = os.path.join( 'results', 'figures', 'scenario_results')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    m_d = pred_ws_list[0]
    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    bobs = obs.loc[obs.obsnme.str.contains('bud'), :]
    bobs.loc[:, 'datetime'] = pd.to_datetime(bobs.datetime)

    pdf = PdfPages(os.path.join(fdir, 'net_budget_swww_scns.pdf'))

    inobs = bobs.loc[bobs.obsnme.apply(lambda x: '_in' in x or 'from' in x and 'bud' in x), :].copy()
    outobs = bobs.loc[bobs.obsnme.apply(lambda x: '_out' in x or 'to_' in x and 'bud' in x), :].copy()
    inobs['k'] = inobs.obsnme.str.split('-').str[-1].astype(int)
    outobs['k'] = outobs.obsnme.str.split('-').str[-1].astype(int)

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to specific layer if specified
    # !!! Need to add layer information to the observations....
    if lay:
        ins = inobs.loc[inobs.k == lay, :].copy()
        outs = outobs.loc[outobs.k == lay, :].copy()
    else:
        ins = inobs.copy()
        outs = outobs.copy()

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

    pr_ins = scn_results_dict['scn01']._df.loc[:, ins.obsnme]
    pr_outs = scn_results_dict['scn01']._df.loc[:, outs.obsnme]
    pt_ins = scn_results_dict['scn01']._df.loc[:, ins.obsnme]
    pt_outs = scn_results_dict['scn01']._df.loc[:, outs.obsnme]

    # if column sums to zero, drop it
    pr_ins = pr_ins.loc[:, (pr_ins != 0).any(axis=0)]
    pr_outs = pr_outs.loc[:, (pr_outs != 0).any(axis=0)]
    pt_ins = pt_ins.loc[:, (pt_ins != 0).any(axis=0)]
    pt_outs = pt_outs.loc[:, (pt_outs != 0).any(axis=0)]

    in_types = pr_ins.columns.str.split(':').str[3].unique().str.replace('_datetime', '', regex=True).str.replace('zbly_', '', regex=True).tolist()
    out_types = pr_outs.columns.str.split(':').str[3].unique().str.replace('_datetime', '', regex=True).str.replace('zbly_', '', regex=True).tolist()

    # remove sto from types:
    in_types = [s for s in in_types if 'sto' not in s]
    out_types = [s for s in out_types if 'sto' not in s]

    # Hack for ASR
    in_types.append('wel_wel_0_in')

    # Check types
    print(in_types)
    print(out_types)

    factor = 0.00002296 * 365.25 # cf to acre-ft per year

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

    colors = [
        "#b7bbc0",  # scn01 - grey (baseline)
        "#f1a924",  # scn02 - orange (full permit use)
        "#9e1818",  # scn03 - red (ultimate drought)
        "#5a812c",  # scn04 - green (drought + asr at 500 acft/yr)
        "#49b6e9",  # scn05 - light blue (drought + asr at 200 acft/yr)
        "#5a812c",  # scn06 - purple (full + def/pend + ASR)
        "#d64fa1",  # scn07 - magenta / pink
        ]

    darker_colors = [
        "#7f8286",  # scn01 - dark grey
        "#d16217",  # scn02 - dark orange
        "#6e1212",  # scn03 - dark red
        "#3e5a1f",  # scn04 - dark green
        "#1f6f9c",  # scn05 - dark blue
        "#3e5a1f",  # scn06 - dark purple
        "#8f2f6b",  # scn07 - dark magenta
        ]

    # plot water balance by layer
    if not lay:
        pdf = PdfPages(os.path.join(fdir, 'water_balance_net.pdf'))
    else:
        pdf = PdfPages(os.path.join(fdir, f'water_balance_layer_{lay}.pdf'))
    fig, ax = plt.subplots(figsize=(9, 4))
    for ws in pred_ws_list:
        scn_tag = ws.split('_')[2]
        scn_num = int(scn_tag.replace('scn',''))

        # Skip GHB scenarios
        if scn_num in [3,4,5]:
            continue


        pt_ins = scn_results_dict[scn_tag]._df.loc[:, ins.obsnme]
        pt_outs = scn_results_dict[scn_tag]._df.loc[:, outs.obsnme]

        in_df = pd.DataFrame()
        for i, in_type in enumerate(in_types):
            mask = pt_ins.columns.str.contains(in_type)
            cols = pt_ins.columns[mask]
            ins_map = ins.set_index('obsnme')
            dates = ins_map.loc[cols, 'datetime'].to_numpy()
            for j in pt_ins.index:
                s = pt_ins.loc[str(j), mask].to_numpy()
                if np.isclose(np.nansum(s), 0.0):  # skip if sums to zero (treats NaNs as 0)
                    continue
                bud = s * factor
                temp_df = pd.DataFrame({'datetime': dates, 'value': bud, 'realization': str(j), 'type': in_type})
                in_df = pd.concat([in_df, temp_df], ignore_index=True)
        # print(in_df.type.unique())
        out_df = pd.DataFrame()
        for i, out_type in enumerate(out_types):
            mask = pt_outs.columns.str.contains(out_type)
            cols = pt_outs.columns[mask]
            outs_map = outs.set_index('obsnme')
            dates = outs_map.loc[cols, 'datetime'].to_numpy()

            for j in pt_outs.index:
                s = pt_outs.loc[str(j), mask].to_numpy()
                if np.isclose(np.nansum(s), 0.0):  # skip if sums to zero (treats NaNs as 0)
                    continue
                bud = s * factor
                temp_df = pd.DataFrame({'datetime': dates, 'value': bud, 'realization': str(j), 'type': out_type})
                out_df = pd.concat([out_df, temp_df], ignore_index=True)

        # --- Convert monthly periods from yearly to monthly values
        in_df['year'] = pd.DatetimeIndex(in_df['datetime']).year
        out_df['year'] = pd.DatetimeIndex(out_df['datetime']).year
        in_df.loc[in_df['year'] >= 2000, 'value'] /= 12
        out_df.loc[out_df['year'] >= 2000, 'value'] /= 12

        print('--'*10)
        print(out_df.type.unique())
        print(scn_tag)
        net_df = pd.DataFrame()
        reals = in_df['realization'].unique()
        for real in reals:
            in_real = in_df.loc[in_df['realization'] == real, :].copy()
            out_real = out_df.loc[out_df['realization'] == real, :].copy()

            # Plot yearly totals
            in_tots = in_real.groupby('year')['value'].sum()
            out_tots = out_real.groupby('year')['value'].sum()
            bnet = in_tots - out_tots
            temp_df = pd.DataFrame({'year': bnet.index, 'net_budget': bnet.values, 'realization': real})
            net_df = pd.concat([net_df, temp_df], ignore_index=True)

        net_vals_acft = net_df.pivot(index='year', columns='realization', values='net_budget')

        # quantiles
        if plot_quantiles:
            q10 = np.percentile(net_vals_acft.values, 10, axis=1)
            q90 = np.percentile(net_vals_acft.values, 90, axis=1)
            years = net_vals_acft.index.values
            ax.fill_between(years[0:-1], q10[0:-1], q90[0:-1], color=colors[scn_num-1], alpha=0.2, zorder=1)
            ax.plot(years[0:-1], q10[0:-1], color=colors[scn_num-1], linestyle='--', lw=1.0, zorder=5)
            ax.plot(years[0:-1], q90[0:-1], color=colors[scn_num-1], linestyle='--', lw=1.0, zorder=5)

        # plot base:
        # Cut out last year #!!! Need to look into this...
        years = years[0:-1]
        net_bud = net_vals_acft[['base']].values[0:-1]
        ax.plot(years, net_bud, color=darker_colors[scn_num-1], lw=1.5, zorder=10)

    ax.set_ylabel('Net water budget (acre-ft/yr)\n(More negative values indicate greater outflow)', fontsize=11)

    # add vertical line for predictive period:
    ax.axvline(x=2024, color='grey', linestyle='-.', linewidth=1)
    ax.text(2024, ax.get_ylim()[0] + 10, '-> Predictive period', fontsize=10,
            ha='left', va='bottom', color='grey')
    ax.axhline(0,color='grey',linewidth=0.8,ls='--')
    # comma formateed y-axis labels:
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
    ax.get_xaxis().set_tick_params(direction='in')
    ax.tick_params(axis='both',direction='in', which='major', labelsize=9)
    ax.grid()
    # add space for legend
    fig.subplots_adjust(bottom=0.15)
    leg_s1  = mlines.Line2D([], [], color=colors[0], lw=1.5, label='Scenario 0 - Baseline')
    leg_s2  = mlines.Line2D([], [], color=colors[1], lw=1.5, label='Scenario 1 - Full Authorized Use')
    leg_s4  = mlines.Line2D([], [], color=colors[6], lw=1.5, label='Scenario 2 - Pending & Deferred Permits')
    leg_s5  = mlines.Line2D([], [], color=colors[5], lw=1.5, label='Scenario 3 - ASR')
    # --- SPLIT INTO TWO LEGENDS ---
    scenario_handles = [leg_s1, leg_s2, leg_s4, leg_s5]

    # Legend A: scenarios (two columns), centered at bottom
    fig.legend(handles=scenario_handles,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.01),  # x=center, y a bit above bottom edge
               ncol=2,
               frameon=True,
               framealpha=0.6,
               fontsize=8,
               )

    pdf.savefig(fig,bbox_inches='tight')
    plt.savefig('water_balance_net.png',
                dpi=250,
                bbox_inches='tight'
                )
    plt.close(fig)
    pdf.close()

# ------------------------------
# ---- Zone budget functions
# ------------------------------
def write_zbud_nam_file(zb_path, cbcPath, grbPath):
    with open(os.path.join(zb_path, 'zbud.nam'), 'w') as f:
        f.write('BEGIN ZONEBUDGET\n')
        f.write(f"  BUD '{cbcPath}'\n")
        f.write('  ZON zones.dat\n')
        f.write(f"  GRB '{grbPath}'\n")
        f.write('END ZONEBUDGET\n')

def write_zone_file(zb_path, zones, ncpl):
    with open(os.path.join(zb_path, 'zones.dat'), 'w') as f:
        f.write('BEGIN DIMENSIONS\n')
        f.write(f'  NCELLS {ncpl}\n')
        f.write('END DIMENSIONS\n')
        f.write('\n')
        f.write('BEGIN GRIDDATA\n')
        f.write('  IZONE\n')
        f.write('  INTERNAL\n')
        np.savetxt(f, zones, fmt='%i')
        f.write('END GRIDDATA\n')

def run_zonebudget(zb_path, org_d):
    import subprocess
    os.chdir(zb_path)

    subprocess.call([os.path.join(os.getcwd(), 'zbud6')])

    os.chdir(org_d)

def run_zb_by_layer(w_d='',modnm='swww', plot=True,scen_tag=''):

    # plot forward run ZB results
    print(f'\n\n\nRun ZB and plot results for layered Model\n\n\n')
    sim = flopy.mf6.MFSimulation.load(sim_ws=w_d,load_only=['dis'])
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
    zon_path = os.path.join(w_d,'results', 'zb', f'layer_zb')

    if not os.path.exists(zon_path):
        os.makedirs(zon_path)

    prep_deps(zon_path) # add zb exe
    # Make zone file
    ncpl = nrow * ncol * nlay
    write_zone_file(zon_path, zon_file.zone.astype(int).values, ncpl)

    # Make ZB nam file
    org_d = os.getcwd()
    cbb = os.path.join(org_d, w_d, f'{modnm}.cbb')
    grb = os.path.join(org_d, w_d, f'{modnm}.dis.grb')
    write_zbud_nam_file(zon_path, cbb, grb)

    # Run ZB
    run_zonebudget(zon_path, org_d)

    if plot:
        zb = pd.read_csv(os.path.join(zon_path,'zbud.csv'))
        # Remove storage terms
        zb = zb.loc[:,~zb.columns.str.contains('STO')]
        # Convert ft3/d to acre-ft/yr
        zb.iloc[:,4:] = zb.iloc[:,4:] / 43560 * 365.25
        zb.columns = zb.columns.str.replace('ZONE', 'Layer',regex=True)

        start_datetime = pd.to_datetime(sim.tdis.start_date_time.array)

        IN_COLORS = [
            '#8c510a',  # brown
            '#35978f',  # green-blue teal
            '#01665e',  # darker teal/blue
            '#3288bd',  # blue
            '#7b3294',  # purple
            '#762a83',  # darker purple
            '#542788',  # deep purple
            '#2d004b',  # almost black-purple
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
        ]

        # plot of water balance by layer
        pdf = PdfPages(os.path.join(zon_path, f'ZB_by_lay_{scen_tag}.pdf'))
        plt.rcParams.update({'font.size': 10})
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))

        for lay in range(nlay):
            df = zb[zb.zone == lay + 1]
            # keep only columns that have some mass
            df = df.loc[:, [f for f in df.columns if df[f].sum() > 1]]

            outs = df.loc[:, (df.columns.str.contains('-OUT')) | (df.columns.str.contains('TO'))].copy()
            if outs.columns.str.contains('WEL').sum() > 1:
                # calculate a total pumping column
                outs['WEL-OUT-TOTAL'] = outs.loc[:, outs.columns.str.contains('WEL')].sum(axis=1)

            ins = df.loc[:, (df.columns.str.contains('-IN')) | (df.columns.str.contains('FROM'))].copy()

            dts = (pd.to_datetime(start_datetime) + pd.to_timedelta(df.totim, unit='d'))

            # sort by magnitude (descending), drop all-zero columns
            tmp = outs.sum()
            out_mag = tmp[tmp != 0].sort_values(ascending=False)
            outs = outs.loc[:, out_mag.index.values]

            tmp = ins.sum()
            in_mag = tmp[tmp != 0].sort_values(ascending=False)
            ins = ins.loc[:, in_mag.index.values]

            # Get mins/maxes for y-lims per-layer plots
            if ins.shape[1] > 0 and outs.shape[1] > 0:
                ymax = np.max(np.concatenate([ins.max().values, outs.max().values]))
            elif ins.shape[1] > 0:
                ymax = ins.max().max()
            elif outs.shape[1] > 0:
                ymax = outs.max().max()
            else:
                # nothing to plot for this layer; skip
                continue
            ymax = max(10, float(ymax))  # guard lower bound for log scale

            # Choose discrete colors (slice to count)
            cin  = IN_COLORS[:min(len(ins.columns), 8)]
            cout = OUT_COLORS[:min(len(outs.columns), 8)]

            # per-layer figure (Ins vs Outs)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # --- Fix monthly periods
            # Ins
            ins.index = dts
            ins.index = pd.to_datetime(ins.index)
            ins.loc[ins.index >= pd.Timestamp("2000-01-01"),[x for x in ins.columns if x != 'datetime']] /= 12
            ins = ins.resample('YS').sum()
            ins = ins.head(len(ins)-1)

            # Outs
            outs.index = dts
            outs.index = pd.to_datetime(outs.index)
            outs.loc[outs.index >= pd.Timestamp("2000-01-01"),[x for x in outs.columns if x != 'datetime']] /= 12
            outs = outs.resample('YS').sum()
            outs = outs.head(len(outs)-1)

            # plot ins - highest to lowest volumes
            for i, col in enumerate(ins.columns):
                ax1.plot(ins.index, ins[col], color=cin[i % len(cin)], label=col, linewidth=1.8)
            ax1.axhline(pd.Timestamp('2024-01-01'),color='k',ls='--',linewidth=1)
            ax1.set_yscale('log')
            ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
            ax1.set_ylim([1, ymax * 1.05])
            ax1.set_title(f'Layer {lay + 1} - Ins')
            ax1.set_ylabel('acre-ft per year')
            ax1.legend(loc='center right', fontsize=8)

            # plot outs - highest to lowest volumes
            for i, col in enumerate(outs.columns):
                if col == 'WEL-OUT-TOTAL':
                    ax2.plot(outs.index, outs[col], color='k', linestyle='--', label=col, zorder=3, linewidth=1.8)
                else:
                    ax2.plot(outs.index, outs[col], color=cout[i % len(cout)], label=col, linewidth=1.8)
            ax2.axhline(pd.Timestamp('2024-01-01'),color='k',ls='--',linewidth=1)
            ax2.set_yscale('log')
            ax2.grid(True, which='both', linestyle=':', linewidth=0.7)
            ax2.set_title(f'Layer/Zone {lay + 1} - Outs')
            ax2.set_ylim([1, ymax * 1.05])
            ax2.legend(loc='center right', fontsize=8)

            pdf.savefig(fig)
            plt.close(fig)

            # plot volume sum for each layer on the combined figure
            if ins.shape[1] > 0:
                ax3.plot(ins.index, ins.sum(axis=1), label=f'Layer {lay + 1}', linewidth=1.8)
            if outs.shape[1] > 0:
                ax4.plot(outs.index, outs.sum(axis=1), label=f'Layer {lay + 1}', linewidth=1.8)

        # combined figure formatting
        for ax in (ax3, ax4):
            ax.grid(True, which='both', linestyle=':', linewidth=0.7)
            ax.set_yscale('log')

        ax3.set_title('Volume In by Layer')
        ax3.set_ylabel('acre-ft per year')
        ax3.legend()

        ax4.set_title('Volume Out by Layer')
        ax4.legend()

        fig2.suptitle('Total Volume by Layer')
        pdf.savefig(fig2)
        plt.close(fig2)
        pdf.close()


# ---------------------------------
# --- Create head drawdown maps
# Built to run locally, not cluster
# ---------------------------------
def plot_head_drawdowns():
    scen_dict = {
        '01':'Baseline',
        '02':'Full Authorized Use',
        '06':'Pending & Deferred Permits',
        '07':'ASR'
        }

    # Load model
    sim = flopy.mf6.MFSimulation.load(sim_ws='model_ws/swww_clean', exe_name='mf6', load_only=['dis'])
    m = sim.get_model("swww")
    botm = m.dis.botm.array
    top_sw = botm[1,:,:]
    # Load shapefiles
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww')
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'sw_ww_modelgrid.shp'))
    modelgrid = modelgrid.set_crs(2265)
    # drains = gpd.read_file(os.path.join(g_d, 'RIV_lines.shp'))
    asr_wells = gpd.read_file(os.path.join(g_d,'asr_locations_13.shp'))

    # Load the processed the PODs
    full_aloc_wells = ['15006106BBC',
                       '15006201DDA8',
                       '15006201DDB2',
                       '15006201DDC2',
                       '15006201DDDD2',
                       '15106131CBC',
                       '15106131CCC',
                       '15106334CAA',
                       ]
    all_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','wells_avg_pump.shp')).set_crs(2265)
    full_aloc_pods = all_pods.loc[all_pods['Well'].isin(full_aloc_wells)]
    def_pend_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','def_pend_pods.shp'))

    # Load baseline scenario ending heads
    hds_f = os.path.join('scenario_hds','scen01','swww.hds')
    baseline_hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0,558))
    baseline_hds = np.where(baseline_hds==1e30, np.nan, baseline_hds)

    # add to modelgrid
    modelgrid['depth_sw_top_baseline'] = baseline_hds[2,:,:].flatten() - top_sw.flatten()

    # Get basemap
    west, south, east, north = modelgrid.to_crs(3857).total_bounds
    img, ext = cx.bounds2img(west,
                             south,
                             east,
                             north,
                             source=cx.providers.USGS.USTopo
                             )
    img, ext = cx.warp_tiles(img,ext,t_crs=2265)

    # First calc vmin and vmax
    for i in [0,1,5,6]:

        # Read scenario hds file
        hds_f = os.path.join('scenario_hds',f'scen0{i+1}','swww.hds')
        hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0,558))
        hds = np.where(hds==1e30,np.nan,hds)

        # Load into the modelgrid
        modelgrid['sw_hds'] = hds[2,:,:].flatten()
        modelgrid['depth_sw_top'] = modelgrid['sw_hds'] - top_sw.flatten()

        if i == 1:
            vmin = np.nanmin(modelgrid['depth_sw_top'])
            vmax = np.nanmax(modelgrid['depth_sw_top'])
        else:
            if np.nanmin(modelgrid['depth_sw_top']) < vmin:
                vmin = np.nanmin(modelgrid['depth_sw_top'])
            if np.nanmax(modelgrid['depth_sw_top']) < vmax:
                vmin = np.nanmax(modelgrid['depth_sw_top'])

    fix,axex = plt.subplots(1,4,figsize=(18,7))

    print(vmin)
    print(vmax)

    # Then loop again and plot
    for i in [0,1,5,6]:
        
        # Read scenario hds file
        hds_f = os.path.join('scenario_hds',f'scen0{i+1}','swww.hds')
        hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0,558))
        hds = np.where(hds==1e30,np.nan,hds)

        # Load into the modelgrid
        modelgrid['sw_hds'] = hds[2,:,:].flatten()
        modelgrid['depth_sw_top'] = modelgrid['sw_hds'] - top_sw.flatten()

        # Init figure
        fig,ax = plt.subplots(1,2,figsize=(10,6))

        # Plot water level elevation above top of Spiritwood
        modelgrid.loc[modelgrid['depth_sw_top'] > 0].plot(column='depth_sw_top',
                                                           legend=True,
                                                           ax=ax[0],
                                                           cmap='Spectral',
                                                           legend_kwds=dict(shrink=0.7),
                                                           # Constant colobars
                                                           vmin=vmin,
                                                           vmax=vmax
                                                           )

        gpd.GeoDataFrame(geometry=[
            modelgrid.loc[modelgrid['depth_sw_top'] <= 0].union_all()
            ]).plot(ax=ax[0],
                   color='red',
                   hatch='\\\\'
                   )

        # Plot change in WL compared to baseline
        # modelgrid['change'] = modelgrid['depth_sw_top_baseline'] - modelgrid['depth_sw_top']
        # modelgrid.plot(column='change',
        #                legend=True,
        #                ax=ax[1],
        #                legend_kwds=dict(shrink=0.7),
        #                cmap='coolwarm',
        #                # scheme='Quantiles',     # or 'EqualInterval', 'FisherJenks'
        #                # k=4,
        #                )

        # Basemaps
        ylims = ax[0].get_ylim()
        xlims = ax[0].get_xlim()
        for a in ax:
            a.imshow(img,
                     extent=ext,
                     origin="upper",
                     zorder=0,
                     alpha=0.6)
            a.set_ylim(ylims)
            a.set_xlim(xlims)
            a.set_xticks([])
            a.set_yticks([])
            a.legend()
            # Add well information
            all_pods.loc[all_pods['k'] == 2].plot(ax=a,
                                                  color='k',
                                                  markersize=6,
                                                  # edgecolor='k'
                                                  )

            if i in [1,3,5]:
                full_aloc_pods.plot(ax=a,
                                    color='cyan',
                                    markersize=7,
                                    # edgecolor='k'
                                    )
            if i in [3,5]:
                def_pend_pods.loc[def_pend_pods['k'] == 2].plot(ax=a,
                                                                color='blueviolet',
                                                                markersize=7,
                                                                # edgecolor='k',
                                                                )
            if i == 5:
                asr_wells.plot(ax=a,
                               markersize=7,
                               color='darkgreen'
                               )

        # Labels
        ax[0].set_title("Height Remaining above Top of Spiritwood")
        ax[1].set_title("Additional Drawdown Compared to Baseline,\nPositive Implies Reduction of Water Levels")
        scen = scen_dict[f"0{i+1}"]
        # fig.suptitle(f"Remaining Confined Thickness of Spiritwood at End of Predictive Period (12/01/2043)\nScenario: {scen}",y=0.96,
        #              fontsize=11)

        # Custom legend
        handles = [Line2D([],[],color='k',marker='o',ls='',label='Baseline Irrigation Wells')]

        if i in [1,3,5]:
            handles.append(Line2D([],[],color='cyan',marker='o',ls='',label='Municipal/Rural Wells'))

        if i in [3,5]:
            handles.append(Line2D([],[],color='blueviolet',marker='o',ls='',label='Pending/Deferred Permits'))
        if i == 5:
            handles.append(Line2D([],[],color='darkgreen',marker='o',ls='',label='ASR Wells'))

        # Always want this at the end
        handles.append(mpatches.Patch(facecolor='red',edgecolor='k',label='Unconfined Conditions',hatch=r'\\\\'))

        ax[0].legend(handles=handles,
                     loc=3,
                     frameon=True)

        plt.tight_layout()

        plt.savefig(f'scenario_hds/figures/scen_{i+1}_drawdown.png',
                    bbox_inches='tight',
                    dpi=250)


# Export a raster file of the depth to top of Spiritwood
def export_depth_sw_top_rasters(modelgrid_gdf,
                                top_sw_2d,
                                scen_idxs=(0, 1, 3, 5),
                                hds_root="scenario_hds",
                                out_dir="scenario_hds/rasters",
                                kstpkper=(0, 558),
                                layer_index=2,          # the layer you used for Spiritwood heads (hds[2,:,:])
                                nodata=-9999.0,
                                drawdown=False,
                                plot_max=False
                                ):
    import rasterio
    from rasterio.transform import from_origin

    if plot_max:
        kstpkper = (0, 553)

    os.makedirs(out_dir, exist_ok=True)

    mg = modelgrid_gdf.copy()

    # --- grid dims from row/column fields (assumes 1-based in shapefile, like you used before)
    if "row" not in mg.columns or "column" not in mg.columns:
        raise ValueError("modelgrid_gdf must have 'row' and 'column' fields to build a raster grid.")

    nrow = int(mg["row"].max())
    ncol = int(mg["column"].max())

    # --- cell size from polygon bounds (robust median)
    b = mg.geometry.bounds
    dx = float(np.median(b["maxx"] - b["minx"]))
    dy = float(np.median(b["maxy"] - b["miny"]))

    # --- raster georeference from modelgrid bounds
    minx, miny, maxx, maxy = mg.total_bounds
    transform = from_origin(minx, maxy, dx, dy)

    crs = mg.crs
    if crs is None:
        raise ValueError("modelgrid_gdf.crs is None. Set CRS on the modelgrid GeoDataFrame first (e.g., EPSG:2265).")

    # --- ensure top_sw is (nrow, ncol)
    top_sw_2d = np.asarray(top_sw_2d)
    if top_sw_2d.shape != (nrow, ncol):
        raise ValueError(f"top_sw_2d shape {top_sw_2d.shape} does not match (nrow, ncol)=({nrow}, {ncol}).")

    # Precompute i/j (0-based)
    mg["i"] = mg["row"].astype(int) - 1
    mg["j"] = mg["column"].astype(int) - 1

    # load base heads if exporting drawdown
    if drawdown:
        hds_f = os.path.join('scenario_hds','scen01','swww.hds')
        baseline_hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=kstpkper)
        baseline_hds = np.where(baseline_hds==1e30, np.nan, baseline_hds)
        baseline_hds = baseline_hds[layer_index,:,:]

    for i in scen_idxs:
        hds_f = os.path.join(hds_root, f"scen0{i+1}", "swww.hds")
        hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=kstpkper)
        hds = np.where(hds == 1e30, np.nan, hds)

        # depth_sw_top = head(layer_index) - top_sw
        depth = hds[layer_index, :, :] - top_sw_2d  # (nrow, ncol)

        # Write GeoTIFF
        if drawdown:
            if plot_max:
                out_tif = os.path.join(out_dir, f"drawdown_sw_scen0{i+1}_Jul43.tif")
            else:
                out_tif = os.path.join(out_dir, f"drawdown_sw_scen0{i+1}_Dec43.tif")
            dd = baseline_hds - hds[layer_index, :, :]
            dd = np.where(np.isfinite(dd), dd, nodata).astype("float32")
        else:
            if plot_max:
                out_tif = os.path.join(out_dir, f"depth_sw_top_scen0{i+1}_Jul43.tif")
            else:
                out_tif = os.path.join(out_dir, f"depth_sw_top_scen0{i+1}_Dec43.tif")
            depth_out = np.where(np.isfinite(depth), depth, nodata).astype("float32")

        with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            height=nrow,
            width=ncol,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="deflate",
            predictor=2,
        ) as dst:
            dst.write(depth_out if not drawdown else dd, 1)

        print(f"Wrote: {out_tif}")

# ----------------------------------------------
# --- Plot remaining water level above top of SW
# ----------------------------------------------
def plot_head_remaining(plot_max=False):

    # Only plot these scenario indices (0-based): 0,1,3,5
    scen_idxs = [0, 1, 3, 5]

    # Load model
    sim = flopy.mf6.MFSimulation.load(sim_ws='model_ws/swww_clean', exe_name='mf6', load_only=['dis'])
    m = sim.get_model("swww")
    botm = m.dis.botm.array
    top_sw = botm[1, :, :]  # as you had it

    # Load shapefiles / grids
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww')
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'sw_ww_modelgrid.shp')).set_crs(2265)

    # Export as rasters
    export_depth_sw_top_rasters(
        modelgrid_gdf=modelgrid,   # your sw_ww_modelgrid.shp GeoDataFrame
        top_sw_2d=top_sw,          # botm[1,:,:] from your DIS
        scen_idxs=(0, 1, 3, 5),
        out_dir="scenario_hds/rasters",
        plot_max=plot_max
    )

    asr_wells = gpd.read_file(os.path.join(g_d, 'asr_locations_13.shp'))

    full_aloc_wells = [
        '15006106BBC', '15006201DDA8', '15006201DDB2', '15006201DDC2',
        '15006201DDDD2', '15106131CBC', '15106131CCC', '15106334CAA',
    ]
    all_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','wells_avg_pump.shp')).set_crs(2265)
    full_aloc_pods = all_pods.loc[all_pods['Well'].isin(full_aloc_wells)]
    def_pend_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','def_pend_pods.shp')).set_crs(2265)

    # Basemap once
    west, south, east, north = modelgrid.to_crs(3857).total_bounds
    img, ext = cx.bounds2img(west, south, east, north, source=cx.providers.USGS.USTopo)
    img, ext = cx.warp_tiles(img, ext, t_crs=2265)

    # ------------------------------------------------------------
    # Pass 1: compute a shared vmin/vmax across the four scenarios
    # ------------------------------------------------------------
    vmin = np.inf
    vmax = -np.inf

    for i in scen_idxs:
        hds_f = os.path.join('scenario_hds', f'scen0{i+1}', 'swww.hds')
        hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0, 558 if not plot_max else 553))
        hds = np.where(hds == 1e30, np.nan, hds)

        depth = hds[2, :, :] - top_sw  # height above top of Spiritwood
        vmin = min(vmin, np.nanmin(depth))
        vmax = max(vmax, np.nanmax(depth))

    # (Optional) If you only care about positive "remaining thickness" colors,
    # you can clamp vmin to 0 so the colorbar starts at 0:
    # vmin = max(0.0, vmin)

    # ------------------------------------------------------------
    # Pass 2: plot in a single 1x4 figure + one shared colorbar
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 7))
    cmap = 'Spectral'
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for ax, i in zip(axes, scen_idxs):
        # Read scenario heads
        hds_f = os.path.join('scenario_hds', f'scen0{i+1}', 'swww.hds')
        hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0, 558 if not plot_max else 553))
        hds = np.where(hds == 1e30, np.nan, hds)

        # Attach values to modelgrid
        modelgrid['sw_hds'] = hds[2, :, :].flatten()
        modelgrid['depth_sw_top'] = modelgrid['sw_hds'] - top_sw.flatten()

        # Plot only cells with confined conditions (>0) using shared vmin/vmax
        modelgrid.loc[modelgrid['depth_sw_top'] > 0].plot(
            column='depth_sw_top',
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            legend=False,
        )

        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        # Basemap
        ax.imshow(img, extent=ext, origin="upper", zorder=0, alpha=0.6)

        # reset view
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # Hatch unconfined (<=0)
        unconf = modelgrid.loc[modelgrid['depth_sw_top'] <= 0]
        if len(unconf) > 0:
            gpd.GeoDataFrame(geometry=[unconf.union_all()], crs=modelgrid.crs).plot(
                ax=ax,
                color='red',
                hatch='\\\\',
                alpha=0.0,  # keep hatch visible without filling solid red
            )

        # Overlay wells (same logic you already had, but only one axis now)
        all_pods.loc[all_pods['k'] == 2].plot(ax=ax, color='k', markersize=6)

        if i in [1, 3, 5]:
            full_aloc_pods.plot(ax=ax, color='cyan', markersize=7)
        if i in [3, 5]:
            def_pend_pods.loc[def_pend_pods['k'] == 2].plot(ax=ax, color='blueviolet', markersize=7)
        if i == 5:
            asr_wells.plot(ax=ax, markersize=7, color='darkgreen')

        # Cosmetics
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")  # you said you'll handle titles in PPT

    # Shared colorbar across all axes, horizontal below plots
    fig.subplots_adjust(bottom=0.18)

    # [left, bottom, width, height]
    cax = fig.add_axes([0.25, 0.07, 0.5, 0.05])

    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation="horizontal"
                        )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Depth of Water to Top of Spiritwood (ft)",
                   fontsize=12)

    # One legend (put on first axis)
    handles = [Line2D([], [], color='k', marker='o', ls='', label='Baseline Irrigation Wells')]
    handles.append(Line2D([], [], color='cyan', marker='o', ls='', label='Municipal/Rural Wells'))
    handles.append(Line2D([], [], color='blueviolet', marker='o', ls='', label='Pending/Deferred Permits'))
    handles.append(Line2D([], [], color='darkgreen', marker='o', ls='', label='ASR Wells'))
    handles.append(mpatches.Patch(facecolor='none', edgecolor='k', hatch=r'\\\\', label='Unconfined Conditions'))
    axes[0].legend(handles=handles, loc=3, frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig('scenario_hds/figures/remaining_thickness_4scenarios.png', bbox_inches='tight', dpi=250)


def shifted_cmap(cmap, midpoint=0.5, name='shifted_cmap'):
    """
    Shift a colormap so that midpoint (0–1) is mapped to white.
    """
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    reg_index = np.linspace(0, 1, 257)
    shift_index = np.hstack([
        np.linspace(0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    return LinearSegmentedColormap(name, cdict)


# ---------------------------------------------------------------------------
# --- Similar to above but showing height of SW for all scenarios in one plot
# ---------------------------------------------------------------------------
def plot_sw_drawdowns_unified(layer=2,
                              plot_max=False,
                              plot_pressure_head=False):
    scen_dict = {
        '01':'Baseline Conditions',
        '02':'Full Authorized Use',
        '03':'Deferred and Pending Permit Full Use',
        '04':'Rural, Municipal, and Pending/Deferred Full Use',
        '05':'Full Permit Use with GHB Stage Uncertainty',
        '06':'ASR',
        '07':'Pending & Deferred Permits',
        }

    # Load model
    sim = flopy.mf6.MFSimulation.load(sim_ws='model_ws/swww_clean', exe_name='mf6', load_only=['dis'])
    m = sim.get_model("swww")
    botm = m.dis.botm.array
    top_sw = botm[1,:,:]
    # Load shapefiles
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww')
    # sw_extent = gpd.read_file(os.path.join(g_d,'sw_extent_SJ.shp')).to_crs(2265)
    # ww_extent = gpd.read_file(os.path.join(g_d,'warwick_boundary_model.shp')).to_crs(2265)
    # k_barrier = gpd.read_file(os.path.join(g_d,'HFB_V7.shp')).to_crs(2265)
    # rch_windows = gpd.read_file(os.path.join(g_d,'sw_recharge_window_large.shp')).to_crs(2265)
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'sw_ww_modelgrid.shp'))
    modelgrid = modelgrid.set_crs(2265)

    # Export as raster
    # Export as rasters
    export_depth_sw_top_rasters(
        modelgrid_gdf=modelgrid,   # your sw_ww_modelgrid.shp GeoDataFrame
        top_sw_2d=top_sw,          # botm[1,:,:] from your DIS
        scen_idxs=(0, 1, 3, 5),
        out_dir="scenario_hds/rasters",
        drawdown=True,
        plot_max=plot_max
    )

    # drains = gpd.read_file(os.path.join(g_d, 'RIV_lines.shp'))
    asr_wells = gpd.read_file(os.path.join(g_d,'asr_locations_13.shp'))
    # Load the processed the PODs
    full_aloc_wells = ['15006106BBC',
                       '15006201DDA8',
                       '15006201DDB2',
                       '15006201DDC2',
                       '15006201DDDD2',
                       '15106131CBC',
                       '15106131CCC',
                       '15106334CAA',
                       ]
    all_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','wells_avg_pump.shp')).set_crs(2265)
    full_aloc_pods = all_pods.loc[all_pods['Well'].isin(full_aloc_wells)]
    full_aloc_pods = full_aloc_pods.loc[full_aloc_pods['k'] == layer]
    def_pend_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','def_pend_pods.shp'))
    def_pend_pods = def_pend_pods.loc[def_pend_pods['k'] == layer]
    # Add baseline drawdown to modelgrid
    hds_f = os.path.join('scenario_hds','scen01','swww.hds')
    baseline_hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0, 558 if not plot_max else 553))
    baseline_hds = np.where(baseline_hds==1e30, np.nan, baseline_hds)
    modelgrid['depth_sw_top_baseline'] = baseline_hds[layer,:,:].flatten() - top_sw.flatten()

    # Get basemap
    west, south, east, north = modelgrid.to_crs(3857).total_bounds
    img, ext = cx.bounds2img(west,
                             south,
                             east,
                             north,
                             source=cx.providers.USGS.USTopo
                             )
    img, ext = cx.warp_tiles(img,ext,t_crs=2265)

    # Init figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Which scenario indices you actually plot (1,3,5 in your logic)
    plot_is = [1, 6, 5]

    # -----------------------------
    # Pass 1: compute global vmin/vmax
    # -----------------------------
    all_drawdown = []

    for i in plot_is:
        hds_f = os.path.join('scenario_hds', f'scen0{i+1}', 'swww.hds')
        hds = flopy.utils.HeadFile(hds_f).get_data(kstpkper=(0, 558 if not plot_max else 553))
        hds = np.where(hds == 1e30, np.nan, hds)

        # same computations as your plot loop
        sw_hds = hds[layer, :, :].flatten()
        depth_sw_top = sw_hds - top_sw.flatten()
        if plot_pressure_head:
            drawdown = depth_sw_top
        else:
            drawdown = modelgrid['depth_sw_top_baseline'] - depth_sw_top

        all_drawdown.append(drawdown)

    # Global limits respected by all plots
    vmin = np.nanmin(np.concatenate(all_drawdown))
    vmax = np.nanmax(np.concatenate(all_drawdown))
    # Overriding vmax so that 10-ft tick shows up
    vmin = -10
    if plot_pressure_head:
        vmin = 0
    # !!! Forcing vmax to be the greatest value from July period for both
    vmax = 71.66696099325895
    
    # print(vmax)
    
    # Shared norm/cmap
    midpoint = (0 - vmin) / (vmax - vmin)
    cmap = shifted_cmap(plt.cm.coolwarm, midpoint=midpoint)
    if plot_pressure_head:
        cmap = mcolors.LinearSegmentedColormap.from_list(
                "viridis_soft",
                [
                    "#b5de2b",  # soft yellow-green
                    "#6ece58",  # light green
                    "#35b779",  # green
                    "#1f9e89",  # green-teal
                    "#26828e",  # teal
                    "#31688e",  # blue
                    "#3b528b",  # muted blue-purple
                ],
                N=256
            )
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # -----------------------------
    # Pass 2: make the plots
    # -----------------------------
    for ax, i, drawdown in zip(axes, plot_is, all_drawdown):

        # Attach to modelgrid for plotting
        modelgrid['drawdown'] = drawdown
        
        if plot_pressure_head:
            _modelgrid = modelgrid.loc[modelgrid['drawdown']>0]
            # Plot WITHOUT legend (we will add one shared colorbar)    
            _modelgrid.plot(column='drawdown',
                            ax=ax,
                            cmap=cmap,
                            norm=norm,
                            legend=False,
                            # vmin=vmin,
                            # vmax=vmax
                            )
        else:
            # Plot WITHOUT legend (we will add one shared colorbar)    
            modelgrid.plot(column='drawdown',
                            ax=ax,
                            cmap=cmap,
                            norm=norm,
                            legend=False,
                            # vmin=vmin,
                            # vmax=vmax
                            )
        
        if plot_pressure_head:
            unconf = modelgrid.loc[modelgrid['drawdown'] <= 0]
            if len(unconf) > 0:
                gpd.GeoDataFrame(geometry=[unconf.union_all()], crs=modelgrid.crs).plot(
                    ax=ax,
                    color='red',
                    hatch='\\\\',
                    alpha=0.0,  # keep hatch visible without filling solid red
                )
        
        # Basemap (keep your order the same)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        if layer == 0:
            xlims = (np.float64(2384339.2924208436), np.float64(2483650.5924208434))
            ylims = (np.float64(265532.096321634), np.float64(380966.096321634))
        ax.imshow(img, extent=ext, origin="upper", zorder=0, alpha=0.6)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add well information
        all_pods.loc[all_pods['k'] == layer].plot(ax=ax, color='k', markersize=6)

        # Conditionaly add some map elements
        full_aloc_pods.plot(ax=ax, color='cyan', markersize=6)
        if i in [5,6]:
            def_pend_pods.loc[def_pend_pods['k'] == layer].plot(ax=ax, color='magenta', markersize=6)
        if i == 5 and layer == 2:
            asr_wells.plot(ax=ax,
                           markersize=7,
                           color='orange'
                           )

        # Labels
        scen = scen_dict[f"0{i+1}"]
        ax.set_title(scen, fontsize=11)

        # Custom legend (well markers only)
        handles = [Line2D([], [], color='k', marker='o', ls='', label='Baseline Wells')]

        handles.append(Line2D([], [], color='cyan', marker='o', ls='', label='Municipal/Rural Wells'))

        if i in [5, 6]:
            handles.append(Line2D([], [], color='magenta', marker='o', ls='', label='Pending/Deferred Permits'))
        if i == 5 and layer == 2:
            handles.append(Line2D([],[],color='orange',marker='o',ls='',label='ASR Wells'))


        ax.legend(handles=handles, loc=3, frameon=True)

    # --------------------------------------------
    # One shared horizontal colorbar across bottom
    # --------------------------------------------
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required by matplotlib for ScalarMappable colorbar

    fig.subplots_adjust(bottom=0.18)
    
    # [left, bottom, width, height]
    cax = fig.add_axes([0.25, -0.05, 0.5, 0.05])

    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation="horizontal"
                        )
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Remaining Pressure Head (ft)" if plot_pressure_head else "Drawdown from Baseline (ft)")
        
    plt.tight_layout()
    plt.savefig('sw_drawdown_dec2043.png' if not plot_max else 'sw_drawdown_jul2043.png',
                dpi=250,
                bbox_inches='tight'
                )


# plot_sw_drawdowns_unified(plot_max=True)
# plot_sw_drawdowns_unified(plot_max=False)


# --- Doesnt really belong here, but easy to add
def make_sccenario_setup_figs():
    # Load shapefiles
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww')
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'sw_ww_modelgrid.shp'))
    modelgrid = modelgrid.set_crs(2265)
    sw_extent = gpd.read_file(os.path.join(g_d,'sw_extent_SJ.shp')).to_crs(2265)
    sw_extent = sw_extent.clip(modelgrid.union_all())
    ww_extent = gpd.read_file(os.path.join(g_d,'warwick_boundary_model.shp')).to_crs(2265)
    asr_wells = gpd.read_file(os.path.join(g_d,'asr_locations_13.shp'))
    # Load the processed the PODs
    full_aloc_wells = ['15006106BBC',
                       '15006201DDA8',
                       '15006201DDB2',
                       '15006201DDC2',
                       '15006201DDDD2',
                       '15106131CBC',
                       '15106131CCC',
                       '15106334CAA',
                       ]
    all_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','wells_avg_pump.shp')).set_crs(2265)
    full_aloc_pods = all_pods.loc[all_pods['Well'].isin(full_aloc_wells)]
    def_pend_pods = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','def_pend_pods.shp'))

    # Make Scenario 2 Figure
    fig,ax = plt.subplots(figsize=(8,8))

    full_aloc_pods.plot(ax=ax,
                        label='Municipal and Rural Wells',
                        color='red',
                        edgecolor='k')
    ww_extent.boundary.plot(ax=ax,
                            color='blue',
                            label='Warwick Extent')
    sw_extent.boundary.plot(ax=ax,
                            color='k',
                            label='Spiritwood Extent')
    ax.legend(frameon=True)
    ax.set_yticks([])
    ax.set_xticks([])
    cx.add_basemap(ax=ax,
                   crs=2265,
                   source=cx.providers.USGS.USTopo,
                   attribution='')


    # Make Scenario 3 Figure
    fig,ax = plt.subplots(figsize=(8,8))

    asr_wells.plot(ax=ax,
                        label='Municipal and Rural Wells',
                        color='red',
                        edgecolor='k')
    ww_extent.boundary.plot(ax=ax,
                            color='blue',
                            label='Warwick Extent')
    sw_extent.boundary.plot(ax=ax,
                            color='k',
                            label='Spiritwood Extent')

    ax.legend(frameon=True)
    ax.set_yticks([])
    ax.set_xticks([])
    cx.add_basemap(ax=ax,
                   crs=2265,
                   source=cx.providers.USGS.USTopo,
                   attribution='')


# make_sccenario_setup_figs()


#%%
# plot_sw_drawdowns_unified()
# plot_sw_drawdowns_unified(plot_max=True,layer=2)

# plot_sw_drawdowns_unified(plot_pressure_head=True,
#                           plot_max=True
#                           )


#%%
# -----------------
# ---- Main
# -----------------
if __name__ == "__main__":
    modnm = 'swww'
    post_ws = 'master_flow_MasterRun3_forward_run_base_temp'

    noptmax_flow = 4
    par_post_nm = f'{modnm}.{noptmax_flow}.par.jcb'
    obs_post_nm = f'{modnm}.{noptmax_flow}.obs.jcb'

    # run controls:
    prep_predict = True
    prep_scn = True
    run_en = True
    run_all = True
    use_condor = True

    plot_predict = True

    if use_condor:
        num_reals_flow = 20
        num_workers_flow = 20
        hostname = '10.99.10.30'
        pid = os.getpid()
        current_time = int(time.time())
        seed_value = (pid + current_time) % 65536
        np.random.seed(seed_value)
        port = random.randint(4001, 4999)
        
    else:
        num_reals_flow = 6
        num_workers_flow = 6
        hostname = None
        port = None

    local = True

    all_scenarios = ['full_alloc_def_pend_realUse', 'full_alloc_asr_def_pend', 'baseline', 'full_allocation']

    all_scenarios = ['baseline']

    if run_all:
        print('*** running predictions for flow-ies ***')
        if use_condor:
            print(f'port #: {port}')
        for scenario in all_scenarios:
            if scenario == 'full_allocation':
                # Scenario 2: Full Permit Use for Rural and Municipal permits
                pred_ws = os.path.join('post_ies_scn02_full_alloc')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    lst_mod_well_files = write_full_alloc_well_files(modnm, pred_ws)
                    lst_mod_ghb_files = write_baseline_ghb_files(modnm, pred_ws)
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)

                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)


            elif scenario == 'deferred_pending':
                # Scenario 3: Deferred + Pending permits + on top of 5-year median pumping
                pred_ws = os.path.join('post_ies_scn03_deferred_pending')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    lst_mod_well_files = write_def_pend_well_files(modnm, pred_ws)
                    lst_mod_ghb_files = write_baseline_ghb_files(modnm, pred_ws)
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)

                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)

            elif scenario == 'full_alloc_def_pend':
                # Scenario 4: Full Permit Use for Rural and Municipal permits + defered and pending
                pred_ws = os.path.join('post_ies_scn04_full_alloc_def_pend')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    # Write WEL files with full allocation and deferred + pending
                    lst_mod_well_files = write_full_alloc_plus_def_pend_well_files(modnm, pred_ws)

                    # Baseline GHB files
                    lst_mod_ghb_files = write_baseline_ghb_files(modnm, pred_ws)

                    # Write mean recharge files
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    # Remove GHB and WEL from mult to model
                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)

                    # Check scenario pumping
                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                # Run ensemble with new inputs
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)

            elif scenario == 'full_alloc_def_pend_realUse':
                # Scenario 2: Full Permit Use for Rural and Municipal permits
                pred_ws = os.path.join('post_ies_scn07_full_alloc_def_pend_realUse')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                # Same as full allocation but w/ GHB adjustment (lower K to simulate hb uncertainty)
                if prep_scn:
                    # Write WEL files with full allocation and deferred + pending
                    # !!! Only 50% of defered pending to match realistic historical use rates
                    lst_mod_well_files = write_full_alloc_plus_def_pend_well_files(modnm, pred_ws, mult=0.50)

                    # Baseline GHB files
                    lst_mod_ghb_files = write_baseline_ghb_files(modnm, pred_ws)

                    # Write mean recharge files
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    # Remove GHB and WEL from mult to model
                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)

                    # Check scenario pumping
                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                # Run ensemble with new inputs
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)

            elif scenario == 'full_allocation_ghb_low_stage':
                # Scenario 5: Full Permit Use for Rural and Municipal permits w/ lowering GHB stage following past 20-year decline rate
                pred_ws = os.path.join('post_ies_scn05_full_alloc_ghb_lowStage')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                # Same as full allocation but w/ GHB adjustment (lower K to simulate hb uncertainty)
                if prep_scn:
                    # Edit pumping to reflect posterior
                    lst_mod_well_files = write_full_alloc_well_files(modnm, pred_ws)

                    # Edit GHB to reflect posterior and lower the stage following estimated past 20-year decline rates
                    lst_mod_ghb_files = write_drought_ghb_files(modnm, pred_ws)

                    # Write mean recharge files
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    # Remove GHB and WEL from mult to model
                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)

                    # Check scenario pumping
                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                # Run ensemble with new inputs
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)

            elif scenario == 'baseline':
                # Scenario 1: Baseline - take posterior 2020-2025 pumping forward
                pred_ws = os.path.join('post_ies_scn01_baseline')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    lst_mod_well_files = write_baseline_well_files(modnm, pred_ws)
                    lst_mod_ghb_files = write_baseline_ghb_files(modnm, pred_ws)
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)
                    nopt0chk_baseline(post_ws, pred_ws, modnm, noptmax_flow)

                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)

            elif scenario == 'full_alloc_asr_def_pend':
                # Scenario 6: ASR
                pred_ws = os.path.join('post_ies_scn06_full_alloc_asr_def_pend')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    # Write WEL files with full allocation and deferred + pending + ASR
                    lst_mod_well_files = write_full_alloc_def_pend_plus_asr_well_files(modnm, pred_ws, mult=0.50)

                    # Baseline GHB files
                    lst_mod_ghb_files = write_baseline_ghb_files(modnm, pred_ws)

                    # Write mean recharge files
                    modified_rch_files = write_baseline_rch_files(modnm, pred_ws)

                    # Remove GHB and WEL from mult to model
                    lst_mod_files = lst_mod_well_files + lst_mod_ghb_files
                    modify_mult2mod(pred_ws, lst_mod_files)

                    # Check scenario pumping
                    nopt0chk_full_permit_use(post_ws,pred_ws, modnm, noptmax_flow)

                # Run ensemble with new inputs
                if run_en:
                    run_ies(pred_ws, modnm=modnm, m_d=pred_ws+'_ensemble',num_workers=num_workers_flow,
                            num_reals=num_reals_flow,niceness=False,noptmax=-1,init_lam=None,
                            local=local,use_condor=use_condor, hostname=hostname,port=port,
                            par_post_nm=par_post_nm, obs_post_nm=obs_post_nm)


    if plot_predict:
        # get dirs that end in '_ensemble'
        pred_ws_list = [d for d in os.listdir('.') if d.startswith('post_ies_scn') and d.endswith('_ensemble')]

        # plot_scn_hydrograpghs(pred_ws_list, modnm='swww', plt_base_only=False, plot_quantiles=True)

        plot_scn_net_water_budgets(pred_ws_list, modnm='swww', plot_quantiles=True)

        # Plot zone budget for each scenario
        # for ws in pred_ws_list:
        #     ws = '_'.join(ws.split('_')[0:-1])
        #     scen_tag = ' '.join(ws.split('_')[3:])
        #     run_zb_by_layer(w_d=ws,modnm='swww', plot=True,scen_tag=scen_tag)





