import os
import sys

from scipy.fft import dst
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
from master_flow_gwv_sspmp_highdim_nozn_allobs_forward_run_base.pyemu import pst
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
import wahp04_process_plot_results as wpp
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


# ------------------------------------------------------- #
# Baseline Scenario Functions
# ------------------------------------------------------- #
def write_baseline_well_files(modnm, pred_ws):
    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
        well_path = os.path.join(pred_ws, f"wahp7ly.{prefix}")
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
    pst = pyemu.Pst(os.path.join(pred_ws,'wahp7ly.pst'))
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_par_en'] = f'{modnm}.{noptmax_flow}.par.jcb'
    pst.pestpp_options["ies_obs_en"] = f'{modnm}.{noptmax_flow}.obs.jcb'
    pst.write(os.path.join(pred_ws,'wahp7ly.pst'),version=2)
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


# ------------------------------------------------------- #
# Full Permit Use Scenario Functions
# ------------------------------------------------------- #
def write_full_alloc_well_files(modnm, pred_ws):
    print("writing full allocation well files...")

    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
    cow_path = os.path.join(pred_ws, "wahp7ly.cow")  # adjust if needed

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
    # minn_path = os.path.join(pred_ws, "wahp7ly.minn")  
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
    # car_path = os.path.join(pred_ws, "wahp7ly.car")  
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
    pst = pyemu.Pst(os.path.join(pred_ws,'wahp7ly.pst'))
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_par_en'] = f'{modnm}.{noptmax_flow}.par.jcb'
    pst.pestpp_options["ies_obs_en"] = f'{modnm}.{noptmax_flow}.obs.jcb'
    pst.write(os.path.join(pred_ws,'wahp7ly.pst'),version=2)
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
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
    car_path = os.path.join(pred_ws, "wahp7ly.car")  
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

# ------------------------------------------------------- #
# ASR Scenario Functions
# ------------------------------------------------------- #
def write_asr_well_files(modnm, pred_ws,rate_acftyr=500,k=5,i=83,j=38):
    print("writing ASR well files...")

    # load in stress period info:
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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
    spd = pd.read_csv(os.path.join('tables','annual_stress_period_info.csv'))
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

def znbud_by_ly_process(modnm='wahp7ly'):
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

def rewrite_zbud_ins_file(modnm='wahp7ly', pred_ws='.'):
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
def plot_inset(ax=[],wl_loc=None,grp=[],cpts=[],aq_extent=[],drains=[],asr_loc=None):
    aq_extent.boundary.plot(ax=ax,edgecolor='black',linewidth=1.25,label='Model area')
    drains.plot(ax=ax,edgecolor='blue',linewidth=0.5,zorder=9,label='Rivers and streams')   
    cpts.plot(ax=ax, edgecolor='black',facecolor='grey',linewidth=0.3,markersize=5,label='Water level observations',zorder=10)
    
    # turn off x and y labels:
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    # remove plot border:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove ticks:
    ax.tick_params(axis='both', which='both', length=0)
    
    # move subplot if needed:
    box = ax.get_position()
    ax.set_position([box.x0-.09, box.y0-0.01,box.width+0.15, box.height+0.15])
    
    if wl_loc is not None:
        df = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
        points = [Point(xy) for xy in zip(df['x_2265'], df['y_2265'])]
        sites = gpd.GeoDataFrame(data=df,
                                geometry=points)
        sites = sites.set_crs(2265)#.to_crs(2265)
        grp_num = sites['group number']
        k = sites['k']
        sites['id'] = 'grpid:'+grp_num.astype(str) + '_k:' + k.astype(str)
        wpt = sites.loc[sites.id == wl_loc]
        wpt.plot(ax=ax, edgecolor='black', facecolor='orange',markersize=25,zorder=11,label='Highlighted water level observation')
    if asr_loc is not None:
        # keep as GeoDataFrame: note the double brackets [[0]]
        asr_pos1 = asr_loc.iloc[[0]]
        asr_pos1.plot(
            ax=ax,
            edgecolor='black',
            facecolor=asr_pos1['color'],
            markersize=25,
            zorder=11,
            label='Scn. 4 - ASR location',
        )

        asr_pos2 = asr_loc.iloc[[1]]
        asr_pos2.plot(
            ax=ax,
            edgecolor='black',
            facecolor=asr_pos2['color'],
            markersize=25,
            zorder=12,
            label='Scn. 5 - ASR location',
        )
        
    lb1 = mlines.Line2D([], [], color='orange', marker='o',markeredgecolor='black', linestyle='None', markersize=8, label='Highlighted water\nlevel observation') 
    lb2 = mlines.Line2D([], [], color='grey', marker='o',markeredgecolor='black', linestyle='None', markersize=5, label='Water level\nobservations') 
    lb3 = mlines.Line2D([], [], color='blue', linestyle='-', label='Rivers and streams') 
    lb4 = mpatches.Patch(facecolor='white',linewidth=1,edgecolor='black', label='Model area')
    
    if asr_loc is None:
        leg = ax.legend(handles=[lb1, lb2, lb3, lb4], loc='lower right', frameon=True)
    else:
        lb5 = mlines.Line2D([], [], color=asr_pos1['color'].values[0], marker='o',markeredgecolor='black', linestyle='None', markersize=8, label='Scn. 4 ASR loc') 
        lb6 = mlines.Line2D([], [], color=asr_pos2['color'].values[0], marker='o',markeredgecolor='black', linestyle='None', markersize=8, label='Scn. 5 ASR loc')
        leg = ax.legend(handles=[lb1, lb2, lb3, lb4, lb5, lb6], loc='lower right', frameon=True)
        
    leg.get_frame().set_facecolor('grey')       # Background color
    leg.get_frame().set_alpha(0.5)              # Transparency
    leg.get_frame().set_edgecolor('black')      # Dark outline color
    leg.get_frame().set_linewidth(2.0)          # Thickness of outline
    leg.set_bbox_to_anchor((.5, 0.05))


def plot_scn_hydrograpghs(pred_ws_list, modnm='wahp7ly',plot_quantiles=True,plt_base_only=False,zoom_predict=False):
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


def plot_scn_wbv_water_budgets(pred_ws_list, modnm='wahp7ly',plot_quantiles=True):
    fdir = os.path.join( 'results', 'figures', 'scenario_results')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    lay = 5 
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
    
    
    pdf = PdfPages(os.path.join(fdir, f'net_budget_wbv_scns.pdf'))
    

    factor = 0.00002296 * 365.25 # cf to acre-ft per year

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
    years10 = mdates.YearLocator(10)
    years20 = mdates.YearLocator(20)
    years_fmt = mdates.DateFormatter('%Y')


    # plot of water balance by layer
    pdf = PdfPages(os.path.join(fdir, 'water_balance_by_layer.pdf'))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for ws in pred_ws_list:
        scn_tag = ws.split('_')[2]
        scn_num = int(scn_tag.replace('scn',''))

        pst = pyemu.Pst(os.path.join(ws, f'{modnm}.pst'))
        obs = pst.observation_data
        bobs = obs.loc[obs.obsnme.str.contains('zbud'), :]
        bobs.loc[:, 'datetime'] = pd.to_datetime(bobs.datetime)

        inobs = bobs.loc[bobs.obsnme.apply(lambda x: '-in' in x or 'from' in x and 'bud' in x), :].copy()
        outobs = bobs.loc[bobs.obsnme.apply(lambda x: '-out' in x or 'to_' in x and 'bud' in x), :].copy()
        inobs['k'] = inobs.obsnme.str.split('-').str[-1].astype(int)
        outobs['k'] = outobs.obsnme.str.split('-').str[-1].astype(int)

        ins = inobs.loc[inobs.obsnme.apply(lambda x: f'zn-{lay}' in x), :].copy()
        outs = outobs.loc[outobs.obsnme.apply(lambda x: f'zn-{lay}' in x), :].copy()

        pt_ins = scn_results_dict[scn_tag]._df.loc[:, ins.obsnme]
        pt_outs = scn_results_dict[scn_tag]._df.loc[:, outs.obsnme]

        pt_ins = pt_ins.loc[:, (pt_ins != 0).any(axis=0)]
        pt_outs = pt_outs.loc[:, (pt_outs != 0).any(axis=0)]

        in_types = pt_ins.columns.str.split(':').str[3].unique().str.replace('_datetime', '', regex=True).str.replace('zbly_', '', regex=True).tolist()
        out_types = pt_outs.columns.str.split(':').str[3].unique().str.replace('_datetime', '', regex=True).str.replace('zbly_', '', regex=True).tolist()

        # remove sto from types:
        in_types = [s for s in in_types if 'sto' not in s]
        out_types = [s for s in out_types if 'sto' not in s]
    
        lay = 5 # fix to just plot WBV for now

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

        in_df['year'] = pd.DatetimeIndex(in_df['datetime']).year
        out_df['year'] = pd.DatetimeIndex(out_df['datetime']).year
        print('--'*10)
        print(out_df.type.unique())
        print(scn_tag)
        net_df = pd.DataFrame()
        reals = in_df['realization'].unique()
        for real in reals:
            in_real = in_df.loc[in_df['realization'] == real, :].copy()
            out_real = out_df.loc[out_df['realization'] == real, :].copy()

            in_tots = in_real.groupby('year')['value'].sum()
            out_tots = out_real.groupby('year')['value'].sum()
            bnet = in_tots - out_tots
            temp_df = pd.DataFrame({'year': bnet.index, 'net_budget': bnet.values, 'realization': real})
            net_df = pd.concat([net_df, temp_df], ignore_index=True)

        net_vals_acft = net_df.pivot(index='year', columns='realization', values='net_budget')
        years = net_vals_acft.index.values
        # quantiles
        if plot_quantiles:
            q10 = np.percentile(net_vals_acft.values, 10, axis=1)
            q90 = np.percentile(net_vals_acft.values, 90, axis=1)
            
            ax.fill_between(years, q10, q90, color=colors[scn_num-1], alpha=0.2, zorder=1)
            ax.plot(years, q10, color=colors[scn_num-1], linestyle='--', lw=1.0, zorder=5)
            ax.plot(years, q90, color=colors[scn_num-1], linestyle='--', lw=1.0, zorder=5)
        
        # plot base:
        ax.plot(years, net_vals_acft[['base']].values, color=darker_colors[scn_num-1], lw=1.5, zorder=10)

    ax.set_ylabel('Net water budget (acre-ft/yr)\n(More negative values indicate greater outflow)', fontsize=11)

    # add vertical line for predictive period:
    ax.axvline(x=2025, color='grey', linestyle='-.', linewidth=1)
    ax.text(2025, ax.get_ylim()[0] + 10, '-> Predictive period', fontsize=10,
            ha='left', va='bottom', color='grey')
    
    # comma formateed y-axis labels:
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
    ax.get_xaxis().set_tick_params(direction='in')
    ax.tick_params(axis='both',direction='in', which='major', labelsize=9)

    # add space for legend
    fig.subplots_adjust(bottom=0.15)
    leg_s1  = mlines.Line2D([], [], color=darker_colors[0], lw=1.5, label='Scenario 1 - Baseline')
    leg_s2  = mlines.Line2D([], [], color=darker_colors[1], lw=1.5, label='Scenario 2 - Full permit use')
    leg_s3  = mlines.Line2D([], [], color=darker_colors[2], lw=1.5, label='Scenario 3 - Full permit use with drought')
    leg_s4  = mlines.Line2D([], [], color=darker_colors[3], lw=1.5, label='Scenario 4 - Drought with ASR (500 acft/yr)')
    leg_s5  = mlines.Line2D([], [], color=darker_colors[4], lw=1.5, label='Scenario 5 - Drought with ASR (200 acft/yr)')
    leg_s6  = mlines.Line2D([], [], color=darker_colors[5], lw=1.5, label='Scenario 6 - COW centered ASR (500 acft/yr)')

    # --- SPLIT INTO TWO LEGENDS ---
    scenario_handles = [leg_s1, leg_s2, leg_s3, leg_s4, leg_s5, leg_s6]

    # Legend A: scenarios (two columns), centered at bottom
    legA = fig.legend(
    handles=scenario_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.001),  # x=center, y a bit above bottom edge
    ncol=2,
    frameon=True,
    framealpha=0.6,
    fontsize=8,
    )

    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()


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


def plot_scn_maxdd(pred_ws_list, modnm='wahp7ly',plot_quantiles=True,plt_base_only=False):
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
    modnm = 'wahp7ly'
    post_ws = 'master_flow_gwv_sspmp_highdim_noWR_final_wss_reweight_forward_run_base'

    noptmax_flow = 6
    par_post_nm = f'{modnm}.{noptmax_flow}.par.jcb'
    obs_post_nm = f'{modnm}.{noptmax_flow}.obs.jcb'

    # run controls:
    scenario = 'full_allocation'
    prep_predict = True
    prep_scn = True
    run_en = True
    run_all = False
    use_condor = True

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

    all_scenarios = ['asr_10_well_at225']#['baseline','full_allocation','drought']#[,,'drought']#,'drought+asr500','drought+asr500cow']
    #all_scenarios = ['drought+asr500','drought+asr500cow']

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
                # Scenario 1: Baseline - take posterior 2020-2025 pumping forward, also want to add 561.5 AFY to COW to account for new planned use
                pred_ws = os.path.join('post_ies_scn01_baseline')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    lst_mod_well_files = write_baseline_well_files(modnm, pred_ws)
                    modify_mult2mod(pred_ws, lst_mod_well_files)
                    nopt0chk_baseline(post_ws,pred_ws, modnm, noptmax_flow)

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

            elif scenario == 'drought':
                print('--- preparing drought scenario ---')
                # Scenario 3: Ultimate Drought with full permit use
                pred_ws = os.path.join('post_ies_scn03_drought')

                if prep_predict:
                    init_predict_ws(modnm, noptmax_flow, post_ws, pred_ws)
                    prep_deps(pred_ws)
                if prep_scn:
                    base_well_files = write_baseline_well_files(modnm, pred_ws)
                    lst_mod_well_files = write_cargill_drought_use_well_files(modnm, pred_ws)
                    all_well_files = base_well_files + lst_mod_well_files
                    all_well_files = list(dict.fromkeys(all_well_files))
                    
                    # modify rivs, ghbs, and rch for drought:
                    ghb_file = write_ghb_files(modnm, pred_ws)
                    lst_mod_well_files = all_well_files + ghb_file
                    riv_files = write_riv_files(modnm, pred_ws)
                    lst_mod_well_files = lst_mod_well_files + riv_files
                    rch_files, sp_mod = write_rch_files(modnm, pred_ws) # do not remove rch, leave mults on cux they insignifcant and then do not break anything
                
                    # par_data = pd.read_csv(os.path.join(pred_ws, f'{modnm}.par_data.csv'))
                    # par_data.loc[(par_data['pargp'] == 'rcht') & (par_data['kper'].isin(sp_mod)), 'parlbnd'] = 0.99
                    # par_data.loc[(par_data['pargp'] == 'rcht') & (par_data['kper'].isin(sp_mod)), 'parubnd'] = 1.01
                    # par_data.loc[(par_data['pargp'] == 'rcht') & (par_data['kper'].isin(sp_mod)), 'parval1'] = 1.0
                    
                    modify_mult2mod(pred_ws, lst_mod_well_files)
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

                    rewrite_zbud_ins_file(modnm='wahp7ly', pred_ws=pred_ws)

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

                    rewrite_zbud_ins_file(modnm='wahp7ly', pred_ws=pred_ws)

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

                    rewrite_zbud_ins_file(modnm='wahp7ly', pred_ws=pred_ws)

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

        plot_scn_hydrograpghs(pred_ws_list, modnm='wahp7ly',plot_quantiles=True,plt_base_only=False,zoom_predict=False)

        plot_scn_wbv_water_budgets(pred_ws_list, modnm='wahp7ly', plot_quantiles=False)

        plot_zone_bud_ly_budget(pred_ws_list)

        plot_zone_bud_terms_by_scenario_like_base(
           pred_ws_list,
            modnm,
            layer_k=5,                 # focus on this layer via "zn-{layer_k}" (set None for all layers)
            x_start="2020-01-01",      # start date for x-axis and data trimming; set None to keep all
            max_plots_per_page=6,
            ylim=(1, 10**6),           # semilogy y-limits
            tol=1e-12
            )

        plot_scn_maxdd(pred_ws_list, modnm='wahp7ly')



