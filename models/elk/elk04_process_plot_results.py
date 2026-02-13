
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import platform
print('Env path order:')
for path in sys.path:
    print(path)
    
import glob
import pyemu
import flopy
from datetime import datetime
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
years10 = mdates.YearLocator(10)
years20 = mdates.YearLocator(20)
years1 = mdates.YearLocator()
import matplotlib.ticker as ticker
years_fmt = mdates.DateFormatter('%Y')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import multiprocessing as mp
import numpy as np
import pandas as pd
# Set some pandas options
pd.set_option('expand_frame_repr', False)
from numbers import Integral   

import shutil
import geopandas as gpd
import calendar

from datetime import timedelta
import matplotlib as mpl
#mpl.rcParams['axes.formatter.limits'] = (-10,20)
#from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

#from scipy import stats
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from shapely import LineString
from shapely import Point
import shapefile
import re
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
from typing import Sequence, Union
from typing import List, Tuple, Any, Dict
from typing import Optional
from collections import defaultdict




# figure props:
def set_graph_specifications():
    rc_dict = {'font.family': 'DejaVu Sans',
               'axes.labelsize': 9,
               'axes.titlesize': 9,
               'axes.linewidth': 0.5,
               'xtick.labelsize': 8,
               'xtick.top': True,
               'xtick.bottom': True,
               'xtick.major.size': 7.2,
               'xtick.minor.size': 3.6,
               'xtick.major.width': 0.5,
               'xtick.minor.width': 0.5,
               'xtick.direction': 'in',
               'ytick.labelsize': 8,
               'ytick.left': True,
               'ytick.right': True,
               'ytick.major.size': 7.2,
               'ytick.minor.size': 3.6,
               'ytick.major.width': 0.5,
               'ytick.minor.width': 0.5,
               'ytick.direction': 'in',
               'pdf.fonttype': 42,
               'savefig.dpi': 300,
               'savefig.transparent': True,
               'legend.fontsize': 9,
               'legend.frameon': False,
               'legend.markerscale': 1.
               }
    mpl.rcParams.update(rc_dict)

def set_map_specifications():
    rc_dict = {'font.family': 'DejaVu Sans',
               'axes.labelsize': 9,
               'axes.titlesize': 9,
               'axes.linewidth': 0.5,
               'xtick.labelsize': 7,
               'xtick.top': True,
               'xtick.bottom': True,
               'xtick.major.size': 7.2,
               'xtick.minor.size': 3.6,
               'xtick.major.width': 0.5,
               'xtick.minor.width': 0.5,
               'xtick.direction': 'in',
               'ytick.labelsize': 7,
               'ytick.left': True,
               'ytick.right': True,
               'ytick.major.size': 7.2,
               'ytick.minor.size': 3.6,
               'ytick.major.width': 0.5,
               'ytick.minor.width': 0.5,
               'ytick.direction': 'in',
               'pdf.fonttype': 42,
               'savefig.dpi': 300,
               'savefig.transparent': True,
               'legend.fontsize': 9,
               'legend.frameon': False,
               'legend.markerscale': 1.
               }
    mpl.rcParams.update(rc_dict)

def _set_axis_style(ax, labels):
    '''helper function to assign style parameters to matplotlib axis object
    Args:
        ax (obj): matplotlib axis object
        labels (list): list of strings to use as axis tick labels
    '''
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.5)

def prep_deps(d):
    '''copy exes to a directory based on platform
    Args:
        d (str): directory to copy into
    Note:
        currently only mf6 for mac and win and mp7 for win are in the repo.
            need to update this...
    '''
    # copy in deps and exes
    if 'window' in platform.platform().lower():
        bd = os.path.join('..','..','bin', 'win')
        
    elif 'linux' in platform.platform().lower():
        bd = os.path.join('..','..','bin', 'linux')
        
    else:
        bd = os.path.join('..','..','bin', 'mac')
        
    for f in os.listdir(bd):
            shutil.copy2(os.path.join(bd, f), os.path.join(d, f))

    try:
        shutil.rmtree(os.path.join(d,'flopy'))
    except:
        pass

    try:
        shutil.copytree(os.path.join('..','..','dependencies','flopy'), os.path.join(d,'flopy'))
    except:
        pass

    try:
        shutil.rmtree(os.path.join(d,'pyemu'))
    except:
        pass

    try:
        shutil.copytree(os.path.join('..','..','dependencies','pyemu'), os.path.join(d,'pyemu'))
    except:
        pass

def get_ies_obs_dict(m_d='master_ies', pst=None, modnm='elk_2lay'):
    obs_df_dict = {}

    pst = pyemu.Pst(os.path.join(m_d,f'{modnm}.pst'))
    if pst.control_data.noptmax == -1:
        itrs = [0]
    else:
        itrs = range(pst.control_data.noptmax+1)

    for i in itrs:
        jcbName = os.path.join(m_d,f'{modnm}.{i}.obs.jcb')
        if os.path.exists(jcbName):
            print(f'loading itr {i}')
            jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
            obs_df_dict[i] = jcb
    return obs_df_dict

def get_ies_prior(m_d='master_ies', pst=None):
    obs_df_dict = {}

    pst = pyemu.Pst(os.path.join(m_d,'elk_2lay.pst'))

    jcbName = os.path.join(m_d,f'elk_2lay.0.obs.jcb')
 
    jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
    obs_df_dict = jcb
    return obs_df_dict

def get_ies_par_dict(m_d='master_ies', pst=None):
    par_df_dict = {}

    pst = pyemu.Pst(os.path.join(m_d,'elk_2lay.pst'))
    if pst.control_data.noptmax == -1:
        itrs = [0]
    else:
        itrs = range(pst.control_data.noptmax+1)

    for i in itrs:
        jcbName = os.path.join(m_d,f'elk_2lay.{i}.par.jcb')
        if os.path.exists(jcbName):
            print(f'loading itr {i}')
            jcb = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=jcbName)
            par_df_dict[i] = jcb
    return par_df_dict
  
def plot_1_to_1_ss(m_d, obsdict,modnm='elk_2lay'):
    # pst = pyemu.Pst(os.path.join(m_d, 'elk_2lay.pst'))

    o_d = os.path.join(m_d, 'results', 'figures', 'one2one_plots')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    #pwl = pd.read_csv(os.path.join(m_d, 'tables', 'processed_WL_timeseries.csv'))

    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis'])
    m = sim.get_model(modnm)

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    gwobs = obs.loc[(obs['obgnme'].str.contains('sshds'))].copy()
    
    gwobs['datetime'] = pd.to_datetime(gwobs['obsnme'].str.extract(r'_datetime:(\d{4}-\d{2}-\d{2})')[0])
    gwobs[['id', 'k', 'i', 'j']] = (gwobs['obsnme'].str.extract(r'ssh_id:(\d+)_k:(\d+)_i:(\d+)_j:(\d+)'))
    
    gwobs['k'].unique()

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(float)
        gwobs = gwobs.loc[pd.notna(gwobs[col])]
        print(gwobs[col].unique())
        gwobs[col] = gwobs[col].astype(int)

    top = m.dis.top.array
    botm = m.dis.botm.array
    nlay = m.dis.nlay.data
    itrmx = max(obsdict)
    
    if nlay == 4:
        layers = {
            0: 'WSS',
            1: 'WBV',
            2: 'DC',
            3: 'WR',
            }
    elif nlay == 6:
        layers = {
            0: 'WSS-1',
            1: 'WSS-2',
            2: 'WSS-3',
            3: 'WBV',
            4: 'DC',
            5: 'WR'
            }   
    elif nlay == 7:
        layers = {
            0: 'WSS-1',
            1: 'WSS-2',
            2: 'WSS-3',
            3: 'CONF',
            4: 'WBV',
            5: 'DC',
            6: 'WR',
            7: 'LWR'
            }
        
   
    gwo_grps = gwobs.groupby('k')

    print(gwo_grps.groups.keys())
    with PdfPages(os.path.join(o_d,'steadystate_o_v_s_by_layer.pdf')) as pdf:
        for nm, grp in gwo_grps:
            layer = nm
            lynm = layers[layer].replace('_', ' ')
        
            if 'WSS' in lynm:
                lynm = lynm.replace('WSS','Wahpeton Shallow Sand')

            # keep only weight-1 observations
            wobs = grp.loc[grp.weight == 1, :].reset_index(drop=True)
            if len(wobs) == 0:
                wobs = grp.reset_index(drop=True)
                wobs = wobs.iloc[[0], :]
            #if wobs.empty:
            #    continue
        
            # pull simulated ensemble columns for these obsnmes
            vals = obsdict[itrmx].loc[:, obsdict[itrmx].columns.isin(grp.obsnme.values)].T
            vals.index.name = 'obsnme'
            
            # merge to get obsval beside every ensemble column
            vals_mrg = (
                pd.merge(vals.reset_index(), wobs[['obsnme', 'obsval']], on='obsnme')
                .dropna(subset=['obsval'])
            )

            # columns we want in the final plot DataFrame
            upd_cols = np.append(vals.columns, 'obsval')
            vals_mrg = vals_mrg.loc[:, upd_cols]

            # full data range (observed + all simulated)
            vmin, vmax = vals_mrg.values.min(), vals_mrg.values.max()

            # ───────────────────────── plot ─────────────────────────
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(111)

            # ensemble cloud
            for col in vals.columns:
                ax1.scatter(vals_mrg['obsval'], vals_mrg[col],
                            c='b', alpha=0.01, s=5)

            # optional 'base' column if present
            if 'base' in vals_mrg.columns:
                ax1.scatter(vals_mrg['obsval'], vals_mrg['base'],
                            color='k', s=5, label='Base of posterior')

            # 1:1 reference line
            ax1.plot([vmin, vmax], [vmin, vmax], 'k-', lw=0.5)

            # square axes spanning the full range
            ax1.set_xlim(vmin, vmax)
            ax1.set_ylim(vmin, vmax)
            ax1.set_aspect('equal', adjustable='box')

            # labels, ticks, title
            ax1.set_xlabel(r'$\bf{Observed}$ SS groundwater level (ft)', fontsize=10)
            ax1.set_ylabel(r'$\bf{Simulated}$ SS groundwater level (ft)', fontsize=10)
            ax1.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{int(x):,}')
            )
            ax1.get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{int(x):,}')
            )
            ax1.tick_params(axis='both', which='both', direction='in', labelsize=9)
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.set_title(f'{lynm} – SS groundwater-level 1:1 plot', fontsize=12)

            fig.tight_layout()

            pdf.savefig(fig)
            
            # save first ➜ then (optionally) show
            outfile = os.path.join(o_d, f'{lynm}_ss.png')
            fig.savefig(outfile, dpi=300)
            print(f'Saved {outfile}')

            # Uncomment if you want to view interactively while scripting
            # plt.show()

            plt.close(fig)
    pdf.close()

def plot_1_to_1_sspmp(m_d, obsdict, modnm='elk_2lay'):
    # pst = pyemu.Pst(os.path.join(m_d, 'elk_2lay.pst'))

    o_d = os.path.join(m_d, 'results', 'figures', 'one2one_plots')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    #pwl = pd.read_csv(os.path.join(m_d, 'tables', 'processed_WL_timeseries.csv'))

    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6',load_only=['dis'])
    m = sim.get_model('elk_2lay')

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    #gwobs = obs.loc[obs['obgnme'].str.contains('ssphds'),:].copy()  # *** NOTE - this is a hard-coded filter
    gwobs = obs.loc[obs['obsnme'].str.contains('sspmp') ,:].copy()  # *** NOTE - this is a hard-coded filter
    print(gwobs.obsnme.values)
    gwobs = obs.loc[obs['obgnme']=='wbvsspmphds' ,:].copy()
    print(gwobs.obgnme.unique())
    gwobs['datetime'] = pd.to_datetime(gwobs['obsnme'].str.extract(r'_datetime:(\d{4}-\d{2}-\d{2})')[0])
    gwobs[['id', 'k', 'i', 'j']] = (gwobs['obsnme'].str.extract(r'sspmp_id:(\d+)_k:(\d+)_i:(\d+)_j:(\d+)').astype(int))

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)

    top = m.dis.top.array
    botm = m.dis.botm.array
    nlay = m.dis.nlay.data
    itrmx = max(obsdict)
    
    if nlay == 4:
        layers = {
            0: 'WSS',
            1: 'WBV',
            2: 'DC',
            3: 'WR',
            }
    elif nlay == 6:
        layers = {
            0: 'WSS-1',
            1: 'WSS-2',
            2: 'WSS-3',
            3: 'WBV',
            4: 'DC',
            5: 'WR'
            }   
        
   
    gwo_grps = gwobs.groupby('k')

    print(gwo_grps.groups.keys())
    with PdfPages(os.path.join(o_d,'steadystate_with_pumping__o_v_s_by_layer.pdf')) as pdf:
        for nm, grp in gwo_grps:
            layer = nm
            lynm = layers[layer].replace('_', ' ')
        
            if 'WSS' in lynm:
                lynm = lynm.replace('WSS','Wahpeton Shallow Sand')

            # keep only weight-1 observations
            wobs = grp.reset_index(drop=True)
            #if wobs.empty:
            #    continue
        
            # pull simulated ensemble columns for these obsnmes
            vals = obsdict[itrmx].loc[:, obsdict[itrmx].columns.isin(grp.obsnme.values)].T
            vals.index.name = 'obsnme'

            # merge to get obsval beside every ensemble column
            vals_mrg = (
                pd.merge(vals.reset_index(), wobs[['obsnme', 'obsval']], on='obsnme')
                .dropna(subset=['obsval'])
            )

            # columns we want in the final plot DataFrame
            upd_cols = np.append(vals.columns, 'obsval')
            vals_mrg = vals_mrg.loc[:, upd_cols]

            # full data range (observed + all simulated)
            vmin, vmax = vals_mrg.values.min(), vals_mrg.values.max()

            # ───────────────────────── plot ─────────────────────────
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(111)

            # ensemble cloud
            for col in vals.columns:
                ax1.scatter(vals_mrg['obsval'], vals_mrg[col],
                            c='b', alpha=0.01, s=5)

            # optional 'base' column if present
            if 'base' in vals_mrg.columns:
                ax1.scatter(vals_mrg['obsval'], vals_mrg['base'],
                            color='k', s=5, label='Base of posterior')

            # 1:1 reference line
            ax1.plot([vmin, vmax], [vmin, vmax], 'k-', lw=0.5)

            # square axes spanning the full range
            ax1.set_xlim(vmin, vmax)
            ax1.set_ylim(vmin, vmax)
            ax1.set_aspect('equal', adjustable='box')

            # labels, ticks, title
            ax1.set_xlabel(r'$\bf{Observed}$ SS groundwater level (ft)', fontsize=10)
            ax1.set_ylabel(r'$\bf{Simulated}$ SS groundwater level (ft)', fontsize=10)
            ax1.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{int(x):,}')
            )
            ax1.get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{int(x):,}')
            )
            ax1.tick_params(axis='both', which='both', direction='in', labelsize=9)
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.set_title(f'{lynm} – SS groundwater-level 1:1 plot', fontsize=12)

            fig.tight_layout()

            pdf.savefig(fig)
            
            # save first ➜ then (optionally) show
            outfile = os.path.join(o_d, f'{lynm}_ss_with_pumping.png')
            fig.savefig(outfile, dpi=300)
            print(f'Saved {outfile}')

            # Uncomment if you want to view interactively while scripting
            # plt.show()

            plt.close(fig)
    pdf.close()

def plot_obs_v_sim_flux(m_d, obsdict):
    fdir = os.path.join(m_d,'results','figures','obs_v_sim_heads')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    pst = pyemu.Pst(os.path.join(m_d,'elk_2lay.pst'))

    iters = list(obsdict.keys())
    iters.sort()
    iters = [iters[0], iters[-2]]
    noise = pyemu.ObservationEnsemble.from_binary(pst=obsdict[iters[0]].pst,
                                                  filename=os.path.join(m_d, 'elk_2lay.obs+noise.jcb'))
    obs_df = pd.read_csv(os.path.join('data', 'processed_streamflow_timeseries_baseflow.csv'))
    obs_df['station_id'] = obs_df['obsnme'].str.split('_').str[3]
    
    pdc = pd.read_csv(os.path.join(m_d, 'elk_2lay.pdc.csv'))
    pdc_obs = set(pdc.loc[:, 'name'].str.lower().to_list())

    obs = pst.observation_data
    flobs = obs.loc[obs.oname.str.contains('drn'),:].copy()
    flobs['datetime'] = pd.to_datetime(flobs.datetime.values)
    start_datetime = '1979-12-31'
    dts = (pd.to_datetime(start_datetime) + pd.to_timedelta(np.float64(flobs.time.values),unit='d'))
    flobs['datetime'] = dts

    gages = []
    for name in flobs.obsnme:
        test = name.split('drn_')
        gages.append(test[2].split('_')[0])

    gages = np.unique(gages)
    pdf = PdfPages(os.path.join(fdir, f'obs_v_sim_fluxes.pdf'))

    # plot prior and posterior gages
    for gage in gages:
        if gage == 'drn':
            gage_id = '8127000'
        else:
            gage_id = gage

        uobs = flobs.loc[flobs.obsnme.str.contains(gage), :].copy()
        
        uobs.sort_values(by='datetime', inplace=True)
        #oobs = uobs.loc[uobs.observed == True, :]
        wobs = uobs.loc[uobs.weight > 0, :]
        dts = uobs.datetime.values
        pr_vals = obsdict[0].loc[:, uobs.obsnme].values
        pt_vals = obsdict[iters[-1]].loc[:, uobs.obsnme].values
        base_vals = obsdict[iters[-1]].loc[:,uobs.obsnme].loc['base',:].values

        fig,ax = plt.subplots(1,1,figsize=(8.5,5.5))

        #[ax.plot(dts, pr_vals[i], '0.5', lw=0.5, alpha=0.5) for i in range(len(pr_vals))]
        #ax.plot(dts, pr_vals[0], '0.5', lw=0.5, alpha=0.5, label='prior')
        [ax.plot(dts, pt_vals[i], 'b', lw=0.5, alpha=0.5) for i in range(len(pt_vals))]
        ax.plot(dts, pt_vals[0], 'b', lw=0.5, label='Posterior ensemble')
        ax.plot(dts, base_vals, '--',color='orange', lw=1, label='Base of posterior')
        
        station = obs_df.loc[obs_df['station_id'] == gage_id]
        station.datetime = pd.to_datetime(station.datetime)
        ax.plot(station.datetime, station.obsval, marker='o', color='grey', label='Estimated observed baseflow')
        
        stnm = station['station_id'].values[0]
        #ax.plot(wobs.datetime,wobs.obsval, marker='*',lw=0, color='r', label='measured')
        ax.legend()
        ax.set_title(f'USGS Gauge ID: {gage_id}',loc='left')
        ax.set_ylabel('Baseflow (cfs)')
        
        #ax.set_ylim([0,800])
        pdf.savefig(fig)
        fig.savefig(os.path.join(fdir,f'{stnm}_comp.png'))
        plt.close(fig)
        plt.clf()
        plt.cla()
    pdf.close()

def plot_simple_obs_v_sim(m_d, modnm='elk_2lay'):
    obsdict = get_ies_obs_dict(m_d=m_d, modnm=modnm)
    pst = pyemu.Pst(os.path.join(m_d,f'{modnm}.pst'))
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,'elk_2lay.obs+noise.jcb'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('hifreq|ext|elev'),:].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    aobs = pd.read_csv(os.path.join(m_d,'elk_2lay.adjusted.obs_data.csv'),index_col=0)
    conflicts = set(gwobs.loc[gwobs.weight>0,'obsnme'].tolist()) - set(aobs.loc[aobs.weight>0,:].index.tolist())
    print(conflicts)

    for col in ['k','i','j']:
        gwobs[col] = gwobs[col].astype(int)
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d)
    m = sim.get_model()
    top = m.dis.top.array
    botm = m.dis.botm.array
    itrmx = max(obsdict)
    usites = gwobs[gwobs.usecol.str.contains('freq')].usecol.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site':usites},index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.usecol==usite,:]
        usitedf.loc[usite,'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight','site'],ascending=False,inplace=True)

    usites = usitedf['site'].values
    #print(usites)
    with PdfPages(os.path.join(m_d,'simple_o_v_s.pdf')) as pdf:
        for site in usites:
            uobs = gwobs.loc[gwobs.usecol==site,:].copy()
            uobs.sort_values(by='datetime',inplace=True)
            k,i,j = uobs.k.values[0],uobs.i.values[0],uobs.j.values[0]
            oobs = uobs.loc[uobs.observed == True, :]
            wobs = oobs.loc[oobs.weight > 0, :]
            dts = uobs.datetime.values
            vals = obsdict[0].loc[:,uobs.obsnme].values

            fig,ax = plt.subplots(1,1,figsize=(10,5))
            [ax.plot(dts,vals[i,:],color='0.5',alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            unoise = noise.loc[:, wobs.obsnme].values
            ndts = wobs.datetime
            [ax.plot(ndts, unoise[i, :], color='r', alpha=0.25, lw=0.1) for i in range(unoise.shape[0])]

            if itrmx > 0:
                vals = obsdict[itrmx].loc[:,uobs.obsnme].values
                [ax.plot(dts, vals[i, :], color='b', alpha=0.5, lw=0.1) for i in range(vals.shape[0])]
            ax.scatter(oobs.datetime, oobs.obsval, marker='o', color='r',facecolor='none', s=50,zorder=10)
            ax.scatter(wobs.datetime, wobs.obsval, marker='.', color='r', s=20,zorder=10)
            cobs = wobs.loc[wobs.obsnme.apply(lambda x: x in conflicts),:]
            ax.scatter(cobs.datetime, cobs.obsval, marker='*', color='k', s=50,zorder=10)

            if wobs.shape[0] > 0:
                mn = unoise.min()
                mx = unoise.max()
                ax.set_ylim(mn*0.9,mx*1.1)
            elif oobs.shape[0] > 0:
                mn = oobs.obsval.min()
                mx = oobs.obsval.max()
                ax.set_ylim(mn * 0.9, mx * 1.1)

            t = top[i,j]
            bslice = botm[:,i,j]
            xlim = ax.get_xlim()
            ax.plot(xlim,[t,t],'m--',lw=1.5)
            for b in bslice:
                ax.plot(xlim,[b,b],'c--',lw=1.5,alpha=0.5)
            ax.set_title('usecol:{0}, mx weight: {1}, kij:{4} top: {2}\nbotm:{3}'. \
                         format(site, wobs.weight.max(), t, str(bslice),str((k,i,j))), loc='left')
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            print('...', site, 'num conflicts:', cobs.shape[0])
             
def plot_obs_v_sim_by_well(m_d, obsdict):
    
    fdir = os.path.join(m_d)#,'prelim_figs','ies_figs')
    
    iters = list(obsdict.keys())
    iters.sort()
    iters = [iters[0],iters[-1]]
    noise = pyemu.ObservationEnsemble.from_binary(pst=obsdict[iters[0]].pst,filename=os.path.join(m_d,'elk_2lay.obs+noise.jcb'))

    pdc = pd.read_csv(os.path.join(m_d,'elk_2lay.pdc.csv'))
    pdc_obs = set(pdc.loc[:,'name'].str.lower().to_list())
    
    obs = obsdict[iters[0]].pst.observation_data.copy()
    robs = obs.loc[obs.obsnme.str.contains('freq'),:]
    robs.loc[:,'datetime'] = pd.to_datetime(robs.datetime)
    for col in ['k','i','j']:
        for name in robs.usecol.unique():
            try:
                robs.loc[robs.usecol==name,col] = h_df.loc[h_df.obsprefix==name.replace('-','.'),col].values[0].astype(int)
            except:
                continue
    
    boundary  = gpd.read_file(os.path.join('gis','input_shapefiles',
                                           'CrossTimbersAquiferExtent.shp'))
    # county  = gpd.read_file(os.path.join('gis','input_shapefiles',
    #                                      'County.shp'))
    wells_shp = gpd.read_file(os.path.join('gis','input_shapefiles',
                                           'Wells','Wells_updated_reprj2265_used.shp'))
    drns = gpd.read_file(os.path.join('gis','input_shapefiles','Rivers_Streams','xtim_riv_str_hres_stord_5.shp'))

    drns = drns.to_crs(wells_shp.crs)

    boundary = boundary.to_crs(wells_shp.crs)

    # get extents of boundary
    minx, miny, maxx, maxy = boundary.total_bounds

    county = county.to_crs(wells_shp.crs)

    wells_shp.columns = wells_shp.columns.str.lower()
    wells_shp = wells_shp.rename(columns={'statewelln':'swn'})
    wells_shp.swn = wells_shp.swn.astype(int)
    well_data = pd.read_csv(os.path.join('data','heads_obs_updated_setup.csv'),
                            index_col='wellid')
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6')
    start_datetime = sim.tdis.start_date_time.array
    perlen = sim.tdis.perioddata.array['perlen']
    nper = len(perlen)
    m = sim.get_model('elk_2lay')
    nlay = m.dis.nlay.data
    botm = m.dis.botm.data
    top = m.dis.top.array
    
    
    def set_violin_body_color(vi_parts,color,lw,alpha):
        for vp in vi_parts['bodies']:
            vp.set_facecolor(color)
            vp.set_edgecolor(color)
            vp.set_linewidth(lw)
            vp.set_alpha(alpha)
    
    pdf = PdfPages(os.path.join(fdir,f'obs_v_sim_well_hydrographs.pdf'))
    
    unique_usecols = robs.usecol.unique()
    unique_usecols.sort()
    for usecol in unique_usecols:
        print('...',usecol)
        uobs = robs.loc[robs.usecol==usecol,:].copy()
        # find which k values are associated with nzobs
        keep_k = []
        for k in uobs.k.unique():
            kobs = uobs.loc[uobs.k==k,:]
            if np.any(kobs.observed):
                keep_k.append(k)
        if len(keep_k) == 0:
            continue
        if len(keep_k) > 1:
            print(uobs)
            print(usecol,uobs.k.unique())
            raise Exception('multiple actual obs layers for the same usecol {0}'.format(usecol))
        uobs = uobs.loc[uobs.k==keep_k[0],:]

        assert np.int8(uobs.i.values).std() == 0
        assert np.int8(uobs.j.values).std() == 0
        assert np.int8(uobs.k.values).std() == 0
        uobs.loc[:,'swn'] = uobs.usecol.apply(lambda x: float(x.split('-')[1]))
        assert uobs.swn.std() == 0
        uobs.sort_values(by='datetime',inplace=True)
        dates = uobs.datetime.values
        k,i,j = int(uobs.k.iloc[0]),int(uobs.i.iloc[0]),int(uobs.j.iloc[0])
        fig = plt.figure(figsize=(11,8.5))
        grid = plt.GridSpec(3, 3)
        inset = fig.add_subplot(grid[0,1])
        violin = fig.add_subplot(grid[:,-1])
        ts     = fig.add_subplot(grid[1:,:-1])
        ts.patch.set_alpha(0.0)
        text   = fig.add_subplot(grid[0,0])
        county.boundary.plot(lw=0.25, color='grey',ax=inset)
        boundary.boundary.plot(lw=1.25,color='black',ax=inset)
        drns.plot(lw=.25,color='blue',ax=inset)
        wells_shp.plot(ax=inset, column='k',markersize=2,legend=True)
        wells_shp[wells_shp.swn==uobs.swn.iloc[0]].plot(ax=inset, color='orange',markersize=20,edgecolor='black')
        inset.set_ylim(miny,maxy)
        inset.set_xlim(minx,maxx)
        inset.set_axis_off()
        top_elev=botm[k-1,i,j] if k>0 else top[i,j]
        bot_elev=botm[k,i,j]
        ls_elev =top[i,j]
        
        # todo: add screen info to obs names next run
        try:
            screen_top,screen_bot=well_data.loc[int(uobs.swn.iloc[0]),['screen_top','screen_bot']].drop_duplicates().values
        except:
            screen_top,screen_bot=well_data.loc[int(uobs.swn.iloc[0]),['screen_top','screen_bot']].drop_duplicates().values[0]
        xmin,xmax=0,4
        for l in range(nlay):
            if l==1:
                violin.hlines(botm[l,i,j], color='brown',xmin=xmin,xmax=xmax, label='Model Layer Elevations')
            else:
                violin.hlines(botm[l,i,j], color='brown',xmin=xmin,xmax=xmax)
        violin.hlines(top_elev, color='purple',xmin=xmin,xmax=xmax)
        violin.hlines(bot_elev, color='purple',xmin=xmin,xmax=xmax)
        violin.hlines(ls_elev, color='green',linestyles='dotted',xmin=xmin,xmax=xmax)
        vmin,vmax = violin.get_ylim()
        violin.hlines(screen_top,color='black', linestyles='dashed',xmin=xmin,xmax=xmax)
        violin.hlines(screen_bot,color='black', linestyles='dashed',xmin=xmin,xmax=xmax)
        
        hmax,hmin=-1e30,1e30
        for itr,c in zip(iters,['0.5','b']):
            
            df = obsdict[itr]._df.copy()
            df = df.loc[:,uobs.obsnme]
            
            #pr.drop(pr.index[pr.max(axis=1)>5e3],axis=0,inplace=True)  
            [ts.plot(dates,df.loc[idx,:],c,lw=0.1,alpha=0.75) for idx in df.index[(df.index!='base')]]
            if 'base' in df.index:
               ts.plot(dates,df.loc['base',:],c,lw=0.5,alpha=1.0) 
            vi_parts = violin.violinplot(df.values.flatten(),
                              widths=0.75,showextrema=False)
            set_violin_body_color(vi_parts,c,0.5,0.5)
            if itr != 0:
                hmin,hmax = ts.get_ylim()
                vmin,vmax = violin.get_ylim()
            
        ts.set_ylabel('ft amsl')
        ts.set_ylim(hmin,hmax)
        n_bscreen = 0
        oobs = uobs.loc[uobs.observed==True,:]
        ts.scatter(oobs.datetime,oobs.obsval,marker='^',c='r',s=20,zorder=10)
        nzobs = uobs.loc[uobs.weight > 0,:].copy()
        ts.scatter(nzobs.datetime,nzobs.obsval,marker='o',facecolor='none',edgecolor='r',s=20,zorder=10)
        hmin = min(hmin,oobs.obsval.mean() - oobs.obsval.std()*3)
        hmax = max(hmax,oobs.obsval.mean() + oobs.obsval.std()*3)
        
        nconflicts = 0
        if nzobs.shape[0] > 0:
            nzobs.sort_values(by='datetime',inplace=True)
            nznoise = noise._df.loc[:,nzobs.obsnme].copy()
            [ts.plot(nzobs.datetime.values,nznoise.loc[idx,:],'r',lw=0.1,alpha=0.35) for idx in nznoise.index]
        
            pdcobs = nzobs.loc[nzobs.obsnme.apply(lambda x: x in pdc_obs),:].copy()
            if pdcobs.shape[0] > 0:
                print(pdcobs.shape[0])
                ts.scatter(pdcobs.datetime.values,pdcobs.obsval.values,marker='*',s=50,c='k',zorder=11,label='Prior Data Conflict')
                nconflicts = pdcobs.shape[0]

        ymin = bot_elev-10 if oobs.obsval.min() < screen_bot else screen_bot-10
        violin.set_ylim(ymin,ls_elev+10)
        ts.set_ylim(ymin,ls_elev+10)
        
        violin.scatter(np.ones_like(oobs.obsval),oobs.obsval,marker='^',c='r',s=20,zorder=10)
        if any(oobs.obsval<screen_top):
            violin.scatter(np.ones_like(oobs.obsval[oobs.obsval<screen_top]),oobs.obsval[oobs.obsval<screen_top],marker='^',c='black',s=20,zorder=10,
                        label = 'observed values below screen top')
        violin.legend(loc='lower left')
        violin.set_xlim(0.5,1.5)
        
        violin.set_xticks([])
        delta = (top_elev-bot_elev)/200
        violin.text(1.5-0.25,top_elev+delta, f'Layer {k+1} Top')
        violin.text(1.5-0.30,bot_elev+delta, f'Layer {k+1} Bottom')
        violin.text(0.55,ls_elev+delta, 'Model Top')
        violin.text(0.55,screen_top+delta, 'Screen Top')
        violin.text(0.55,screen_bot+delta, 'Screen Bottom')

        text.text(0,0.5,f'site: {usecol}\nlay:{k+1}, row:{i+1}, col:{j+1}\nobs:{oobs.shape[0]},nzobs:{nzobs.shape[0]}\nconflicts:{nconflicts}', fontsize=15)
        text.set_axis_off()
        pdf.savefig(fig)
        #fig.savefig(os.path.join(fdir, 'well_hydrographs',f'{owsn.split('_')[0]}_comp.png'))
        plt.close(fig)
        plt.clf()
        plt.cla()
    pdf.close()

def plot_zone_histos(m_d, obsdict, logscale=False):
    import matplotlib.ticker as mticker
    fdir = os.path.join(m_d,'results','figures','zone_histos')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    iters = list(obsdict.keys())
    iters.sort()
    obs = obsdict[iters[-1]].pst.observation_data.copy()
    robs = obs.loc[obs.obsnme.str.contains('arr'),:]
    robs = robs.loc[~robs.obsnme.str.contains('arrp'),:].copy()


    robs[['id','k','i','j']] = robs['obsnme'].str.extract(
        r'oname:([^:]+):(\d+)_otype:[^_]+_i:(\d+)_j:(\d+)'
    )
    robs['id'] = robs['id'].str.split('_').str[0]

    robs[['k','i','j']] = robs[['k','i','j']].astype(int)
    
    pars = ['hk','vk','ss','sy']
    parnames = ['k','k33','ss','sy']
    plong = ['Horizontal Hydraulic Conductivity (ft/day)','Vertical Hydraulic Conductivity','Specific Storage','Specific Yield']
    
    par_dict = {}
    cnt = 0
    for par in pars:
        par_dict[par] = [parnames[cnt],plong[cnt]]
        cnt += 1
    log_pars = {'hk','vk','ss'}   # which parameters use log‐x

    # --- load model and data (assumed already in your namespace) ---
    sim     = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis'])
    m       = sim.get_model('elk_2lay')
    idom    = m.dis.idomain.data      # shape (nlay, nrow, ncol)
    pr      = obsdict[0]._df          # prior DataFrame: index = real IDs, cols = obs names
    pt      = obsdict[iters[-1]]._df  # posterior DataFrame
    robs    = robs                      # your obs‐info DataFrame with ['obsnme','k','i','j'], index=obs names
    zon_arr = setup_pst.make_new_zone(mws=m_d, gwf=m)
    reals   = pr.index.values


    # --- begin plotting ---
    with PdfPages(os.path.join(fdir, 'layer_zone_histos.pdf')) as pdf:
        for par in pars:
            short, longname = par_dict[par]
            # select only obs for this parameter
            parobs = robs[robs.obsnme.str.contains(f'{short}_k')]
            if parobs.empty:
                continue

            layers = sorted(parobs.k.unique())
            for k in layers:
                lyobs = parobs[parobs.k == k]

                # decide zones (None for layers 3 & 5)
                if k in (3,5):
                    zones = [None]
                else:
                    # gather active zones for this layer/parameter
                    zones = sorted({
                        int(zon_arr[k,i,j])
                        for i,j in lyobs[['i','j']].values
                        if idom[k,i,j] > 0
                    })
                    if not zones:
                        zones = [None]

                n_z  = len(zones)
                rows = n_z if n_z > 1 else 2
                fig, axes = plt.subplots(rows, 1, figsize=(6, 6), squeeze=False)
                axes = axes.flatten()
                if n_z == 1:
                    axes[1].axis('off')

                for idx, zone in enumerate(zones):
                    ax = axes[idx]

                    # build list of obs columns for this zone
                    if zone is None:
                        # whole layer, only active cells
                        coords = lyobs[['i','j']].values
                        cols = [
                            col for col, (i,j) in zip(lyobs.index, coords)
                            if idom[k,i,j] > 0
                        ]
                        title_zone = '(whole layer)'
                    else:
                        coords = lyobs[['i','j']].values
                        cols = [
                            col for col, (i,j) in zip(lyobs.index, coords)
                            if (zon_arr[k,i,j] == zone) and (idom[k,i,j] > 0)
                        ]
                        title_zone = f'(Zone {zone})'

                    if not cols:
                        continue

                    # extract prior/post values
                    pr_vals = pr.loc[reals, cols].to_numpy().ravel()
                    pt_vals = pt.reindex(index=reals, columns=cols).to_numpy().ravel()
                    pr_vals = pr_vals[~np.isnan(pr_vals)]
                    pt_vals = pt_vals[~np.isnan(pt_vals)]

                    # debug summary
                    print(f'Layer {k+1} {title_zone}: '
                        f'{len(pr_vals)} prior pts, {len(pt_vals)} post pts, '
                        f'prior [{pr_vals.min():.2e}, {pr_vals.max():.2e}], '
                        f'post [{pt_vals.min():.2e}, {pt_vals.max():.2e}]')

                    # choose bins & scale
                    if par in log_pars:
                        # filter out non‐positive
                        pr_vals = pr_vals[pr_vals > 0]
                        pt_vals = pt_vals[pt_vals > 0]
                        allv = np.concatenate([pr_vals, pt_vals])
                        vmin, vmax = allv.min(), allv.max()
                        bins = np.logspace(np.log10(vmin), np.log10(vmax), 50)
                        ax.set_xscale('log')
                        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
                    else:
                        bins = 50
                        ax.xaxis.set_major_formatter(
                            mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
                        )

                    # common y‐axis formatting
                    ax.yaxis.set_major_formatter(
                        mticker.FuncFormatter(lambda y, _: f'{int(y):,}')
                    )

                    # plot histograms
                    ax.hist(pr_vals, bins=bins, alpha=0.75, color='0.5', label='Prior')
                    ax.hist(pt_vals, bins=bins, alpha=0.6, color='b',   label='Posterior')
                    ax.set_title(f'Layer {k+1} {title_zone}', fontsize=10)
                    ax.set_xlabel(longname)
                    ax.set_ylabel('Frequency')
                    ax.legend(loc='upper right', fontsize=8)

                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

def plot_base_params_map(m_d,obsdict,logscale,partype='hk'):
    fdir = os.path.join(m_d)
    iters = list(obsdict.keys())
    iters.sort()
    obs = obsdict[iters[-1]].pst.observation_data.copy()
    robs = obs.loc[obs.obsnme.str.contains('arr'),:]
    robs['datetime'] = pd.to_datetime(
        robs['obsnme'].str.extract(r'_datetime:(\d{4}-\d{2}-\d{2})')[0]
    )

    robs[['id', 'k', 'i', 'j']] = (
        robs['obsnme']
            .str.extract(r'sshid:(\d+)-(\d+)-(\d+)-(\d+)')   # 4 capture groups
            .astype(int)                                     # make them ints
    )

    robs.loc[:,'datetime'] = pd.to_datetime(robs.datetime)
    robs.loc[:,['i','j','k']] = robs.loc[:,['i','j','k']].astype(int)
    
    pars = ['hk','vk','ss','sy']
    parnames = ['k','k33','ss','sy']
    plong = ['Horizontal Hydraulic Conductivity','Vertical Hydraulic Conductivity','Specific Storage','Specific Yield']
    
    par_dict = {}
    cnt = 0
    for par in pars:
        par_dict[par] = [parnames[cnt],plong[cnt]]
        cnt =+ 1
    
    pars = [partype]
    parnames = [par_dict[partype][0]]
    plong = [par_dict[partype][1]][0]
    
    # laod model:
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6')
    start_datetime = sim.tdis.start_date_time.array
    perlen = sim.tdis.perioddata.array['perlen']
    nper=len(perlen)
    dts = (pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(perlen),unit='d')).strftime('%Y%m%d')
    m = sim.get_model('elk_2lay')
    nlay = m.dis.nlay.data
    nrow = m.dis.nrow.data
    ncol = m.dis.ncol.data
    idom = m.dis.idomain.data
    
    # get gis data:
    g_d = os.path.join('..','..','gis')
    wahp_extent = gpd.read_file(os.path.join(g_d, 'input_shps', 'elk', 'elk_boundary_lf.shp'))
    modelgrid = gpd.read_file(os.path.join(g_d, 'output_shps', 'elk', 'elk_cell_size_660ft_epsg2265_rot20.grid.shp'))
    #county = gpd.read_file(os.path.join(g_d,'County.shp'))
    #county = county[county.NAME.isin(counties_in_stdyarea)]
    
    #roads = gpd.read_file(os.path.join(g_d,'tl_2019_48_prisecroads.shp'))
    #opts = gpd.read_file(os.path.join(g_d,'Wells','TWDB_Wells_reprj2265_used.shp'))
    #roads = roads.to_crs(opts.crs)
    #counties = county.to_crs(opts.crs)
    #wahp_extent = wahp_extent.to_crs(opts.crs)
    #counties = counties.overlay(wahp_extent, how='intersection').NAME.unique()
    #i20 = roads[roads.FULLNAME=='I- 20']
    
    cpts = pd.read_csv(os.path.join('data','analyzed', 'processed_ss_head_targs.csv'))
    cpts = gpd.GeoDataFrame(cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=wahp_extent.crs)
    #cpts = cpts.to_crs(opts.crs)
    cpts['k'] = cpts.k.astype(int)
    
    if logscale:
        pdf = PdfPages(os.path.join(fdir,f'modelbase_{pars[0]}__maps_log.pdf'))
    else:
        pdf = PdfPages(os.path.join(fdir,f'modelbase_{pars[0]}_maps.pdf'))
    
    for par,parname in zip(pars,parnames):
        print(f'{par}')
        probs = robs.loc[robs.obsnme.str.contains(f'{par}')]
        for k in range(nlay):
            print(f'Layer {k+1}')
            kprobs = probs.loc[probs.k==k]
            for n in ['base']:
                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,10))
                if 0 in iters:
                    pr = obsdict[0]._df
                    par_arr = np.ones((nrow,ncol))*1e+30
                    for (i,j),col in zip(kprobs.loc[:,['i','j']].values,kprobs.loc[:,['i','j']].index):
                        par_arr[i,j] = pr.loc[n,col]
                    if not logscale:
                        par_arr = 10**par_arr
                    mapview = flopy.plot.PlotMapView(model=m,layer=k,ax=ax1)
                    ly_ibnd = idom[k,:,:]
                    par_arr = np.where(ly_ibnd==0,1e+30,par_arr)
                    mv_arr_pr = mapview.plot_array(par_arr, masked_values=[1e+30],ax=ax1)
                
                    ax1.set_title(f'Prior {plong}:\n Layer {k+1}, Realization: {n}',fontsize=14)

                    quadmesh = mapview.plot_bc('RIV',label='riv', color='blue')
                    if k > 0:
                        quadmesh = mapview.plot_bc('GHB',label='ghb', color='cyan')
                    quadmesh = mapview.plot_ibound(edgecolor='None',color_noflow='grey',color_vpt='red',alpha=0.4)
                    counties.boundary.plot(ax=ax1,edgecolor='grey',alpha=0.2,zorder=10)
                    wahp_extent.boundary.plot(ax=ax1,edgecolor='black',alpha=0.5,zorder=11)
            
                    cpts.loc[(cpts.k==k) & (cpts.tag.str.startswith('hifreq'))].plot(ax=ax1,
                                                                    edgecolor='red',
                                                                    facecolor='red',
                                                                    markersize=5,
                                                                    label='hifreq wl obs')
                    cpts.loc[(cpts.k==k) & (cpts.tag.str.startswith('lofreq'))].plot(ax=ax1,
                                                                    edgecolor='orange',
                                                                    facecolor='orange',
                                                                    markersize=5,
                                                                    label='lofreq wl obs')
                    par_arr = np.where(par_arr==1e+30,np.nan,par_arr)
                    pr_par_mx = np.nanmax(par_arr)
                    pr_par_mn = np.nanmin(par_arr)
            
                if max(iters) > 0:
                    itr = max(iters)
                    pt = obsdict[itr]._df
                    par_arr = np.ones((nrow,ncol))*1e+30
                    for (i,j),col in zip(kprobs.loc[:,['i','j']].values,kprobs.loc[:,['i','j']].index):
                        par_arr[i,j] = pt.loc[n,col] 
                    if not logscale:
                        par_arr = 10**par_arr
                    ly_ibnd = idom[k,:,:]
                    par_arr = np.where(ly_ibnd==0,1e+30,par_arr)
                    mapview = flopy.plot.PlotMapView(model=m,layer=k,ax=ax2)
                    mv_arr_pt = mapview.plot_array(par_arr, masked_values=[1e+30],ax=ax2)
                        
                    ax2.set_title(f'Posterior {plong}:\n Layer {k+1}, Realization: {n}',fontsize=14)
                    quadmesh = mapview.plot_ibound(edgecolor='None',color_noflow='grey',color_vpt='red',alpha=0.4)
                    quadmesh = mapview.plot_bc('RIV',label='riv', color='blue')
                    if k >0:
                        quadmesh = mapview.plot_bc('GHB',label='ghb', color='cyan')
  
                    counties.boundary.plot(ax=ax2,edgecolor='grey',alpha=0.2,zorder=10)
                    wahp_extent.boundary.plot(ax=ax2,edgecolor='black',alpha=0.5,zorder=11)
            
                    cpts.loc[(cpts.k==k) & (cpts.tag.str.startswith('hifreq'))].plot(ax=ax2,
                                                                    edgecolor='red',
                                                                    facecolor='red',
                                                                    markersize=5,
                                                                    label='hifreq wl obs')
                    cpts.loc[(cpts.k==k) & (cpts.tag.str.startswith('lofreq'))].plot(ax=ax2,
                                                                    edgecolor='orange',
                                                                    facecolor='orange',
                                                                    markersize=5,
                                                                    label='lofreq wl obs')
                    ax2.legend(loc='upper left',fontsize=12)
                    
                    par_arr = np.where(par_arr==1e+30,np.nan,par_arr)
                    pt_par_mx = np.nanmax(par_arr)
                    pt_par_mn = np.nanmin(par_arr)
                
                pmx = max(pr_par_mx,pt_par_mx)
                pmn = min(pr_par_mn,pt_par_mn)

                mv_arr_pr.set_clim(pmn, pmx)
                mv_arr_pt.set_clim(pmn, pmx)

                norm = matplotlib.colors.Normalize(vmin=pmn,vmax=pmx)
                    
                # if logscale:
                #     sm = plt.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=pmn,vmax=pmx),
                #                             cmap=mv_arr.cmap)
                # else:
                sm = plt.cm.ScalarMappable(norm=norm, cmap=mv_arr_pt.cmap)
                sm.set_array([])
                ax1.get_yaxis().set_visible(False)
                ax1.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                ax2.get_xaxis().set_visible(False)
                ax1.axis('off')
                ax2.axis('off')
                cax = fig.add_axes([0.95, 0.1, 0.01, 0.3])
                cb = fig.colorbar(sm, shrink=0.35, cax=cax,orientation='vertical')
                cb.ax.tick_params(labelsize=12)
                cax.yaxis.set_ticks_position('left')
                cax.yaxis.set_label_position('right')
                cax.set_ylabel(f'{par} (ft/d)',fontsize=12)                
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)  
        pdf.close()

def plot_phi_sequence(m_d, modnm='elk_2lay'):
    fdir = os.path.join(m_d,'results','figures')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10,3.5))
    ax = axes[0]
    phi = pd.read_csv(os.path.join(m_d,f'{modnm}.phi.actual.csv'),index_col=0)
    phi.index = phi.total_runs
    itrs = phi.index/phi.index.min()
    phi.iloc[:,6:].apply(np.log10).plot(legend=False,lw=0.5,color='k', ax=ax)
    ax.set_title(r'Actual $\Phi$')
    ax.set_ylabel(r'log $\Phi$')
    # right
    ax = axes[-1]
    phi = pd.read_csv(os.path.join(m_d,f'{modnm}.phi.meas.csv'),index_col=0)
    phi.index = phi.total_runs
    phi.iloc[:,6:].apply(np.log10).plot(legend=False,lw=0.2,color='r', ax=ax)
    ax.set_title(r'Measured+Noise $\Phi$')
    fig.tight_layout()
    fig.savefig(os.path.join(fdir,'phi.pdf'))
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(4.5,4))
    phi = pd.read_csv(os.path.join(m_d,f'{modnm}.phi.actual.csv'),index_col=0)
    phi.index = phi.total_runs - phi.total_runs.min()
    itrs = phi.index/phi.index.min()
    phi.iloc[:,6:].apply(np.log10).plot(legend=False,lw=0.5,color='grey', ax=ax,label='Realizations')
    ax.set_ylabel(r'log $\Phi$')
    ax.set_xlabel('Total runs')
    # plot vertical lines at index values:
    for itr in phi.index.values:
        if itr != 0:
            ax.axvline(itr, color='k', lw=1.0, ls='--')
    ax.set_xlim(0,phi.index.max())
    
    # plot base:
    base = np.log10(phi['base'])
    ax.plot(phi.index,base,'k',lw=1.0,label='Base Realization')
    # comma format x-axis:
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
    fig.tight_layout()
    fig.savefig(os.path.join(fdir,'phi_totalruns_w_itrs_marked.pdf'))

def plot_wateruse(m_d='.', modnm='elk_2lay', noptmax=0, max_fail=0):
    '''Function to plot specified and simulated annual water use rates by county and aquifer unit
    Args:
        m_d (str): relative path to master directory
        o_d (str): relative path to output directory
        modnm (str): name of MODFLOW model
        noptmax (int): pest parameter estimation iteration to plot
        max_fail (int): maximum threshold of convergence failures allowed before dropping realization from plots
    '''      
    print('\n')
    print('plotting sim wateruse from {0} iteration {1}'.format(m_d, noptmax))
    print('\n')

    pst_name = f'{modnm}.pst'

    o_d = os.path.join(m_d,'results','figures','pumping')
    if os.path.exists(o_d) == False:
        os.makedirs(o_d)
    
    fill_plot = True
    #fill_plot = False

    hide_forecast = True
    # hide_forecast = False

    xmin = pd.to_datetime('19700101', format='%Y%m%d')
    xmax = pd.to_datetime('20400101', format='%Y%m%d')

    pst = pyemu.Pst(os.path.join(m_d, pst_name))
    
    obs = pd.read_csv(os.path.join(m_d, 'listbudget_flx_obs.csv'), index_col=0)
    obs['datetime'] = pd.to_datetime(obs.index, format='%Y%m%d')

    if hide_forecast:
        obs.loc[:,'datetime1'] = pd.to_datetime(obs.loc[:,'datetime'], format='%Y%m%d')
        obs = obs.loc[obs['datetime1'] < pd.to_datetime('1-1-2023')]
        xmax = pd.to_datetime('20230101', format='%Y%m%d')

    use_ins = [x for x in obs.columns if 'in' in x]
    use_rej = [x for x in obs.columns if 'rej' in x]
    use_simins = [x for x in obs.columns if 'simin' in x]
    use_types = [x.split('-')[0] for x in use_simins]
    
    obs_en = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(m_d,f'{modnm}.{noptmax}.obs.jcb'))
    pr_en = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(m_d,f'{modnm}.0.obs.jcb'))
    use_ins_en = [x for x in obs_en.columns if any([y in x for y in use_ins])]
    
    
    obs_en = obs_en.loc[:, use_ins_en].copy()
    obs_en = pd.DataFrame.from_records((obs_en.to_records()))
    obs_en.index = obs_en.pop('index')
    #convert to acre-feet from cubic feet per year
    obs_en *= -0.00837926
    
    pr_en = pr_en.loc[:, use_ins_en].copy()
    pr_en = pd.DataFrame.from_records((pr_en.to_records()))
    pr_en.index = pr_en.pop('index')
    #convert to acre-feet from cubic feet per day
    pr_en *= -0.00837926
    
    clr = '0.5'
    
    ut_dict = {'wel': 'All pumping'}
    
    with PdfPages(os.path.join(o_d, f'{modnm}_pumping.pdf')) as pdf:

        ax_per_page = 1
        ncols = 1
        fig, axes = plt.subplots(int(ax_per_page / ncols), ncols, figsize=(4.5, 4.5), dpi=300)
        ax_count = 0
        pg_count = 0
        plt_count = 0        
        
        for ut in use_types:
            utnm = ut_dict[ut]
            ax = axes
            cols =  [x for x in obs_en.columns if (f':{ut}-' in x) and ('simin' in x)]
            en_grp = obs_en.loc[:, cols].copy()
            en_dts = pd.to_datetime(en_grp.columns.str.split(':').str[-1], format='%Y%m%d')
            en_dts = pd.DataFrame(en_dts, columns=['datetime'])
            en_dts = en_dts.sort_values(by='datetime')
            en_grp_cols = en_grp.columns
            en_grp_df = pd.DataFrame(en_grp_cols, columns=['cols'])
            en_grp_df = en_grp_df.iloc[en_dts.index]
            en_grp_df['datetime'] = en_dts['datetime'].values
            
            cols =  [x for x in pr_en.columns if (f':{ut}-' in x) and ('uin' in x)]
            pr_grp = pr_en.loc[:, cols].copy()
            pr_dts = pd.to_datetime(pr_grp.columns.str.split(':').str[-1], format='%Y%m%d')
            pr_dts = pd.DataFrame(pr_dts, columns=['datetime'])
            pr_dts = pr_dts.sort_values(by='datetime')
            pr_grp_cols = pr_grp.columns
            pr_grp_df = pd.DataFrame(pr_grp_cols, columns=['cols'])
            pr_grp_df = pr_grp_df.iloc[pr_dts.index]
            pr_grp_df['datetime'] = pr_dts['datetime'].values
            
            en_grp = obs_en.loc[:, en_grp_df.cols].copy()
            pr_grp = pr_en.loc[:, pr_grp_df.cols].copy()
            
            # plot range of ensemble simulated values
            if fill_plot:
                ax.fill_between(en_grp_df.datetime,en_grp.min(axis=0).values,
                                en_grp.max(axis=0).values, facecolor='lightblue', alpha=0.5, label='Ensemble range',zorder=3)
            else:
                [ax.plot(en_dts, en_grp.loc[i, :], color='lightblue', lw=0.025, alpha=0.1,zorder=2) for i in en_grp.index.values]
            if 'base' in en_grp.index:
                ax.plot(en_dts, en_grp.loc['base', :], color='navy', lw=1.25, label='Base realization',zorder=10)
            
            # plot prior:
            ax.plot(pr_dts, pr_grp.loc['base', :], color='black', lw=1.25, label='Prior base realization',zorder=4)
            [ax.plot(pr_dts, pr_grp.loc[i, :], color='grey', lw=0.25, alpha=0.5,zorder=1) for i in pr_grp.index.values if i != 'base']


            # plot original model input
            #ax.plot(wdf.loc[:, 'datetime'], wdf.loc[:, grp1], color='k', lw=0.5, ls='--')
                
            # set x axis formatting
            ax.xaxis.set_major_locator(years20)
            ax.xaxis.set_major_formatter(years_fmt)
            ax.set_xlim(xmin, xmax)     
            
            # comma format y axis
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                  
            ax.set_title(f'{utnm} ', loc='left')
            
            if plt_count == 0:
                ax.legend(loc='upper left')
            
            ax_count += 1
            plt_count += 1
            
            if ax_count % 2 != 0:
                ax.set_ylabel('Pumping (acre-feet)')

        plt.tight_layout()
        pdf.savefig()
        plt.savefig(os.path.join(o_d, 'wahp_water_use.png'), dpi=300)
        plt.close(fig)

def plot_water_budget_ss(m_d, obsdict):
    fdir = os.path.join(m_d,'results','figures','water_budget')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    iters = list(obsdict.keys())
    iters.sort()
    iters = [iters[0], iters[-1]]

    obs = obsdict[iters[0]].pst.observation_data.copy()
    bobs = obs.loc[obs.obsnme.str.contains('bud'),:]
    bobs['datetime'] = pd.to_datetime(
        bobs['obsnme'].str.extract(r'_datetime:(\d{4}-\d{2}-\d{2})')[0]
    )
    bobs.loc[:,'datetime'] = pd.to_datetime(bobs.datetime)

    pdf = PdfPages(os.path.join(fdir, f'budget_ss.pdf'))
    bud_cats = bobs.obgnme.unique()
    bud_cats = {
        s.split(':')[3].split('_datetime')[0]
        for s in bobs.obsnme
    }
    n = 0

    #print(cat)
    cobs = bobs.loc[bobs.obsnme.apply(lambda x: '1970-01-01' in x and 'bud' in x), :].copy()


    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    if 0 in iters:
        pr = obsdict[0]._df.loc[:, cobs.obsnme]
        pr = pr.loc[:,pr.sum()!=0]
        pr = pr.loc[:,~pr.columns.str.contains('total|percent')]
        pr.columns = ['drns', 'rch', 'riv_in', 'riv_out']
        # convert ft3/d to acre-ft/yr
        pr = pr/43560*365.25


        [ax.plot(pr.columns, pr.loc[str(i)], '0.5', marker='o', alpha=0.2, lw=0) for i in pr.index[(pr.index != 'base')]]
        base = pr.loc['base', :]
        ax.plot(pr.columns, base, '0.5', marker='o', lw=0, label='prior')
    if max(iters) > 0:
        itr = max(iters)
        pt = obsdict[itr]._df.loc[:, cobs.obsnme]
        pt = pt.loc[:,pt.sum()!=0]
        pt = pt.loc[:,~pt.columns.str.contains('total|percent')]
        pt.columns = ['drns', 'rch', 'riv_in', 'riv_out']
        # convert ft3/d to acre-ft/yr
        pt = pt/43560*365.25


        [ax.plot(pt.columns, pt.loc[str(i)], c='blue', marker='o', lw=0, alpha=0.3) for i in pt.index[(pt.index != 'base')]]
        base = pt.loc['base', :]
        ax.plot(pt.columns, base, 'b', marker='o', lw=0, label='posterior')

        # ax.set_ylabel('afy')
        if pr.sum().sum() != 0:
            ax.legend(loc='upper right', fontsize=10)
            #ax.set_title(f'Budget Term: {cat}', fontsize=18, loc='left')
            ax.semilogy()
            ax.grid()
            ax.set_ylabel('Water Budget SS (acre-ft/yr)', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)

            pdf.savefig(fig)
            plt.close(fig)
        # n+=1

    pdf.close()

    ########################################
    #   Plot Pie Chart ins/outs of post-base
    ########################################
    cf2acreft   = 0.000810714          # ft³ d⁻¹ → ac-ft yr⁻¹
    types       = ['in', 'out']        # build an “inflows” and an “outflows” pie
    target_year = 1970                 # change if you need 1980, etc.

    # key  →  (long label , colour)
    pie_meta = {
        'drns'    : ('Base-flow / Drain package',       'darkgreen'),
        'rch'     : ('Recharge (RCH)',                  'cornflowerblue'),
        'riv_in'  : ('River seepage – inflow (RIV)',    'green'),
        'riv_out' : ('River seepage – outflow (RIV)',   'orange'),
    }

    # ─── helper: pull the “base” member for a given year and convert units ───
    def _year_base_acft(cat_obsnmes, year=target_year):
        itr  = max(iters)                                        # posterior iter
        ens  = obsdict[itr]._df.loc[:, list(cat_obsnmes)]        # (real × dates)

        base = ens.loc['base'].reset_index()                     # → ['obsnme', 0]
        base.columns = ['obsnme', 'value']

        base['datetime'] = pd.to_datetime(
            base['obsnme'].str.split(':').str[-1], format='%Y-%m-%d'
        )
        sel = base[base['datetime'].dt.year == year]
        if sel.empty:
            return 0.0
        return sel['value'].iloc[0] * cf2acreft                  # ac-ft yr⁻¹
    # ───────────────────────────────────────────────────────────────

    # pre-compute the unique budget-term tags in the file
    all_tags = {s.split(':')[3] for s in bobs.obsnme}

    for io in types:
        pdf = PdfPages(os.path.join(fdir, f'budget_piechart_{io}_ss.pdf'))
        labels, values, colours = [], [], []

        # all tags that match the direction and aren’t storage / totals
        pattern = '_in_datetime' if io == 'in' else '_out_datetime'
        wanted  = sorted(
            t for t in all_tags
            if t.endswith(pattern)
            and not t.startswith(('sto', 'total', 'percent'))
        )

        for cat_tag in wanted:
            short_raw = cat_tag.split('_datetime')[0]            # e.g. rcha_in

            # map raw tag → pie_meta key
            short = {
                'drn_in'  : 'drns',
                'drn_out' : 'drns',
                'rcha_in' : 'rch',
                'rcha_out': 'rch',
                'rch_in'  : 'rch',
                'rch_out' : 'rch',
            }.get(short_raw, short_raw)  # default to itself (riv_in/riv_out)

            if short not in pie_meta:
                continue

            # grab obsnmes for this tag
            mask    = (bobs['obsnme'].str.contains(cat_tag)
                    & bobs['obsnme'].str.contains('bud'))
            cat_obs = bobs.loc[mask, 'obsnme']
            if cat_obs.empty:
                continue

            acft = _year_base_acft(cat_obs)
            if acft == 0:
                continue

            long_label, colour = pie_meta[short]
            labels .append(f'{long_label}\n{acft:,.0f} ac-ft yr⁻¹')
            values .append(acft)
            colours.append(colour)

        # nothing to plot?
        if not values:
            pdf.close()
            continue

        # ───────── draw pie ─────────
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, _, _ = ax.pie(
            values, colors=colours, startangle=-30,
            autopct='%1.1f%%', pctdistance=0.75,
            wedgeprops=dict(edgecolor='black')
        )

        # pretty labels outside with leader lines
        kw = dict(arrowprops=dict(arrowstyle='-'),
                bbox=dict(boxstyle='round,pad=0.3', fc='w', ec='k', lw=0.72),
                zorder=0, va='center')
        for w, lbl in zip(wedges, labels):
            ang = (w.theta2 + w.theta1) / 2
            x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            ax.annotate(lbl, xy=(x, y), xytext=(1.25 * x, 1.25 * y),
                        ha='center', **kw)

        ax.set_title(f'Steady-State Water Budget – {io.title()}flows ({target_year})',
                    loc='left', fontsize=14)
        ax.axis('equal')
        pdf.savefig(fig)
        plt.close(fig)
        pdf.close()

def plot_layer_one2one_wdepth(m_d,obsdict, modnm='elk_2lay'):
    o_d = os.path.join(m_d, 'results', 'figures', 'one2one_plots')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    #pwl = pd.read_csv(os.path.join(m_d, 'tables', 'processed_WL_timeseries.csv'))

    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6',load_only=['dis'])
    m = sim.get_model(modnm)
    nlay = m.dis.nlay.array
    if nlay == 4:
        layers = {
            0: 'WSS',
            1: 'WBV',
            2: 'DC',
            3: 'WR',
            }
    elif nlay == 6:
        layers = {
            0: 'WSS-1',
            1: 'WSS-2',
            2: 'WSS-3',
            3: 'WBV',
            4: 'DC',
            5: 'WR'
            }
    elif nlay == 7:
        layers = {
            0: 'WSS-1',
            1: 'WSS-2',
            2: 'WSS-3',
            3: 'WSS-confining',
            4: 'WBV',
            5: 'DC',
            6: 'WR'
            }

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('trans|sspmp|sshds'), :].copy()  # *** NOTE - this is a hard-coded filter
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    
    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])
    
    aobs = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'), index_col=0)
    conflicts = set(gwobs.loc[gwobs.weight > 0, 'obsnme'].tolist()) - set(aobs.loc[aobs.weight > 0, :].index.tolist())
    print(conflicts)

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)

    top = m.dis.top.array
    botm = m.dis.botm.array
    itrmx = max(obsdict)
    usites = gwobs.id.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site': usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)

    usites = usitedf['site'].values

    gwobs = gwobs.loc[gwobs.id.isin(usites), :]
    #gwo_grps = gwobs.oname.unique()
    gwo_grps = gwobs.groupby('k')
    with PdfPages(os.path.join(o_d, 'layer_one2one_plots_dpth2wl.pdf')) as pdf:
        for nm, grp in gwo_grps:
            ly = int(nm)
            unq_onms = grp.oname.unique()
            for onm in unq_onms:
                oobs = grp[grp.oname == onm].copy()
                wobs = oobs.loc[oobs.weight > 0, :]
                wobs = wobs.reset_index(drop=True)
                if len(wobs) == 0:
                    continue
                vals = obsdict[itrmx].loc[:, obsdict[itrmx].columns.isin(wobs.obsnme.values)].T
                base = vals.loc[:, 'base'].reset_index()
                base = base.rename(columns={'index': 'obsnme'})
                base_mrg = pd.merge(base, wobs, on='obsnme', how='left')
                base = base_mrg.loc[base_mrg.obsval.notnull(), :]
                base['model_top'] = top[base.i.astype(int), base.j.astype(int)]
                # calculate depth to water:
                base['depth_to_water'] = base.model_top - base.obsval
                base['resid'] = base.obsval - base.base
                base = base[['obsnme', 'obgnme', 'base', 'obsval', 'resid', 'depth_to_water']]

                fig = plt.figure(figsize=(10, 6))
                gs = gridspec.GridSpec(1, 3)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1:])
                # plot one to one plot:
                # first plot scatter with color by depth_to_water:
                # divergent colormap centered on zero:
                try:
                    divnorm = mpl.colors.TwoSlopeNorm(vmin=base['depth_to_water'].min(), vcenter=0,
                                                    vmax=base['depth_to_water'].max())
                except:
                    divnorm = mpl.colors.TwoSlopeNorm(vmin=-0.1, vcenter=0,
                                                    vmax=base['depth_to_water'].max())
                cbar = plt.cm.ScalarMappable(cmap='seismic', norm=divnorm)
                cmap = plt.cm.get_cmap('seismic')
                ax1.scatter(base.obsval,base.base,c=base['depth_to_water'],cmap=cmap,s=20,norm=divnorm,edgecolors='black',linewidth=0.1)
                # add one to one line:
                xmn = min([base.obsval.min(),base.base.min()])
                xmx = max([base.obsval.max(),base.base.max()])
                ax1.set_xlim(xmn, xmx)
                ax1.set_ylim(xmn, xmx)
                ax1.plot([xmx, xmn], [xmx, xmn], 'k-', lw=0.5)
                ax1.set_xlabel(r'$\bf{Observed}$ groundwater level, in feet' + '\nabove average sea level', fontsize=10)
                ax1.set_ylabel(r'$\bf{Simulated}$ groundwater level, in feet' + '\nabove average sea level', fontsize=10)
                ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax1.tick_params(axis='both', which='major', labelsize=9)
                ax1.tick_params(axis='both', which='both', direction='in')
                ax1.xaxis.set_ticks_position('both')
                ax1.yaxis.set_ticks_position('both')
                # set axis equal:
                ax1.set_aspect('equal', adjustable='box')
                ax1.set_title(f'Layer {int(nm)+1}\nTarget Type: {onm}\n groundwater-level observation\n1:1 plot', fontsize=12)
              

                cbar.set_array([])
                cbar.set_clim(-25, base['depth_to_water'].max())
                cbar = plt.colorbar(cbar, ax=ax1, orientation='horizontal', pad=0.2)
                # cbar label font size:
                cbar.ax.tick_params(labelsize=9)
                # add colorbar label:
                cbar.set_label('Depth to water table (feet)', fontsize=10)

                # add hist plot of residuals:
                ax2.hist(base.resid, bins=40, edgecolor='black', linewidth=0.5)
                # add vertical line at mean of residuals:
                ax2.axvline(base.resid.mean(), color='black', linestyle='-', linewidth=1.5)
                # add text label for mean of residuals:
                text_x = base.resid.median()
                text_y = ax2.get_ylim()[1] * 0.9
                ax2.text(
                        text_x,
                        text_y,
                        f'Mean residual: {base.resid.mean():.2f} feet',
                        fontsize=9,
                        ha='left',
                        va='top',
                        color='black',  # Set text color to black
                        bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.3')  # Grey background, black outline
                    )
                ax2.annotate(
                    '',  # No text
                    xy=(text_x + 5, text_y-1),  # Arrow tip (closer to text)
                    xytext=(text_x, text_y-1),  # Arrow start (further left)
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )
                ax2.set_title(f'Groundwater-level observation residuals', fontsize=12)
                ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax2.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax2.tick_params(axis='both', which='major', labelsize=9)
                ax2.tick_params(axis='both', which='both', direction='in')
                ax2.xaxis.set_ticks_position('both')
                ax2.yaxis.set_ticks_position('both')
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_xlabel('Groundwater-level residual (observed minus simulated),\nin feet above average sea level',
                            fontsize=10)
                plt.tight_layout()
                pdf.savefig()
                plt.savefig(os.path.join(o_d, f'ly{nm}_{onm}_one2one_with_depth.png'))
                plt.close(fig)

def plot_layer_one2one_bywell(m_d,obsdict, modnm='elk_2lay'):
    o_d = os.path.join(m_d, 'results', 'figures', 'one2one_plots')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    #pwl = pd.read_csv(os.path.join(m_d, 'tables', 'processed_WL_timeseries.csv'))

    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6',load_only=['dis'])
    m = sim.get_model(modnm)
    nlay = m.dis.nlay.array

    layers = {
        0: 'Soils/Clay/Silt',
        1: 'Elk Valley Aquifer',
        }

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('trans|sspmp|sshds'), :].copy()  # *** NOTE - this is a hard-coded filter
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    
    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])
    
    aobs = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'), index_col=0)
    conflicts = set(gwobs.loc[gwobs.weight > 0, 'obsnme'].tolist()) - set(aobs.loc[aobs.weight > 0, :].index.tolist())
    print(conflicts)

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)

    top = m.dis.top.array
    botm = m.dis.botm.array
    itrmx = max(obsdict)
    usites = gwobs.id.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site': usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)

    usites = usitedf['site'].values

    gwobs = gwobs.loc[gwobs.id.isin(usites), :]
    #gwo_grps = gwobs.oname.unique()
    gwo_grps = gwobs.groupby('k')
    with PdfPages(os.path.join(o_d, 'layer_one2one_plots_bywell.pdf')) as pdf:
        for nm, grp in gwo_grps:
            ly = int(nm)
            unq_onms = grp.oname.unique()
            for onm in unq_onms:
                oobs = grp[grp.oname == onm].copy()
                wobs = oobs.loc[oobs.weight > 0, :]
                wobs = wobs.reset_index(drop=True)
                if len(wobs) == 0:
                    continue
                vals = obsdict[itrmx].loc[:, obsdict[itrmx].columns.isin(wobs.obsnme.values)].T
                base = vals.loc[:, 'base'].reset_index()
                base = base.rename(columns={'index': 'obsnme'})
                base_mrg = pd.merge(base, wobs, on='obsnme', how='left')
                base = base_mrg.loc[base_mrg.obsval.notnull(), :]
                base['model_top'] = top[base.i.astype(int), base.j.astype(int)]
                # calculate depth to water:
                base['depth_to_water'] = base.model_top - base.obsval
                base['resid'] = base.obsval - base.base
                base = base[['obsnme', 'obgnme', 'base', 'obsval', 'resid', 'depth_to_water','id']]

                fig = plt.figure(figsize=(10, 6))
                gs = gridspec.GridSpec(1, 3)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1:])
                # plot one to one plot:
                # first plot scatter with color by depth_to_water:
                ids = base['id'].astype('category')  # or 'id' if that column exists
                unique_ids = sorted(gwobs['id'].unique())  # or use usites if already defined earlier
                n_colors = len(unique_ids)
                if n_colors <= 20:
                    cmap = plt.cm.get_cmap('tab20', n_colors)
                elif n_colors <= 256:
                    cmap = plt.cm.get_cmap('tab20b', n_colors)
                else:
                    cmap = plt.cm.get_cmap('hsv', n_colors)
                color_map = {uid: cmap(i % n_colors) for i, uid in enumerate(unique_ids)}
                base['color'] = base['id'].map(color_map)

                # Scatter plot with unique colors per site ID
                ax1.scatter(
                    base.obsval,
                    base.base,
                    c=base['color'],
                    s=20,
                    edgecolors=None,
                    linewidth=0.1
                )
                xmn = min([base.obsval.min(), base.base.min()])
                xmx = max([base.obsval.max(), base.base.max()])
                ax1.plot([xmn, xmx], [xmn, xmx], 'k-', lw=0.5)

                # Optional legend showing which color corresponds to which ID
                handles = [
                    plt.Line2D([0], [0], marker='o', color='w',
                            label=str(uid), markerfacecolor=color_map[uid],
                            markersize=5, markeredgecolor='black')
                    for uid in unique_ids
                ]
                #ax1.legend(handles=handles, title='Site ID', bbox_to_anchor=(.05, -0.5),
                #        loc='lower left', fontsize=7, ncol=15)
                # add one to one line:
                # xmn = min([base.obsval.min(),base.base.min()])
                # xmx = max([base.obsval.max(),base.base.max()])
                ax1.set_xlim(xmn, xmx)
                ax1.set_ylim(xmn, xmx)
                #ax1.plot([xmx, xmn], [xmx, xmn], 'k-', lw=0.5)
                ax1.set_xlabel(r'$\bf{Observed}$ groundwater level, in feet' + '\nabove average sea level', fontsize=10)
                ax1.set_ylabel(r'$\bf{Simulated}$ groundwater level, in feet' + '\nabove average sea level', fontsize=10)
                ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax1.tick_params(axis='both', which='major', labelsize=9)
                ax1.tick_params(axis='both', which='both', direction='in')
                ax1.xaxis.set_ticks_position('both')
                ax1.yaxis.set_ticks_position('both')
                # set axis equal:
                ax1.set_aspect('equal', adjustable='box')
                ax1.set_title(f'Layer {int(nm)+1}\nTarget Type: {onm}\n groundwater-level observation\n1:1 plot', fontsize=12)
              

                #cbar.set_array([])
                #cbar.set_clim(-25, base['depth_to_water'].max())
                #cbar = plt.colorbar(cbar, ax=ax1, orientation='horizontal', pad=0.2)
                # cbar label font size:
                #cbar.ax.tick_params(labelsize=9)
                # add colorbar label:
                #cbar.set_label('Depth to water table (feet)', fontsize=10)

                # add hist plot of residuals:
                ax2.hist(base.resid, bins=40, edgecolor='black', linewidth=0.5)
                # add vertical line at mean of residuals:
                ax2.axvline(base.resid.mean(), color='black', linestyle='-', linewidth=1.5)
                # add text label for mean of residuals:
                text_x = base.resid.median()
                text_y = ax2.get_ylim()[1] * 0.9
                ax2.text(
                        text_x,
                        text_y,
                        f'Mean residual: {base.resid.mean():.2f} feet',
                        fontsize=9,
                        ha='left',
                        va='top',
                        color='black',  # Set text color to black
                        bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.3')  # Grey background, black outline
                    )
                ax2.annotate(
                    '',  # No text
                    xy=(text_x + 5, text_y-1),  # Arrow tip (closer to text)
                    xytext=(text_x, text_y-1),  # Arrow start (further left)
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )
                ax2.set_title(f'Groundwater-level observation residuals', fontsize=12)
                ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax2.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                ax2.tick_params(axis='both', which='major', labelsize=9)
                ax2.tick_params(axis='both', which='both', direction='in')
                ax2.xaxis.set_ticks_position('both')
                ax2.yaxis.set_ticks_position('both')
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_xlabel('Groundwater-level residual (observed minus simulated),\nin feet above average sea level',
                            fontsize=10)
                plt.tight_layout()
                pdf.savefig()
                plt.savefig(os.path.join(o_d, f'ly{nm}_{onm}_one2one_bywell.png'))
                plt.close(fig)

def plot_water_budget(m_d, obsdict, pie_year=2023):
    fdir = os.path.join(m_d, 'results', 'figures', 'water_budget')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    iters = list(obsdict.keys())
    iters.sort()
    iters = [iters[0], iters[-1]]

    obs = obsdict[iters[0]].pst.observation_data.copy()
    bobs = obs.loc[obs.obsnme.str.contains('bud'), :]
    bobs.loc[:, 'datetime'] = pd.to_datetime(bobs.datetime)

    pdf = PdfPages(os.path.join(fdir, f'budget_ts.pdf'))
    bud_cats = bobs.usecol.unique()
    bud_cats = set([s.split(':')[3].split('_t')[0] for s in bobs.obsnme])
    n = 0
    for cat in bud_cats:
        print(cat)
        cobs = bobs.loc[bobs.obsnme.apply(lambda x: cat in x and 'bud' in x), :].copy()
        if cat == 'riv_in_datetime' or cat == 'riv_out_datetime':
            cobs = cobs.loc[cobs.obsnme.str.contains(':riv')]
        assert cobs.shape[0] > 0
        cobs.sort_values(by='datetime', inplace=True)
        dates = cobs.datetime.values
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        if 0 in iters:
            pr = obsdict[0]._df.loc[:, cobs.obsnme]
            if pr.sum().sum() != 0:
                [ax.plot(dates, pr.loc[str(i)] * 0.00002296 * 365.25, '0.5', lw=0.3, alpha=0.2) for i in
                 pr.index[(pr.index != 'base')]]
                base = pr.loc['base', :]
                ax.plot(dates, base * 0.00002296 * 365.25, '0.5', lw=1.0, label='prior')
        if max(iters) > 0:
            itr = max(iters)
            pt = obsdict[itr]._df.loc[:, cobs.obsnme]
            if pt.sum().sum() != 0:
                [ax.plot(dates, pt.loc[str(i)] * 0.00002296 * 365.25, c='blue', lw=0.15, alpha=0.3) for i in
                 pt.index[(pt.index != 'base')]]
                base = pt.loc['base', :]
                ax.plot(dates, base * 0.00002296 * 365.25, 'b', lw=1.0, label='posterior')

        # ax.set_ylabel('afy')
        if pr.sum().sum() != 0:
            ax.legend(loc='upper right')
            ax.set_title(f'Budget Term: {cat}', fontsize=18, loc='left')
            ax.semilogy()
            ax.grid()
            ax.set_ylabel('acre-ft/yr')

            pdf.savefig(fig)
            plt.close(fig)
        # n+=1

    pdf.close()

    ############################
    #   Plot ins/outs
    ############################
    types = ['in', 'out']
    for type in types:
        pdf = PdfPages(os.path.join(fdir, f'budget_{type}_ts.pdf'))
        cats = [x for x in bud_cats if f'_{type}_' in x and 'total' not in x and 'sto' not in x]
        colors_pr = ['lightcoral', 'navajowhite', 'cornsilk', 'yellowgreen', 'lightgreen', 'lightsteelblue', 'lavender',
                     'thistle', 'lavenderblush', 'mistyrose', 'lightgrey']
        colors_pt = ['red', 'orange', 'gold', 'greenyellow', 'green', 'b', 'slateblue', 'purple', 'magenta', 'crimson',
                     'darkgrey']
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5))
        ax.set_yscale('log')
        j = 0
        for cat in cats:
            print(cat)
            cobs = bobs.loc[bobs.obsnme.apply(lambda x: cat in x and 'bud' in x), :].copy()
            if cat == 'riv_in_datetime' or cat == 'riv_out_datetime':
                cobs = cobs.loc[cobs.obsnme.str.contains(':riv')]
            assert cobs.shape[0] > 0
            cobs.sort_values(by='datetime', inplace=True)
            dates = cobs.datetime.values
            if 0 in iters:
                pr = obsdict[0]._df.loc[:, cobs.obsnme]
                if pr.sum().sum() != 0:
                    [ax.plot(dates, pr.loc[str(i)] * 0.00002296 * 365.25, c=colors_pr[j], lw=0.05, alpha=0.1) for i in
                     pr.index[(pr.index != 'base')]]
                    ax2.plot(dates, pr.loc['base'] * 0.00002296 * 365.25, c=colors_pr[j],
                             linestyle='--', lw=0.1, alpha=0.2, )
                    base = pr.loc['base', :]
                    ax2.plot(dates, base * 0.00002296 * 365.25, c=colors_pt[j],
                             linestyle='--', lw=1, )
                    # label=f'pr {cat.split(f'_{type}_datetime')[0]} mean: {np.round(np.mean(base)* 0.00002296 *365.25,0)}')
                    # ax.plot(dates, base,'0.5', lw=1.0)
            if max(iters) > 0:
                itr = max(iters)
                pt = obsdict[itr]._df.loc[:, cobs.obsnme]
                if pt.sum().sum() != 0:
                    [ax.plot(dates, pt.loc[str(i)] * 0.00002296 * 365.25, c=colors_pt[j], lw=0.05, alpha=0.2) for i in
                     pt.index[(pt.index != 'base')]]
                    base = pt.loc['base', :]
                    ax.plot(dates, base * 0.00002296 * 365.25, c=colors_pt[j], lw=1.0,
                            label=cat.split(f'_{type}_datetime')[0])
                    ax2.plot(dates, base * 0.00002296 * 365.25, c=colors_pt[j], lw=1.0,
                             label=f"pt {cat.split(f'_{type}_datetime')[0]} mean: {np.round(np.mean(base) * 0.00002296 * 365.25, 0)}")
                    j += 1

        ax.legend(loc='upper right')
        ax.set_ylabel('acre-ft/yr', fontsize=14)
        ax.set_title(f'Budget Term - {type}', fontsize=18, loc='left')
        ax.semilogy()
        ax.grid()

        ax2.legend(loc='lower right')
        ax2.set_ylabel('acre-ft/yr', fontsize=14)
        ax2.set_title(f'Budget Term - {type}', fontsize=18, loc='left')
        ax2.semilogy()
        ax2.grid()

        pdf.savefig(fig)
        pdf.savefig(fig2)
        plt.close(fig2)
        pdf.close()

    ########################################
    #   Plot Pie Chart ins/outs of post-base
    ########################################
    cf2acreft = 0.00002296 * 365.25  # 0.000810714
    types = ['in', 'out']

    for type in types:
        pdf = PdfPages(os.path.join(fdir, f'budget_piechart_{type}_{pie_year}.pdf'))
        cats = [x for x in bud_cats if f'_{type}_' in x and 'total' not in x and 'sto' not in x]
        cats_shrt = [x.split(f'_{type}_datetime')[0] for x in cats]
        cats_dict = {'otriv': ['Otter Tail Stream seepage\n(River package)', 'darkgreen', 'inflow/outflow'],
                     'riv': ['Stream seepage\n(River package)', 'green', 'inflow/outflow'],
                     'drn': ['Stream Seepage\n(Drain package)', 'darkgreen', 'outflow'],
                     'wel_cob': ['City of Breckenridge\n(Well package)', 'chocolate', 'outflow'],
                     'wel_car': ['Cargill Inc.\n(Well package)', 'sandybrown', 'outflow'],
                     'wel_minn': ['Minn-Dak\n(Well package)', 'peachpuff', 'outflow'],
                     'wel_cow': ['City of Wahpeton\n(Well package)', 'moccasin', 'outflow'],
                     'wel_malt': ['Malt Corp.\n(Well package)', 'orange', 'outflow'],
                     'rcha': ['Recharge\n(Recharge package)', 'cornflowerblue', 'input'],
                     'ghb': ['Regional GW Flow\n(GHB package)', 'grey', 'input'],
                     }

        fig, ax = plt.subplots(1, 1, figsize=(11, 10))
        if type == 'in':
            axtitle = 'Inflows'
        else:
            axtitle = 'Outflows'

        j = 0
        labels = []
        budgets = []
        colors = []
        # manually sort cats:
        if type == 'out':
            cats = ['drn_ag_out_datetime',
                    'otriv_out_datetime',
                    'riv_out_datetime',
                    'wel_car_out_datetime',
                    'wel_cow_out_datetime',
                    'wel_cob_out_datetime',
                    'wel_malt_out_datetime',
                    'wel_minn_out_datetime',
                    'ghb_out_datetime', ]
        else:
            cats = ['rcha_in_datetime',
                    'otriv_in_datetime',
                    'riv_in_datetime',
                    'wel_car_in_datetime',
                    'wel_cow_in_datetime',
                    'wel_cob_in_datetime',
                    'wel_malt_in_datetime',
                    'wel_minn_in_datetime',
                    'ghb_in_datetime', ]

        for cat in cats:
            print(cat)
            cshrt = cat.split(f'_{type}_datetime')[0]
            long_nm = cats_dict[cshrt][0]
            clr = cats_dict[cshrt][1]
            io = cats_dict[cshrt][2]
            if io == 'input/outflow':
                if type == 'in':
                    io = 'inflow'
                else:
                    io = 'outflow'
                    long_nm = long_nm.replace('Stream Seepage', 'Baseflow')
            cobs = bobs.loc[bobs.obsnme.apply(lambda x: cat in x and 'bud' in x), :].copy()
            if cat == 'riv_in_datetime' or cat == 'riv_out_datetime':
                cobs = cobs.loc[cobs.obsnme.str.contains(':riv')]
            assert cobs.shape[0] > 0
            cobs.sort_values(by='datetime', inplace=True)
            dates = cobs.datetime.values

            if max(iters) > 0:
                itr = max(iters)
                pt = obsdict[itr]._df.loc[:, cobs.obsnme]
                if pt.sum().sum() != 0:
                    base = pt.loc['base', :]
                    base = base.reset_index()
                    base = base.rename(columns={'index': 'oname'})
                    base['datetime'] = pd.to_datetime(base.oname.str.split(':').str[-1], format='%Y-%m-%d')
                    base['year'] = base.datetime.dt.year
                    base = base.loc[base.year == pie_year, :]
                    acft = base.base * cf2acreft
                    labels.append(long_nm)
                    budgets.append(acft.values[0])
                    colors.append(clr)
                    # ax.plot(dates, base, c=colors_pt[j], lw=1.0, label=cat.split(f'_{type}_datetime')[0])
                    j += 1

        explode = np.zeros(len(budgets))  # if explode set one of the zeros to 0.1

        for i in range(len(budgets)):
            bdg = round(budgets[i])
            labels[i] = labels[i] + f'\n{bdg:,.0f} (acre-ft)'

        wedges, texts, autopct = ax.pie(budgets, explode=explode, colors=colors, autopct='%1.1f%%', startangle=-40,
                                        wedgeprops=dict(edgecolor='black'), labeldistance=1.2)

        bbox_props = dict(boxstyle='square,pad=0.5', fc='w', ec='k', lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'), bbox=bbox_props, zorder=0, va='center')

        spacing = 1.35  # Adjust this value as needed
        label_texts = []
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            annotation = ax.annotate(labels[i], xy=(x, y), xytext=(spacing * x, spacing * y),
                                     horizontalalignment='center', verticalalignment='center', fontsize=8, **kw)
            label_texts.append(annotation)

        for text in autopct:
            text.set_color('white')

        narrow_ptch = []
        for patch, txt in zip(wedges, autopct):
            # the angle at which the text is located
            ang = (patch.theta2 + patch.theta1) / 2.
            # new coordinates of the text, 0.7 is the distance from the center
            x = patch.r * 0.7 * np.cos(ang * np.pi / 180)
            y = patch.r * 0.7 * np.sin(ang * np.pi / 180)
            txt.set_fontsize(9)
            # if patch is narrow enough, move text to new coordinates
            if (patch.theta2 - patch.theta1) < 10.:
                narrow_ptch.append(patch)
        if len(narrow_ptch) > 2:
            # get new postions betwen range of 10 and 80 and splt by len of narrow patches:
            new_positions = np.linspace(0.1, 0.8, len(narrow_ptch))
            cnt = 0
            for patch, txt in zip(wedges, autopct):
                # the angle at which the text is located
                ang = (patch.theta2 + patch.theta1) / 2.
                # new coordinates of the text, 0.7 is the distance from the center
                x = patch.r * 0.7 * np.cos(ang * np.pi / 180)
                y = patch.r * 0.7 * np.sin(ang * np.pi / 180)
                # if patch is narrow enough, move text to new coordinates
                if (patch.theta2 - patch.theta1) < 10.:
                    npos = new_positions[cnt]
                    x = patch.r * npos * np.cos(ang * np.pi / 180)
                    y = patch.r * npos * np.sin(ang * np.pi / 180)
                    txt.set_position((x, y))
                    txt.set_fontsize(10)
                    cnt += 1
            cnt = 0
            ang_add = np.linspace(-75, 150, len(wedges))
            for patch, txt in zip(wedges, label_texts):
                # the angle at which the text is located
                ang = (patch.theta2 + patch.theta1) / 2.
                # new coordinates of the text, 0.7 is the distance from the center
                x = patch.r * np.cos(ang * np.pi / 180)
                y = patch.r * np.sin(ang * np.pi / 180)
                # if patch is narrow enough, move text to new coordinates
                if (patch.theta2 - patch.theta1) < 10.:
                    angp = ang_add[cnt]
                    nang = (patch.theta2 + patch.theta1) / 2. + angp
                    y += np.sin(np.deg2rad(nang))
                    x += np.cos(np.deg2rad(nang))
                    txt.set_position((x, y))
                    cnt += 1

        ax.set_title(f'Budget Terms \n  - {axtitle} in {pie_year}', fontsize=18, loc='left')
        plt.axis('equal')
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)
        pdf.close()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_selected_budget_timeseries_posterior_only(
    obsdict,
    stress_period_table_path,
    out_dir,
    date_col="end_datetime",
    verbose=True,
    fill_missing_with_zero=True,
):
    """
    Posterior-only budget time-series plotting for selected terms.

    Fixes included
    -------------
    1) Robust across annual->monthly transitions:
       - Collapse duplicate obs by *datetime* (sum all obs with same datetime).
       - Align collapsed series to stress-period dates by datetime match (no reshape).

    2) Auto-pick the correct SP date column (start_datetime vs end_datetime) by
       comparing overlap with bobs['datetime'].
       - This prevents "SP1 = 0" caused by mismatched start/end conventions.

    Notes
    -----
    - Assumes MODFLOW budget outputs in CFD (cubic feet/day) for flows/rates.
    - Ensemble members plotted in blue (semi-transparent); posterior base plotted in black.
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Grab posterior dataframe
    # -------------------------
    iters = sorted(obsdict.keys())
    last_itr = iters[-1]
    pst = obsdict[last_itr].pst
    obs = pst.observation_data.copy()

    # Keep only "bud" observations
    bobs = obs.loc[obs.obsnme.astype(str).str.contains("bud", na=False), :].copy()

    # Ensure datetime column exists/parsed
    if "datetime" not in bobs.columns:
        raise ValueError("Expected 'datetime' column in pst.observation_data for budget plotting.")
    bobs["datetime"] = pd.to_datetime(bobs["datetime"], errors="coerce")

    pt = obsdict[last_itr]._df  # rows: members + "base", cols: obsnme

    # -------------------------
    # Stress period table
    # -------------------------
    sp = pd.read_csv(stress_period_table_path, sep=None, engine="python")
    
    # Optional: drop steady-state stress periods (often SP1 init)
    if "steady_state" in sp.columns:
        ss = sp["steady_state"].astype(str).str.upper().isin(["TRUE", "T", "1", "YES"])
        if ss.any() and verbose:
            print(f"[info] dropping {ss.sum()} steady_state stress periods from plots")
        sp = sp.loc[~ss, :].reset_index(drop=True)

    for dc in ["start_datetime", "end_datetime"]:
        if dc in sp.columns:
            sp[dc] = pd.to_datetime(sp[dc], errors="coerce")

    if "perlen" not in sp.columns:
        raise ValueError("stress-period table must include a 'perlen' column (days).")

    # --- AUTO-PICK date_col based on overlap with budget obs datetimes ---
    # normalize to date-only to ignore time-of-day mismatches
    obs_dates = pd.Index(pd.to_datetime(bobs["datetime"]).dropna().unique()).normalize()

    def _match_count(colname):
        if colname not in sp.columns:
            return -1
        spd = pd.Index(pd.to_datetime(sp[colname]).dropna().unique()).normalize()
        return len(obs_dates.intersection(spd))

    c_start = _match_count("start_datetime")
    c_end = _match_count("end_datetime")

    if verbose:
        print(f"[info] datetime matches (normalized): start_datetime={c_start}, end_datetime={c_end}")

    # If user passed a column name that doesn't exist, fall back to best
    if date_col not in sp.columns:
        date_col = "end_datetime" if c_end >= c_start else "start_datetime"

    # Override if the other column clearly matches better
    if c_end > c_start:
        date_col = "end_datetime"
    elif c_start > c_end:
        date_col = "start_datetime"

    if verbose:
        print(f"[info] using date_col='{date_col}' for alignment/plots")

    # Sort SP table in a stable way
    if "stress_period" in sp.columns:
        sp = sp.sort_values("stress_period").reset_index(drop=True)
    else:
        sp = sp.sort_values(date_col).reset_index(drop=True)

    sp_dates = pd.to_datetime(sp[date_col]).copy()
    perlen_days_sp = sp["perlen"].astype(float).values
    years_sp = pd.to_datetime(sp[date_col]).dt.year.values

    # -------------------------
    # Conversions
    # -------------------------
    SECONDS_PER_DAY = 86400.0
    CUFT_PER_ACFT = 43560.0
    GAL_PER_CUFT = 7.48052
    MIN_PER_DAY = 1440.0

    def cfd_to_cfs(x):
        return x / SECONDS_PER_DAY

    def cfd_to_gpm(x):
        # CFD -> gal/day -> gpm
        return (x * GAL_PER_CUFT) / MIN_PER_DAY

    def annual_agg_from_sp(cfd_vals, years, perlen):
        """
        Given per-stress-period CFD values aligned to stress periods, compute:
          - annual total volume (ac-ft)
          - annual average rate (cfs)
          - annual average gpm
        """
        df = pd.DataFrame({"year": years, "cfd": cfd_vals, "perlen": perlen})
        df["vol_cuft"] = df["cfd"] * df["perlen"]
        df["af"] = df["vol_cuft"] / CUFT_PER_ACFT

        out = []
        for y, g in df.groupby("year"):
            days_in_year = 366.0 if pd.Timestamp(int(y), 12, 31).is_leap_year else 365.0
            vol_cuft = g["vol_cuft"].sum()
            afy = g["af"].sum()
            avg_cfs = vol_cuft / (days_in_year * SECONDS_PER_DAY)
            avg_gpm = cfd_to_gpm(vol_cuft / days_in_year)
            out.append((y, afy, avg_cfs, avg_gpm))

        out = pd.DataFrame(out, columns=["year", "afy", "avg_cfs", "avg_gpm"]).sort_values("year")
        out["year_dt"] = pd.to_datetime(out["year"].astype(int).astype(str) + "-07-01")
        return out

    # -------------------------
    # Collapse-by-datetime helper (core fix)
    # -------------------------
    def collapse_by_datetime(dates, mem_2d, base_1d, how="sum"):
        dt = pd.to_datetime(pd.Series(dates))
        udates = np.sort(dt.dropna().unique())

        base_u = np.zeros(len(udates), dtype=float)
        mem_u = np.zeros((mem_2d.shape[0], len(udates)), dtype=float)

        for ii, d in enumerate(udates):
            sel = (dt == d).to_numpy()
            if how == "sum":
                base_u[ii] = float(np.nansum(base_1d[sel]))
                mem_u[:, ii] = np.nansum(mem_2d[:, sel], axis=1)
            elif how == "mean":
                base_u[ii] = float(np.nanmean(base_1d[sel]))
                mem_u[:, ii] = np.nanmean(mem_2d[:, sel], axis=1)
            else:
                raise ValueError("how must be 'sum' or 'mean'")
        return pd.to_datetime(udates).values.astype("datetime64[ns]"), mem_u, base_u

    def align_to_sp_dates(dates_u, mem_u, base_u, title):
        """
        Align collapsed series (indexed by datetime) to the SP table datetimes.
        Uses normalized date-only matching to avoid time-of-day mismatches.
        """
        idx_sp = pd.DatetimeIndex(pd.to_datetime(sp_dates)).normalize()
        idx_u  = pd.DatetimeIndex(pd.to_datetime(dates_u)).normalize()

        base_al = np.full(len(idx_sp), np.nan, dtype=float)
        take = idx_sp.get_indexer(idx_u)
        ok = take >= 0
        base_al[take[ok]] = base_u[ok]

        mem_al = np.full((mem_u.shape[0], len(idx_sp)), np.nan, dtype=float)
        mem_al[:, take[ok]] = mem_u[:, ok]

        if verbose:
            nmiss = int(np.isnan(base_al).sum())
            if nmiss > 0:
                print(f"[warn] {title}: {nmiss} stress periods had no matching obs by datetime (left as NaN).")

        return mem_al, base_al


    # -------------------------
    # Series definitions
    # -------------------------
    series_specs = [
        ("DRN AG out", "zbly_drn_ag-drn-out_datetime", "cfs_afy"),
        ("DRN WL out", "zbly_drn_wl-drn-out_datetime", "cfs_afy"),
        ("DRN N out",  "zbly_drn_n-drn-out_datetime",  "cfs_afy"),
        ("DRN MN out", "zbly_drn_mn-drn-out_datetime", "cfs_afy"),
        ("DRN MS out", "zbly_drn_ms-drn-out_datetime", "cfs_afy"),
        ("DRN S out",  "zbly_drn_s-drn-out_datetime",  "cfs_afy"),

        ("RIV Hazen net (in-out)", ("riv_hazen_in_datetime", "riv_hazen_out_datetime"), "cfs_afy"),
        ("RIV Goose net (in-out)", ("riv_goose_in_datetime", "riv_goose_out_datetime"), "cfs_afy"),
        ("RIV Turtle net (in-out)", ("riv_turtle_in_datetime", "riv_turtle_out_datetime"), "cfs_afy"),

        ("WEL out", "wel_wel_out_datetime", "cfs_gpm"),
        ("RCH in",  "rcha_in_datetime",     "afy_gpm"),
    ]

    # -------------------------
    # Helpers to find obsnmes and extract posterior series (+ their datetimes)
    # -------------------------
    def find_obsnmes_for_cat(cat_substring):
        m = bobs["obsnme"].astype(str).apply(lambda x: (cat_substring in x) and ("bud" in x))
        return bobs.loc[m, "obsnme"].astype(str).tolist()

    def extract_posterior_series_with_dates(cols):
        if len(cols) == 0:
            return None, None, None

        # dates for these columns (preserve `cols` order)
        tmp = bobs.loc[bobs["obsnme"].astype(str).isin(cols), ["obsnme", "datetime"]].copy()
        tmp["obsnme"] = tmp["obsnme"].astype(str)
        tmp = tmp.set_index("obsnme").reindex(cols)
        dates = pd.to_datetime(tmp["datetime"], errors="coerce").values

        sub = pt.loc[:, cols]
        if "base" not in sub.index:
            raise ValueError("Posterior dataframe does not have a 'base' row.")
        base = sub.loc["base", :].values.astype(float)
        mem_idx = [i for i in sub.index if str(i) != "base"]
        members = sub.loc[mem_idx, :].values.astype(float)

        return dates, members, base

    # -------------------------
    # Plotting outputs
    # -------------------------
    perSP_pdf = os.path.join(out_dir, "budget_selected_perSP.pdf")
    ann_avg_pdf = os.path.join(out_dir, "budget_selected_annual_avg.pdf")
    ann_tot_pdf = os.path.join(out_dir, "budget_selected_annual_totals.pdf")

    with PdfPages(perSP_pdf) as pdf_sp, PdfPages(ann_avg_pdf) as pdf_ann_avg, PdfPages(ann_tot_pdf) as pdf_ann_tot:
        for title, cat, unit_mode in series_specs:

            # Extract per-observation CFD series
            if isinstance(cat, tuple):
                cols_in = find_obsnmes_for_cat(cat[0])
                cols_out = find_obsnmes_for_cat(cat[1])

                if len(cols_in) == 0 or len(cols_out) == 0:
                    if verbose:
                        print(f"[skip] {title}: missing in/out cats ({cat[0]} or {cat[1]})")
                    continue

                d_in, mem_in, base_in = extract_posterior_series_with_dates(cols_in)
                d_out, mem_out, base_out = extract_posterior_series_with_dates(cols_out)

                # collapse each by datetime (sum across layers/zones/etc)
                d_in_u, mem_in_u, base_in_u = collapse_by_datetime(d_in, mem_in, base_in, how="sum")
                d_out_u, mem_out_u, base_out_u = collapse_by_datetime(d_out, mem_out, base_out, how="sum")

                # align in/out to common dates, then net = in - out
                idx_in = pd.Index(pd.to_datetime(d_in_u))
                idx_out = pd.Index(pd.to_datetime(d_out_u))
                common = np.intersect1d(idx_in.normalize().values, idx_out.normalize().values)

                if common.size == 0:
                    if verbose:
                        print(f"[skip] {title}: no common datetimes between in/out series after collapsing.")
                    continue

                # reindex to common (normalized)
                take_in = idx_in.normalize().get_indexer(pd.to_datetime(common))
                take_out = idx_out.normalize().get_indexer(pd.to_datetime(common))

                mem_cfd_u = mem_in_u[:, take_in] - mem_out_u[:, take_out]
                base_cfd_u = base_in_u[take_in] - base_out_u[take_out]
                dates_u = pd.to_datetime(common).values.astype("datetime64[ns]")

            else:
                cols = find_obsnmes_for_cat(cat)
                if len(cols) == 0:
                    if verbose:
                        print(f"[skip] {title}: missing cat {cat}")
                    continue

                d_raw, mem_raw, base_raw = extract_posterior_series_with_dates(cols)
                dates_u, mem_cfd_u, base_cfd_u = collapse_by_datetime(d_raw, mem_raw, base_raw, how="sum")

            # Align to stress-period table by datetime
            mem_cfd, base_cfd = align_to_sp_dates(dates_u, mem_cfd_u, base_cfd_u, title)

            if fill_missing_with_zero:
                mem_cfd = np.nan_to_num(mem_cfd, nan=0.0)
                base_cfd = np.nan_to_num(base_cfd, nan=0.0)

            dates = sp_dates.values
            years = years_sp
            perlen = perlen_days_sp

            # -------------------------
            # Per-stress-period plots
            # -------------------------
            if unit_mode == "cfs_afy":
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                ax.plot(dates, cfd_to_cfs(mem_cfd).T, color="blue", alpha=0.15, linewidth=0.4)
                ax.plot(dates, cfd_to_cfs(base_cfd), color="black", linewidth=1.3, label="posterior base")
                ax.set_title(f"{title} (per stress period) — CFS")
                ax.set_ylabel("CFS")
                ax.grid(True, alpha=0.3)
                pdf_sp.savefig(fig)
                plt.close(fig)

            elif unit_mode == "cfs_gpm":
                # CFS
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                ax.plot(dates, cfd_to_cfs(mem_cfd).T, color="blue", alpha=0.15, linewidth=0.4)
                ax.plot(dates, cfd_to_cfs(base_cfd), color="black", linewidth=1.3, label="posterior base")
                ax.set_title(f"{title} (per stress period) — CFS")
                ax.set_ylabel("CFS")
                ax.grid(True, alpha=0.3)
                pdf_sp.savefig(fig)
                plt.close(fig)

                # GPM
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                ax.plot(dates, cfd_to_gpm(mem_cfd).T, color="blue", alpha=0.15, linewidth=0.4)
                ax.plot(dates, cfd_to_gpm(base_cfd), color="black", linewidth=1.3, label="posterior base")
                ax.set_title(f"{title} (per stress period) — GPM")
                ax.set_ylabel("GPM")
                ax.grid(True, alpha=0.3)
                pdf_sp.savefig(fig)
                plt.close(fig)

            elif unit_mode == "afy_gpm":
                # per-SP AFY-equivalent rate
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                mem_afy_rate = mem_cfd * 365.25 / CUFT_PER_ACFT
                base_afy_rate = base_cfd * 365.25 / CUFT_PER_ACFT
                ax.plot(dates, mem_afy_rate.T, color="blue", alpha=0.15, linewidth=0.4)
                ax.plot(dates, base_afy_rate, color="black", linewidth=1.3, label="posterior base")
                ax.set_title(f"{title} (per stress period) — AFY-equivalent rate")
                ax.set_ylabel("AFY (equiv. rate)")
                ax.grid(True, alpha=0.3)
                pdf_sp.savefig(fig)
                plt.close(fig)

                # per-SP GPM
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                ax.plot(dates, cfd_to_gpm(mem_cfd).T, color="blue", alpha=0.15, linewidth=0.4)
                ax.plot(dates, cfd_to_gpm(base_cfd), color="black", linewidth=1.3, label="posterior base")
                ax.set_title(f"{title} (per stress period) — GPM")
                ax.set_ylabel("GPM")
                ax.grid(True, alpha=0.3)
                pdf_sp.savefig(fig)
                plt.close(fig)

            # -------------------------
            # Annual aggregation plots
            # -------------------------
            ann_base = annual_agg_from_sp(base_cfd, years, perlen)
            ann_members = [annual_agg_from_sp(mem_cfd[k, :], years, perlen) for k in range(mem_cfd.shape[0])]

            if unit_mode == "cfs_afy":
                # annual avg CFS
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for am in ann_members:
                    ax.plot(am["year_dt"].values, am["avg_cfs"].values, color="blue", alpha=0.15, linewidth=0.6)
                ax.plot(ann_base["year_dt"].values, ann_base["avg_cfs"].values, color="black", linewidth=1.4)
                ax.set_title(f"{title} (annual) — average CFS")
                ax.set_ylabel("CFS")
                ax.grid(True, alpha=0.3)
                pdf_ann_avg.savefig(fig)
                plt.close(fig)

                # annual AFY totals
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for am in ann_members:
                    ax.plot(am["year_dt"].values, am["afy"].values, color="blue", alpha=0.15, linewidth=0.6)
                ax.plot(ann_base["year_dt"].values, ann_base["afy"].values, color="black", linewidth=1.4)
                ax.set_title(f"{title} (annual) — total AFY")
                ax.set_ylabel("acre-ft/year")
                ax.grid(True, alpha=0.3)
                pdf_ann_tot.savefig(fig)
                plt.close(fig)

            elif unit_mode == "cfs_gpm":
                # annual avg CFS
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for am in ann_members:
                    ax.plot(am["year_dt"].values, am["avg_cfs"].values, color="blue", alpha=0.15, linewidth=0.6)
                ax.plot(ann_base["year_dt"].values, ann_base["avg_cfs"].values, color="black", linewidth=1.4)
                ax.set_title(f"{title} (annual) — average CFS")
                ax.set_ylabel("CFS")
                ax.grid(True, alpha=0.3)
                pdf_ann_avg.savefig(fig)
                plt.close(fig)

                # annual avg GPM
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for am in ann_members:
                    ax.plot(am["year_dt"].values, am["avg_gpm"].values, color="blue", alpha=0.15, linewidth=0.6)
                ax.plot(ann_base["year_dt"].values, ann_base["avg_gpm"].values, color="black", linewidth=1.4)
                ax.set_title(f"{title} (annual) — average GPM")
                ax.set_ylabel("GPM")
                ax.grid(True, alpha=0.3)
                pdf_ann_avg.savefig(fig)
                plt.close(fig)

            elif unit_mode == "afy_gpm":
                # annual AFY totals
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for am in ann_members:
                    ax.plot(am["year_dt"].values, am["afy"].values, color="blue", alpha=0.15, linewidth=0.6)
                ax.plot(ann_base["year_dt"].values, ann_base["afy"].values, color="black", linewidth=1.4)
                ax.set_title(f"{title} (annual) — total AFY")
                ax.set_ylabel("acre-ft/year")
                ax.grid(True, alpha=0.3)
                pdf_ann_tot.savefig(fig)
                plt.close(fig)

                # annual avg GPM
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for am in ann_members:
                    ax.plot(am["year_dt"].values, am["avg_gpm"].values, color="blue", alpha=0.15, linewidth=0.6)
                ax.plot(ann_base["year_dt"].values, ann_base["avg_gpm"].values, color="black", linewidth=1.4)
                ax.set_title(f"{title} (annual) — average GPM")
                ax.set_ylabel("GPM")
                ax.grid(True, alpha=0.3)
                pdf_ann_avg.savefig(fig)
                plt.close(fig)

    if verbose:
        print(f"Wrote:\n  {perSP_pdf}\n  {ann_avg_pdf}\n  {ann_tot_pdf}")




def plot_zone_bud_ly_by_perctot(m_d, obsdict, top_n=6, pdf_name='percent_in_out_by_layer.pdf'):
    """
    Layer-by-layer percent-of-total inflow/outflow plots (storage excluded).
    One page per layer: left panel = inflows %, right panel = outflows %.
    Legends are placed below each subplot, centered, in two columns.
    """
    # --- Output path
    fdir = os.path.join(m_d, 'results', 'figures', 'water_budget', 'zone_budget')
    os.makedirs(fdir, exist_ok=True)
    pdf_path = os.path.join(fdir, pdf_name)

    # --- Iterations
    iters = sorted(list(obsdict.keys()))
    if not iters:
        print("obsdict is empty; nothing to plot.")
        return
    it_first, it_last = iters[0], iters[-1]

    # --- Observations (metadata)
    obs = obsdict[it_first].pst.observation_data.copy()
    bobs = obs.loc[obs.obsnme.str.contains('zbud', case=False), :].copy()
    if bobs.empty:
        print("No 'zbud' observations found; nothing to plot.")
        return
    bobs['datetime'] = pd.to_datetime(bobs['datetime'])

    # --- Split inflow/outflow
    is_in  = bobs.obsnme.apply(lambda x: (('-in' in x) or ('from' in x)) and ('bud' in x))
    is_out = bobs.obsnme.apply(lambda x: (('-out' in x) or ('to' in x)) and ('bud' in x))
    inobs  = bobs.loc[is_in, :].copy()
    outobs = bobs.loc[is_out, :].copy()

    # --- Extract "type" (4th field), strip suffix/prefix
    def type_from_obsnme(s: pd.Series) -> pd.Series:
        return (
            s.str.split(':').str[3]
             .str.replace('_datetime', '', regex=True)
             .str.replace('zbly_', '', regex=True)
        )

    inobs['type']  = type_from_obsnme(inobs.obsnme)
    outobs['type'] = type_from_obsnme(outobs.obsnme)

    # --- Exclude storage
    inobs  = inobs.loc[~inobs['type'].str.contains(r'(sto-ss|sto)', case=False, regex=True)].copy()
    outobs = outobs.loc[~outobs['type'].str.contains(r'(sto-ss|sto)', case=False, regex=True)].copy()

    # --- Extract layer number
    def extract_layer(s: pd.Series) -> pd.Series:
        z = s.str.extract(r'zn-(\d+)', expand=False)
        z = z.fillna(s.str.extract(r'-(\d+)$', expand=False))
        return pd.to_numeric(z, errors='coerce').astype('Int64')

    inobs['layer']  = extract_layer(inobs.obsnme)
    outobs['layer'] = extract_layer(outobs.obsnme)
    inobs  = inobs.dropna(subset=['layer']).copy()
    outobs = outobs.dropna(subset=['layer']).copy()

    # --- Meta (avoids merge ambiguity)
    in_meta = inobs.drop_duplicates(subset='obsnme')[['obsnme', 'datetime', 'type', 'layer']].set_index('obsnme')
    out_meta = outobs.drop_duplicates(subset='obsnme')[['obsnme', 'datetime', 'type', 'layer']].set_index('obsnme')

    # --- Latest iteration values
    df_in_vals  = obsdict[it_last]._df.loc[:, in_meta.index]  if not in_meta.empty  else pd.DataFrame()
    df_out_vals = obsdict[it_last]._df.loc[:, out_meta.index] if not out_meta.empty else pd.DataFrame()

    # --- Helper: join with meta
    def _base_join(df_vals: pd.DataFrame, meta_idxed: pd.DataFrame) -> pd.DataFrame:
        if df_vals.empty or meta_idxed.empty:
            return pd.DataFrame(columns=['obsnme','datetime','type','layer','value'])
        idx_to_use = 'base' if 'base' in df_vals.index else df_vals.index[0]
        left = (
            df_vals.loc[idx_to_use]
            .rename('value')
            .rename_axis('obsnme')
            .reset_index()
        )
        return (
            left.merge(meta_idxed, left_on='obsnme', right_index=True, how='left')
                .sort_values(['layer','datetime'])
        )

    base_in_all  = _base_join(df_in_vals,  in_meta)
    base_out_all = _base_join(df_out_vals, out_meta)

    # --- Colors
    IN_COLORS = ['#8c510a','#35978f','#01665e','#3288bd','#7b3294','#762a83','#542788','#2d004b','#382001','#0e0d0d']
    OUT_COLORS = ['#d9a900','#f46d43','#d73027','#a50026','#e7298a','#ce1256','#67001f','#3f007d','#382001','#0e0d0d']

    # --- Utilities
    def consolidate(df):
        if df.empty: return pd.DataFrame()
        g = df.groupby(['datetime','type'], as_index=False)['value'].sum()
        return g.pivot(index='datetime', columns='type', values='value').sort_index().fillna(0.0)

    def to_percent(wide):
        if wide.empty: return wide
        row_sum = wide.sum(axis=1).replace(0.0,np.nan)
        return (wide.divide(row_sum, axis=0)*100.0).fillna(0.0)

    def topn_with_other(wide, pct, n):
        if wide.empty or pct.empty: return pd.DataFrame()
        mags = wide.sum(axis=0).sort_values(ascending=False)
        keep = mags.head(n).index.tolist()
        show = pct[keep].copy()
        if pct.shape[1] > len(keep):
            other = 100.0 - show.sum(axis=1)
            show['Other'] = other.clip(lower=0.0)
        return show

    # --- Plot
    layers = sorted(pd.unique(pd.concat([base_in_all['layer'], base_out_all['layer']], ignore_index=True).dropna()))
    if len(layers) == 0:
        print("No parsed layers found to plot.")
        return

    with PdfPages(pdf_path) as pdf:
        matplotlib.rcParams.update({'font.size': 12})

        for lay in layers:
            df_in  = base_in_all.loc[base_in_all['layer']==lay,['datetime','type','value']]
            df_out = base_out_all.loc[base_out_all['layer']==lay,['datetime','type','value']]

            wide_in, wide_out = consolidate(df_in), consolidate(df_out)
            pct_in, pct_out   = to_percent(wide_in), to_percent(wide_out)
            pct_in_show, pct_out_show = topn_with_other(wide_in,pct_in,top_n), topn_with_other(wide_out,pct_out,top_n)

            if pct_in_show.empty and pct_out_show.empty: continue

            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,7),sharex=False)

            if not pct_in_show.empty:
                ax1.stackplot(
                    pct_in_show.index,
                    *[pct_in_show[c].values for c in pct_in_show.columns],
                    labels=pct_in_show.columns,
                    colors=IN_COLORS[:pct_in_show.shape[1]],
                )
                ax1.set_title(f'Layer {int(lay)} — Inflows %')
                ax1.set_ylabel('Percent')
                ax1.set_ylim(0,100)
                ax1.grid(True, linestyle=':', linewidth=0.7)
                ax1.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5,-0.15),
                    ncol=2,
                    fontsize=8,
                    frameon=False
                )

            if not pct_out_show.empty:
                ax2.stackplot(
                    pct_out_show.index,
                    *[pct_out_show[c].values for c in pct_out_show.columns],
                    labels=pct_out_show.columns,
                    colors=OUT_COLORS[:pct_out_show.shape[1]],
                )
                ax2.set_title(f'Layer {int(lay)} — Outflows %')
                ax2.set_ylim(0,100)
                ax2.grid(True, linestyle=':', linewidth=0.7)
                ax2.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5,-0.15),
                    ncol=2,
                    fontsize=8,
                    frameon=False
                )

            fig.autofmt_xdate()
            fig.subplots_adjust(bottom=0.25)  # leave space for legends
            fig.suptitle(f'ZoneBudget Percent Contributions — Layer {int(lay)} (Latest Iteration, Base Run)')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved layer-by-layer percent-of-total plots:\n  {pdf_path}")

def plot_parm_histos_with_quantile_obs(m_d, pst_name, noptmax):
    '''Function to plot ensemble distributions of summary statistics for effective MODFLOW inputs generated by multiplier parameters
    Args:
        m_d (str): relative path to master directory
        o_d (str): relative path to output directory
        pst_name (str): pest control file name
        noptmax (int): pest parameter estimation iteration to plot
        max_fail (int): maximum threshold of convergence failures allowed before dropping realization from plots
    ''' 
    print('\n')
    print('plotting mf input for {0} iteration {1}'.format(m_d, noptmax))
    print('\n')
    
    o_d = os.path.join(m_d,'results','figures','mf_input_tracking')
    if os.path.exists(o_d) == False:
        os.makedirs(o_d)
        
    pst = pyemu.Pst(os.path.join(m_d, pst_name))
    oe = pyemu.ObservationEnsemble.from_binary(filename=os.path.join(m_d, '{0}.{1}.obs.jcb'.format(pst_name.split('.')[0], noptmax)), pst=pst)
    oe = oe.loc[:,[x for x in oe.columns if x.startswith('oname:mfin')]]
    df = pd.DataFrame.from_records((oe.to_records()))
    df.index = df.pop('index')
    grps = np.unique([x.split(':')[-1] for x in df.columns])
    grps = [x for x in grps if x[-2] != '_']
    grps = sorted(grps, key=lambda x: (0 if x.startswith('k_') else 1 if x.startswith('k33_') else 2, x))
    grps = np.array(grps)
    

    if noptmax > 0:
        oe_pr = pyemu.ObservationEnsemble.from_binary(filename=os.path.join(m_d, '{0}.0.obs.jcb'.format(pst_name.split('.')[0])), pst=pst)
        oe_pr = oe_pr.loc[:,[x for x in oe_pr.columns if x.startswith('oname:mfin')]]
        df_pr = pd.DataFrame.from_records((oe_pr.to_records()))
        df_pr.index = df_pr.pop('index')
    
    parm_fullnm = {'k_':'Horizonal hydraulic conductivity (ft/day)',
                   'k33_':'Anisotropy Ratio Kv/Kh (-)',
                   'ss_':'Specific storage (1/ft)',
                   'sy_':'Specific yield (-)'}
      
    with PdfPages(os.path.join(o_d, '{0}_{1}_mf_input_stats_histos_by_zone.pdf'.format(m_d, noptmax))) as pp:
        for grp in grps:
            df1 = df.loc[:,[x for x in df.columns if all((grp in x, x[-2] != '_'))]]
            df1.rename(columns={x:x.split(':')[3].replace('_input', '') for x in df1.columns}, inplace=True)
            nlb = df1.pop('near_lbnd').mean()
            nub = df1.pop('near_ubnd').mean()
            lb = df1.pop('lower_bound').iloc[0]
            ub = df1.pop('upper_bound').iloc[0]
            labels = ['min', 'qnt25', 'qnt50', 'qnt75' , 'max']
            data = df1.filter(labels).values
            # flatten data fro histo plot:
            data = data.flatten()
            

            
            if noptmax > 0:
                from matplotlib.ticker import LogFormatter
               

                df1_pr = df_pr.loc[:,[x for x in df_pr.columns if all((grp in x, x[-2] != '_'))]]
                df1_pr.rename(columns={x:x.split(':')[3].replace('_input', '') for x in df1_pr.columns}, inplace=True)
                nlb_pr = df1_pr.pop('near_lbnd').mean()
                nub_pr = df1_pr.pop('near_ubnd').mean()
                data_pr = df1_pr.filter(labels).values
                
                f, ax = plt.subplots(1,1,figsize=(8,4), dpi=300)
                ax.xaxis.set_major_formatter(LogFormatter(10))

                # Create log-spaced bins
                log_bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)

                # Plot histogram with log-spaced bins
                ax.hist(data, bins=log_bins, color='b', alpha=0.4, edgecolor='k', linewidth=0.5)

                # get base real data:
                base = np.unique(df1.loc[df1.index == 'base', labels].values.flatten())
                for b in base:
                    # plot vertical line at base:
                    ax.axvline(b, color='b', linestyle='--', linewidth=1.5, label='base')
                ax.set_xscale('log')
                max_freq = ax.get_ylim()[1]
                max_freq = np.ceil(max_freq / 10) * 10
                ax.vlines([ub, lb], 1, max_freq, linestyle='-', lw=1.0, colors='r', label='ultimate bounds')
                ax.legend(loc='upper left')
                
                # if grp startswith parm_fullnm key then get the value
                for key in parm_fullnm.keys():
                    if grp.startswith(key):
                        grp_full_nm = parm_fullnm[key] +': Layer ' + str(int(grp.split('_k')[1])+1)
                        break
                
                ax.set_title('{0}\n near lbnd: {1} prior/{2} post, near ubnd: {3} prior/{4} post'.format(grp_full_nm, int(nlb), int(nlb_pr), int(nub), int(nub_pr)))
            
            pp.savefig(f)
            plt.close()

def plot_parm_violins(m_d, pst_name, noptmax):
    '''Function to plot ensemble distributions of summary statistics for effective MODFLOW inputs generated by multiplier parameters
    Args:
        m_d (str): relative path to master directory
        o_d (str): relative path to output directory
        pst_name (str): pest control file name
        noptmax (int): pest parameter estimation iteration to plot
        max_fail (int): maximum threshold of convergence failures allowed before dropping realization from plots
    ''' 
    print('\n')
    print('plotting mf input for {0} iteration {1}'.format(m_d, noptmax))
    print('\n')
    
    o_d = os.path.join(m_d,'results','figures','mf_input_tracking')
    if os.path.exists(o_d) == False:
        os.makedirs(o_d)
        
    pst = pyemu.Pst(os.path.join(m_d, pst_name))
    oe = pyemu.ObservationEnsemble.from_binary(filename=os.path.join(m_d, '{0}.{1}.obs.jcb'.format(pst_name.split('.')[0], noptmax)), pst=pst)
    oe = oe.loc[:,[x for x in oe.columns if x.startswith('oname:mfin')]]
    df = pd.DataFrame.from_records((oe.to_records()))
    df.index = df.pop('index')
    grps = np.unique([x.split(':')[-1] for x in df.columns])
    grps = [x for x in grps if x[-2] != '_']
    grps0 = np.unique(['_'.join(x.split('_')[:-1]) for x in grps])


    if noptmax > 0:
        oe_pr = pyemu.ObservationEnsemble.from_binary(filename=os.path.join(m_d, '{0}.0.obs.jcb'.format(pst_name.split('.')[0])), pst=pst)
        oe_pr = oe_pr.loc[:,[x for x in oe_pr.columns if x.startswith('oname:mfin')]]
        df_pr = pd.DataFrame.from_records((oe_pr.to_records()))
        df_pr.index = df_pr.pop('index')
         
    with PdfPages(os.path.join(o_d, '{0}_{1}_mf_input_stats_comp.pdf'.format(m_d, noptmax))) as pp:
        for grp in grps:
            print(grp)
            df1 = df.loc[:,[x for x in df.columns if all((grp in x, x[-2] != '_'))]]
            df1.rename(columns={x:x.split(':')[3].replace('_input', '') for x in df1.columns}, inplace=True)
            nlb = df1.pop('near_lbnd').mean()
            nub = df1.pop('near_ubnd').mean()
            lb = df1.pop('lower_bound').iloc[0]
            ub = df1.pop('upper_bound').iloc[0]
            labels = ['min', 'qnt25', 'qnt50', 'qnt75' , 'max']
            data = df1.filter(labels).values

            f, ax = plt.subplots(1,1,figsize=(8,4), dpi=300)

            if noptmax > 0:
                df1_pr = df_pr.loc[:,[x for x in df_pr.columns if all((grp in x, x[-2] != '_'))]]
                df1_pr.rename(columns={x:x.split(':')[3].replace('_input', '') for x in df1_pr.columns}, inplace=True)
                nlb_pr = df1_pr.pop('near_lbnd').mean()
                nub_pr = df1_pr.pop('near_ubnd').mean()
                data_pr = df1_pr.filter(labels).values
                
                vp_pr = ax.violinplot(dataset=data_pr)
                for pc in vp_pr['bodies']:
                    pc.set_facecolor('0.5')
                    #pc.set_edgecolor('k')
                    pc.set_alpha(0.4)
                for partname in ('cbars','cmins','cmaxes'):
                    vpt = vp_pr[partname]
                    vpt.set_edgecolor('0.5')
                    vpt.set_linewidth(0.5)
            
            vp = ax.violinplot(dataset=data)
            for pc in vp['bodies']:
                pc.set_facecolor('b')
                #pc.set_edgecolor('k')
                pc.set_alpha(0.4)
            for partname in ('cbars','cmins','cmaxes'):
                vpt = vp[partname]
                vpt.set_edgecolor('b')
                vpt.set_linewidth(0.5)

            ax.set_title('{0} near lbnd: {1} prior/{2} post, near ubnd: {3} prior/{4} post'.format(grp, int(nlb), int(nlb_pr), int(nub), int(nub_pr)))
            ax.set_yscale('log')

            ax.hlines([ub, lb], 1, len(df1.columns), linestyle='--', lw=0.5, colors='k')
            
            ax.annotate(text='upper bound', xy=(0.35, ub))
            ax.annotate(text='lower bound', xy=(0.35, lb))
            
            _set_axis_style(ax, labels)
            pp.savefig(f)
            plt.close()

def plot_inset(ax=[],wl_loc=None,grp=[],cpts=[],aq_extent=[],drains=[]):
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
        grp_num = sites['group_number']
        k = sites['k']
        sites['id'] = 'grpid:'+grp_num.astype(str) + '_k:' + k.astype(str)
        wpt = sites.loc[sites.id == wl_loc]
        wpt.plot(ax=ax, edgecolor='black', facecolor='orange',markersize=25,zorder=11,label='Highlighted water level observation')
 
    lb1 = mlines.Line2D([], [], color='orange', marker='o',markeredgecolor='black', linestyle='None', markersize=8, label='Highlighted water\nlevel observation') 
    lb2 = mlines.Line2D([], [], color='grey', marker='o',markeredgecolor='black', linestyle='None', markersize=5, label='Water level\nobservations') 
    lb3 = mlines.Line2D([], [], color='blue', linestyle='-', label='Rivers and streams') 
    lb4 = mpatches.Patch(facecolor='white',linewidth=1,edgecolor='black', label='Model area')
    leg = ax.legend(handles=[lb1, lb2, lb3, lb4], loc='lower right', frameon=True)
    leg = ax.legend(handles=[lb1, lb2, lb3, lb4], loc='lower right', frameon=True)
    leg.get_frame().set_facecolor('grey')       # Background color
    leg.get_frame().set_alpha(0.5)              # Transparency
    leg.get_frame().set_edgecolor('black')      # Dark outline color
    leg.get_frame().set_linewidth(2.0)          # Thickness of outline
    leg.set_bbox_to_anchor((.5, 0.05))

def plot_vert_xsec(ax, m_d, m, wl_loc=None, mwl=0, cpts=[]):
    if wl_loc is None:
        print('No water level location provided for vertical cross section plot')
        return
    
    df = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    points = [Point(xy) for xy in zip(df['x_2265'], df['y_2265'])]
    sites = gpd.GeoDataFrame(data=df,
                            geometry=points)
    sites = sites.set_crs(2265)#.to_crs(2265)
#    sites.to_file('data/analyzed/wl_obs.shp')
    grp_num = sites['group_number']
    k = sites['k']
    sites['id'] = 'grpid:'+grp_num.astype(str) + '_k:' + k.astype(str)
    wpt = sites.loc[sites.id == wl_loc]
    
    # sites = pd.read_csv(os.path.join('data', 'raw', 'sites.txt'),
    #                     sep='\t')
    # sites.columns = sites.columns.str.lower()

    # wpt = cpts.loc[cpts.obsprefix==wl_loc]
    wpt = sites.loc[sites.id == wl_loc]
    wpt['screen_top'] = wpt['top_screen'].values[0]
    wpt['screen_bot'] = wpt['bot_screen'].values[0]

    id = m.dis.idomain.array
    top = m.dis.top.array
    botm = m.dis.botm.array
    nlay = m.dis.nlay.data

    layers = {
        0: 'Soils/Clay/Silt',
        1: 'Elk Valley Aquifer',
    }
    
    wpt['i'] = wpt['row']-1
    wpt['j'] = wpt['col']-1
    i = wpt.i.astype(int).values[0]
    j = wpt.j.astype(int).values[0]
    idom = id[:, i, j]
    struct = np.array([top[i, j], *botm[:, i, j]])
    thk = np.diff(struct)
    mod_top = top[i, j]
    scr_top = wpt.screen_top.values[0]
    scr_bot = wpt.screen_bot.values[0]
    df = pd.DataFrame(data={'layer': layers.values(), 'bots': thk})
    df.layer = df.layer.str.replace('_', ' ')
    df = df.set_index('layer').T
    df = df.reset_index(drop=True)
    df = df.reset_index()

    sns.barplot(x='index', y='Soils/Clay/Silt', data=df, color='tan', label='Soils/Clay/Silt', bottom=mod_top,
                ax=ax)
    top_bar = mod_top + df['Soils/Clay/Silt'].values[0]
    sns.barplot(x='index', y='Elk Valley Aquifer', data=df, color='lightblue', label='Elk Valley Aquifer', bottom=top_bar,
                ax=ax)
   
    # plot vertical line between screen top and bottom:
    ax.plot([0, 0], [mod_top, mod_top - scr_bot], color='grey', linewidth=4)
    ax.plot([0, 0], [mod_top - scr_top, mod_top - scr_bot], color='black', linewidth=4, dashes=[0.5, 0.2],
            label='Screened Interval')
    ax.plot(0, mwl, 'bx', markersize=6, label='Mean Water Level')
    ax.legend(loc='lower right', bbox_to_anchor=(2.32, -0.08), ncol=1)
    ax.set_ylabel(f'Model Structure (feet ASL)')
    ax.set_xlabel('')
    ax.set_ylim([(mod_top - scr_bot) - 20, mod_top + 10])
    # ax1.set_title(f'Layer Structure at: row:{row}, col:{col}',fontsize=12)
    ax.set_xticklabels('')
    ax.tick_params(axis='both', which='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))

    box = ax.get_position()
    ax.set_position([box.x0 + 0.02, box.y0 + 0.06, box.width, box.height])

def plot_fancy_obs_v_sim(m_d, obsdict, modnm='elk_2lay',plot_hdiff=False,plt_noise=True,plt_pr=True):
    o_d = os.path.join(m_d, 'results', 'figures', 'obs_vs_sim')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, f'{modnm}.obs+noise.jcb'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('trans'), :].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    gwobs = gwobs.loc[gwobs.weight>0,:]
    noise = pst.ies.noise
    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])
    aobs = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'), index_col=0)
    conflicts = set(gwobs.loc[gwobs.weight > 0, 'obsnme'].tolist()) - set(aobs.loc[aobs.weight > 0, :].index.tolist())
    print(conflicts)

    if plot_hdiff:
        # add head difference obs:
        hdiff = obs.loc[obs.obgnme.str.contains('hdiff'), :].copy()
        hdiff['datetime'] = pd.to_datetime(hdiff.datetime.values)
        hdiff.loc[hdiff.obgnme.str.contains('hdiff'),'id'] = hdiff.loc[hdiff.obgnme.str.contains('hdiff'),'obsnme'].apply(lambda x: x.split('hdiff_')[1].split('_i')[0])


    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis'])
    m = sim.get_model(modnm)

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)
    top = m.dis.top.array
    botm = m.dis.botm.array
    itrmx = max(obsdict)
    usites = gwobs.id.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site': usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)

    # load shapefiles needed for plotting:
    cpts = gpd.read_file(os.path.join('data', 'analyzed', 'transient_well_targets_lookup.csv'))
    cpts = gpd.GeoDataFrame(cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=2265)
    # read in obsprefix info:
    obspre = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'))
    obspre['row'] = obspre['i'] + 1
    obspre['col'] = obspre['j'] + 1
    cpts['row'] = cpts['row'].astype(int)
    cpts['col'] = cpts['col'].astype(int)
    cpts = cpts.merge(obspre, left_on=['row','col'], right_on=['row','col'], how='left')
    #cpts['id'] = cpts
    cpts = cpts.groupby(['obsprefix']).last().reset_index()
    cpts.columns = cpts.columns.str.lower()
    
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'elk')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'elk')
    aq_extent = gpd.read_file(os.path.join(g_d, 'elk_boundary.shp'))
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'elk_cell_size_660ft_epsg2265_rot20.grid.shp'))
    drains = gpd.read_file(os.path.join(g_d, 'drn_raw.shp'))
    aq_extent = aq_extent.to_crs(modelgrid.crs)

    wls = pd.read_csv(os.path.join('data', 'analyzed','transient_well_targets_lookup.csv'))
    wls['grp_id'] = 'grpid:' + wls['group_number'].astype(str) + '_k:' + (wls['model_layer']-1).astype(str)

    usites = usitedf['site'].values
    with PdfPages(os.path.join(o_d, 'obs_v_sim_inset_vcross.pdf')) as pdf:
        for site in usites:
            # limit to just target wells:
            if site not in wls['grp_id'].values:
                continue
            uobs = gwobs.loc[gwobs.id == site, :].copy()
            uobs.sort_values(by='datetime', inplace=True)
            if plot_hdiff:
                grp_num = 'grp.' + site.split('_k:')[0].split('grpid:')[1]
                uhdiff = hdiff.loc[hdiff.id.str.contains(grp_num+'_'), :].copy()
                if len(uhdiff) > 0:
                    uhdiff.sort_values(by='datetime', inplace=True)
                    ohdiff = uhdiff.loc[uhdiff.observed == True, :]
    
            k, i, j = uobs.k.values[0], uobs.i.values[0], uobs.j.values[0]
            oobs = uobs.loc[uobs.observed == True, :]
            wobs = oobs.loc[oobs.weight > 0, :]
            if wobs.shape[0] == 0:
                continue
            dts = uobs.datetime.values
            vals = obsdict[itrmx].loc[:, uobs.obsnme].values

            fig = plt.figure(figsize=(11, 8))
            gs = gridspec.GridSpec(5, 6)
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            ax2 = fig.add_subplot(gs[0:2, 2])
            ax3 = fig.add_subplot(gs[0:2, 4:])
            if plot_hdiff:
                ax4 = fig.add_subplot(gs[2:4, :]) 
                ax5 = fig.add_subplot(gs[4:, :], sharex=ax4)      # BOTTOM shares x with ax4
                # hide x tick labels on the top axis (so only bottom shows dates)
                ax4.tick_params(axis='x', labelbottom=False)
            else:
                ax4 = fig.add_subplot(gs[2:, :])

            ax3.set_xticklabels('')
            ax3.set_yticklabels('')
            ax3.tick_params(axis='both', which='both', direction='in')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.tick_params(axis='both', which='both', length=0)

            plot_inset(ax=ax1, wl_loc=site, cpts=cpts, aq_extent=aq_extent, drains=drains)
            plot_vert_xsec(ax2, m_d, m, wl_loc=site, mwl=wobs.obsval.mean(), cpts=cpts)

            # [ax4.plot(dts,vals[i,:],color='0.5',alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            unoise = noise.loc[:, wobs.obsnme].values
            ndts = wobs.datetime
            # [ax4.plot(ndts, unoise[i, :], color='r', alpha=0.25, lw=0.1) for i in range(unoise.shape[0])]
            if itrmx > 0:
                vals = obsdict[itrmx].loc[:, uobs.obsnme].values
                [ax4.plot(dts, vals[i, :], color='b', alpha=0.5, lw=0.1,zorder=3) for i in range(vals.shape[0])]
                # plot base:
                min_val = vals.min()
                max_val = vals.max()
                ax4.scatter(oobs.datetime, oobs.obsval, marker='.', color='k', s=30, zorder=8)
                ax4.scatter(wobs.datetime, wobs.obsval, marker='.', color='r', s=40, zorder=10)
                base_vals = obsdict[itrmx].loc[:, uobs.obsnme].loc['base', :].values
                ax4.plot(dts, base_vals, color='orange', lw=1.5, zorder=10)
                if plt_noise:
                    nvals = noise.loc[:,oobs.obsnme].values
                    [ax4.plot(oobs.datetime,nvals[i,:],'r',alpha=0.3,lw=0.2,zorder=2) for i in range(nvals.shape[0])]
                if plt_pr:
                    prvals = pst.ies.obsen0.loc[:,oobs.obsnme].values
                    [ax4.plot(dts,prvals[i,:],'0.5',alpha=0.5,lw=0.2,zorder=1) for i in range(prvals.shape[0])]

                # ax4.scatter(wobs.datetime, wobs.obsval, marker='.', color='r', s=20,zorder=10)
                # cobs = wobs.loc[wobs.obsnme.apply(lambda x: x in conflicts),:]
                # ax4.scatter(cobs.datetime, cobs.obsval, marker='*', color='k', s=50,zorder=10)
                mx_obs = oobs.obsval.max()
                mn_obs = oobs.obsval.min()
             
                #ult_mx = max(mx_obs, max_val)
                #ult_mn = min(mn_obs, min_val)
                midpoint = np.median([mx_obs, mn_obs])
                #if midpoint + 30 > ult_mx:
                ult_mx = midpoint + 15
                #if midpoint - 30 < ult_mn:
                ult_mn = midpoint - 15

                ax4.set_ylim(ult_mn - 10, ult_mx + 10)
                ax4.yaxis.set_major_locator(ticker.MultipleLocator(30))  # Major ticks every 50
                ax4.yaxis.set_minor_locator(ticker.MultipleLocator(10))  # Minor ticks every 10

            t = top[i, j]
            bslice = botm[:, i, j]
            xlim = ax4.get_xlim()
            ax4.plot(xlim, [t, t], '--', color='tan', lw=1.5)
            for b in bslice:
                ax4.plot(xlim, [b, b], 'c--', lw=1.5, alpha=0.5)

            ax4.xaxis.set_major_locator(years10)
            ax4.xaxis.set_major_formatter(years_fmt)
            ax4.get_xaxis().set_tick_params(direction='in')
            ax4.tick_params(axis='both',direction='in', which='major', labelsize=11)
            ax4.set_ylabel('Water level (ft-ASL)', fontsize=12)
            ax4.tick_params(axis='both', which='major', labelsize=11)
            #ax4.tick_params(axis='x', labelbottom=False)
            # comma formateed y-axis labels:
            ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
            # add vertical line for predictive period:
            ax4.axvline(x=pd.to_datetime('2025-01-01'), color='grey', linestyle='-.', linewidth=1)
            ax4.text(pd.to_datetime('2025-01-01'), ax4.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                     ha='left', va='bottom', color='grey')
            ax4.set_xlim(pd.to_datetime('1965-01-01'), xlim[1])

            # make custom ax4 legend:
            leg1 = mlines.Line2D([], [], color='red', marker='o', linestyle='', markersize=5,
                                 label='Observed water level (weighted)')
            leg2 = mlines.Line2D([], [], color='black', marker='o', linestyle='', markersize=4,
                                 label='Observed water level (unweighted)')
            leg3 = mlines.Line2D([], [], color='blue', linestyle='-', label='Posterior ensemble')
            leg4 = mlines.Line2D([], [], color='orange', linestyle='-', label='Base of the Posterior')
            leg5 = mlines.Line2D([], [], color='black', linestyle='--', label='Layer contacts')

            legend1 = ax4.legend(handles=[leg1, leg2, leg3, leg4, leg5], loc='lower left', ncol=1, fontsize=8, frameon=True,framealpha=0.5)
            ax4.add_artist(legend1)
            shared_colors = ['#2C6B30','#377EB8','#4DAF4A','#984EA3','#FF7F00','#FFFF33','#A65628',]
            
            # load raw wl data:
            try:
                water_levels = pd.read_csv(os.path.join('data','raw','obs_data','elk_valley_water_level_data.csv'))
            except:
                water_levels = pd.read_parquet(os.path.join('data','raw','wahp_waterlevels.parquet'))
            # Filter
            water_levels['date_meas'] = pd.to_datetime(water_levels['Date_Measured'])
            water_levels = water_levels.set_index('date_meas')
            water_levels.loc[water_levels['Water_Level(NAVD88)']>10000,'Water_Level(NAVD88)'] = np.nan
            water_levels.loc[water_levels['Water_Level(NAVD88)']<-1000,'Water_Level(NAVD88)'] = np.nan
            water_levels = water_levels.dropna()

            grp_num = int(site.split('_')[0].split(':')[1])
            k = int(site.split('_')[1].split(':')[1])
            sites_grp = wls[wls['group_number'] == int(grp_num)]
            grp_full = sites_grp.copy()
            sites_grp = sites_grp[sites_grp['model_layer']-1 == int(k)]
            color_cnt = 0
            leg6 = []
            for site_id in sites_grp['location'].unique():
                wl_data = water_levels.loc[water_levels['Location']==site_id,'Water_Level(NAVD88)']

                # Daily resample on high-frequency transducer data
                if len(wl_data) > 1200:
                    wl_data_resamp = wl_data.resample('D').mean()
                    wl_data_resamp = wl_data_resamp.dropna()
                    ax4.plot(wl_data_resamp,
                                marker='o',
                                ms=4,
                                color=shared_colors[color_cnt],
                                label=site_id,
                                alpha=0.5)
                    color_cnt += 1
                else:
                    ax4.plot(wl_data,
                                marker='o',
                                ms=4,
                                color=shared_colors[color_cnt],
                                label=site_id,
                                alpha=0.5)
                    color_cnt += 1
                leg6.append(mlines.Line2D([], [], color=shared_colors[color_cnt-1], marker='o', linestyle='-', markersize=5,
                                      label=site_id))
            legend2 = ax4.legend(handles=leg6, loc='upper left')

            if wobs.weight.max() == 1.0:
                wobs_bool = True
            else:
                wobs_bool = False
            
            aq_key = {0: "Soils/Clay/Silt",1: "Elk Valley Aquifer"}
            current_aq = aq_key.get(k)

            # sort grp_full by model_layer
            grp_full = grp_full.sort_values(by=['model_layer'])

            ax3.text(0.5, 0.75,
                     f'{current_aq}\n Group: {grp_num}\n Layer: {k + 1}, \nRow: {i + 1}, Column: {j + 1}\nModel top: {t}\n\n',
                     fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
            
            # add text that icludes grp_full [loc_id, assigned aquifer]:
            ax3.text(0.5, 0.27, 'All wells in group:\n' + '\n'.join(
                [f"{row['location']} - {aq_key.get(row['model_layer']-1)}" for idx, row in grp_full.iterrows()]),
                     fontsize=10, ha='center', va='center', color='black', transform=ax3.transAxes)
            
            if plot_hdiff and len(uhdiff) > 0:
                # fill out k column using obsnme:
                uhdiff = uhdiff.loc[uhdiff.datetime >= uobs.datetime.min(), :]
                uhdiff = uhdiff.loc[uhdiff.datetime <= uobs.datetime.max(), :]
                if uhdiff.shape[0] > 0:
                    hdvals = obsdict[itrmx].loc[:, uhdiff.obsnme].values
                    if itrmx > 0:
                        [ax5.plot(uhdiff.datetime, hdvals[i, :], color='b', alpha=0.5, lw=0.1) for i in range(hdvals.shape[0])]
                        # plot base:
                        base_vals = obsdict[itrmx].loc[:, uhdiff.obsnme].loc['base', :].values
                        ax5.plot(uhdiff.datetime, base_vals, color='orange', lw=1.25, zorder=10)
                        min_val = hdvals.min()
                        max_val = hdvals.max()
                        ax5.scatter(ohdiff.datetime, ohdiff.obsval, marker='.', color='r', s=30, zorder=8)
                        aq_pair = uhdiff.obsnme.values[0].split(':')[3].split('_')[2].upper()
                        ax5.set_ylabel(f'Vertical gradient\n for aquifer pair:\n {aq_pair} (feet)', fontsize=11)

                        ax5.xaxis.set_major_locator(years10)
                        ax5.xaxis.set_major_formatter(years_fmt)
                        ax5.xaxis.set_ticks_position('both') 
                        ax5.get_xaxis().set_tick_params(direction='out')
                        ax5.tick_params(axis='both',direction='in', which='major', labelsize=11)
                        ax5.axvline(x=pd.to_datetime('2025-01-01'), color='grey', linestyle='-.', linewidth=1)
                        ax5.text(pd.to_datetime('2025-01-01'), ax5.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                                ha='left', va='bottom', color='grey')
            pdf.savefig()
            plt.close(fig)
    pdf.close()


def plot_fancy_obs_v_sim_base(m_d, obsdict, modnm='elk_2lay',plot_hdiff=False,plt_noise=False,plt_pr=False):
    o_d = os.path.join(m_d, 'results', 'figures', 'obs_vs_sim')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join(m_d, f'{modnm}.obs+noise.jcb'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('trans'), :].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    gwobs = gwobs.loc[gwobs.weight>0,:]
    noise = pst.ies.noise
    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])
    aobs = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'), index_col=0)
    conflicts = set(gwobs.loc[gwobs.weight > 0, 'obsnme'].tolist()) - set(aobs.loc[aobs.weight > 0, :].index.tolist())
    print(conflicts)

    if plot_hdiff:
        # add head difference obs:
        hdiff = obs.loc[obs.obgnme.str.contains('hdiff'), :].copy()
        hdiff['datetime'] = pd.to_datetime(hdiff.datetime.values)
        hdiff.loc[hdiff.obgnme.str.contains('hdiff'),'id'] = hdiff.loc[hdiff.obgnme.str.contains('hdiff'),'obsnme'].apply(lambda x: x.split('hdiff_')[1].split('_i')[0])


    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis'])
    m = sim.get_model(modnm)

    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)
    top = m.dis.top.array
    botm = m.dis.botm.array
    itrmx = max(obsdict)
    usites = gwobs.id.unique()
    usites.sort()

    usitedf = pd.DataFrame({'site': usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)

    # load shapefiles needed for plotting:
    cpts = gpd.read_file(os.path.join('data', 'analyzed', 'transient_well_targets_lookup.csv'))
    cpts = gpd.GeoDataFrame(cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=2265)
    # read in obsprefix info:
    obspre = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'))
    obspre['row'] = obspre['i'] + 1
    obspre['col'] = obspre['j'] + 1
    cpts['row'] = cpts['row'].astype(int)
    cpts['col'] = cpts['col'].astype(int)
    cpts = cpts.merge(obspre, left_on=['row','col'], right_on=['row','col'], how='left')
    #cpts['id'] = cpts
    cpts = cpts.groupby(['obsprefix']).last().reset_index()
    cpts.columns = cpts.columns.str.lower()
    
    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'elk')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'elk')
    aq_extent = gpd.read_file(os.path.join(g_d, 'elk_boundary.shp'))
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'elk_cell_size_660ft_epsg2265_rot20.grid.shp'))
    drains = gpd.read_file(os.path.join(g_d, 'drn_raw.shp'))
    aq_extent = aq_extent.to_crs(modelgrid.crs)

    wls = pd.read_csv(os.path.join('data', 'analyzed','transient_well_targets_lookup.csv'))
    wls['grp_id'] = 'grpid:' + wls['group_number'].astype(str) + '_k:' + (wls['model_layer']-1).astype(str)

    usites = usitedf['site'].values
    with PdfPages(os.path.join(o_d, 'obs_v_sim_inset_vcross_base.pdf')) as pdf:
        for site in usites:
            # limit to just target wells:
            if site not in wls['grp_id'].values:
                continue
            uobs = gwobs.loc[gwobs.id == site, :].copy()
            uobs.sort_values(by='datetime', inplace=True)
            if plot_hdiff:
                grp_num = 'grp.' + site.split('_k:')[0].split('grpid:')[1]
                uhdiff = hdiff.loc[hdiff.id.str.contains(grp_num+'_'), :].copy()
                if len(uhdiff) > 0:
                    uhdiff.sort_values(by='datetime', inplace=True)
                    ohdiff = uhdiff.loc[uhdiff.observed == True, :]
    
            k, i, j = uobs.k.values[0], uobs.i.values[0], uobs.j.values[0]
            oobs = uobs.loc[uobs.observed == True, :]
            wobs = oobs.loc[oobs.weight > 0, :]
            if wobs.shape[0] == 0:
                continue
            dts = uobs.datetime.values
            vals = obsdict[itrmx].loc[:, uobs.obsnme].values

            fig = plt.figure(figsize=(11, 8))
            gs = gridspec.GridSpec(5, 6)
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            ax2 = fig.add_subplot(gs[0:2, 2])
            ax3 = fig.add_subplot(gs[0:2, 4:])
            if plot_hdiff:
                ax4 = fig.add_subplot(gs[2:4, :]) 
                ax5 = fig.add_subplot(gs[4:, :], sharex=ax4)      # BOTTOM shares x with ax4
                # hide x tick labels on the top axis (so only bottom shows dates)
                ax4.tick_params(axis='x', labelbottom=False)
            else:
                ax4 = fig.add_subplot(gs[2:, :])

            ax3.set_xticklabels('')
            ax3.set_yticklabels('')
            ax3.tick_params(axis='both', which='both', direction='in')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.tick_params(axis='both', which='both', length=0)

            plot_inset(ax=ax1, wl_loc=site, cpts=cpts, aq_extent=aq_extent, drains=drains)
            plot_vert_xsec(ax2, m_d, m, wl_loc=site, mwl=wobs.obsval.mean(), cpts=cpts)

            # [ax4.plot(dts,vals[i,:],color='0.5',alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            unoise = noise.loc[:, wobs.obsnme].values
            ndts = wobs.datetime
            # [ax4.plot(ndts, unoise[i, :], color='r', alpha=0.25, lw=0.1) for i in range(unoise.shape[0])]
            if itrmx > 0:
                vals = obsdict[itrmx].loc[:, uobs.obsnme].values
                [ax4.plot(dts, vals[i, :], color='b', alpha=0.5, lw=0.1,zorder=3) for i in range(vals.shape[0])]
                # plot base:
                min_val = vals.min()
                max_val = vals.max()
                ax4.scatter(oobs.datetime, oobs.obsval, marker='.', color='k', s=30, zorder=8)
                ax4.scatter(wobs.datetime, wobs.obsval, marker='.', color='r', s=40, zorder=10)
                base_vals = obsdict[itrmx].loc[:, uobs.obsnme].loc['base', :].values
                ax4.plot(dts, base_vals, color='orange', lw=1.5, zorder=10)
                if plt_noise:
                    nvals = noise.loc[:,oobs.obsnme].values
                    [ax4.plot(oobs.datetime,nvals[i,:],'r',alpha=0.3,lw=0.2,zorder=2) for i in range(nvals.shape[0])]
                if plt_pr:
                    prvals = pst.ies.obsen0.loc[:,oobs.obsnme].values
                    [ax4.plot(dts,prvals[i,:],'0.5',alpha=0.5,lw=0.2,zorder=1) for i in range(prvals.shape[0])]

                # ax4.scatter(wobs.datetime, wobs.obsval, marker='.', color='r', s=20,zorder=10)
                # cobs = wobs.loc[wobs.obsnme.apply(lambda x: x in conflicts),:]
                # ax4.scatter(cobs.datetime, cobs.obsval, marker='*', color='k', s=50,zorder=10)
                mx_obs = oobs.obsval.max()
                mn_obs = oobs.obsval.min()
             
                #ult_mx = max(mx_obs, max_val)
                #ult_mn = min(mn_obs, min_val)
                midpoint = np.median([mx_obs, mn_obs])
                #if midpoint + 30 > ult_mx:
                ult_mx = midpoint + 15
                #if midpoint - 30 < ult_mn:
                ult_mn = midpoint - 15

                ax4.set_ylim(ult_mn - 10, ult_mx + 10)
                ax4.yaxis.set_major_locator(ticker.MultipleLocator(30))  # Major ticks every 50
                ax4.yaxis.set_minor_locator(ticker.MultipleLocator(10))  # Minor ticks every 10

            t = top[i, j]
            bslice = botm[:, i, j]
            xlim = ax4.get_xlim()
            ax4.plot(xlim, [t, t], '--', color='tan', lw=1.5)
            for b in bslice:
                ax4.plot(xlim, [b, b], 'c--', lw=1.5, alpha=0.5)

            ax4.xaxis.set_major_locator(years10)
            ax4.xaxis.set_major_formatter(years_fmt)
            ax4.get_xaxis().set_tick_params(direction='in')
            ax4.tick_params(axis='both',direction='in', which='major', labelsize=11)
            ax4.set_ylabel('Water level (ft-ASL)', fontsize=12)
            ax4.tick_params(axis='both', which='major', labelsize=11)
            #ax4.tick_params(axis='x', labelbottom=False)
            # comma formateed y-axis labels:
            ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
            # add vertical line for predictive period:
            ax4.axvline(x=pd.to_datetime('2025-01-01'), color='grey', linestyle='-.', linewidth=1)
            ax4.text(pd.to_datetime('2025-01-01'), ax4.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                     ha='left', va='bottom', color='grey')
            ax4.set_xlim(pd.to_datetime('1965-01-01'), xlim[1])

            # make custom ax4 legend:
            leg1 = mlines.Line2D([], [], color='red', marker='o', linestyle='', markersize=5,
                                 label='Observed water level (weighted)')
            leg2 = mlines.Line2D([], [], color='black', marker='o', linestyle='', markersize=4,
                                 label='Observed water level (unweighted)')
            leg3 = mlines.Line2D([], [], color='blue', linestyle='-', label='Posterior ensemble')
            leg4 = mlines.Line2D([], [], color='orange', linestyle='-', label='Base of the Posterior')
            leg5 = mlines.Line2D([], [], color='black', linestyle='--', label='Layer contacts')

            legend1 = ax4.legend(handles=[leg1, leg2, leg3, leg4, leg5], loc='lower left', ncol=1, fontsize=8, frameon=True,framealpha=0.5)
            ax4.add_artist(legend1)
            shared_colors = ['#2C6B30','#377EB8','#4DAF4A','#984EA3','#FF7F00','#FFFF33','#A65628',]
            
            # load raw wl data:
            try:
                water_levels = pd.read_csv(os.path.join('data','raw','obs_data','elk_valley_water_level_data.csv'))
            except:
                water_levels = pd.read_parquet(os.path.join('data','raw','wahp_waterlevels.parquet'))
            # Filter
            water_levels['date_meas'] = pd.to_datetime(water_levels['Date_Measured'])
            water_levels = water_levels.set_index('date_meas')
            water_levels.loc[water_levels['Water_Level(NAVD88)']>10000,'Water_Level(NAVD88)'] = np.nan
            water_levels.loc[water_levels['Water_Level(NAVD88)']<-1000,'Water_Level(NAVD88)'] = np.nan
            water_levels = water_levels.dropna()

            grp_num = int(site.split('_')[0].split(':')[1])
            k = int(site.split('_')[1].split(':')[1])
            sites_grp = wls[wls['group_number'] == int(grp_num)]
            grp_full = sites_grp.copy()
            sites_grp = sites_grp[sites_grp['model_layer']-1 == int(k)]
            color_cnt = 0
            leg6 = []
            for site_id in sites_grp['location'].unique():
                wl_data = water_levels.loc[water_levels['Location']==site_id,'Water_Level(NAVD88)']

                # Daily resample on high-frequency transducer data
                if len(wl_data) > 1200:
                    wl_data_resamp = wl_data.resample('D').mean()
                    wl_data_resamp = wl_data_resamp.dropna()
                    ax4.plot(wl_data_resamp,
                                marker='o',
                                ms=4,
                                color=shared_colors[color_cnt],
                                label=site_id,
                                alpha=0.5)
                    color_cnt += 1
                else:
                    ax4.plot(wl_data,
                                marker='o',
                                ms=4,
                                color=shared_colors[color_cnt],
                                label=site_id,
                                alpha=0.5)
                    color_cnt += 1
                leg6.append(mlines.Line2D([], [], color=shared_colors[color_cnt-1], marker='o', linestyle='-', markersize=5,
                                      label=site_id))
            legend2 = ax4.legend(handles=leg6, loc='upper left')

            if wobs.weight.max() == 1.0:
                wobs_bool = True
            else:
                wobs_bool = False
            
            aq_key = {0: "Soils/Clay/Silt",1: "Elk Valley Aquifer"}
            current_aq = aq_key.get(k)

            # sort grp_full by model_layer
            grp_full = grp_full.sort_values(by=['model_layer'])

            ax3.text(0.5, 0.75,
                     f'{current_aq}\n Group: {grp_num}\n Layer: {k + 1}, \nRow: {i + 1}, Column: {j + 1}\nModel top: {t}\n\n',
                     fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)
            
            # add text that icludes grp_full [loc_id, assigned aquifer]:
            ax3.text(0.5, 0.27, 'All wells in group:\n' + '\n'.join(
                [f"{row['location']} - {aq_key.get(row['model_layer']-1)}" for idx, row in grp_full.iterrows()]),
                     fontsize=10, ha='center', va='center', color='black', transform=ax3.transAxes)
            
            if plot_hdiff and len(uhdiff) > 0:
                # fill out k column using obsnme:
                uhdiff = uhdiff.loc[uhdiff.datetime >= uobs.datetime.min(), :]
                uhdiff = uhdiff.loc[uhdiff.datetime <= uobs.datetime.max(), :]
                if uhdiff.shape[0] > 0:
                    hdvals = obsdict[itrmx].loc[:, uhdiff.obsnme].values
                    if itrmx > 0:
                        [ax5.plot(uhdiff.datetime, hdvals[i, :], color='b', alpha=0.5, lw=0.1) for i in range(hdvals.shape[0])]
                        # plot base:
                        base_vals = obsdict[itrmx].loc[:, uhdiff.obsnme].loc['base', :].values
                        ax5.plot(uhdiff.datetime, base_vals, color='orange', lw=1.25, zorder=10)
                        min_val = hdvals.min()
                        max_val = hdvals.max()
                        ax5.scatter(ohdiff.datetime, ohdiff.obsval, marker='.', color='r', s=30, zorder=8)
                        aq_pair = uhdiff.obsnme.values[0].split(':')[3].split('_')[2].upper()
                        ax5.set_ylabel(f'Vertical gradient\n for aquifer pair:\n {aq_pair} (feet)', fontsize=11)

                        ax5.xaxis.set_major_locator(years10)
                        ax5.xaxis.set_major_formatter(years_fmt)
                        ax5.xaxis.set_ticks_position('both') 
                        ax5.get_xaxis().set_tick_params(direction='out')
                        ax5.tick_params(axis='both',direction='in', which='major', labelsize=11)
                        ax5.axvline(x=pd.to_datetime('2025-01-01'), color='grey', linestyle='-.', linewidth=1)
                        ax5.text(pd.to_datetime('2025-01-01'), ax5.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                                ha='left', va='bottom', color='grey')
            pdf.savefig()
            plt.close(fig)
    pdf.close()

def base_posterior_param_forward_run(m_d0, noptmax):

    m_d = m_d0 + '_forward_run_base'

    print('copying dir {0} to {1}'.format(m_d0, m_d))
    shutil.copytree(m_d0, m_d, ignore=shutil.ignore_patterns('*.cbb', '*.hds', '*.log', '*.lst', '*.rec', '*.rei', '*_obs.csv','*.pdf'))

    modnm = 'elk_2lay'
    m_d0 = None
    start_datetime = pd.to_datetime('12-31-1979')
    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))

    # Update parameter set
    pst.parrep(parfile=os.path.join(m_d, f'{modnm}.{noptmax}.base.par'))
    # pst.control_data.noptmax = 0
    pst.write(os.path.join(m_d, f'{modnm}.pst'), version=2)
    prep_deps(m_d)
    pyemu.os_utils.run(f'pestpp-ies {modnm}.pst', cwd=m_d)
    # pyemu.os_utils.run('python forward_run.py', cwd=m_d)


def model_packages_to_shp_joined(
    d: str = ".",
    grid_shp: str = os.path.join(
        "..", "..", "gis", "output_shps", "elk", "elk_cell_size_660ft_epsg2265_rot20.grid.shp"
    ),
    debug: bool = False,
    bc_first_sp_only: bool = True,  # affects RIV/DRN/GHB only; WEL is always multi-SP
) -> None:
    """
    Join MF6 model attributes to an existing grid shapefile (authoritative geometry/CRS).

    Indexing:
      - grid: (i,j) 0-based; (row,col) 1-based; node 1-based
      - model arrays: 0-based

    Outputs -> <d>/output_shapefiles/model_packages/ :
      - npf.(shp|gpkg): grid + idom_*, thk_* + kh_*, k33_*, kv_*, T_*
      - sto.(shp|gpkg): grid + idom_*, ss_*, sy_*
      - rch.(shp|gpkg): grid + idom_* + rch#### (one col per SP, truncated to last MAX_SP_FIELDS)
      - per-instance boundary shapefiles (only cells where boundary exists):
          • WEL: per-SP (wlQ####; truncated to last MAX_SP_FIELDS)
          • RIV/DRN/GHB:
              - per-SP columns (e.g. rvS####, rvB####, rvC####; truncated to last MAX_SP_FIELDS)
              - if bc_first_sp_only=True → single columns per family (no SP numbering)
    """

    # -----------------------------
    # Config
    # -----------------------------
    MAX_SP_FIELDS = 240  # cap number of per-SP fields in shapefiles to avoid dBASE limits

    def _dbg(*args):
        if debug:
            print(*args)

    # short, shapefile-safe field names for BC SP columns
    def _bc_field_name(family: str, logical: str, spno: int | None = None) -> str:
        fam = family.lower()
        fam_code = {
            "riv": "rv",
            "drn": "dr",
            "ghb": "gh",
            "wel": "wl",
        }.get(fam[:3], fam[:2])

        log = logical.lower()
        log_code = {
            "stage": "S",
            "rbot": "B",
            "cond": "C",
            "elev": "E",
            "bhead": "H",
            "q": "Q",
            "rate": "Q",
            "flux": "Q",
        }.get(log, log[:1].upper() if log else "X")

        if spno is None:
            # no SP number (first_sp_only)
            name = f"{fam_code}{log_code}"
        else:
            name = f"{fam_code}{log_code}{spno:04d}"

        # ensure <=10 chars
        return name[:10]

    # -----------------------------
    # Load model
    # -----------------------------
    print("Loading MF6 simulation/model…")
    sim = flopy.mf6.MFSimulation.load(sim_ws=d)
    mf = sim.get_model("elk_2lay")
    dis = mf.dis

    # arrays
    k_arr = np.asarray(mf.npf.k.array)
    k33_arr = np.asarray(mf.npf.k33.array)
    top = np.asarray(dis.top.array)
    botm = np.asarray(dis.botm.array)
    try:
        idomain = np.asarray(dis.idomain.array)
    except Exception:
        idomain = np.ones_like(k_arr, dtype=int)

    # storage params
    ss_arr = getattr(getattr(mf, "sto", None), "ss", None)
    sy_arr = getattr(getattr(mf, "sto", None), "sy", None)
    ss_arr = (
        np.asarray(ss_arr.array)
        if (ss_arr is not None and getattr(ss_arr, "array", None) is not None)
        else None
    )
    sy_arr = (
        np.asarray(sy_arr.array)
        if (sy_arr is not None and getattr(sy_arr, "array", None) is not None)
        else None
    )

    nlay, nrow, ncol = k_arr.shape
    _dbg(f"Grid shape: nlay={nlay}, nrow={nrow}, ncol={ncol}")

    # layer thickness (model-derived)
    thk = np.empty_like(k_arr, dtype=float)
    for kk in range(nlay):
        top_k = top if kk == 0 else botm[kk - 1]
        thk[kk] = np.maximum(0.0, top_k - botm[kk])

    # -----------------------------
    # Load grid shapefile
    # -----------------------------
    _dbg("Reading grid shapefile:", grid_shp)
    grid = gpd.read_file(grid_shp)

    if "i" in grid.columns and "j" in grid.columns:
        gi = pd.to_numeric(grid["i"], errors="coerce").fillna(0).astype(int)
        gj = pd.to_numeric(grid["j"], errors="coerce").fillna(0).astype(int)
    elif "row" in grid.columns and "col" in grid.columns:
        gi = pd.to_numeric(grid["row"], errors="coerce").fillna(1).astype(int) - 1
        gj = pd.to_numeric(grid["col"], errors="coerce").fillna(1).astype(int) - 1
    else:
        raise ValueError("Grid shapefile must contain (i,j) or (row,col) columns.")

    grid["__key__"] = list(zip(gi, gj))
    _dbg(
        f"Grid features: {len(grid)}; "
        f"min(i,j)=({gi.min()},{gj.min()}), max(i,j)=({gi.max()},{gj.max()})"
    )

    # 2-D lookup (i,j) → row index in grid GDF
    row_index_2d = np.full((nrow, ncol), -1, dtype=int)
    gi_np, gj_np = gi.to_numpy(), gj.to_numpy()
    for r_idx, (ii0, jj0) in enumerate(zip(gi_np, gj_np)):
        if 0 <= ii0 < nrow and 0 <= jj0 < ncol:
            row_index_2d[ii0, jj0] = r_idx

    out_dir = os.path.join(d, "output_shapefiles", "model_packages")
    os.makedirs(out_dir, exist_ok=True)
    _dbg("Output directory:", out_dir)

    # -----------------------------
    # Writers
    # -----------------------------
    def _remove_shp(path_no_ext: str):
        for e in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = path_no_ext + e
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _write_both(gdf: gpd.GeoDataFrame, out_path: str):
        base = os.path.splitext(out_path)[0]
        _remove_shp(base)
        # DBF-friendly: cast bools
        for c in gdf.columns:
            if c == "geometry":
                continue
            if pd.api.types.is_bool_dtype(gdf[c]):
                gdf[c] = gdf[c].astype(int)
        gdf.to_file(base + ".shp")
        gdf.to_file(base + ".gpkg", driver="GPKG")
        _dbg("  wrote:", base + ".shp", "and", base + ".gpkg")

    # -----------------------------
    # Helpers
    # -----------------------------
    flat_idx = gi_np * ncol + gj_np  # row-major flat pos for (nrow,ncol)

    def add_layerwise_columns(
        df: gpd.GeoDataFrame, arr3d: np.ndarray, prefix: str
    ) -> gpd.GeoDataFrame:
        flat = arr3d.reshape(nlay, -1)
        vals = flat[:, flat_idx]  # (nlay, N)
        for kk in range(nlay):
            df[f"{prefix}_{kk}"] = vals[kk].astype(float)
        return df

    # ============================================================
    # NPF
    # ============================================================
    _dbg("Building NPF…")
    npf_df = grid.copy()
    npf_df = add_layerwise_columns(npf_df, idomain, "idom")
    npf_df = add_layerwise_columns(npf_df, k_arr, "kh")
    npf_df = add_layerwise_columns(npf_df, k33_arr, "k33")
    for kk in range(nlay):
        kh_col, k33_col = f"kh_{kk}", f"k33_{kk}"
        npf_df[f"kv_{kk}"] = npf_df[kh_col].astype(float) * npf_df[k33_col].astype(float)
        thk_vals = thk[kk].reshape(-1)[flat_idx]
        npf_df[f"T_{kk}"] = npf_df[kh_col].astype(float) * thk_vals
    _write_both(npf_df, os.path.join(out_dir, "npf.shp"))

    # ============================================================
    # STO (storage only)
    # ============================================================
    _dbg("Building STO…")
    sto_df = grid.copy()
    sto_df = add_layerwise_columns(sto_df, idomain, "idom")
    if ss_arr is not None:
        sto_df = add_layerwise_columns(sto_df, ss_arr, "ss")
    if sy_arr is not None:
        sto_df = add_layerwise_columns(sto_df, sy_arr, "sy")
    _write_both(sto_df, os.path.join(out_dir, "sto.shp"))

    # ============================================================
    # RCH
    # ============================================================
    _dbg("Building RCH…")
    rch_df = grid.copy()
    rch_df = add_layerwise_columns(rch_df, idomain, "idom")

    rch_pkg = mf.rch
    raw = None
    if hasattr(rch_pkg, "recharge") and hasattr(rch_pkg.recharge, "array"):
        raw = np.asarray(rch_pkg.recharge.array)
    elif hasattr(rch_pkg, "rech") and hasattr(rch_pkg.rech, "array"):
        raw = np.asarray(rch_pkg.rech.array)

    if raw is not None:
        _dbg("  RCH raw shape:", raw.shape)
        if raw.ndim == 2 and raw.shape == (nrow, ncol):
            raw = raw[None, ...]
        elif raw.ndim == 3 and raw.shape[1:] == (nrow, ncol):
            pass
        elif raw.ndim == 4 and raw.shape[-2:] == (nrow, ncol):
            raw = raw.sum(axis=1)  # sum across layers
        else:
            raise ValueError(f"Unsupported RCH shape {raw.shape}")

        rflat = raw.reshape(raw.shape[0], -1)
        vals_all = rflat[:, flat_idx]  # (n_sp, N)
        n_sp_total = vals_all.shape[0]

        if n_sp_total > MAX_SP_FIELDS:
            start = n_sp_total - MAX_SP_FIELDS
            _dbg(
                f"  RCH: {n_sp_total} SPs exceeds MAX_SP_FIELDS={MAX_SP_FIELDS}; "
                f"keeping last {MAX_SP_FIELDS} (offset={start})."
            )
        else:
            start = 0

        for local_idx in range(start, n_sp_total):
            spno = local_idx + 1  # 1-based SP number
            fld = f"rch{spno:04d}"  # e.g., rch0001 (<=10 chars)
            rch_df[fld] = vals_all[local_idx].astype(float)

    _write_both(rch_df, os.path.join(out_dir, "rch.shp"))

    # ============================================================
    # Boundary instance discovery (by package_type)
    # ============================================================
    def _title(pkg, family: str) -> str:
        for attr in ("pname", "package_name", "name"):
            try:
                val = getattr(pkg, attr)
                if isinstance(val, (list, tuple)) and val:
                    val = val[0]
                if isinstance(val, str) and val.strip():
                    return val.strip().lower()
            except Exception:
                pass
        fn = getattr(pkg, "filename", None)
        if isinstance(fn, str) and fn:
            stem = os.path.splitext(os.path.basename(fn))[0]
            if "." in stem:
                stem = stem.split(".")[-1]
            return stem.lower()
        return family

    def _find_instances_by_type(model, want: str) -> list[str]:
        want = want.lower()
        names = []
        try:
            for nm in model.get_package_list():
                try:
                    pk = model.get_package(nm)
                except Exception:
                    continue
                ptype = str(getattr(pk, "package_type", "")).lower()
                if ptype == want or ptype == f"{want}6" or want in ptype:
                    names.append(nm)
        except Exception:
            pass
        return names

    # Common index extractor → (ii,jj)
    _node_cache: Dict[int, Tuple[int, int]] = {}
    _cellid_cache: Dict[Any, Tuple[int, int]] = {}

    def _extract_ij(arr, names: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray] | None:
        if "i" in names and "j" in names:
            return arr[names["i"]].astype(int), arr[names["j"]].astype(int)
        if "row" in names and "column" in names:
            return arr[names["row"]].astype(int) - 1, arr[names["column"]].astype(int) - 1
        if "node" in names:
            nodes = arr[names["node"]].astype(int) - 1
            uniq = np.unique(nodes)
            for n in uniq:
                if n not in _node_cache:
                    try:
                        _k, _i, _j = mf.modelgrid.get_lrc(int(n))
                        _node_cache[n] = (int(_i), int(_j))
                    except Exception:
                        _node_cache[n] = None
            pairs = [_node_cache.get(int(n), (np.nan, np.nan)) for n in nodes]
            ii = np.array([p[0] for p in pairs], dtype=float)
            jj = np.array([p[1] for p in pairs], dtype=float)
            ok = ~np.isnan(ii) & ~np.isnan(jj)
            return ii[ok].astype(int), jj[ok].astype(int)
        if "cellid" in names:
            cids = arr[names["cellid"]]
            uniq = list({c for c in cids})
            for cid in uniq:
                if cid in _cellid_cache:
                    continue
                try:
                    if hasattr(cid, "__len__"):
                        if len(cid) == 3:
                            _cellid_cache[cid] = (int(cid[1]), int(cid[2]))
                        elif len(cid) == 2:
                            lyr, icell = int(cid[0]), int(cid[1])
                            try:
                                _i, _j = mf.modelgrid.get_ij(icell)
                            except Exception:
                                _k, _i, _j = mf.modelgrid.get_lrc(icell)
                            _cellid_cache[cid] = (int(_i), int(_j))
                        else:
                            _cellid_cache[cid] = None
                    else:
                        try:
                            _k, _i, _j = mf.modelgrid.get_lrc(int(cid))
                            _cellid_cache[cid] = (int(_i), int(_j))
                        except Exception:
                            try:
                                _i, _j = mf.modelgrid.get_ij(int(cid))
                                _cellid_cache[cid] = (int(_i), int(_j))
                            except Exception:
                                _cellid_cache[cid] = None
                except Exception:
                    _cellid_cache[cid] = None
            pairs = [_cellid_cache.get(cid, (np.nan, np.nan)) for cid in cids]
            ii = np.array([p[0] for p in pairs], dtype=float)
            jj = np.array([p[1] for p in pairs], dtype=float)
            ok = ~np.isnan(ii) & ~np.isnan(jj)
            return ii[ok].astype(int), jj[ok].astype(int)
        return None

    def _write_boundary_instance(
        pkg, family: str, fields: Dict[str, str], first_sp_only: bool
    ):
        """
        For RIV/DRN/GHB:
          - if first_sp_only=True → write only first SPD period (no SP suffix, e.g. rvS, rvB, rvC).
          - else → per-SP fields, truncated to last MAX_SP_FIELDS SPs.
        """
        title = _title(pkg, family)

        spd = getattr(pkg, "stress_period_data", None)
        if spd is None or getattr(spd, "data", None) is None:
            _dbg(f"  {title}: [SKIP] no SPD")
            return

        per_keys = sorted(list(spd.data.keys()))  # 0-based SP indices
        if not per_keys:
            _dbg(f"  {title}: [SKIP] empty SPD")
            return

        collapse = first_sp_only and family in {"riv", "drn", "ghb"}
        target_keys = per_keys[:1] if collapse else per_keys
        nper_out = len(target_keys)

        N = len(grid)
        present = np.zeros((nrow, ncol), dtype=bool)
        store = {
            fname: np.full((nper_out, N), np.nan, dtype=float) for fname in fields.keys()
        }

        # fill store[logical][ip, row_idx]
        for ip, per in enumerate(target_keys):
            arr = spd.data[per]
            if arr is None or len(arr) == 0:
                continue
            names = {n.lower(): n for n in arr.dtype.names}

            active_mask = None
            existed = False
            for logical, spdname in fields.items():
                lname = spdname.lower()
                if lname in names:
                    existed = True
                    v = arr[names[lname]]
                    if np.issubdtype(v.dtype, np.floating):
                        v = np.nan_to_num(v, nan=0.0)
                    m = v != 0.0
                    active_mask = m if active_mask is None else (active_mask | m)
            if not existed or active_mask is None or not active_mask.any():
                continue

            arr_act = arr[active_mask]
            names_act = {n.lower(): n for n in arr_act.dtype.names}
            ij = _extract_ij(arr_act, names_act)
            if ij is None:
                continue
            ii_a, jj_a = ij
            if len(ii_a) == 0:
                continue

            inb = (ii_a >= 0) & (ii_a < nrow) & (jj_a >= 0) & (jj_a < ncol)
            if not inb.any():
                continue
            rows = row_index_2d[ii_a[inb], jj_a[inb]]
            ok = rows >= 0
            if not ok.any():
                continue
            rows = rows[ok]
            present[ii_a[inb][ok], jj_a[inb][ok]] = True

            for logical, spdname in fields.items():
                lname = spdname.lower()
                if lname in names_act:
                    v = arr_act[names_act[lname]][inb][ok].astype(float, copy=False)
                    store[logical][ip, rows] = v

        if not present.any():
            _dbg(f"  {title}: [SKIP] no active cells detected.")
            return

        ii_p, jj_p = np.nonzero(present)
        keys = set(map(tuple, np.c_[ii_p, jj_p]))
        sel = grid["__key__"].isin(keys)
        df = grid.loc[sel].copy()
        if df.empty:
            _dbg(f"  {title}: [WARN] active but no grid intersection.")
            return

        idx_sel = sel.to_numpy().nonzero()[0]

        if collapse:
            # just one "period" (the first in SPD)
            for logical, arr2d in store.items():
                col = _bc_field_name(family, logical, spno=None)
                df[col] = arr2d[0, idx_sel]
        else:
            for logical, arr2d in store.items():
                vals = arr2d[:, idx_sel]  # (nper_out, N_sel)
                n_sp_total = vals.shape[0]
                if n_sp_total > MAX_SP_FIELDS:
                    offset = n_sp_total - MAX_SP_FIELDS
                    _dbg(
                        f"  {title}/{family}-{logical}: {n_sp_total} SPs > "
                        f"MAX_SP_FIELDS={MAX_SP_FIELDS}; keeping last {MAX_SP_FIELDS} "
                        f"(offset={offset})."
                    )
                else:
                    offset = 0

                for local_idx in range(offset, n_sp_total):
                    per_global = target_keys[local_idx]  # 0-based SP index
                    spno = per_global + 1
                    fld = _bc_field_name(family, logical, spno)
                    df[fld] = vals[local_idx]

        _write_both(df, os.path.join(out_dir, f"{title}.shp"))
        _dbg(f"  {title}: wrote {len(df)} cells (collapse={collapse})")

    # ---- WEL writer (uses true SP numbers, compressed field names, and truncation) ----
    def _write_wel_instance(pkg):
        family = "wel"
        title = _title(pkg, "wel")
        spd = getattr(pkg, "stress_period_data", None)
        if spd is None or not hasattr(spd, "array"):
            _dbg(f"[WEL] {title}: [SKIP] no SPD.array")
            return

        nper = mf.nper
        seq = list(spd.array)  # per-SP content; may contain None
        if len(seq) != nper:
            _dbg(f"[WEL] {title}: [WARN] SPD.array len {len(seq)} != nper {nper}")

        # pick value field from first non-None entry
        value_field = None
        for a in seq:
            if a is not None and hasattr(a, "dtype") and a.dtype.names:
                names = {n.lower(): n for n in a.dtype.names}
                for cand in ("q", "rate", "flux"):
                    if cand in names:
                        value_field = cand
                        break
            if value_field:
                break
        if value_field is None:
            _dbg(f"[WEL] {title}: [SKIP] no q/rate/flux field found")
            return

        def _ij_from_arr(a):
            names = {n.lower(): n for n in a.dtype.names}
            if "i" in names and "j" in names:
                return a[names["i"]].astype(int), a[names["j"]].astype(int)
            if "row" in names and "column" in names:
                return a[names["row"]].astype(int) - 1, a[names["column"]].astype(int) - 1
            if "node" in names:
                nodes = a[names["node"]].astype(int) - 1
                ii = np.full(len(nodes), -1, dtype=int)
                jj = np.full(len(nodes), -1, dtype=int)
                for idx, n in enumerate(nodes):
                    try:
                        _k, _i, _j = mf.modelgrid.get_lrc(int(n))
                        ii[idx], jj[idx] = int(_i), int(_j)
                    except Exception:
                        pass
                return ii, jj
            if "cellid" in names:
                cids = a[names["cellid"]]
                ii = np.full(len(cids), -1, dtype=int)
                jj = np.full(len(cids), -1, dtype=int)
                for idx, cid in enumerate(cids):
                    try:
                        if hasattr(cid, "__len__") and len(cid) >= 3:
                            # Treat row/col as already 0-based (consistent with _extract_ij)
                            ii[idx] = int(cid[1])
                            jj[idx] = int(cid[2])
                        else:
                            _k, _i, _j = mf.modelgrid.get_lrc(int(cid))
                            ii[idx], jj[idx] = int(_i), int(_j)
                    except Exception:
                        pass
                return ii, jj
            return None

        N = len(grid)
        present = np.zeros((nrow, ncol), dtype=bool)
        store = np.zeros((nper, N), dtype=float)  # zeros for missing SPs

        for kper in range(nper):
            arr = seq[kper] if kper < len(seq) else None
            if arr is None or len(arr) == 0:
                _dbg(f"[WEL] {title}: kper={(kper+1):04d} → None/empty (zeros)")
                continue

            names = {n.lower(): n for n in arr.dtype.names}
            if value_field not in names:
                _dbg(f"[WEL] {title}: kper={(kper+1):04d} → no '{value_field}' (zeros)")
                continue

            vals = arr[names[value_field]].astype(float, copy=False)
            vals = np.nan_to_num(vals, nan=0.0)
            active = vals != 0.0
            if not active.any():
                _dbg(f"[WEL] {title}: kper={(kper+1):04d} → all zero (skip)")
                continue

            act = arr[active]
            ij = _ij_from_arr(act)
            if ij is None:
                _dbg(f"[WEL] {title}: kper={(kper+1):04d} → no i/j (skip)")
                continue

            ii_a, jj_a = ij
            inb = (ii_a >= 0) & (ii_a < nrow) & (jj_a >= 0) & (jj_a < ncol)
            if not inb.any():
                _dbg(f"[WEL] {title}: kper={(kper+1):04d} → out-of-bounds (skip)")
                continue

            rows = row_index_2d[ii_a[inb], jj_a[inb]]
            ok = rows >= 0
            if not ok.any():
                _dbg(f"[WEL] {title}: kper={(kper+1):04d} → no grid match (skip)")
                continue

            rows = rows[ok]
            present[ii_a[inb][ok], jj_a[inb][ok]] = True
            vals_sp = act[names[value_field]][inb][ok].astype(float, copy=False)

            store[kper, rows] = vals_sp
            _dbg(
                f"[WEL] {title}: kper={(kper+1):04d} rows_in={len(arr)} "
                f"nonzero={active.sum()} mapped={len(rows)} "
                f"sumQ={float(np.nansum(vals_sp)):.2f}"
            )

        if not present.any():
            _dbg(f"[WEL] {title}: [SKIP] no active well cells across all SPs")
            return

        ii_p, jj_p = np.nonzero(present)
        keys = set(map(tuple, np.c_[ii_p, jj_p]))
        sel = grid["__key__"].isin(keys)
        df = grid.loc[sel].copy()
        idx_sel = sel.to_numpy().nonzero()[0]

        # keep last MAX_SP_FIELDS only
        n_sp_total = nper
        if n_sp_total > MAX_SP_FIELDS:
            offset = n_sp_total - MAX_SP_FIELDS
            _dbg(
                f"[WEL] {title}: {n_sp_total} SPs > MAX_SP_FIELDS={MAX_SP_FIELDS}; "
                f"keeping last {MAX_SP_FIELDS} (offset={offset})."
            )
        else:
            offset = 0

        for kper in range(offset, nper):
            spno = kper + 1
            fld = _bc_field_name(family, "q", spno)
            df[fld] = store[kper, idx_sel]

        _write_both(df, os.path.join(out_dir, f"{title}.shp"))
        _dbg(
            f"[WEL] {title}: wrote {len(df)} cells with "
            f"{min(nper, MAX_SP_FIELDS)} SP columns (zeros where missing)"
        )

    # ---- Discover names by package_type and write each family ----
    riv_names = _find_instances_by_type(mf, "riv")
    print(f"[RIV] instances via get_package_list: {riv_names}")
    for nm in riv_names:
        _write_boundary_instance(
            mf.get_package(nm),
            "riv",
            {"stage": "stage", "rbot": "rbot", "cond": "cond"},
            first_sp_only=bc_first_sp_only,
        )

    drn_names = _find_instances_by_type(mf, "drn")
    print(f"[DRN] instances via get_package_list: {drn_names}")
    for nm in drn_names:
        _write_boundary_instance(
            mf.get_package(nm),
            "drn",
            {"elev": "elev", "cond": "cond"},
            first_sp_only=bc_first_sp_only,
        )

    ghb_names = _find_instances_by_type(mf, "ghb")
    print(f"[GHB] instances via get_package_list: {ghb_names}")
    for nm in ghb_names:
        _write_boundary_instance(
            mf.get_package(nm),
            "ghb",
            {"bhead": "bhead", "cond": "cond"},
            first_sp_only=bc_first_sp_only,
        )

    wel_names = _find_instances_by_type(mf, "wel")
    print(f"[WEL] instances via get_package_list: {wel_names}")
    for nm in wel_names:
        _write_wel_instance(mf.get_package(nm))

    print(f"Done. Shapefiles written to: {out_dir}")




       
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

def run_zb_by_layer(w_d='',modnm='elk_2lay', plot=True):
   
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
        pdf = PdfPages(os.path.join(zon_path, 'water_balance_by_layer.pdf'))
        matplotlib.rcParams.update({'font.size': 10})
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

            # plot ins - highest to lowest volumes
            for i, col in enumerate(ins.columns):
                ax1.plot(dts, ins[col], color=cin[i % len(cin)], label=col, linewidth=1.8)
            ax1.set_yscale('log')
            ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
            ax1.set_ylim([10, ymax * 1.05])
            ax1.set_title(f'Layer {lay + 1} - Ins')
            ax1.set_ylabel('acre-ft per year')
            ax1.legend(loc='center right', fontsize=8)

            # plot outs - highest to lowest volumes
            for i, col in enumerate(outs.columns):
                if col == 'WEL-OUT-TOTAL':
                    ax2.plot(dts, outs[col], color='k', linestyle='--', label=col, zorder=3, linewidth=1.8)
                else:
                    ax2.plot(dts, outs[col], color=cout[i % len(cout)], label=col, linewidth=1.8)
            ax2.set_yscale('log')
            ax2.grid(True, which='both', linestyle=':', linewidth=0.7)
            ax2.set_title(f'Layer/Zone {lay + 1} - Outs')
            ax2.set_ylim([10, ymax * 1.05])
            ax2.legend(loc='center right', fontsize=8)

            pdf.savefig(fig)
            plt.close(fig)

            # plot volume sum for each layer on the combined figure
            if ins.shape[1] > 0:
                ax3.plot(dts, ins.sum(axis=1), label=f'Layer {lay + 1}', linewidth=1.8)
            if outs.shape[1] > 0:
                ax4.plot(dts, outs.sum(axis=1), label=f'Layer {lay + 1}', linewidth=1.8)

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

def plot_simple_obs_v_sim_base(m_d):

    pst = pyemu.Pst(os.path.join(m_d,'elk_2lay.pst'))

    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('freq|ext|elev'),:].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)

    heads_file = 'elk_2lay.hds'
    hdsobj = flopy.utils.binaryfile.HeadFile(os.path.join(m_d, heads_file))
    hds = hdsobj.get_alldata()
    hds[hds == 1e+30] = np.nan
    times = hdsobj.get_times()
    dates = pd.to_datetime('12-31-1979')+ pd.to_timedelta(times, unit='d')


    for col in ['k','i','j']:
        gwobs[col] = gwobs[col].astype(int)
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d)
    m = sim.get_model()
    top = m.dis.top.array
    botm = m.dis.botm.array
    # itrmx = max(obsdict)
    usites = gwobs[gwobs.usecol.str.contains('freq')].usecol.unique()
    usites.sort()
    hobs = m.obs[2].continuous.data['elk_2lay.head.obs.output.csv']

    usitedf = pd.DataFrame({'site':usites},index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.usecol==usite,:]
        usitedf.loc[usite,'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight','site'],ascending=False,inplace=True)

    usites = usitedf['site'].values
    #print(usites)
    with PdfPages(os.path.join(m_d,'simple_o_v_base.pdf')) as pdf:
        for site in usites:
            uobs = gwobs.loc[gwobs.usecol==site,:].copy()
            uobs.sort_values(by='datetime',inplace=True)
            k,i,j = uobs.k.values[0],uobs.i.values[0],uobs.j.values[0]
            oobs = uobs.loc[uobs.observed == True, :]
            wobs = oobs.loc[oobs.weight > 0, :]
            dts = uobs.datetime.values
            vals = hds[:,k,i,j]

            fig,ax = plt.subplots(1,1,figsize=(10,5))
            # [ax.plot(dts,vals[i,:],color='0.5',alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            # unoise = noise.loc[:, wobs.obsnme].values
            ndts = wobs.datetime
            # [ax.plot(ndts, wobs.obsval.values, color='r', alpha=0.25, lw=0.1) ]

            [ax.plot(dates, vals, color='b', alpha=0.8, lw=1, label='Simulated')]
            ax.scatter(oobs.datetime, oobs.obsval, marker='o', color='r',facecolor='none', s=50,zorder=10, label='Observed')
            ax.scatter(wobs.datetime, wobs.obsval, marker='.', color='r', s=20,zorder=10, label='Weighted')
            # cobs = wobs.loc[wobs.obsnme.apply(lambda x: x in conflicts),:]
            # ax.scatter(cobs.datetime, cobs.obsval, marker='*', color='k', s=50,zorder=10)

            # if wobs.shape[0] > 0:
            #     mn = unoise.min()
            #     mx = unoise.max()
            #     ax.set_ylim(mn*0.9,mx*1.1)
            # elif oobs.shape[0] > 0:
            #     mn = oobs.obsval.min()
            #     mx = oobs.obsval.max()
            #     ax.set_ylim(mn * 0.9, mx * 1.1)

            mn = np.nanmin(np.concatenate([oobs.obsval.values, vals]))
            mx = np.nanmax(np.concatenate([oobs.obsval.values, vals]))
            ax.set_ylim(mn * 0.99, mx * 1.01)

            t = top[i,j]
            bslice = botm[:,i,j]
            xlim = ax.get_xlim()
            ax.plot(xlim,[t,t],'m--',lw=1.5)
            for b in bslice:
                ax.plot(xlim,[b,b],'c--',lw=1.5,alpha=0.5)
            ax.set_title('usecol:{0}, mx weight: {1}, kij:{4} top: {2}\nbotm:{3}'. \
                         format(site, wobs.weight.max(), t, str(bslice),str((k,i,j))), loc='left')
            ax.grid()
            ax.legend()

            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            print('...', site)

def plot_base_water_table(d=''):
    '''This plots the water table where the lowest-elevation saturated
    zone is interpreted to be the water table and any saturated zones
    above it are interpreted to be non-physical artifacts of the
    newton-raphson approach to passing recharge through dry cells.

    Saves water-table plots in 'outdir' for specified print times.

    Using top elevation from dis package also plots depth to water to 'outdir'
    '''

    sim = flopy.mf6.MFSimulation.load(sim_ws=d, exe_name='mf6', load_only=['DIS', 'DRN', 'NPF',])
    mf = sim.get_model('elk_2lay')
    #perlen = np.floor(np.cumsum([entry[0] for entry in sim.tdis.perioddata.array]) / 365.0) + 2007

    outdir = os.path.join(d, 'results', 'figures', 'water_table')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    heads_file = 'elk_2lay.hds'
    hdsobj = flopy.utils.binaryfile.HeadFile(os.path.join(d, heads_file))
    hds = hdsobj.get_alldata()
    hds[hds == 1e+30] = np.nan
    times = hdsobj.get_times()


    mf._modelgrid._angrot = 0.0
    mf._modelgrid._xoff = 1468894.0
    mf._modelgrid._yoff = 6327767.0

    timeidxs = {'1980s': np.arange(0,10),
                '1990s': np.arange(10,20),
                '2000s': np.arange(20,30),
                '2010s': np.arange(30,40),
                '2020s': np.arange(40,50),
                '2030s': np.arange(50,60),
                '2040s': np.arange(60,64)}
    levels = np.arange(400, 2000, 50)
    levels_depth = [-40, -20, 0, 20, 40, 60, 80, 100]


    top = mf.dis.top.array

    for decade, timeidx in timeidxs.items():

        # Prepare array of heads representing water table keying off saturated thickness
        satthick = hds[timeidx] - mf.dis.botm.data  # heads below cell bottom will have negative sat thicknesses
        satthick = np.nanmean(satthick, axis=0)
        wt = np.full(satthick.shape[1:], np.nan)  # easiest to fill empty array recursively
        depth_wt = wt.copy()  # empty array for depth to water table

        for rowidx in range(satthick.shape[1]):
            for colidx in range(satthick.shape[2]):
                stack = satthick[:, rowidx, colidx].copy()
                if np.isnan(stack).all():
                    continue
                if (stack > 0).all():
                    wtlayidx = 0
                else:
                    wtlayidx = len(stack) - (stack[
                                             ::-1] < 0).argmax()# this will be the first (top to bottom) positive value after the last negative value
                    if wtlayidx ==11 and stack[1]>0:
                        wtlayidx = 1


                if wtlayidx == 10:
                    print(rowidx, colidx)
                wt[rowidx, colidx] = np.nanmean(hds[timeidx, wtlayidx, rowidx, colidx,], axis=0)
                depth_wt = top - wt

        # Export depth to WT as shapefile
        grid = gpd.read_file(os.path.join('gis','output_shapefiles', 'model_packages', 'top.shp'))
        grid['top'] = depth_wt.reshape([mf.dis.nrow.data * mf.dis.ncol.data])
        grid.rename(columns={'top':'wt_depth'}, inplace=True)
        # geometry = grid.geometry
        grid.to_file(os.path.join('gis','output_shapefiles', 'model_packages', 'wt_depth.shp'),)

        # flood check
        flood_chk = depth_wt.copy()
        flood_chk = np.where(flood_chk<-1,np.nan,flood_chk)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8),)# dpi=500)
        mapview = flopy.plot.PlotMapView(model=mf, layer=1)
        mapview.plot_grid(alpha=0.1)
        quadmesh = mapview.plot_ibound()
        quadmesh = mapview.plot_array(flood_chk, alpha=0.5,) # vmax=500, vmin=-500)
        quadmesh = mapview.plot_bc('DRN', color='blue', alpha = 0.5)
        # contour_set = mapview.contour_array(depth_wt, levels=levels_depth, colors='blue')


        plt.title(f'Flood Check - Decade {decade}', fontdict={'fontsize': 14})
        cb = plt.colorbar(quadmesh, shrink=0.8)
        cb.ax.set_title('Depth\nElev. (m)')
        # ax.set_ylim([7.220e6, 7.225e6])
        # ax.set_xlim([7.10e5, 7.16e5])
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(os.path.join(outdir, f'Decade_{decade}_flood_chk.png'))
        # plt.show()
        plt.close(fig)



        # Plot water table
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # , dpi=500)
        mapview = flopy.plot.PlotMapView(model=mf, layer=1)
        # mapview.plot_grid(alpha=0.1)
        quadmesh = mapview.plot_ibound()
        quadmesh = mapview.plot_array(wt, cmap='terrain', vmax =2500 )
        cb = plt.colorbar(quadmesh, shrink=0.8)
        #quadmesh = mapview.plot_bc('DRN', color='blue')
        contour_set = mapview.contour_array(wt, levels=levels, colors='black', alpha=0.5, lw=0.1)


        plt.title(f'Decade {decade} Average Water Table', fontdict={'fontsize': 14})
        plt.clabel(contour_set, fmt='%.1f', colors='black', fontsize=8)

        cb.ax.set_title('Elev. (m)')
        # ax.set_ylim([7.217e6, 7.226e6])
        # ax.set_xlim([7.09e5, 7.215e5])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'Decade_{decade}.png'))
        # plt.show()
        plt.close(fig)

        # Plot depth to water table
        fig, ax = plt.subplots(1, 1, figsize=(8, 8),)# dpi=500)
        mapview = flopy.plot.PlotMapView(model=mf, layer=1)
        mapview.plot_grid(alpha=0.1)
        quadmesh = mapview.plot_ibound()
        quadmesh = mapview.plot_array(depth_wt, alpha=0.5, vmax=500, vmin=-500)
        # contour_set = mapview.contour_array(depth_wt, levels=levels_depth, colors='blue')


        plt.title(f'Depth to Water Table - Decade {decade}', fontdict={'fontsize': 14})
        cb = plt.colorbar(quadmesh, shrink=0.8)
        cb.ax.set_title('Depth\nElev. (m)')
        # ax.set_ylim([7.220e6, 7.225e6])
        # ax.set_xlim([7.10e5, 7.16e5])
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(os.path.join(outdir, f'Decade_{decade}_wt_depth'))
        # plt.show()
        plt.close(fig)

def write_summary_tables(m_d='master', pst_name='elk_2lay.pst', noptmax=0, max_fail=2):
    '''writes parameter and observation summary tables using pyemu methods
    Args:
        m_d (str): relative path to master directory
        pst_name (str): pest control file name
        noptmax (int): pest parameter estimation iteration to plot
        max_fail (int): maximum threshold of convergence failures allowed before dropping realization from plots
    ''' 
    o_d = os.path.join(m_d,'results','tables')
    if os.path.exists(o_d) == False:
        os.makedirs(o_d)
    pst = pyemu.Pst(os.path.join(m_d, pst_name))
    pst.write_par_summary_table(filename=os.path.join(o_d, '{0}_{1}_par_summary.xlsx'.format(m_d, noptmax)))
    pst.write_obs_summary_table(filename=os.path.join(o_d, '{0}_{1}_obs_summary.xlsx'.format(m_d, noptmax)))

def plot_hds(
    model_ws: Union[str, os.PathLike],
    sim_name: str = 'elk_2lay',
    kstpkper: tuple[int, int] = (0, 0),
    layers: Union[str, Sequence[int]] = 'all',
    show_bc: bool = True,
    bc_colors: dict[str, str] | None = None,
    out_dir: str = 'fig_heads',
    dpi: int = 300,
):
    '''
    Colour-flood head maps with boundary conditions, compatible with
    older FloPy versions (uses PlotMapView.plot_bc).

    Parameters
    ----------
    model_ws
        Workspace containing the simulation.
    sim_name
        GWF model name (same as *.nam* prefix).
    kstpkper
        (kstp, kper) tuple to extract from *.hds*.
    layers
        'all' or iterable of 0-based layer indices.
    show_bc
        Toggle RIV, DRN, GHB, CHD overlays.
    bc_colors
        Dict of colours for each BC.
    out_dir
        Sub-folder for PNGs.
    dpi
        Output resolution.
    '''
    bc_colors = bc_colors or dict(
        riv='dodgerblue',
        drn='darkorange',
        ghb='limegreen',
        chd='magenta',
    )

    model_ws = os.fspath(model_ws)

    # 1 - Load sim & heads
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, exe_name='mf6')
    gwf = sim.get_model(sim_name)
    grid = gwf.modelgrid
    nlay = gwf.dis.nlay.data

    if layers == 'all':
        layers = range(nlay)

    hfile = flopy.utils.HeadFile(os.path.join(model_ws, f'{sim_name}.hds'))
    heads = hfile.get_data(kstpkper=kstpkper)
    idomain = grid.idomain                           # inactive mask (≤0)

    # Prepare array of  saturated thickness
    satthick = heads - gwf.dis.botm.data  # heads below cell bottom will have negative sat thicknesses



    # fig, ax = plt.subplots(1, 1, figsize=(8, 8),)# dpi=500)
    # mapview = flopy.plot.PlotMapView(model=mf, layer=1)
    # mapview.plot_grid(alpha=0.1)
    # quadmesh = mapview.plot_ibound()
    # quadmesh = mapview.plot_array(flood_chk, alpha=0.5,) # vmax=500, vmin=-500)
    # quadmesh = mapview.plot_bc("DRN", color='blue', alpha = 0.5)


    # 2 - Output folder
    figdir = os.path.join(model_ws, out_dir)
    os.makedirs(figdir, exist_ok=True)

    cmap = mpl.colormaps.get_cmap('viridis')

    # FloPy BC names expected by plot_bc
    tag2ftype = dict(riv='RIV', drn='DRN', ghb='GHB', chd='CHD')

    for k in layers:
        fig, ax = plt.subplots(figsize=(8, 7))
        mview = flopy.plot.PlotMapView(model=gwf, layer=k, ax=ax)

        satthk = satthick[k]
     
        # mask inactive + fill values
        active = idomain[k] > 0
        h_msk = np.ma.masked_where(~active | (heads[k] >= 1e20), heads[k])
        sat_msk = np.ma.masked_where(~active | (satthk >= 0.0), satthk)

        vmin, vmax = np.nanmin(h_msk), np.nanmax(h_msk)
        mview.plot_array(h_msk, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap,
                                  norm=mpl.colors.Normalize(vmin, vmax)),
            ax=ax, shrink=0.8, label='Head (ft)'
        )

        # 3 - Boundary conditions via plot_bc
        if show_bc:
            for tag, ftype in tag2ftype.items():
                if hasattr(gwf, tag):
                    # only required args: name (ftype) and optional color
                    mview.plot_bc(
                        name=ftype,
                        kper=0,
                        color=bc_colors[tag],
                        plotAll=False,          # keep only current layer
                    )
                    # dummy handle for legend
                    ax.scatter([], [], marker='o', facecolors='none',
                               edgecolors=bc_colors[tag], label=ftype)

        quadmesh = mview.plot_array(sat_msk, alpha=0.5,cmap='Reds', vmin=-10, vmax=0)
        # cosmetics & save
        ax.set_title(f'{sim_name} – Layer {k + 1}', loc='left')
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (ft)')
        ax.set_ylabel('Northing (ft)')
        if show_bc:
            ax.legend(fontsize=8, frameon=True)

        fig.tight_layout()
        png = os.path.join(figdir, f'heads_layer_{k + 1:02d}.png')
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
        print('saved →', png)
        
def run_a_real(m_d,real_name='base',noptmax=None,case='elk_2lay'):
    '''

    Parameters:
        master_d (str): the directory with ies results
        real_name (str): the name of a realization in the posterior parameter ensemble.  Default is 'base'
        noptmax (int): the ies iteration results to treat as the posterior. Default is pst.control_data.noptmax

    '''
    pst = pyemu.Pst(os.path.join(m_d,'{0}.pst'.format(case)))
    if noptmax is None:
        phidf = pd.read_csv(os.path.join(m_d,'{0}.phi.actual.csv'.format(case)))
        noptmax = phidf.iteration.max()
    pe_file = os.path.join(m_d,'{0}.{1}.par.jcb'.format(case,noptmax))
    assert os.path.exists(pe_file),pe_file
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=pe_file)    
    assert real_name in pe.index

    pst.parameter_data.loc[:,'parval1'] = pe.loc[real_name,pst.par_names].values
    pst.control_data.noptmax = 0
    pst_name = 'real_{0}.pst'.format(real_name)
    pst.write(os.path.join(m_d,pst_name),version=2)
    pyemu.os_utils.run('pestpp-ies {0}'.format(pst_name),cwd=m_d)

def plot_base_params_map_mel(
    model_ws: Union[str, os.PathLike],
    partype: str,
    sim_name: str = 'elk_2lay',
    layers: Union[str, Sequence[int]] = 'all',
    kstpkper: tuple[int, int] = (0, 0),          # for recharge only
    logscale_hk: bool = True,
    cmap_overrides: dict[str, str] | None = None,
    out_dir: str = 'fig_params',
    dpi: int = 300,
):
    '''
    Colour-flood maps of HK, VK, SS, SY or RECH ― one PNG per layer ― with the
    Wahpeton project outline shown as a grey basemap.
    '''

    # ── helper to unwrap LayeredArray/Util3D → ndarray
    def _to_array(obj):
        return obj.array if hasattr(obj, 'array') else obj

    # ── colour maps
    cmaps = dict(hk='plasma', vk='plasma', ss='viridis',
                 sy='YlGn', rech='Blues')
    if cmap_overrides:
        cmaps.update({k.lower(): v for k, v in cmap_overrides.items()})

    partype = partype.lower()
    if partype not in {'hk', 'vk', 'ss', 'sy', 'rech'}:
        raise ValueError('partype must be hk, vk, ss, sy or rech')

    # ── load simulation & model
    model_ws = os.fspath(model_ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, exe_name='mf6')
    gwf = sim.get_model(sim_name)
    grid = gwf.modelgrid
    idomain = grid.idomain

    nlay = gwf.dis.nlay.data
    if layers == 'all':
        layers = range(nlay)

    # ── read outline shapefile once and match CRS
    gis_dir = os.path.join('..', '..', 'gis')
    outline_fp = os.path.join(gis_dir, 'input_shps',
                              'elk', 'elk_boundary_lf.shp')
    outline = gpd.read_file(outline_fp)


    # ── fetch requested parameter array
    if partype == 'hk':
        arr3d = _to_array(gwf.npf.k)

    elif partype == 'vk':
        raw = getattr(gwf.npf, 'k33', None) or getattr(gwf.npf, 'vk', None)
        if raw is None:
            raise ValueError('Model has neither K33 nor VK defined.')
        arr3d = _to_array(raw)

    elif partype == 'ss':
        arr3d = _to_array(gwf.sto.ss)

    elif partype == 'sy':
        arr3d = _to_array(gwf.sto.sy)

    elif partype == 'rech':
        if not hasattr(gwf, 'rch'):
            raise ValueError('Model has no RCH package.')

        full = _to_array(gwf.rch.recharge)          # 2-, 3- or 4-D
        if full.ndim == 4:                          # (nper, nlay, nrow, ncol)
            arr2d = full[kstpkper[1], 0, :, :]
        elif full.ndim == 3:                        # (nper, nrow, ncol)
            arr2d = full[kstpkper[1]]
        else:                                       # steady 2-D
            arr2d = full

        arr3d = arr2d[np.newaxis, ...]              # fake 3-D for loop
        layers = [0]

    # ── output folder
    figdir = os.path.join(model_ws, out_dir)
    os.makedirs(figdir, exist_ok=True)
    cmap = mpl.colormaps.get_cmap(cmaps[partype])

    # ── loop over layers
    for k in layers:
        data = arr3d[k]
        active = idomain[min(k, idomain.shape[0]-1)] > 0
        data_m = np.ma.masked_where(~active |
                                    (data <= -1e30) | (data >= 1e30), data)

        # colour norm
        if partype in {'hk', 'vk'} and logscale_hk:
            data_m = np.ma.masked_where(data_m <= 0, data_m)
            norm = mpl.colors.LogNorm(vmin=data_m.min(), vmax=data_m.max())
        else:
            norm = mpl.colors.Normalize(vmin=data_m.min(), vmax=data_m.max())

        # ── plotting
        fig, ax = plt.subplots(figsize=(8, 7))

        # 2) parameter colour-flood
        mview = flopy.plot.PlotMapView(model=gwf, layer=k, ax=ax)
        mview.plot_array(data_m, cmap=cmap, norm=norm)
                # 1) grey outline basemap
        outline.plot(ax=ax, facecolor='none', edgecolor='0.4', linewidth=1.0)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=ax, shrink=0.8,
                     label=('Recharge (ft/d)' if partype == 'rech'
                            else partype.upper()))

        title = (f'{sim_name} – RECH (kstpkper {kstpkper})'
                 if partype == 'rech'
                 else f'{sim_name} – {partype.upper()} layer {k + 1}')
        ax.set_title(title, loc='left')
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (ft)')
        ax.set_ylabel('Northing (ft)')
        fig.tight_layout()

        fname = (f'rech_kstpkper_{kstpkper[0]}_{kstpkper[1]}.png'
                 if partype == 'rech'
                 else f'{partype}_layer_{k + 1:02d}.png')
        png = os.path.join(figdir, fname)
        fig.savefig(png, dpi=dpi,transparent=False)
        plt.close(fig)
        print('saved →', png)

def plot_simple_timeseries(m_d, modnm='elk_2lay',noptmax=None):
    pst = pyemu.Pst(os.path.join(m_d,f'{modnm}.pst'))
    #obsen = pst.ies.obsen
    noise = pst.ies.noise
    obs = pst.observation_data
    obs = obs.loc[obs.oname=='transhds',:]
    obs['datetime'] = pd.to_datetime(obs.datetime)
    nzobs = obs.loc[obs.weight>0,:]
    groups = nzobs.groupby(['grpid','k']).count().sort_values('obsnme',ascending=False).index.values
    
    fout = os.path.join(m_d,'results','figures','timeseries')
    if os.path.exists(fout) == False:
        os.makedirs(fout)
    
    itr = noptmax
    if itr is None:
        itr = pst.ies.phiactual.iteration.max()
    with PdfPages(os.path.join(fout,'timeseries.pdf')) as pdf:
        for group in groups:
            grp = group[0]
            k = group[1]
            gobs = obs.loc[(obs['grpid']==grp) & (obs['k']==k),:].copy()
            gobs.sort_values(by='datetime',inplace=True)
            fig,ax = plt.subplots(1,1,figsize=(7,7))
    
            nzobs = gobs.loc[gobs.weight>0,:]
            nvals = noise.loc[:,nzobs.obsnme].values
            dts = nzobs.datetime.values
            [ax.plot(dts,nvals[i,:],'r',alpha=0.3,lw=0.2) for i in range(nvals.shape[0])]
    
            dts = gobs.datetime.values
            ptvals = pst.ies.__getattr__('obsen{0}'.format(itr)).loc[:,gobs.obsnme].values
            # if any value in ptvals is >1500 or less than 2000 then drop the dataset
            ptvals = np.where(ptvals>980,np.nan,ptvals)
            ptvals = np.where(ptvals<-1,np.nan,ptvals)

            [ax.plot(dts,ptvals[i,:],'b',alpha=0.5,lw=0.2) for i in range(ptvals.shape[0])]
            
            ax.scatter(nzobs.datetime,nzobs.obsval,marker='^',s=50,c='r',zorder=10)
            
            ylim = ax.get_ylim()
            prvals = pst.ies.obsen0.loc[:,gobs.obsnme].values
            [ax.plot(dts,prvals[i,:],'0.5',alpha=0.5,lw=0.2) for i in range(prvals.shape[0])]
            ax.set_ylim(ylim)

            ax.set_title('site: {0} layer {1}'.format(grp, k), loc='left')
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            print(group)

def plot_simple_1to1(m_d, modnm='elk_2lay'):
    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    # noise = pst.ies.obsen
    noise = pst.ies.noise
    obs = pst.observation_data
    obs = obs.loc[obs.weight > 0, :]
    odict = obs.obsval.to_dict()
    groups = obs.obgnme.unique()
    groups.sort()
    itrs = pst.ies.phiactual.iteration.values

    outdir = os.path.join(m_d,'results','figures','one2one_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with PdfPages(os.path.join(outdir, '1to1_seq.pdf')) as pdf:
        for group in groups:
            if 'trans' in group:
                continue
            gobs = obs.loc[obs.obgnme == group]
            for itr in [itrs[0], itrs[-1]]:
                # oe = obsen.loc[obsen.index.get_level_values(0)==itr,:]
                oe = pst.ies.__getattr__('obsen{0}'.format(itr))
                base_dict = None
                if 'base' in oe.index:
                    base_dict = oe.loc['base', :].to_dict()
                    # print(base_dict)
                    # exit()

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                for oname in gobs.obsnme:
                    nvals = noise.loc[:, oname].values
                    svals = oe.loc[:, oname].values
                    if oname == gobs.obsnme.values[0]:
                        ax.scatter([odict[oname] for _ in range(nvals.shape[0])], nvals, marker='.', s=15, c='r',
                                   alpha=0.1, label='observed+noise')
                        ax.scatter([odict[oname] for _ in range(svals.shape[0])], svals, marker='.', s=15, c='b',
                                   alpha=0.1, label='simulated')
                    else:
                        ax.scatter([odict[oname] for _ in range(nvals.shape[0])], nvals, marker='.', s=15, c='r',
                                   alpha=0.1, )
                        ax.scatter([odict[oname] for _ in range(svals.shape[0])], svals, marker='.', s=15, c='b',
                                   alpha=0.1)
                    if base_dict is not None:
                        # ax.scatter(odict[oname],oe.loc['base',oname],marker='^',s=60,c='b')
                        if oname == gobs.obsnme.values[0]:
                            ax.scatter(odict[oname], base_dict[oname], marker='^', s=50, c='b',
                                       label='base realization')
                        else:
                            ax.scatter(odict[oname], base_dict[oname], marker='^', s=50, c='b', )
                for ax in fig.axes:
                    for col in ax.collections:  # scatter returns collections
                        col.set_rasterized(True)
                ax.set_title('iterationL: {0} group:{1}'.format(itr, group), loc='left')
                mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
                mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([mn, mx], [mn, mx], 'k--', lw=3.0)
                ax.set_ylim(mn, mx)
                ax.set_xlim(mn, mx)
                ax.grid()
                ax.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)

    with PdfPages(os.path.join(outdir, '1to1_grouped_seq.pdf')) as pdf:
        for group in ['pmp', 'sshds']:
            if group == 'pmp':
                gobs = obs.loc[obs.obgnme.str.contains('pmp'), :]
            else:
                gobs = obs.loc[~obs.obgnme.str.contains('pmp|trans'), :]
            for itr in [itrs[0], itrs[-1]]:
                # oe = obsen.loc[obsen.index.get_level_values(0)==itr,:]
                oe = pst.ies.__getattr__('obsen{0}'.format(itr))
                base_dict = None
                if 'base' in oe.index:
                    base_dict = oe.loc['base', :].to_dict()
                    # print(base_dict)
                    # exit()

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                for oname in gobs.obsnme:
                    nvals = noise.loc[:, oname].values
                    svals = oe.loc[:, oname].values
                    if oname == gobs.obsnme.values[0]:
                        ax.scatter([odict[oname] for _ in range(nvals.shape[0])], nvals, marker='.', s=15, c='r',
                                   alpha=0.1, label='observed+noise')
                        ax.scatter([odict[oname] for _ in range(svals.shape[0])], svals, marker='.', s=15, c='b',
                                   alpha=0.1, label='simulated')
                    else:
                        ax.scatter([odict[oname] for _ in range(nvals.shape[0])], nvals, marker='.', s=15, c='r',
                                   alpha=0.1, )
                        ax.scatter([odict[oname] for _ in range(svals.shape[0])], svals, marker='.', s=15, c='b',
                                   alpha=0.1)
                    if base_dict is not None:
                        # ax.scatter(odict[oname],oe.loc['base',oname],marker='^',s=60,c='b')
                        if oname == gobs.obsnme.values[0]:
                            ax.scatter(odict[oname], base_dict[oname], marker='^', s=50, c='b',
                                       label='base realization')
                        else:
                            ax.scatter(odict[oname], base_dict[oname], marker='^', s=50, c='b', )
                if group == 'pmp':
                    ax.set_title('iterationL: {0} group:SS pumping'.format(itr), loc='left')
                else:
                    ax.set_title('iterationL: {0} group:SS predevelopment'.format(itr), loc='left')
                
                for ax in fig.axes:
                    for col in ax.collections:  # scatter returns collections
                        col.set_rasterized(True)
                mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
                mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([mn, mx], [mn, mx], 'k--', lw=3.0)
                ax.set_ylim(mn, mx)
                ax.set_xlim(mn, mx)
                ax.grid()
                ax.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)

def plot_simple_par_histo(m_d, modnm='elk_2lay'):
    pst = pyemu.Pst(os.path.join(m_d,f'{modnm}.pst'))
    par = pst.parameter_data
    adjpar = par.loc[par.partrans.apply(lambda x: x in ['none','log']),:]
    groups = adjpar.pargp.unique()
    groups.sort()
    pr = pst.ies.paren0

    fout = os.path.join(m_d,'results','figures','param_distribs')
    if os.path.exists(fout) == False:
        os.makedirs(fout)
    itrs = pst.ies.phiactual.iteration
    with PdfPages(os.path.join(fout,'simple_par_histo.pdf')) as pdf:

        #for pname in pst.adj_par_names:
        for group in groups:
            gpar = adjpar.loc[adjpar.pargp==group,:].copy()

            lb = gpar.parlbnd.min()
            ub = gpar.parubnd.max()
            if gpar.partrans.iloc[0] != 'none':
                lb = np.log10(lb)
                ub = np.log10(ub)
            for itr in itrs[1:]:
                pt = pst.ies.__getattr__('paren{0}'.format(itr))
                fig,ax = plt.subplots(1,1,figsize=(6,6))
                if gpar.partrans.iloc[0] == 'none':
                    ax.hist(pr.loc[:,gpar.parnme].values.flatten(),bins=20,facecolor='0.5',edgecolor='none',alpha=0.5,density=True)
                    ax.hist(pt.loc[:,gpar.parnme].values.flatten(),bins=20,facecolor='b',edgecolor='none',alpha=0.5,density=True)
                    ax.set_xlabel('')

                else:
                    ax.hist(np.log10(pr.loc[:,gpar.parnme].values.flatten()),bins=20,facecolor='0.5',edgecolor='none',alpha=0.5,density=True)
                    ax.hist(np.log10(pt.loc[:,gpar.parnme].values.flatten()),bins=20,facecolor='b',edgecolor='none',alpha=0.5,density=True)
                    ax.set_xlabel('$log_{10}$')
                ylim = ax.get_ylim()
                ax.plot([lb,lb],ylim,'k--',lw=3)
                ax.plot([ub,ub],ylim,'k--',lw=3)
                ax.set_ylim(ylim)
                ax.set_title('iteration:{0} pname:{1}, npar:{2}'.format(itr,group,gpar.shape[0]),loc='left')
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                print(itr,group)

def plot_array_histo(m_d, modnm='elk_2lay', partype='k',
                     noptmax=None, logbool=True):
    import os
    import numpy as np
    import flopy
    import pyemu
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # -------------------------
    # DEBUG HEADER
    # -------------------------
    print("\n" + "-" * 80)
    print(f"[plot_array_histo] START")
    print(f"[plot_array_histo] m_d     = {m_d}")
    print(f"[plot_array_histo] modnm   = {modnm}")
    print(f"[plot_array_histo] partype = {partype}")
    print(f"[plot_array_histo] noptmax = {noptmax}")
    print(f"[plot_array_histo] logbool = {logbool}")

    # ------- Load or define all independent variables ------- #
    print("[plot_array_histo] loading simulation/packages...")
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=m_d,
        exe_name='mf6',
        load_only=['dis', 'npf', 'sto', 'chd', 'obs'],
        verbosity_level=0,
    )

    m = flopy.mf6.MFSimulation().load(sim_ws=m_d, load_only=['npf', 'sto', 'chd', 'obs']).get_model()
    idom = m.dis.idomain.array
    idom[idom < 0] = 0
    print(f"[plot_array_histo] idomain shape = {idom.shape}, active cells = {int((idom > 0).sum())}")

    # ------- PEST objects / groups ------- #
    pst_path = os.path.join(m_d, f'{modnm}.pst')
    print(f"[plot_array_histo] pst path = {pst_path} (exists={os.path.exists(pst_path)})")
    pst = pyemu.Pst(pst_path)

    obs = pst.observation_data.copy()
    obgnmes = obs.obgnme.unique()
    print(f"[plot_array_histo] obs count = {obs.shape[0]}, unique obgnme = {len(obgnmes)}")

    # show a few example groups (helps verify naming)
    ex = list(obgnmes[:10])
    print(f"[plot_array_histo] example obgnme (first 10): {ex}")

    groups = [group for group in obgnmes if group.startswith(f'{partype}_k:')]
    print(f"[plot_array_histo] looking for prefix '{partype}_k:' -> groups_found={len(groups)}")
    if len(groups) > 0:
        print(f"[plot_array_histo] first matching group: {groups[0]}")
    else:
        print(f"[plot_array_histo] WARNING: no matching groups for partype='{partype}'. "
              f"This will produce an empty PDF unless you skip/return.")

    # ------- zones ------- #
    zon_arr = idom.copy()
    zon_arr[zon_arr == 0] = -9999
    # NOTE: assumes exactly 2 layers; keep as-is for now
    zon_arr[0, :, :] = np.where(zon_arr[0, :, :] > 0, 0, zon_arr[0, :, :])  # layer 1
    zon_arr[1, :, :] = np.where(zon_arr[1, :, :] > 0, 1, zon_arr[1, :, :])  # layer 2

    # ------- dictionaries ------- #
    zon_dict = {0: 'Soils/Clay/Silt',
                1: 'Elk Valley Aquifer'}
    par_dict = {'k33': ['Anisotropy Ratio', '[-]'],
                'k': ['Horizontal Hydraulic Conductivity', '[ft/d]'],
                'ss': ['Specific Storage', '[1/ft]'],
                'sy': ['Specific Yield', '[-]']}

    # ------- ensembles ------- #
    prior_path = os.path.join(m_d, f'{modnm}.0.obs.jcb')
    print(f"[plot_array_histo] prior ensemble path = {prior_path} (exists={os.path.exists(prior_path)})")
    prior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=prior_path)._df
    print(f"[plot_array_histo] prior df shape = {prior.shape}")

    posterior = None
    post_path = None
    if noptmax:
        post_path = os.path.join(m_d, f'{modnm}.{noptmax}.obs.jcb')
        print(f"[plot_array_histo] posterior ensemble path = {post_path} (exists={os.path.exists(post_path)})")
        posterior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=post_path)._df
        print(f"[plot_array_histo] posterior df shape = {posterior.shape}")
    else:
        print("[plot_array_histo] noptmax not provided -> posterior will not be plotted")

    # ------- output paths ------- #
    SAVEDIR = os.path.join(m_d, 'results', 'figures', 'param_distribs')
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
        print(f"[plot_array_histo] created SAVEDIR: {SAVEDIR}")
    else:
        print(f"[plot_array_histo] SAVEDIR exists: {SAVEDIR}")

    if logbool:
        pdf_path = os.path.join(SAVEDIR, f'{partype}_log10_distribs.pdf')
    else:
        pdf_path = os.path.join(SAVEDIR, f'{partype}_distribs.pdf')

    print(f"[plot_array_histo] pdf_path = {pdf_path}")
    pdf = PdfPages(pdf_path)

    # counters for debugging
    n_fig_saved = 0
    n_png_saved = 0

    # ------- main loop ------- #
    for idx, group in enumerate(groups):
        print(f"[plot_array_histo] --- layer_index={idx} group='{group}' ---")
        zones = zon_arr[idx]
        uniq_z = np.unique(zones)
        print(f"[plot_array_histo] unique zones (incl nodata) = {uniq_z}")

        # reorder columns
        cols = []
        for i in range(m.dis.nrow.data):
            for j in range(m.dis.ncol.data):
                cols.append(f'oname:{group}_otype:arr_i:{i}_j:{j}')

        # verify columns exist in ensemble
        missing = [c for c in cols[:50] if c not in prior.columns]  # sample-check first 50 for speed
        if missing:
            print(f"[plot_array_histo] WARNING: some expected columns not found in prior (sample of first 50). "
                  f"Example missing: {missing[:3]}")
        else:
            print(f"[plot_array_histo] columns look present (sample-check first 50)")

        try:
            pr_ = prior.loc[:, cols]
        except Exception as e:
            print(f"[plot_array_histo] ERROR selecting prior columns for group '{group}': {e}")
            continue

        pt_ = None
        if posterior is not None:
            try:
                pt_ = posterior.loc[:, cols]
            except Exception as e:
                print(f"[plot_array_histo] ERROR selecting posterior columns for group '{group}': {e}")
                pt_ = None

        # base reshape
        try:
            pr_base = pr_.loc['base', :].values.reshape((m.dis.nrow.data, m.dis.ncol.data)) * idom[idx]
            pt_base = (pt_.loc['base', :].values.reshape((m.dis.nrow.data, m.dis.ncol.data)) * idom[idx]) if pt_ is not None else None
        except Exception as e:
            print(f"[plot_array_histo] WARNING: could not reshape base for group '{group}': {e}")
            pr_base = np.nan
            pt_base = np.nan

        # ensemble arrays
        pr_arr = pr_.values.reshape((pr_.shape[0], m.dis.nrow.data, m.dis.ncol.data)) * idom[idx]
        pt_arr = (pt_.values.reshape((pt_.shape[0], m.dis.nrow.data, m.dis.ncol.data)) * idom[idx]) if pt_ is not None else None

        for zone in np.unique(zones):
            if zone < -1:
                continue

            zone_arr = np.where(zones == zone, 1, 0)
            pr_zn = pr_arr * zone_arr
            pt_zn = pt_arr * zone_arr if pt_arr is not None else None

            prb_zn = pr_base * zone_arr
            ptb_zn = pt_base * zone_arr if pt_arr is not None else None

            pr_zn[pr_zn == 0] = np.nan
            prb_zn[prb_zn == 0] = np.nan
            if pt_zn is not None:
                pt_zn[pt_zn == 0] = np.nan
                ptb_zn[ptb_zn == 0] = np.nan

            # flatten + basic sanity
            v_pr = pr_zn.flatten()
            n_nans = int(np.isnan(v_pr).sum())
            print(f"[plot_array_histo]   zone={zone} prior values: n={v_pr.size}, n_nan={n_nans}, "
                  f"nanmean={np.nanmean(v_pr):.3e}")

            if pt_zn is not None:
                v_pt = pt_zn.flatten()
                n_nans_pt = int(np.isnan(v_pt).sum())
                print(f"[plot_array_histo]   zone={zone} post  values: n={v_pt.size}, n_nan={n_nans_pt}, "
                      f"nanmean={np.nanmean(v_pt):.3e}")

            # Create histogram
            fig, ax = plt.subplots(dpi=200)

            ax.hist(
                v_pr,
                color='k',
                edgecolor='k',
                bins=25,
                alpha=0.15,
                weights=np.zeros_like(v_pr) + 1. / v_pr.size,
                label='prior'
            )
            ax.axvline(np.nanmean(prb_zn), color='k', linestyle='--', linewidth=1, label='base prior mean')

            if pt_zn is not None:
                v_pt = pt_zn.flatten()
                ax.hist(
                    v_pt,
                    color='b',
                    edgecolor='b',
                    bins=25,
                    alpha=0.5,
                    weights=np.zeros_like(v_pt) + 1. / v_pt.size,
                    label='posterior'
                )
                ax.axvline(np.nanmean(ptb_zn), color='b', linestyle='--', linewidth=1, label='base posterior mean')

            if logbool:
                ax.set_xscale('log')

            # titles/labels (guard if partype not in dict)
            if partype in par_dict:
                ptitle = par_dict[partype][0]
                punit = par_dict[partype][1]
            else:
                ptitle = partype
                punit = ""

            zname = zon_dict.get(zone, str(zone))

            ax.set_title(
                f'{ptitle} Layer: {idx + 1}, Zone: {zname}\n'
                f'Prior Base Mean:{np.nanmean(prb_zn):.2e}\n'
                f'Posterior Base Mean:{np.nanmean(ptb_zn):.2e}\n',
                fontsize=10
            )
            ax.set_ylabel('Relative Frequency', fontsize=12)
            ax.set_xlabel(f'{ptitle} {punit}', fontsize=12)
            ax.grid()
            ax.legend(fontsize=10)
            plt.tight_layout()

            # save png + pdf
            if logbool:
                png_path = os.path.join(SAVEDIR, f'{partype}_log10_lay{idx + 1}_zn{zone}.png')
            else:
                png_path = os.path.join(SAVEDIR, f'{partype}_lay{idx + 1}_zn{zone}.png')

            plt.savefig(png_path)
            n_png_saved += 1
            pdf.savefig(fig)
            n_fig_saved += 1
            plt.close(fig)

            if n_fig_saved % 10 == 0:
                print(f"[plot_array_histo] saved {n_fig_saved} figures so far...")

    # close pdf
    pdf.close()
    print(f"[plot_array_histo] DONE. figs_saved={n_fig_saved}, pngs_saved={n_png_saved}")
    print(f"[plot_array_histo] pdf exists? {os.path.exists(pdf_path)}")
    if os.path.exists(pdf_path):
        print(f"[plot_array_histo] pdf size (bytes) = {os.path.getsize(pdf_path)}")
    print("-" * 80)

def plot_all_par_histo(
    m_d,
    modnm="elk_2lay",
    noptmax=None,
    log_mode="auto",          # "auto" | True | False
    bins=40,
    max_pars_per_pdf=250,     # keep PDFs from getting insane
    out_subdir=("results", "figures", "param_distribs_all"),
    group_by="pargp",         # "pargp" | "none"
    include_tied=False,
    include_fixed=False,
    exclude_regex=None,       # e.g. r"^DUM|junk"
    verbose=True,
):
    """
    Plot prior vs posterior histograms for ALL PEST parameters using parameter ensembles.

    Reads:
      - <modnm>.0.par.jcb
      - <modnm>.<noptmax>.par.jcb  (if noptmax is not None)

    Outputs:
      - PDFs (one per group or one total) + optional per-parameter PNGs if you add that later.

    Notes:
      - Uses log10 x-scale for parameters that look log-distributed (auto) or if log_mode=True.
      - Skips parameters not present in the ensemble columns.
      - Skips tied/fixed parameters unless requested.
    """
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pyemu

    def _is_log_candidate(series):
        """Heuristic: positive-only and spans >= ~2 orders of magnitude."""
        x = series.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size < 10:
            return False
        q1, q99 = np.nanpercentile(x, [1, 99])
        if q1 <= 0 or q99 <= 0:
            return False
        return (q99 / q1) >= 100.0  # ~2 orders of magnitude

    # -------------------------
    # Setup + load PST
    # -------------------------
    pst_path = os.path.join(m_d, f"{modnm}.pst")
    if not os.path.exists(pst_path):
        raise FileNotFoundError(f"Missing pst file: {pst_path}")
    pst = pyemu.Pst(pst_path)
    par = pst.parameter_data.copy()

    # filters
    if not include_tied and "partied" in par.columns:
        par = par.loc[par.partied == False, :]
    if not include_fixed and "partrans" in par.columns:
        # fixed typically means not estimated; depends on your convention
        par = par.loc[par.partrans.str.lower() != "fixed", :]

    if exclude_regex:
        rx = re.compile(exclude_regex)
        par = par.loc[~par.parnme.apply(lambda s: bool(rx.search(s))), :]

    parnmes = par.parnme.tolist()
    if verbose:
        print("\n" + "-" * 80)
        print("[plot_all_par_histo] START")
        print(f"[plot_all_par_histo] m_d     = {m_d}")
        print(f"[plot_all_par_histo] modnm   = {modnm}")
        print(f"[plot_all_par_histo] noptmax = {noptmax}")
        print(f"[plot_all_par_histo] candidates after filters = {len(parnmes)}")

    # -------------------------
    # Load parameter ensembles
    # -------------------------
    prior_path = os.path.join(m_d, f"{modnm}.0.par.jcb")
    if not os.path.exists(prior_path):
        raise FileNotFoundError(
            f"Missing prior parameter ensemble: {prior_path}\n"
            f"Hint: you may be pointing at obs ensembles (*.obs.jcb). This function needs *.par.jcb."
        )
    prior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=prior_path)._df

    posterior = None
    post_path = None
    if noptmax is not None:
        post_path = os.path.join(m_d, f"{modnm}.{noptmax}.par.jcb")
        if not os.path.exists(post_path):
            raise FileNotFoundError(f"Missing posterior parameter ensemble: {post_path}")
        posterior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=post_path)._df

    if verbose:
        print(f"[plot_all_par_histo] prior df shape = {prior.shape}")
        if posterior is not None:
            print(f"[plot_all_par_histo] post  df shape = {posterior.shape}")

    # Keep only parameters that actually exist in ensemble columns
    parnmes_in = [p for p in parnmes if p in prior.columns]
    missing = sorted(set(parnmes) - set(parnmes_in))
    if verbose:
        print(f"[plot_all_par_histo] in prior columns = {len(parnmes_in)}")
        if missing:
            print(f"[plot_all_par_histo] WARNING missing in prior columns (showing up to 15): {missing[:15]}")

    if len(parnmes_in) == 0:
        raise ValueError("No parameters found in the prior ensemble columns after filtering.")

    # -------------------------
    # Output directory + grouping
    # -------------------------
    out_dir = os.path.join(m_d, *out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if group_by == "pargp" and "pargp" in par.columns:
        groups = {}
        for p in parnmes_in:
            g = par.loc[p, "pargp"]
            groups.setdefault(g, []).append(p)
    else:
        groups = {"all_parameters": parnmes_in}

    if verbose:
        print(f"[plot_all_par_histo] out_dir = {out_dir}")
        print(f"[plot_all_par_histo] groups  = {len(groups)}")
        print("-" * 80)

    # -------------------------
    # Plotting
    # -------------------------
    for gname, plist in groups.items():
        # split into chunks so a single PDF doesn't get too huge
        chunks = [plist[i:i + max_pars_per_pdf] for i in range(0, len(plist), max_pars_per_pdf)]

        for ci, chunk in enumerate(chunks, start=1):
            pdf_name = f"par_hists__{gname}__chunk{ci:02d}.pdf" if len(chunks) > 1 else f"par_hists__{gname}.pdf"
            pdf_path = os.path.join(out_dir, pdf_name)

            if verbose:
                print(f"[plot_all_par_histo] writing {pdf_path}  (npar={len(chunk)})")

            figs_saved = 0
            with PdfPages(pdf_path) as pdf:
                for p in chunk:
                    pr = prior[p].astype(float)
                    pt = posterior[p].astype(float) if posterior is not None and p in posterior.columns else None

                    # choose log scaling
                    if log_mode == "auto":
                        use_log = _is_log_candidate(pr) or (_is_log_candidate(pt) if pt is not None else False)
                    elif isinstance(log_mode, bool):
                        use_log = log_mode
                    else:
                        use_log = False

                    # basic stats for title
                    pr_base = np.nan
                    if "base" in prior.index:
                        try:
                            pr_base = float(prior.loc["base", p])
                        except Exception:
                            pr_base = np.nan

                    pt_base = np.nan
                    if pt is not None and "base" in posterior.index:
                        try:
                            pt_base = float(posterior.loc["base", p])
                        except Exception:
                            pt_base = np.nan

                    # plot
                    fig, ax = plt.subplots(dpi=200)

                    pr_vals = pr.to_numpy()
                    pr_vals = pr_vals[np.isfinite(pr_vals)]

                    # If log, must be positive
                    if use_log:
                        pr_plot = pr_vals[pr_vals > 0]
                    else:
                        pr_plot = pr_vals

                    if pr_plot.size == 0:
                        plt.close(fig)
                        continue

                    ax.hist(
                        pr_plot,
                        bins=bins,
                        alpha=0.20,
                        weights=np.zeros_like(pr_plot) + 1.0 / pr_plot.size,
                        label="prior"
                    )

                    if np.isfinite(pr_base) and (not use_log or pr_base > 0):
                        ax.axvline(pr_base, linestyle="--", linewidth=1, label="base prior")

                    if pt is not None:
                        pt_vals = pt.to_numpy()
                        pt_vals = pt_vals[np.isfinite(pt_vals)]
                        pt_plot = pt_vals[pt_vals > 0] if use_log else pt_vals

                        if pt_plot.size > 0:
                            ax.hist(
                                pt_plot,
                                bins=bins,
                                alpha=0.45,
                                weights=np.zeros_like(pt_plot) + 1.0 / pt_plot.size,
                                label="posterior"
                            )
                            if np.isfinite(pt_base) and (not use_log or pt_base > 0):
                                ax.axvline(pt_base, linestyle="--", linewidth=1, label="base posterior")

                    if use_log:
                        ax.set_xscale("log")

                    # nice title labels if available
                    row = pst.parameter_data.loc[p] if p in pst.parameter_data.index else None
                    pargp = row.pargp if row is not None and "pargp" in row.index else ""
                    partrans = row.partrans if row is not None and "partrans" in row.index else ""

                    ax.set_title(f"{p}   (pargp={pargp}, trans={partrans})", fontsize=10)
                    ax.set_ylabel("Relative Frequency")
                    ax.set_xlabel("Parameter value")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=9)
                    plt.tight_layout()

                    pdf.savefig(fig)
                    figs_saved += 1
                    plt.close(fig)

            if verbose:
                print(f"[plot_all_par_histo] saved pages = {figs_saved}  exists={os.path.exists(pdf_path)}  "
                      f"size={os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'NA'}")

    if verbose:
        print("[plot_all_par_histo] DONE\n")


def plot_simple_par_maps(m_d,modnm,noptmax=None):
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis','npf', 'sto', 'chd', 'obs'],)
    m = flopy.mf6.MFSimulation().load(sim_ws=m_d, load_only=['npf', 'sto', 'chd', 'obs']).get_model()  # for base values
    nrow = m.dis.nrow.data
    ncol = m.dis.ncol.data
    idom = m.dis.idomain.array
    idom[idom<0] = 0

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data
    props = ['k','k33','ss','sy']
    aobs = obs.loc[obs.oname.apply(lambda x: x in props),:].copy()
    for col in ['k','i','j']:
        aobs[col] = aobs[col].astype(int)
    pr = pst.ies.obsen0
    pr.index = pr.index.map(str)
    if noptmax is None:
        noptmax = pst.ies.phiactual.iteration.max()
    pt = pst.ies.__getattr__('obsen{0}'.format(noptmax))
    pt.index = pt.index.map(str)
    
    reals_to_plot = []
    
    for i in range(4):
        reals_to_plot.append(pt.index[i])
    if 'base' in pt.index and 'base' not in reals_to_plot:
        reals_to_plot.append('base')
    cmap = plt.get_cmap('magma')
    cmap.set_bad('k')
    with PdfPages(os.path.join(m_d,'simple_prop_maps.pdf')) as pdf:
        for prop in props:
            pobs = aobs.loc[aobs.oname==prop,:].copy()
            assert pobs.shape[0] > 0
            uks = pobs.k.unique()
            uks.sort()
            for k in uks:
                kobs = pobs.loc[pobs.k==k,:]

                prarr = np.zeros((nrow,ncol))
                prarr[kobs.i,kobs.j] = np.log10(pr.loc[:,kobs.obsnme].values).mean(axis=0)
                ptarr = np.zeros((nrow,ncol))
                ptarr[kobs.i,kobs.j] = np.log10(pt.loc[:,kobs.obsnme].values).mean(axis=0)
                
                prarr[idom[k,:,:]==0] = np.nan
                ptarr[idom[k,:,:]==0] = np.nan
                mx = max(np.nanmax(prarr),np.nanmax(ptarr))
                mn = min(np.nanmin(prarr),np.nanmin(ptarr))
                
                fig,axes = plt.subplots(1,2,figsize=(10,5))
                cb = axes[0].imshow(prarr,cmap=cmap,vmin=mn,vmax=mx)
                plt.colorbar(cb,ax=axes[0])
                cb = axes[1].imshow(ptarr,cmap=cmap,vmin=mn,vmax=mx)
                plt.colorbar(cb,ax=axes[1])
                axes[0].set_title('prior prop:{0} layer{1} mean'.format(prop,k+1),loc='left')
                axes[1].set_title('post prop:{0} lay:{1} mean'.format(prop,k+1),loc='left')
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                

                prarr = np.zeros((nrow,ncol))
                prarr[kobs.i,kobs.j] = np.log10(pr.loc[:,kobs.obsnme].values).std(axis=0)
                ptarr = np.zeros((nrow,ncol))
                ptarr[kobs.i,kobs.j] = np.log10(pt.loc[:,kobs.obsnme].values).std(axis=0)
                
                prarr[idom[k,:,:]==0] = np.nan
                ptarr[idom[k,:,:]==0] = np.nan
                mx = max(np.nanmax(prarr),np.nanmax(ptarr))
                mn = min(np.nanmin(prarr),np.nanmin(ptarr))
                
                fig,axes = plt.subplots(1,2,figsize=(10,5))
                cb = axes[0].imshow(prarr,cmap=cmap,vmin=mn,vmax=mx)
                plt.colorbar(cb,ax=axes[0])
                cb = axes[1].imshow(ptarr,cmap=cmap,vmin=mn,vmax=mx)
                plt.colorbar(cb,ax=axes[1])
                axes[0].set_title('prior prop:{0} lay:{1} std'.format(prop,k+1),loc='left')
                axes[1].set_title('post prop:{0} lay:{1} std'.format(prop,k+1),loc='left')
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)



                for real in reals_to_plot:
                    prarr = np.zeros((nrow,ncol))
                    prarr[kobs.i,kobs.j] = np.log10(pr.loc[real,kobs.obsnme].values)
                    ptarr = np.zeros((nrow,ncol))
                    ptarr[kobs.i,kobs.j] = np.log10(pt.loc[real,kobs.obsnme].values)
                    
                    prarr[idom[k,:,:]==0] = np.nan
                    ptarr[idom[k,:,:]==0] = np.nan
                    mx = max(np.nanmax(prarr),np.nanmax(ptarr))
                    mn = min(np.nanmin(prarr),np.nanmin(ptarr))
                    
                    fig,axes = plt.subplots(1,2,figsize=(10,5))
                    cb = axes[0].imshow(prarr,cmap=cmap,vmin=mn,vmax=mx)
                    plt.colorbar(cb,ax=axes[0])
                    cb = axes[1].imshow(ptarr,cmap=cmap,vmin=mn,vmax=mx)
                    plt.colorbar(cb,ax=axes[1])
                    axes[0].set_title('prior prop:{0} lay:{2} rl:{1}'.format(prop,real,k+1),loc='left')
                    axes[1].set_title('post prop:{0} lay:{2} rl:{1}'.format(prop,real,k+1),loc='left')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(prop,k,real)

def xsection_comp_pr_pt(m_d, modnm, noptmax=4, partype = 'k'):
    import matplotlib.cm as cm
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis','npf', 'sto', 'chd', 'obs'],
                                      verbosity_level=0,)
    m = sim.get_model(modnm)  # for base values
    nrow = m.dis.nrow.data
    ncol = m.dis.ncol.data
    idom = m.dis.idomain.array
    idom[idom<0] = 0

    par_dict = {'k33': ['Anisotropy Ratio', '[-]'],
                'k': ['Horizontal Hydraulic Conductivity', '[ft/d]'],
                'ss': ['Specific Storage', '[1/ft]'],
                'sy': ['Specific Yield','[-]']}

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data

    pr = pst.ies.obsen0
    pr.index = pr.index.map(str)

    if noptmax is None:
        noptmax = pst.ies.phiactual.iteration.max()
    pt = pst.ies.__getattr__('obsen{0}'.format(noptmax))
    pt.index = pt.index.map(str)


    # reorder columns
    cols = []
    for k in range(m.dis.nlay.data):
        for i in range(m.dis.nrow.data):
            for j in range(m.dis.ncol.data):
                cols.append(f'oname:{partype}_k:{str(k).zfill(3)}_otype:arr_i:{i}_j:{j}')

    pr_b = pr.loc['base', cols]
    pt_b = pt.loc['base', cols] if noptmax else None

    pr_b = pr_b.values.reshape((m.dis.nlay.data, m.dis.nrow.data, m.dis.ncol.data)) * idom
    pr_b[pr_b == 0] = np.nan  # set zeroes to NaN for histogram
    pt_b = pt_b.values.reshape((m.dis.nlay.data, m.dis.nrow.data, m.dis.ncol.data)) * idom if pt_b is not None else None
    pt_b[pt_b == 0] = np.nan  # set zeroes to NaN for histogram

    # updated the modelgrid with the rotation and offset
    grid = m.modelgrid

    fpth = os.path.join('..', '..', 'gis', 'input_shps', 'elk_2lay', 'xsection_lines_pr.shp')
    line = flopy.plot.plotutil.shapefile_get_vertices(fpth)

    xsect_gdf = gpd.read_file(fpth)


    # plot x section
    # Cross Section Dictionary
    xsect_dict = {"South to North A-A'": np.array([(2960432, 245071), (2900832, 315598)]),}

    output_dir = os.path.join(m_d, 'results', 'figures', 'xsections',)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    vmin = np.nanmin([pr_b, pt_b])
    vmax = np.nanmax([pr_b, pt_b])
    for i, name in enumerate(xsect_dict.keys()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # plot prior
        xsect = flopy.plot.PlotCrossSection(model=m, line={'line': line[0]}, geographic_coords=True, ax=ax1)

        cmaps = cm.get_cmap('jet')
        csa_ = xsect.plot_array(pr_b, cmap=cmaps, alpha=0.5, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))

        cb = plt.colorbar(csa_, shrink=0.75)
        # cb.locator = LogLocator(base=10)
        cb.update_ticks()
        cb.set_label(f'{par_dict[partype][0]} {par_dict[partype][1]})', fontsize=10)
        ax1.set_title(f'Prior', fontsize=12)
        # linecollection = xsect.plot_grid()
        patches = xsect.plot_ibound(color_noflow='white')

        # plot posterior
        xsect = flopy.plot.PlotCrossSection(model=m, line={'line': line[0]}, geographic_coords=True, ax=ax2)

        cmaps = cm.get_cmap('jet')
        csa_ = xsect.plot_array(pt_b, cmap=cmaps, alpha=0.5, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))

        cb = plt.colorbar(csa_, shrink=0.75)
        # cb.locator = LogLocator(base=10)
        cb.update_ticks()
        cb.set_label(f'{par_dict[partype][0]} {par_dict[partype][1]}', fontsize=10)
        ax2.set_title(f'Posterior', fontsize=12)

        # linecollection = xsect.plot_grid()
        patches = xsect.plot_ibound(color_noflow='white')

        plt.suptitle(f'{par_dict[partype][0]} - {name}', fontsize=16)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'{partype}_{name}.png'))
        # plt.savefig(
        #     os.path.join(output_dir,  f'Mg_year{peak_year}_{source.upper()}_{name}.svg'))
        plt.close(fig)

def plot_pr(modnm):
    m_d = 'template_flow_gwv_sspmp_highdim_pmpun'

    pst = pyemu.Pst(os.path.join(m_d,f'{modnm}.pst'))
    if pst.control_data.noptmax < 0:
        itrs = [0]
    else:
        itrs = range(pst.control_data.noptmax+1)

    for i in itrs:
        jcbName = os.path.join(m_d,f'{modnm}.obs+noise.jcb')
        if os.path.exists(jcbName):
            print(f'loading itr {i}')
            jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
            obs_en = jcb

    obs = pd.read_csv(os.path.join(m_d, 'listbudget_flx_obs.csv'), index_col=0)
    obs['datetime'] = pd.to_datetime(obs.index, format='%Y%m%d')

    use_ins = [x for x in obs.columns if 'in' in x]
    use_rej = [x for x in obs.columns if 'rej' in x]
    use_simins = [x for x in obs.columns if 'simin' in x]
    use_types = [x.split('-')[0] for x in use_simins]
    
    use_ins_en = [x for x in obs_en.columns if any([y in x for y in use_ins])]
    
    obs_en = obs_en.loc[:, use_ins_en].copy()
    obs_en = pd.DataFrame.from_records((obs_en.to_records()))
    obs_en.index = obs_en.pop('index')
    #convert to acre-feet from cubic feet per year
    obs_en *= -0.00837926
    
    clr = '0.5'
    
    ut_dict = {'car': 'Cargill Inc.', 'malt': 'Froedtert Malt Corp.', 'cow': 'City of Wahpeton', 'minn': 'Minn-Dak Farmers Cooperative', 'cob': 'City of Breckenridge'}
    
    with PdfPages(os.path.join(f'{modnm}_pr_pumping.pdf')) as pdf:

        ax_per_page = 6
        ncols = 2
        fig, axes = plt.subplots(int(ax_per_page / ncols), ncols, figsize=(8.5, 11), dpi=300)
        ax_count = 0
        pg_count = 0
        plt_count = 0        
        
        for ut in use_types:
            utnm = ut_dict[ut]
            #if utnm == 'Mining':
            #    print(babsd)
            ax = axes.flat[ax_count]
            cols =  [x for x in obs_en.columns if (f':{ut}-' in x) and ('simin' in x)]
            en_grp = obs_en.loc[:, cols].copy()
            en_dts = pd.to_datetime(en_grp.columns.str.split(':').str[-1], format='%Y%m%d')
            en_dts = pd.DataFrame(en_dts, columns=['datetime'])
            en_dts = en_dts.sort_values(by='datetime')
            en_grp_cols = en_grp.columns
            en_grp_df = pd.DataFrame(en_grp_cols, columns=['cols'])
            en_grp_df = en_grp_df.iloc[en_dts.index]
            en_grp_df['datetime'] = en_dts['datetime'].values
            
            en_grp = obs_en.loc[:, en_grp_df.cols].copy()
            
            # plot range of ensemble simulated values
      
            ax.fill_between(en_grp_df.datetime,en_grp.min(axis=0).values,
                            en_grp.max(axis=0).values, facecolor=clr, alpha=0.5, label='Ensemble range')
            if 'base' in en_grp.index:
                ax.plot(en_dts, en_grp.loc['base', :], color='black', lw=1.25, label='Base realization')
            
            # plot original model input
            #ax.plot(wdf.loc[:, 'datetime'], wdf.loc[:, grp1], color='k', lw=0.5, ls='--')
                
            # set x axis formatting
            ax.xaxis.set_major_locator(years20)
            ax.xaxis.set_major_formatter(years_fmt)
            #ax.set_xlim(xmin, xmax)     
            
            # comma format y axis
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
                  
            ax.set_title(f'{utnm} pumping ', loc='left')
            
            if plt_count == 0:
                ax.legend(loc='upper left')
            
            ax_count += 1
            plt_count += 1
            
            if ax_count % 2 != 0:
                ax.set_ylabel('Pumping (acre-feet)')

        plt.tight_layout()
        pdf.savefig()
        plt.savefig(os.path.join(o_d, 'wahp_water_use.png'), dpi=300)
        plt.close(fig)


def plot_rech_initial_vs_base_mel(
    model_ws_initial: Union[str, os.PathLike],
    model_ws_base: Union[str, os.PathLike],
    sim_name: str = "elk_2lay",
    out_dir: str = "fig_rech_compare",
    max_kper: Optional[int] = None,
    dpi: int = 300,
) -> None:
    """
    Compare recharge between an initial model and a base/posterior model.

    For each stress period kper:
      - Left panel : initial recharge (ft/d)
      - Right panel: base/posterior recharge (ft/d)
      Both panels share a colour scale based on the combined data.

    Output:
      PNGs written to: <model_ws_base>/<out_dir>/
        → rech_compare_sp0000.png, rech_compare_sp0001.png, ...

    Parameters
    ----------
    model_ws_initial : str or PathLike
        Workspace for the initial MF6 simulation (e.g. 'model_ws/elk_2lay_monthly').
    model_ws_base : str or PathLike
        Workspace for the base/posterior MF6 simulation (e.g. m_d_base).
    sim_name : str, default 'elk_2lay'
        GWF model name in both simulations.
    out_dir : str, default 'fig_rech_compare'
        Subfolder under model_ws_base to store the figures.
    max_kper : int, optional
        If provided, only kper = 0..max_kper-1 are plotted.
        If None, all stress periods are processed.
    dpi : int, default 300
        DPI for saved PNGs.
    """

    model_ws_initial = os.fspath(model_ws_initial)
    model_ws_base = os.fspath(model_ws_base)

    print(f"[RCH COMP] Loading initial simulation from: {model_ws_initial}")
    sim0 = flopy.mf6.MFSimulation.load(sim_ws=model_ws_initial, exe_name="mf6")
    gwf0 = sim0.get_model(sim_name)

    print(f"[RCH COMP] Loading base/posterior simulation from: {model_ws_base}")
    sim1 = flopy.mf6.MFSimulation.load(sim_ws=model_ws_base, exe_name="mf6")
    gwf1 = sim1.get_model(sim_name)

    # --- helper to unwrap LayeredArray/Util3D → ndarray
    def _to_array(obj):
        return obj.array if hasattr(obj, "array") else obj

    # --- recharge extractor: returns (nper, nrow, ncol)
    def _get_rch_3d(gwf) -> np.ndarray:
        if not hasattr(gwf, "rch"):
            raise ValueError("Model has no RCH package.")
        rch = gwf.rch
        full = None
        if hasattr(rch, "recharge") and hasattr(rch.recharge, "array"):
            full = _to_array(rch.recharge)
        elif hasattr(rch, "rech") and hasattr(rch.rech, "array"):
            full = _to_array(rch.rech)
        else:
            raise ValueError("RCH package does not have a recognizable recharge array.")

        full = np.asarray(full, dtype=float)

        # Acceptable shapes:
        #   (nrow, ncol)
        #   (nper, nrow, ncol)
        #   (nper, nlay, nrow, ncol) → sum over layers
        if full.ndim == 2:
            full = full[None, ...]  # (1, nrow, ncol)
        elif full.ndim == 3:
            # (nper, nrow, ncol) – OK
            pass
        elif full.ndim == 4:
            # (nper, nlay, nrow, ncol) – sum across layers
            full = full.sum(axis=1)
        else:
            raise ValueError(f"Unsupported RCH shape: {full.shape}")

        return full

    rch0 = _get_rch_3d(gwf0)
    rch1 = _get_rch_3d(gwf1)

    if rch0.shape != rch1.shape:
        raise ValueError(
            f"Initial and base RCH shapes differ: {rch0.shape} vs {rch1.shape}"
        )

    nper, nrow, ncol = rch0.shape
    print(f"[RCH COMP] RCH shape: nper={nper}, nrow={nrow}, ncol={ncol}")

    # idomain masks (keep active cells in each model)
    idom0 = gwf0.modelgrid.idomain
    idom1 = gwf1.modelgrid.idomain

    # make sure idomain is 2-D per model (take top layer if 3-D)
    def _idomain_2d(idom_arr: np.ndarray) -> np.ndarray:
        idom_arr = np.asarray(idom_arr)
        if idom_arr.ndim == 3:
            return idom_arr[0]
        elif idom_arr.ndim == 2:
            return idom_arr
        else:
            raise ValueError(f"Unexpected idomain shape: {idom_arr.shape}")

    idom0_2d = _idomain_2d(idom0)
    idom1_2d = _idomain_2d(idom1)

    # shared active mask: active in BOTH models
    active_mask = (idom0_2d > 0) & (idom1_2d > 0)

    # --- outline shapefile (same as plot_base_params_map_mel)
    gis_dir = os.path.join("..", "..", "gis")
    outline_fp = os.path.join(gis_dir, "input_shps", "elk", "elk_boundary_lf.shp")
    outline = gpd.read_file(outline_fp)

    # --- optional TDIS date info for page titles
    tdis = sim1.get_package("tdis")
    sp_times = None
    try:
        start_date = pd.to_datetime(getattr(tdis, "start_date_time", "1970-01-01"))
        perlen = np.array(tdis.perioddata.array["perlen"], dtype=float)
        cum_days = np.cumsum(perlen)
        sp_times = [start_date + pd.Timedelta(days=float(d)) for d in cum_days]
    except Exception:
        sp_times = None

    # --- output directory
    figdir = os.path.join(model_ws_base, out_dir)
    os.makedirs(figdir, exist_ok=True)

    cmap = mpl.colormaps.get_cmap("Blues")

    # how many stress periods to plot
    nper_plot = nper if max_kper is None else min(nper, max_kper)

    for kper in range(nper_plot):
        data0 = rch0[kper].copy()
        data1 = rch1[kper].copy()

        # mask inactive / crazy values
        mask_bad0 = (~active_mask) | (data0 <= -1e30) | (data0 >= 1e30)
        mask_bad1 = (~active_mask) | (data1 <= -1e30) | (data1 >= 1e30)

        data0_m = np.ma.masked_where(mask_bad0, data0)
        data1_m = np.ma.masked_where(mask_bad1, data1)

        # combined stats for colour scale
        combo = np.concatenate(
            [data0_m.compressed(), data1_m.compressed()]
        )  # non-masked values
        if combo.size == 0:
            vmin, vmax = 0.0, 1e-6
        else:
            vmin = np.nanpercentile(combo, 2.0)
            vmax = np.nanpercentile(combo, 98.0)
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin == vmax:
                vmin = float(np.nanmin(combo))
                vmax = float(np.nanmax(combo))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin == vmax:
                vmin, vmax = 0.0, 1e-6

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # stats per model for titles
        d0_min, d0_max = (
            float(data0_m.min()) if data0_m.count() > 0 else np.nan,
            float(data0_m.max()) if data0_m.count() > 0 else np.nan,
        )
        d1_min, d1_max = (
            float(data1_m.min()) if data1_m.count() > 0 else np.nan,
            float(data1_m.max()) if data1_m.count() > 0 else np.nan,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        # left: initial
        ax0 = axes[0]
        mview0 = flopy.plot.PlotMapView(model=gwf0, layer=0, ax=ax0)
        mview0.plot_array(data0_m, cmap=cmap, norm=norm)
        outline.plot(ax=ax0, facecolor="none", edgecolor="0.4", linewidth=1.0)
        ax0.set_aspect("equal")
        ax0.set_xlabel("Easting (ft)")
        ax0.set_ylabel("Northing (ft)")
        ax0.set_title(
            f"Initial RECH – kper {kper}\n"
            f"min={d0_min:.3e}, max={d0_max:.3e} ft/d",
            loc="left",
        )

        # right: base/posterior
        ax1 = axes[1]
        mview1 = flopy.plot.PlotMapView(model=gwf1, layer=0, ax=ax1)
        mview1.plot_array(data1_m, cmap=cmap, norm=norm)
        outline.plot(ax=ax1, facecolor="none", edgecolor="0.4", linewidth=1.0)
        ax1.set_aspect("equal")
        ax1.set_xlabel("Easting (ft)")
        ax1.set_ylabel("Northing (ft)")
        ax1.set_title(
            f"Base RECH – kper {kper}\n"
            f"min={d1_min:.3e}, max={d1_max:.3e} ft/d",
            loc="left",
        )

        # shared colourbar
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            shrink=0.8,
            label="Recharge (ft/d)",
        )

        # page-level title w/ optional date
        if sp_times is not None and kper < len(sp_times):
            date_str = sp_times[kper].date().isoformat()
            fig.suptitle(
                f"{sim_name} – Recharge comparison (kper={kper}, end={date_str})",
                fontsize=11,
            )
        else:
            fig.suptitle(
                f"{sim_name} – Recharge comparison (kper={kper})",
                fontsize=11,
            )

        fname = f"rech_compare_sp{kper:04d}.png"
        fpath = os.path.join(figdir, fname)
        fig.savefig(fpath, dpi=dpi, transparent=False)
        plt.close(fig)
        print("saved →", fpath)


def plot_well_pumping_comparison_mel(
    init_ws: str,
    post_ws: str,
    sim_name: str = "elk_2lay",
    wel_pkg_name: str = "wel",
    out_dir: str = "fig_wel_pumping",
    dpi: int = 300,
) -> None:
    """
    For each well cell, plot time series of:
      - Initial model requested rate (solid grey)
      - Posterior model requested rate (solid blue)

    Units:
      - Model rates are in ft^3/day (cfd)
      - Plots are in gpm (cfd / 192.5)

    X-axis:
      - Actual dates, using TDIS start_date_time and stress-period lengths
        (date shown is the *end* of each stress period).

    Plots are labelled by Layer / Row / Column only (no global cellnumber in title).

    Parameters
    ----------
    init_ws : str
        Workspace of the initial model (with WEL package).
    post_ws : str
        Workspace of the posterior/base model.
    sim_name : str, default "elk_2lay"
        GWF model name within the MF6 simulation.
    wel_pkg_name : str, default "wel"
        Package name of the WEL package.
    out_dir : str, default "fig_wel_pumping"
        Output directory for PNGs (created under post_ws).
    dpi : int, default 300
        Figure resolution.
    """
    CFD_TO_GPM = 1.0 / 192.5  # ft^3/day → gpm

    # ------------------------------------------------------------------
    # Helper: load requested rates from a WEL package (by cellnumber)
    #         and also return cellnum → (k,i,j) mapping
    # ------------------------------------------------------------------
    from typing import Dict, Tuple
    import numpy as np
    import pandas as pd
    import re
    import os
    import flopy
    import matplotlib.pyplot as plt

    def _load_requested_from_wel(
        ws: str,
    ) -> tuple[
        Dict[int, np.ndarray],
        Dict[int, np.ndarray],
        flopy.mf6.ModflowGwf,
        Dict[int, Tuple[int, int, int]],
    ]:
        """
        Returns
        -------
        times_dict : {cellnum: np.array(datetime64[ns])}
            Full time axis for all SPs (missing SPs filled with 0 pumping).
        q_dict : {cellnum: np.array(float)}
            Pumping in gpm, same length as times_dict[cellnum].
        gwf : flopy.mf6.ModflowGwf
            Loaded GWF model (for reference).
        cell_index_map : {cellnum: (k, i, j)}
            MF6 0-based layer/row/col for each cellnum.
        """
        sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name="mf6")
        gwf = sim.get_model(sim_name)
        dis = gwf.dis

        # Dimensions
        nlay = int(dis.nlay.data)
        nrow = int(dis.nrow.data)
        ncol = int(dis.ncol.data)

        # ---------------------------
        # Load TDIS & build datetime axis
        # ---------------------------
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
        # Use END of SP for plotting
        sp_end_dates = [d + pd.Timedelta(days=float(pl)) for d, pl in zip(sp_dates, perlen)]

        # ---------------------------
        # Load WEL SPD
        # ---------------------------
        wel = gwf.get_package(wel_pkg_name)
        if wel is None:
            raise ValueError(f"No WEL package named '{wel_pkg_name}' in workspace {ws}")

        # detect rate field
        rate_field = None
        for kper in range(nper):
            try:
                arr = wel.stress_period_data.get_data(kper)
            except Exception:
                arr = None
            if arr is not None and len(arr) > 0:
                names = {n.lower(): n for n in arr.dtype.names}
                for cand in ("q", "rate", "flux"):
                    if cand in names:
                        rate_field = names[cand]
                        break
            if rate_field is not None:
                break
        if rate_field is None:
            raise ValueError("Cannot find q/rate/flux field in WEL SPD")

        # ---------------------------
        # Collect all cellnumbers + store (k,i,j)
        # ---------------------------
        all_cells: set[int] = set()
        sp_cell_q: list[Dict[int, float]] = [dict() for _ in range(nper)]  # [kper][cellnum] = q_cfd
        cell_index_map: Dict[int, Tuple[int, int, int]] = {}

        for kper in range(nper):
            try:
                arr = wel.stress_period_data.get_data(kper)
            except Exception:
                arr = None
            if arr is None or len(arr) == 0:
                continue

            names = {n.lower(): n for n in arr.dtype.names}
            cellids = arr[names["cellid"]]
            qvals = arr[rate_field].astype(float)

            for cid, q in zip(cellids, qvals):
                # MF6 DIS cellid is already 0-based (k, i, j)
                if len(cid) < 3:
                    continue
                k = int(cid[0])
                i = int(cid[1])
                j = int(cid[2])
                if not (0 <= k < nlay and 0 <= i < nrow and 0 <= j < ncol):
                    continue

                node_index = k * (nrow * ncol) + i * ncol + j  # 0-based node index
                cellnum = node_index + 1                       # 1-based cellnumber

                all_cells.add(cellnum)
                sp_cell_q[kper][cellnum] = q

                # Save the index mapping the first time we see this cellnum
                if cellnum not in cell_index_map:
                    cell_index_map[cellnum] = (k, i, j)

        # ---------------------------
        # Build complete time series, filling missing SPs = 0
        # ---------------------------
        times_dict: Dict[int, np.ndarray] = {}
        q_dict: Dict[int, np.ndarray] = {}

        for cellnum in sorted(all_cells):
            ts = []
            qs = []
            for kper in range(nper):
                ts.append(sp_end_dates[kper])
                q_cfd = sp_cell_q[kper].get(cellnum, 0.0)  # missing SP = 0
                qs.append(q_cfd * CFD_TO_GPM)

            times_dict[cellnum] = np.array(ts, dtype="datetime64[ns]")
            q_dict[cellnum] = np.array(qs, dtype=float)

        return times_dict, q_dict, gwf, cell_index_map

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("[INFO] Loading initial requested WEL rates…")
    init_req_times, init_req_q, gwf_init, init_idx = _load_requested_from_wel(init_ws)

    print("[INFO] Loading posterior requested WEL rates…")
    post_req_times, post_req_q, gwf_post, post_idx = _load_requested_from_wel(post_ws)

    if gwf_init.modelgrid.grid_type != gwf_post.modelgrid.grid_type:
        print("[WARN] Initial and posterior grids differ in type; cellnumber mapping assumes same DIS grid.")

    # ------------------------------------------------------------------
    # Build union of all wells (by cellnumber)
    # ------------------------------------------------------------------
    all_cellnums = sorted(set(init_req_q.keys()) | set(post_req_q.keys()))
    if not all_cellnums:
        print("[WARN] No wells found in either WEL package.")
        return

    # Output directory under posterior workspace
    out_dir_full = os.path.join(post_ws, out_dir)
    os.makedirs(out_dir_full, exist_ok=True)

    print(f"[INFO] Plotting {len(all_cellnums)} wells to: {out_dir_full}")

    # ------------------------------------------------------------------
    # Plot each well
    # ------------------------------------------------------------------
    for fig_idx, cellnum in enumerate(all_cellnums, start=1):
        # Requested series
        t_init = init_req_times.get(cellnum, np.array([], dtype="datetime64[ns]"))
        q_init = init_req_q.get(cellnum, np.array([], dtype=float))

        t_post = post_req_times.get(cellnum, np.array([], dtype="datetime64[ns]"))
        q_post = post_req_q.get(cellnum, np.array([], dtype=float))

        if t_init.size == 0 and t_post.size == 0:
            continue

        # Get k,i,j from the stored mapping
        ijk = init_idx.get(cellnum)
        if ijk is None:
            ijk = post_idx.get(cellnum)

        if ijk is not None:
            k, i, j = ijk
            k1, i1, j1 = k + 1, i + 1, j + 1  # human-readable
            wel_label = f"L{k1} R{i1} C{j1}"
            fname_stub = f"wel_L{k1:02d}_R{i1:03d}_C{j1:03d}_fig{fig_idx:03d}"
        else:
            wel_label = "Unknown location"
            fname_stub = f"wel_unknown_fig{fig_idx:03d}"

        # ------------------------------------------------------------------
        # Plot (white background)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        ax.set_facecolor("white")

        # Initial requested (solid grey)
        if t_init.size > 0:
            ax.plot(
                t_init,
                q_init,
                color="black",
                linewidth=1.0,
                label="Initial requested",
            )

        # Posterior requested (solid blue)
        if t_post.size > 0:
            ax.plot(
                t_post,
                q_post,
                color="C0",
                linewidth=1.0,
                label="Posterior requested",
                linestyle="--",  
            )

        ax.set_title(f"Figure {fig_idx}: Well pumping – {wel_label}", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Pumping rate (gpm)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        fig.autofmt_xdate()
        fig.tight_layout()

        png = os.path.join(out_dir_full, f"{fname_stub}.png")
        fig.savefig(png, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"  saved → {png}")

    print("[DONE] Well pumping comparison plots created.")



def plot_total_well_pumping_comparison_mel(
    init_ws: str,
    post_ws: str,
    sim_name: str = "elk_2lay",
    wel_pkg_name: str = "wel",
    out_dir: str = "fig_wel_pumping",
    dpi: int = 300,
) -> None:
    """
    Plot time series of TOTAL well rate (summed over all WEL cells) for each
    stress period, comparing:

      - Initial model total rate (solid grey)
      - Posterior model total rate (solid blue)

    Notes
    -----
    - Rates are summed directly from the WEL package (net sum of q over cells).
      For typical MF6 sign convention:
        * Negative = pumping (extraction)
        * Positive = injection
      So the plotted values are "net total well rate" (gpm).

    - Time axis uses TDIS start_date_time and period lengths; the plotted date
      is the END of each stress period.

    Parameters
    ----------
    init_ws : str
        Workspace of the initial model (with WEL package).
    post_ws : str
        Workspace of the posterior/base model (with WEL package).
    sim_name : str, default "elk_2lay"
        GWF model name within the MF6 simulation.
    wel_pkg_name : str, default "wel"
        Package name of the WEL package.
    out_dir : str, default "fig_wel_pumping"
        Output directory for PNGs (created under post_ws).
    dpi : int, default 300
        Figure resolution.
    """
    CFD_TO_GPM = 1.0 / 192.5  # ft^3/day → gpm

    # ------------------------------------------------------------------
    # Helper: load total WEL rate per stress period for a workspace
    # ------------------------------------------------------------------
    def _load_total_from_wel(ws: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        sp_end_dates : np.ndarray of datetime64[ns]
            End-of-stress-period timestamps.
        total_q_gpm : np.ndarray of float
            Net total rate (sum over all WEL entries) in gpm for each SP.
        """
        sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name="mf6")
        gwf = sim.get_model(sim_name)

        # ---------------------------
        # Load TDIS & build datetime axis
        # ---------------------------
        tdis = sim.get_package("tdis")

        # parse start_date_time
        raw = tdis.start_date_time.get_data() if hasattr(tdis.start_date_time, "get_data") else tdis.start_date_time
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
        # end of each stress period
        sp_end_dates = np.array(
            [d + pd.Timedelta(days=float(pl)) for d, pl in zip(sp_dates, perlen)],
            dtype="datetime64[ns]",
        )

        # ---------------------------
        # Load WEL SPD
        # ---------------------------
        wel = gwf.get_package(wel_pkg_name)
        if wel is None:
            raise ValueError(f"No WEL package named '{wel_pkg_name}' in workspace {ws}")

        # detect rate field once
        rate_field = None
        for kper in range(nper):
            try:
                arr = wel.stress_period_data.get_data(kper)
            except Exception:
                arr = None
            if arr is not None and len(arr) > 0:
                names = {n.lower(): n for n in arr.dtype.names}
                for cand in ("q", "rate", "flux"):
                    if cand in names:
                        rate_field = names[cand]
                        break
            if rate_field is not None:
                break
        if rate_field is None:
            raise ValueError("Cannot find q/rate/flux field in WEL SPD")

        # ---------------------------
        # Sum total rate per SP
        # ---------------------------
        total_q_gpm = np.zeros(nper, dtype=float)

        for kper in range(nper):
            try:
                arr = wel.stress_period_data.get_data(kper)
            except Exception:
                arr = None

            if arr is None or len(arr) == 0:
                # no wells that SP → total stays 0
                continue

            q_cfd = arr[rate_field].astype(float)
            total_q_gpm[kper] = np.sum(q_cfd) * CFD_TO_GPM  # net total

        return sp_end_dates, total_q_gpm

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("[INFO] Loading total WEL rates for initial model…")
    dates_init, qtot_init = _load_total_from_wel(init_ws)

    print("[INFO] Loading total WEL rates for posterior model…")
    dates_post, qtot_post = _load_total_from_wel(post_ws)

    # ------------------------------------------------------------------
    # Basic consistency check on time axes
    # ------------------------------------------------------------------
    same_time_axis = (
        dates_init.shape == dates_post.shape
        and np.all(dates_init == dates_post)
    )
    if not same_time_axis:
        print("[WARN] Initial and posterior TDIS date axes differ; "
              "plotting each against its own dates.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_dir_full = os.path.join(post_ws, out_dir)
    os.makedirs(out_dir_full, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")

    # Initial total
    ax.plot(
        dates_init,
        qtot_init,
        color="black",
        linewidth=1.0,
        label="Initial total WEL rate",
    )

    # Posterior total
    ax.plot(
        dates_post,
        qtot_post,
        color="C0",
        linewidth=1.0,
        label="Posterior total WEL rate",
        linestyle="--",  
    )

    ax.set_title("Total well rate – initial vs posterior", loc="left")
    ax.set_xlabel("Date (end of stress period)")
    ax.set_ylabel("Net total WEL rate (gpm; negative = pumping)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()

    png = os.path.join(out_dir_full, "total_wel_pumping_timeseries.png")
    fig.savefig(png, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[DONE] Total pumping comparison plot saved → {png}")

def plot_rech_timeseries_initial_vs_base_mel(
    model_ws_initial: Union[str, os.PathLike],
    model_ws_base: Union[str, os.PathLike],
    sim_name: str = "elk_2lay",
    out_dir: str = "fig_rech_compare_ts",
    dpi: int = 300,
) -> None:
    """
    Compute and plot domain-average recharge for each stress period and
    annual averages, comparing an initial vs base/posterior model.

    For both initial and base models:
      - Compute domain-average recharge over cells that are active in BOTH models.
      - Units: RCH stored in ft/d → converted to in/yr for plotting.

    Outputs
    -------
    Two PNGs written to <model_ws_base>/<out_dir>/:
      - rech_timeseries_sp.png      (SP-by-SP domain-average recharge)
      - rech_timeseries_annual.png  (time-weighted annual average recharge)

    Notes
    -----
    - Domain average is computed over the shared active mask:
        active_mask = (idomain_initial > 0) & (idomain_base > 0)
    - Annual averages are time-weighted using TDIS perlen:
        mean_year(ft/d) = sum(R_sp(ft/d) * perlen_sp(days)) / sum(perlen_sp)
        then converted to in/yr.
    """

    model_ws_initial = os.fspath(model_ws_initial)
    model_ws_base = os.fspath(model_ws_base)

    print(f"[RCH TS] Loading initial simulation from: {model_ws_initial}")
    sim0 = flopy.mf6.MFSimulation.load(sim_ws=model_ws_initial, exe_name="mf6")
    gwf0 = sim0.get_model(sim_name)

    print(f"[RCH TS] Loading base/posterior simulation from: {model_ws_base}")
    sim1 = flopy.mf6.MFSimulation.load(sim_ws=model_ws_base, exe_name="mf6")
    gwf1 = sim1.get_model(sim_name)

    # --- helper to unwrap LayeredArray/Util3D → ndarray
    def _to_array(obj):
        return obj.array if hasattr(obj, "array") else obj

    # --- recharge extractor: returns (nper, nrow, ncol) in ft/d
    def _get_rch_3d(gwf) -> np.ndarray:
        if not hasattr(gwf, "rch"):
            raise ValueError("Model has no RCH package.")
        rch = gwf.rch
        full = None
        if hasattr(rch, "recharge") and hasattr(rch.recharge, "array"):
            full = _to_array(rch.recharge)
        elif hasattr(rch, "rech") and hasattr(rch.rech, "array"):
            full = _to_array(rch.rech)
        else:
            raise ValueError("RCH package does not have a recognizable recharge array.")

        full = np.asarray(full, dtype=float)

        # Acceptable shapes:
        #   (nrow, ncol)
        #   (nper, nrow, ncol)
        #   (nper, nlay, nrow, ncol) → sum over layers
        if full.ndim == 2:
            full = full[None, ...]  # (1, nrow, ncol)
        elif full.ndim == 3:
            # (nper, nrow, ncol) – OK
            pass
        elif full.ndim == 4:
            # (nper, nlay, nrow, ncol) – sum across layers
            full = full.sum(axis=1)
        else:
            raise ValueError(f"Unsupported RCH shape: {full.shape}")

        return full  # ft/d

    rch0 = _get_rch_3d(gwf0)  # initial
    rch1 = _get_rch_3d(gwf1)  # base/posterior

    if rch0.shape != rch1.shape:
        raise ValueError(
            f"Initial and base RCH shapes differ: {rch0.shape} vs {rch1.shape}"
        )

    nper, nrow, ncol = rch0.shape
    print(f"[RCH TS] RCH shape: nper={nper}, nrow={nrow}, ncol={ncol}")

    # --- idomain masks (keep shared active cells)
    idom0 = gwf0.modelgrid.idomain
    idom1 = gwf1.modelgrid.idomain

    def _idomain_2d(idom_arr: np.ndarray) -> np.ndarray:
        idom_arr = np.asarray(idom_arr)
        if idom_arr.ndim == 3:
            return idom_arr[0]
        elif idom_arr.ndim == 2:
            return idom_arr
        else:
            raise ValueError(f"Unexpected idomain shape: {idom_arr.shape}")

    idom0_2d = _idomain_2d(idom0)
    idom1_2d = _idomain_2d(idom1)

    active_mask = (idom0_2d > 0) & (idom1_2d > 0)

    # --- TDIS timing (use base model's TDIS)
    tdis1 = sim1.get_package("tdis")

    # Parse start date
    raw = getattr(tdis1, "start_date_time", "1970-01-01")
    if hasattr(raw, "get_data"):
        raw = raw.get_data()
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(raw))
    if not m:
        raise ValueError(f"Cannot parse start_date_time from TDIS: {raw!r}")
    start_date = pd.to_datetime(m.group(0))

    perdata = tdis1.perioddata.array
    perlen_days = np.atleast_1d(perdata["perlen"]).astype(float)
    if perlen_days.size != nper:
        raise ValueError("TDIS nper and RCH nper mismatch.")

    sp_start_dates = [start_date]
    for i in range(1, nper):
        sp_start_dates.append(sp_start_dates[-1] + pd.Timedelta(days=float(perlen_days[i - 1])))
    sp_end_dates = np.array(
        [d + pd.Timedelta(days=float(pl)) for d, pl in zip(sp_start_dates, perlen_days)],
        dtype="datetime64[ns]",
    )

    # --- compute domain-average recharge (ft/d) for each SP
    def _domain_avg_ft_per_day(rch: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        out = np.full(nper, np.nan, dtype=float)
        for kper in range(nper):
            data = rch[kper]
            valid_mask = (
                active_mask
                & np.isfinite(data)
                & (data > -1e30)
                & (data < 1e30)
            )
            if np.any(valid_mask):
                out[kper] = float(np.mean(data[valid_mask]))
            else:
                out[kper] = np.nan
        return out

    avg0_ftd = _domain_avg_ft_per_day(rch0, active_mask)
    avg1_ftd = _domain_avg_ft_per_day(rch1, active_mask)

    # --- convert ft/d → in/yr
    FT_PER_DAY_TO_IN_PER_YEAR = 12.0 * 365.0  # assuming 365-day year

    avg0_inyr = avg0_ftd * FT_PER_DAY_TO_IN_PER_YEAR
    avg1_inyr = avg1_ftd * FT_PER_DAY_TO_IN_PER_YEAR

    # --- output directory
    figdir = os.path.join(model_ws_base, out_dir)
    os.makedirs(figdir, exist_ok=True)

    # ==========================================================
    # 1) Stress period time series (SP-by-SP)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(
        sp_end_dates,
        avg0_inyr,
        label="Initial (domain avg)",
        linewidth=1.0,
        color="black",
    )
    ax.plot(
        sp_end_dates,
        avg1_inyr,
        label="Base/posterior (domain avg)",
        linewidth=1.0,
        color="C0",
        linestyle="--", 
    )

    ax.set_title(f"{sim_name} – Domain-average recharge (per stress period)", loc="left")
    ax.set_xlabel("Date (end of stress period)")
    ax.set_ylabel("Recharge (in/yr, domain-average)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()

    sp_png = os.path.join(figdir, "rech_timeseries_sp.png")
    fig.savefig(sp_png, dpi=dpi, transparent=False)
    plt.close(fig)
    print("saved →", sp_png)

    # ==========================================================
    # 2) Annual time-weighted average (calendar-year)
    # ==========================================================
    years = np.array([pd.Timestamp(d).year for d in sp_end_dates], dtype=int)

    annual_dates = []
    annual0_inyr = []
    annual1_inyr = []

    for year in np.unique(years):
        idx = np.where(years == year)[0]
        if idx.size == 0:
            continue

        total_days = np.sum(perlen_days[idx])
        if total_days <= 0:
            continue

        # time-weighted mean ft/d across SPs in this year
        mean0_ftd = np.nansum(avg0_ftd[idx] * perlen_days[idx]) / total_days
        mean1_ftd = np.nansum(avg1_ftd[idx] * perlen_days[idx]) / total_days

        annual0_inyr.append(mean0_ftd * FT_PER_DAY_TO_IN_PER_YEAR)
        annual1_inyr.append(mean1_ftd * FT_PER_DAY_TO_IN_PER_YEAR)

        # use Dec 31 of that year as the x coordinate
        annual_dates.append(pd.Timestamp(year=year, month=12, day=31))

    if len(annual_dates) > 0:
        annual_dates = np.array(annual_dates, dtype="datetime64[ns]")
        annual0_inyr = np.array(annual0_inyr, dtype=float)
        annual1_inyr = np.array(annual1_inyr, dtype=float)

        fig2, ax2 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax2.plot(
            annual_dates,
            annual0_inyr,
            marker="o",
            linestyle="-",
            linewidth=1.0,
            label="Initial (annual avg)",
            color="black",
        )
        ax2.plot(
            annual_dates,
            annual1_inyr,
            marker="o",
            linestyle="--",
            linewidth=1.0,
            label="Base/posterior (annual avg)",
            color="C0",
        )

        ax2.set_title(f"{sim_name} – Domain-average recharge (annual)", loc="left")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Recharge (in/yr, domain-average)")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=8)
        fig2.autofmt_xdate()

        annual_png = os.path.join(figdir, "rech_timeseries_annual.png")
        fig2.savefig(annual_png, dpi=dpi, transparent=False)
        plt.close(fig2)
        print("saved →", annual_png)
    else:
        print("[RCH TS] No annual data could be computed (check TDIS/perlen/year logic).")

def analyze_drn_riv_fluxes_timeseries(
    *,
    sim_ws: str,
    model_name: str = "elk_2lay",
    out_dir: str = "figures",
    model_pkg_shp_dir: str | None = None,
    riv_reach_csv: str | None = os.path.join("data", "raw", "riv_reach_info.csv"),
) -> Dict[str, str]:
    """
    Time-series flux analysis for DRN and RIV:

    Outputs
    -------
    1) RIV reaches (cfs)
       - Uses riv_reach_info.csv with columns: i, j (0-based), row, col (1-based), reach
       - Assumes all RIV cells are in layer 1 (k=0).
       - For each reach:
           * Sum signed q (cfs; + into model, - out) across cells for each time.
           * Page with:
               - small map highlighting reach cells
               - time series of net reach flow (cfs)

    2) drn_wl net time series (cfs)
       - Uses shapefile
         sim_ws/output_shapefiles/model_packages/drn_wl.shp
         with columns i,j (0-based) or row,col (1-based).
       - Assumes drn_wl is in layer 2 (k=1).
       - For each time:
           * Sum signed q (cfs) over all drn_wl cells.
       - One PDF with:
           - map of all drn_wl cells
           - time series of net drn_wl flow (cfs).

    3) drn_ag cell-by-cell time series (in/day)
       - Uses shapefile drn_ag.shp in the same directory.
       - Assumes drn_ag cells are in layer 1 (k=0).
       - For each **cell**, computes:
           flux_inch_per_day = q_ft3_per_day / cell_area_ft2 * 12
           (signed: + into model, - out)
       - One PDF with one page per cell:
           - map highlighting that cell
           - time series of in/day.

    4) drn total (plain drn) net time series (cfs)
       - Uses shapefile drn.shp (layer 1, k=0).
       - For each time:
           * Sum signed q (cfs) over all drn cells.
       - One PDF with:
           - map of all drn cells
           - time series of net drn flow (cfs).

    Returns
    -------
    dict
        Keys and their output PDF paths, e.g.:
            {
              "riv_reaches": "<..._RIV_reach_timeseries.pdf>",
              "drn_wl_timeseries": "<..._DRN_drn_wl_timeseries_cfs.pdf>",
              "drn_ag_cells": "<..._DRN_drn_ag_cell_timeseries_in_per_day.pdf>",
              "drn_total_timeseries": "<..._DRN_drn_total_timeseries_cfs.pdf>",
            }
    """
    os.makedirs(out_dir, exist_ok=True)
    from flopy.utils import CellBudgetFile

    # --- helper: tdis start date parsing ---
    import re

    def _parse_mf6_start_date(tdis) -> pd.Timestamp:
        s = str(getattr(tdis.start_date_time, "data", getattr(tdis, "start_date_time", ""))).strip()
        m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", s)
        if m:
            return pd.to_datetime(m.group(1))
        try:
            return pd.to_datetime(s)
        except Exception:
            raise ValueError(f"Unrecognized TDIS start_date_time: {s!r}")

    # --- helper: vertices for rotated grids ---
    def _vertices_for_pcolor(mg) -> Tuple[np.ndarray, np.ndarray]:
        """Return cell-corner grids (nrow+1, ncol+1), honoring rotation if present."""
        nrow, ncol = int(mg.nrow), int(mg.ncol)
        try:
            XV = np.asarray(mg.xvertices)
            YV = np.asarray(mg.yvertices)
            if XV.shape == (nrow + 1, ncol + 1) and YV.shape == (nrow + 1, ncol + 1):
                return XV, YV
        except Exception:
            pass

        # fallback: infer edges from centers
        xc, yc = np.asarray(mg.xcellcenters), np.asarray(mg.ycellcenters)

        def _edges_from_centers_1d(c1d):
            c = np.ravel(c1d).astype(float)
            inc = (c[-1] > c[0])
            if not inc:
                c = c[::-1]
            mids = 0.5 * (c[1:] + c[:-1])
            first = c[0] - (mids[0] - c[0])
            last = c[-1] + (c[-1] - mids[-1])
            e = np.concatenate([[first], mids, [last]])
            return e if inc else e[::-1]

        xe = _edges_from_centers_1d(xc[0, :])
        ye = _edges_from_centers_1d(yc[:, 0])
        XE, YE = np.meshgrid(xe, ye)
        return XE, YE

    def _auto_cbb_path(ws: str, mnm: str) -> str:
        candidates = [
            os.path.join(ws, f"{mnm}.cbb"),
            os.path.join(ws, f"{mnm}.cbc"),
            os.path.join(ws, f"{mnm}.bud"),
        ]
        candidates.extend(glob.glob(os.path.join(ws, "*.cbb")))
        candidates.extend(glob.glob(os.path.join(ws, "*.cbc")))
        subdir = os.path.join(ws, mnm)
        if os.path.isdir(subdir):
            candidates.extend(glob.glob(os.path.join(subdir, "*.cbb")))
            candidates.extend(glob.glob(os.path.join(subdir, "*.cbc")))
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No CBB/CBC/BUD file found in {ws!r}")

    # --- load sim + model ---
    print(f"[flux] Loading MF6 simulation from {sim_ws!r} ...")
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    gwf = sim.get_model(model_name)
    tdis = sim.get_package("tdis")
    t0 = _parse_mf6_start_date(tdis)

    mg = gwf.modelgrid
    xcc, ycc = np.asarray(mg.xcellcenters), np.asarray(mg.ycellcenters)
    XE, YE = _vertices_for_pcolor(mg)

    nlay = int(mg.nlay)
    nrow = int(mg.nrow)
    ncol = int(mg.ncol)
    ncpl = nrow * ncol

    # cell area matrix (ft^2) for DIS grid
    # delr: length ncol; delc: length nrow
    delr = np.asarray(mg.delr)
    delc = np.asarray(mg.delc)
    if delr.ndim == 0:
        delr = np.full(ncol, float(delr))
    if delc.ndim == 0:
        delc = np.full(nrow, float(delc))
    cell_area = np.outer(delc, delr)  # [i,j] in ft^2

    cbb_path = _auto_cbb_path(sim_ws, model_name)
    print(f"[flux] Using cell-by-cell budget file: {cbb_path}")

    try:
        cbf = CellBudgetFile(cbb_path, precision="double")
    except OSError:
        print("[flux] double-precision read failed; retrying single precision...")
        cbf = CellBudgetFile(cbb_path, precision="single")

    times = np.asarray(cbf.get_times(), float)
    if times.size == 0:
        raise RuntimeError("No times found in cell-by-cell budget file.")

    time_index = pd.to_datetime(t0) + pd.to_timedelta(times, unit="D")

    # optional Elk boundary for maps
    elk_boundary = None
    elk_boundary_shp = os.path.join("..", "..", "gis", "input_shps", "elk", "elk_boundary_lf.shp")
    if gpd is not None and os.path.exists(elk_boundary_shp):
        try:
            elk_boundary = gpd.read_file(elk_boundary_shp)
        except Exception:
            elk_boundary = None

    # --- node -> (k,i,j) decoder for DIS ---
    def _decode_node(node: int) -> tuple[int, int, int]:
        """Convert MF6 DIS global node (1-based) to (k,i,j) 0-based."""
        node0 = int(node) - 1
        k = node0 // ncpl
        icpl = node0 % ncpl
        i = icpl // ncol
        j = icpl % ncol
        return k, i, j

    # helper: simple map plot of a boolean mask (layer-agnostic → 2D)
    def _plot_mask_map(mask2d: np.ndarray, title: str):
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        if elk_boundary is not None:
            try:
                elk_boundary.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)
            except Exception:
                pass
        if np.any(mask2d):
            arr = np.where(mask2d, 1.0, np.nan)
            ax.pcolormesh(XE, YE, arr, shading="auto")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (ft)")
        ax.set_ylabel("Y (ft)")
        ax.set_title(title)
        return fig, ax

    out_paths: Dict[str, str] = {}

    # =====================================================================
    # DRN shapefile masks: drn, drn_ag, drn_wl
    # =====================================================================
    if model_pkg_shp_dir is None:
        model_pkg_shp_dir = os.path.join(sim_ws, "output_shapefiles", "model_packages")

    drn_pkg_defs = {
        # static geographic drains (layer 1)
        "drn_s":  {"layer": 0, "shp": "drn_s.shp"},
        "drn_ms": {"layer": 0, "shp": "drn_ms.shp"},
        "drn_mn": {"layer": 0, "shp": "drn_mn.shp"},
        "drn_n":  {"layer": 0, "shp": "drn_n.shp"},
        # ag and wl drains
        "drn_ag": {"layer": 0, "shp": "drn_ag.shp"},
        "drn_wl": {"layer": 1, "shp": "drn_wl.shp"},
    }
    drn_masks = {}
    for pkg_key, info in drn_pkg_defs.items():
        shp_path = os.path.join(model_pkg_shp_dir, info["shp"])
        mask = np.zeros((nlay, nrow, ncol), dtype=bool)
        if gpd is not None and os.path.exists(shp_path):
            try:
                gdf = gpd.read_file(shp_path)
                # prefer 0-based i,j if present
                if "i" in gdf.columns and "j" in gdf.columns:
                    for _, r in gdf.iterrows():
                        i = int(r["i"])
                        j = int(r["j"])
                        k = info["layer"]
                        if 0 <= k < nlay and 0 <= i < nrow and 0 <= j < ncol:
                            mask[k, i, j] = True
                elif "row" in gdf.columns and "col" in gdf.columns:
                    for _, r in gdf.iterrows():
                        i = int(r["row"]) - 1
                        j = int(r["col"]) - 1
                        k = info["layer"]
                        if 0 <= k < nlay and 0 <= i < nrow and 0 <= j < ncol:
                            mask[k, i, j] = True
            except Exception as e:
                print(f"[flux] WARNING: failed to read {shp_path!r} for {pkg_key}: {e}")
        else:
            print(f"[flux] WARNING: shapefile not found for {pkg_key}: {shp_path}")
        drn_masks[pkg_key] = mask

    # build combined mask for all static DRN packages
    static_keys = ["drn_s", "drn_ms", "drn_mn", "drn_n"]
    mask_drn = np.zeros((nlay, nrow, ncol), dtype=bool)
    for key in static_keys:
        if key in drn_masks:
            mask_drn |= drn_masks[key]
    drn_masks["drn"] = mask_drn  # for backward-compatible use below
    
    # =====================================================================
    # PART 1: drn_wl net time series (cfs)
    # =====================================================================
    if drn_masks["drn_wl"].any():
        print("[flux] Computing drn_wl net time series (cfs) ...")
        mask_wl = drn_masks["drn_wl"]
        series_wl = pd.Series(index=time_index, dtype=float)
        series_wl.loc[:] = 0.0

        for it, totim in enumerate(times):
            try:
                data_list = cbf.get_data(text="DRN", totim=totim)
            except Exception:
                continue
            if not data_list:
                continue

            net_q = 0.0
            for rec in data_list:
                if rec.dtype.names is None:
                    continue
                if "q" in rec.dtype.names:
                    q_raw = np.asarray(rec["q"], float)
                elif "flow" in rec.dtype.names:
                    q_raw = np.asarray(rec["flow"], float)
                else:
                    continue

                q_cfs = q_raw / 86400.0  # convert ft^3/d → ft^3/s
                nodes = rec["node"]
                for node_val, q in zip(nodes, q_cfs):
                    k, i, j = _decode_node(node_val)
                    if not (0 <= k < nlay and 0 <= i < nrow and 0 <= j < ncol):
                        continue
                    if not mask_wl[k, i, j]:
                        continue
                    net_q += q

            series_wl.iloc[it] = net_q

        out_pdf_wl = os.path.join(out_dir, f"{model_name}_DRN_drn_wl_timeseries_cfs.pdf")
        with PdfPages(out_pdf_wl) as pdf:
            mask2d = drn_masks["drn_wl"].any(axis=0)
            fig, ax_map = _plot_mask_map(mask2d, "drn_wl cells")
            pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
            ax.plot(series_wl.index, series_wl.values, lw=1.5)
            ax.axhline(0.0, color="0.5", lw=1.0, linestyle="--")
            ax.set_ylabel("Net drain flow (cfs)\n(+ into model, - out)")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)
            ax.set_title("drn_wl – net flow (all cells, cfs)")
            pdf.savefig(fig); plt.close(fig)

        out_paths["drn_wl_timeseries"] = os.path.abspath(out_pdf_wl)
        print(f"[flux]   Wrote drn_wl timeseries PDF: {out_pdf_wl}")
    else:
        print("[flux] No drn_wl cells found in shapefile; skipping drn_wl TS.")

    # =====================================================================
    # PART 2: drn_ag cell-by-cell time series (in/day)
    # =====================================================================
    if drn_masks["drn_ag"].any():
        print("[flux] Computing drn_ag cell-by-cell time series (in/day) ...")
        mask_ag = drn_masks["drn_ag"]
        # Collect list of ag cells
        ag_cells = [(k, i, j)
                    for k in range(nlay)
                    for i in range(nrow)
                    for j in range(ncol)
                    if mask_ag[k, i, j]]
        n_ag = len(ag_cells)
        print(f"[flux]   Number of drn_ag cells: {n_ag}")

        cell_to_idx = {cell: idx for idx, cell in enumerate(ag_cells)}
        # times × cells array in in/day
        arr_inch_day = np.zeros((len(times), n_ag), dtype=float)

        for it, totim in enumerate(times):
            try:
                data_list = cbf.get_data(text="DRN", totim=totim)
            except Exception:
                continue
            if not data_list:
                continue

            for rec in data_list:
                if rec.dtype.names is None:
                    continue
                if "q" in rec.dtype.names:
                    q_raw = np.asarray(rec["q"], float)  # ft^3/day
                elif "flow" in rec.dtype.names:
                    q_raw = np.asarray(rec["flow"], float)
                else:
                    continue

                nodes = rec["node"]
                for node_val, q_ft3_d in zip(nodes, q_raw):
                    k, i, j = _decode_node(node_val)
                    cell = (k, i, j)
                    idx = cell_to_idx.get(cell)
                    if idx is None:
                        continue
                    # in/day = (ft^3/d) / (ft^2) * 12
                    A = cell_area[i, j] if np.isfinite(cell_area[i, j]) and cell_area[i, j] > 0 else np.nan
                    if not np.isfinite(A) or A <= 0:
                        continue
                    q_in_day = (q_ft3_d / A) * 12.0
                    arr_inch_day[it, idx] += q_in_day  # signed

        out_pdf_ag = os.path.join(out_dir, f"{model_name}_DRN_drn_ag_cell_timeseries_in_per_day.pdf")
        with PdfPages(out_pdf_ag) as pdf:
            mask2d_base = drn_masks["drn_ag"].any(axis=0)

            for idx, (k, i, j) in enumerate(ag_cells):
                ser = pd.Series(arr_inch_day[:, idx], index=time_index)

                fig = plt.figure(figsize=(11, 7), constrained_layout=True)
                gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.6])

                # Map (top, spanning both columns)
                ax_map = fig.add_subplot(gs[0, :])
                if elk_boundary is not None:
                    try:
                        elk_boundary.boundary.plot(ax=ax_map, color="black", linewidth=0.5, alpha=0.5)
                    except Exception:
                        pass

                # highlight this cell
                mask2d = np.zeros((nrow, ncol), dtype=float)
                mask2d_base_faint = np.where(mask2d_base, 0.3, np.nan)
                # faint all ag cells
                ax_map.pcolormesh(XE, YE, mask2d_base_faint, shading="auto", cmap="Greys")
                # strong highlight this one
                mask2d[i, j] = 1.0
                ax_map.pcolormesh(XE, YE, mask2d, shading="auto")
                ax_map.set_aspect("equal", adjustable="box")
                ax_map.set_xlabel("X (ft)")
                ax_map.set_ylabel("Y (ft)")
                ax_map.set_title(f"drn_ag cell (k={k+1}, row={i+1}, col={j+1})")

                # Time series (bottom, spanning both columns)
                ax_ts = fig.add_subplot(gs[1, :])
                ax_ts.plot(ser.index, ser.values, lw=1.5)
                ax_ts.axhline(0.0, color="0.5", lw=1.0, linestyle="--")
                ax_ts.set_ylabel("Drain flux (in/day)\n(+ into model, - out)")
                ax_ts.set_xlabel("Time")
                ax_ts.grid(True, alpha=0.3)
                ax_ts.set_title("drn_ag – cell time series (in/day)")

                pdf.savefig(fig)
                plt.close(fig)

        out_paths["drn_ag_cells"] = os.path.abspath(out_pdf_ag)
        print(f"[flux]   Wrote drn_ag cell timeseries PDF: {out_pdf_ag}")
    else:
        print("[flux] No drn_ag cells found in shapefile; skipping drn_ag cell TS.")

    # =====================================================================
    # PART 3: drn total (plain drn) net time series (cfs)
    # =====================================================================
    if drn_masks["drn"].any():
        print("[flux] Computing total drn (plain .drn) net time series (cfs) ...")
        mask_drn = drn_masks["drn"]
        series_drn = pd.Series(index=time_index, dtype=float)
        series_drn.loc[:] = 0.0

        for it, totim in enumerate(times):
            try:
                data_list = cbf.get_data(text="DRN", totim=totim)
            except Exception:
                continue
            if not data_list:
                continue

            net_q = 0.0
            for rec in data_list:
                if rec.dtype.names is None:
                    continue
                if "q" in rec.dtype.names:
                    q_raw = np.asarray(rec["q"], float)
                elif "flow" in rec.dtype.names:
                    q_raw = np.asarray(rec["flow"], float)
                else:
                    continue

                q_cfs = q_raw / 86400.0
                nodes = rec["node"]
                for node_val, q in zip(nodes, q_cfs):
                    k, i, j = _decode_node(node_val)
                    if not (0 <= k < nlay and 0 <= i < nrow and 0 <= j < ncol):
                        continue
                    if not mask_drn[k, i, j]:
                        continue
                    net_q += q

            series_drn.iloc[it] = net_q

        out_pdf_drn = os.path.join(out_dir, f"{model_name}_DRN_drn_total_timeseries_cfs.pdf")
        with PdfPages(out_pdf_drn) as pdf:
            mask2d = drn_masks["drn"].any(axis=0)
            fig, ax_map = _plot_mask_map(mask2d, "drn (plain) cells")
            pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
            ax.plot(series_drn.index, series_drn.values, lw=1.5)
            ax.axhline(0.0, color="0.5", lw=1.0, linestyle="--")
            ax.set_ylabel("Net drain flow (cfs)\n(+ into model, - out)")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)
            ax.set_title("drn (plain .drn) – net flow (all cells, cfs)")
            pdf.savefig(fig); plt.close(fig)

        out_paths["drn_total_timeseries"] = os.path.abspath(out_pdf_drn)
        print(f"[flux]   Wrote drn total timeseries PDF: {out_pdf_drn}")
    else:
        print("[flux] No drn (plain) cells found in shapefile; skipping drn total TS.")

    # =====================================================================
    # PART 4: RIV reach time series (cfs)
    # =====================================================================
    if riv_reach_csv is not None and os.path.exists(riv_reach_csv):
        print(f"[flux] Computing RIV reach time series from {riv_reach_csv!r} ...")
        rivdf = pd.read_csv(riv_reach_csv)

        # Assume RIV is in layer 1 (k=0)
        reach_cells: Dict[str, set] = {}
        for _, r in rivdf.iterrows():
            i = int(r["i"]) if "i" in rivdf.columns else int(r["row"] - 1)
            j = int(r["j"]) if "j" in rivdf.columns else int(r["col"] - 1)
            k = 0
            reach = str(r["reach"])
            reach_cells.setdefault(reach, set()).add((k, i, j))

        reach_names = sorted(reach_cells.keys())
        if not reach_names:
            print("[flux]   No reaches found in riv_reach_info.csv; skipping reach TS.")
        else:
            # map cell -> reach for fast lookup
            cell_to_reach: Dict[tuple, str] = {}
            for reach, cells in reach_cells.items():
                for cell in cells:
                    cell_to_reach[cell] = reach

            df_reach = pd.DataFrame(index=time_index, columns=reach_names, dtype=float)
            df_reach.loc[:, :] = 0.0

            for it, totim in enumerate(times):
                try:
                    data_list = cbf.get_data(text="RIV", totim=totim)
                except Exception:
                    continue
                if not data_list:
                    continue

                flux_sum = {r: 0.0 for r in reach_names}

                for rec in data_list:
                    if rec.dtype.names is None:
                        continue
                    if "q" in rec.dtype.names:
                        q_raw = np.asarray(rec["q"], float)
                    elif "flow" in rec.dtype.names:
                        q_raw = np.asarray(rec["flow"], float)
                    else:
                        continue

                    q_cfs = q_raw / 86400.0
                    nodes = rec["node"]

                    for node_val, q in zip(nodes, q_cfs):
                        k, i, j = _decode_node(node_val)
                        if not (0 <= k < nlay and 0 <= i < nrow and 0 <= j < ncol):
                            continue
                        reach = cell_to_reach.get((k, i, j))
                        if reach is None:
                            continue
                        flux_sum[reach] += q  # signed: + into model, - out

                for rname in reach_names:
                    df_reach.iloc[it, df_reach.columns.get_loc(rname)] = flux_sum[rname]

            out_pdf_riv = os.path.join(out_dir, f"{model_name}_RIV_reach_timeseries.pdf")
            with PdfPages(out_pdf_riv) as pdf:
                for reach in reach_names:
                    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
                    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.6])

                    # Map panel (top, spanning two columns)
                    ax_map = fig.add_subplot(gs[0, :])
                    if elk_boundary is not None:
                        try:
                            elk_boundary.boundary.plot(ax=ax_map, color="black", linewidth=0.5, alpha=0.5)
                        except Exception:
                            pass

                    mask = np.zeros((nrow, ncol), dtype=float)
                    for (k, i, j) in reach_cells[reach]:
                        if 0 <= i < nrow and 0 <= j < ncol:
                            mask[i, j] = 1.0

                    if np.any(mask > 0):
                        ax_map.pcolormesh(XE, YE, mask, shading="auto")
                    ax_map.set_aspect("equal", adjustable="box")
                    ax_map.set_xlabel("X (ft)")
                    ax_map.set_ylabel("Y (ft)")
                    ax_map.set_title(f"RIV reach: {reach}")

                    # Time series panel (bottom, spanning two columns)
                    ax_ts = fig.add_subplot(gs[1, :])
                    ser = df_reach[reach]
                    ax_ts.plot(df_reach.index, ser.values, lw=1.5)
                    ax_ts.axhline(0.0, color="0.5", lw=1.0, linestyle="--")
                    ax_ts.set_ylabel("Net river leakage (cfs)\n(+ into model, - out)")
                    ax_ts.set_xlabel("Time")
                    ax_ts.grid(True, alpha=0.3)
                    ax_ts.set_title(f"RIV reach {reach} — total flow (cfs)")

                    pdf.savefig(fig)
                    plt.close(fig)

            out_paths["riv_reaches"] = os.path.abspath(out_pdf_riv)
            print(f"[flux]   Wrote RIV reach timeseries PDF: {out_pdf_riv}")
    else:
        print(f"[flux] RIV reach CSV not found or None: {riv_reach_csv}")

    return out_paths

def plot_total_pumping(
    m_d=".",
    modnm="elk_2lay",
    iter_num=4,
    obs_data_csv="elk_2lay.obs_data.csv",
    require_in_suffix=True,             # only *_in terms
    hide_forecast=True,
    fill_plot=True,
    take_abs=False,
    ):
    pst = pyemu.Pst(os.path.join(m_d, f"{modnm}.pst"))

    # --- obs metadata table (this is the source of truth) ---
    od = pd.read_csv(os.path.join(m_d, obs_data_csv))
    od["datetime"] = pd.to_datetime(od["datetime"], errors="coerce")

    # keep budget obs only
    od = od.loc[od["oname"].astype(str).str.lower() == "bud"].copy()
    
    # keep only pumping-related usecols (by prefix)
    u = od["usecol"].astype(str).str.lower()
    keep = False
    keep = (u == 'wel')
    od = od.loc[keep].copy()

    if od.empty:
        raise ValueError(
            "No budget pumping obs found after filtering.\n"
            f"Example usecol values: {sorted(pd.read_csv(os.path.join(m_d, obs_data_csv))['usecol'].dropna().unique())[:30]}"
        )

    if hide_forecast:
        od = od.loc[od["datetime"] < pd.to_datetime("2023-01-01")].copy()

    # group the observation names by datetime
    by_dt = od.groupby("datetime")["obsnme"].apply(list).sort_index()

    # --- load ensembles ---
    pr = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(m_d, f"{modnm}.0.obs.jcb"))
    pt = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(m_d, f"{modnm}.{iter_num}.obs.jcb"))

    def _build_days_in_sp(dts: pd.DatetimeIndex, year_boundary="2000-01-01"):
        dts = pd.DatetimeIndex(pd.to_datetime(dts)).sort_values()
        days = pd.Series(index=dts, dtype=float)
    
        boundary = pd.to_datetime(year_boundary)
    
        # yearly SPs: use actual calendar year length based on the year
        yearly_mask = dts < boundary
        for dt in dts[yearly_mask]:
            days.loc[dt] = 366 if calendar.isleap(dt.year) else 365
    
        # monthly SPs: days in that month
        monthly_mask = ~yearly_mask
        for dt in dts[monthly_mask]:
            days.loc[dt] = dt.days_in_month
    
        return days

    # --- build (realization x datetime) totals ---
    def _total_ts(en):
        series_list = []
        for dt, names in by_dt.items():
            names = [n for n in names if n in en.columns]
            if not names:
                continue
            s = en.loc[:, names].sum(axis=1)  # ft3/day (rate)
            series_list.append(s.rename(dt))
    
        if not series_list:
            raise ValueError("No matching obsnmes found in ensemble columns (check naming / prefixes).")
    
        df = pd.concat(series_list, axis=1)  # rows=real, cols=datetime
        df.columns = pd.to_datetime(df.columns)
        df = df.sort_index(axis=1)
    
        if take_abs:
            df = df.abs()
    
        # ---- convert ft3/day -> acre-ft per SP, then resample to acre-ft/year ----
        days_in_sp = _build_days_in_sp(df.columns, year_boundary="2000-01-01")  # aligned by datetime
        # multiply each column by its SP length (days), then convert ft3->acft
        df_acft_sp = df.mul(days_in_sp.values, axis=1) / 43560.0
    
        # sum to annual totals (ac-ft/year)
        df_acft_yr = df_acft_sp.T.groupby(df_acft_sp.columns.year).sum().T
        # make the x-axis datetime-ish (Jan 1 of each year)
        df_acft_yr.columns = pd.to_datetime(df_acft_yr.columns.astype(str) + "-01-01")
    
        return df_acft_yr

    pr_ts = _total_ts(pr)
    pt_ts = _total_ts(pt)
   
    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    x = pr_ts.columns
    
    # Plot all prior reals
    for i in pr_ts.index:
        if i != "base":
            ax.plot(x, pr_ts.loc[i].values, color="grey", alpha=0.2, lw=0.25, zorder=1)

    # Use a fill for the posterior
    ax.fill_between(x, pt_ts.min(axis=0).values, pt_ts.max(axis=0).values, color='lightblue', alpha=0.5, zorder=3, label="Posterior ensemble range")

    # Plot base reals
    if "base" in pr_ts.index:
        ax.plot(x, pr_ts.loc["base"].values, color="black", lw=1.6, label="Base of prior",zorder=4)
    if "base" in pt_ts.index:
        ax.plot(x, pt_ts.loc["base"].values, color="blue", lw=1.8, label="Base of posterior",zorder=10)

    handles = [Patch(facecolor='lightblue', alpha=0.5, label='Posterior ensemble range'),
               Line2D([],[],color='blue',label='Base of posterior'),
               Line2D([],[],color='grey',alpha=0.5,label='Prior realizations'),
               Line2D([],[],color='black',label='Base of prior')]

    ax.set_title("Total Pumping", loc="left")
    ax.set_ylabel("Pumping (acre-feet)")
    ax.grid(alpha=0.2)
    ax.legend(handles=handles,
              loc="upper left")
    plt.tight_layout()
    with PdfPages('ensemble_pumping.pdf') as pdf:
        pdf.savefig(dpi=250,
                    bbox_inches='tight')
    plt.close(fig)


def plot_total_rch(
    m_d=".",
    modnm="elk_2lay",
    iter_num=4,
    obs_data_csv="elk_2lay.obs_data.csv",
    require_in_suffix=True,
    hide_forecast=True,
    fill_plot=True,
    take_abs=False,
    log_y=False,
    ylims=None,  # <-- NEW (tuple like (ymin, ymax))
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import calendar

    import flopy
    import pyemu
    import tqdm

    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    pst = pyemu.Pst(os.path.join(m_d, f"{modnm}.pst"))

    # --- obs metadata table (source of truth) ---
    od = pd.read_csv(os.path.join(m_d, obs_data_csv))
    od["datetime"] = pd.to_datetime(od["datetime"], errors="coerce")

    # keep budget obs only
    od = od.loc[od["oname"].astype(str).str.lower() == "bud"].copy()

    # keep only recharge-related usecols
    u = od["usecol"].astype(str).str.lower()
    keep = (u == "rcha")
    od = od.loc[keep].copy()

    if od.empty:
        raw = pd.read_csv(os.path.join(m_d, obs_data_csv))
        raise ValueError(
            "No budget recharge obs found after filtering.\n"
            f"Example usecol values: {sorted(raw['usecol'].dropna().unique())[:30]}"
        )

    if hide_forecast:
        od = od.loc[od["datetime"] < pd.to_datetime("2023-01-01")].copy()

    # group the observation names by datetime
    by_dt = od.groupby("datetime")["obsnme"].apply(list).sort_index()

    # --- load ensembles ---
    pr = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(m_d, f"{modnm}.0.obs.jcb"))
    pt = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(m_d, f"{modnm}.{iter_num}.obs.jcb"))

    def _build_days_in_sp(dts: pd.DatetimeIndex, year_boundary="2000-01-01"):
        dts = pd.DatetimeIndex(pd.to_datetime(dts)).sort_values()
        days = pd.Series(index=dts, dtype=float)

        boundary = pd.to_datetime(year_boundary)

        # yearly SPs: use actual calendar year length based on the year
        yearly_mask = dts < boundary
        for dt in dts[yearly_mask]:
            days.loc[dt] = 366 if calendar.isleap(dt.year) else 365

        # monthly SPs: days in that month
        monthly_mask = ~yearly_mask
        for dt in dts[monthly_mask]:
            days.loc[dt] = dt.days_in_month

        return days

    def _active_top_area_ft2(sim_ws, modelname):
        """
        Compute active model area (ft^2) from DIS idomain in the top layer (layer 1).
        Assumes a structured grid (delr/delc).
        """
        sim = flopy.mf6.MFSimulation.load(
            sim_name=f"{modelname}.nam",
            version="mf6",
            exe_name="mf6.exe",
            sim_ws=sim_ws,
            load_only=["dis"],
        )
        gwf_local = sim.get_model(modelname)
        mg = gwf_local.modelgrid

        idom = gwf_local.dis.idomain.array  # (nlay, nrow, ncol)
        active_top = (idom[0, :, :] > 0)

        delr = np.asarray(mg.delr)  # (ncol,)
        delc = np.asarray(mg.delc)  # (nrow,)
        cell_area = delc[:, None] * delr[None, :]  # (nrow, ncol)

        return float(cell_area[active_top].sum())

    # area used to convert volume -> depth
    area_ft2 = _active_top_area_ft2(sim_ws=m_d, modelname=modnm)
    if not np.isfinite(area_ft2) or area_ft2 <= 0:
        raise ValueError(f"Computed active top area is invalid: {area_ft2}")

    def _total_ts_in_per_yr(en):
        """
        Build annual average recharge rate (in/yr) for each realization.
        Assumes the recharge observations represent total recharge volume rate (ft3/day)
        summed over the domain at each stress period.
        """
        series_list = []
        for dt, names in by_dt.items():
            names = [n for n in names if n in en.columns]
            if not names:
                continue
            s = en.loc[:, names].sum(axis=1)  # ft3/day (rate)
            series_list.append(s.rename(dt))

        if not series_list:
            raise ValueError("No matching obsnmes found in ensemble columns (check naming / prefixes).")

        df = pd.concat(series_list, axis=1)  # rows=real, cols=datetime
        df.columns = pd.to_datetime(df.columns)
        df = df.sort_index(axis=1)

        if take_abs:
            df = df.abs()

        # ft3/day -> ft3 per stress period
        days_in_sp = _build_days_in_sp(df.columns, year_boundary="2000-01-01")
        df_ft3_sp = df.mul(days_in_sp.values, axis=1)

        # sum to annual total volume (ft3/yr)
        df_ft3_yr = df_ft3_sp.T.groupby(df_ft3_sp.columns.year).sum().T

        # x-axis as Jan 1 of each year
        df_ft3_yr.columns = pd.to_datetime(df_ft3_yr.columns.astype(str) + "-01-01")

        # convert to depth rate: (ft3/yr)/(ft2)=ft/yr -> in/yr
        df_in_yr = (df_ft3_yr / area_ft2) * 12.0

        return df_in_yr

    pr_ts = _total_ts_in_per_yr(pr)
    pt_ts = _total_ts_in_per_yr(pt)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    x = pr_ts.columns

    # Plot all prior reals
    for i in pr_ts.index:
        if i != "base":
            ax.plot(x, pr_ts.loc[i].values, color="grey", alpha=0.2, lw=0.25, zorder=1)

    # Posterior envelope
    ax.fill_between(
        x,
        pt_ts.min(axis=0).values,
        pt_ts.max(axis=0).values,
        color="lightblue",
        alpha=0.5,
        zorder=3,
        label="Posterior ensemble range",
    )

    # Base reals
    if "base" in pr_ts.index:
        ax.plot(x, pr_ts.loc["base"].values, color="black", lw=1.6, label="Base of prior", zorder=4)
    if "base" in pt_ts.index:
        ax.plot(x, pt_ts.loc["base"].values, color="blue", lw=1.8, label="Base of posterior", zorder=10)

    handles = [
        Patch(facecolor="lightblue", alpha=0.5, label="Posterior ensemble range"),
        Line2D([], [], color="blue", label="Base of posterior"),
        Line2D([], [], color="grey", alpha=0.5, label="Prior realizations"),
        Line2D([], [], color="black", label="Base of prior"),
    ]

    if log_y:
        ax.set_yscale("log")
    
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_title("Model Recharge", loc="left")
    ax.set_ylabel("Recharge (in/yr)")
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left")
    plt.tight_layout()

    out_pdf = os.path.join(m_d,"results", "figures", "ensemble_rch_in_per_yr.pdf")
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_pdf


if __name__ == '__main__':

    m_d = 'master_flow_08_highdim_restrict_bcs_flood_full_final_rch'
    modnm = 'elk_2lay'
    #plot_simple_par_maps(m_d,modnm)
    #exit()

    plot_ies = False
    plot_sensitivity = False
    plot_base = False
    plot_fancy = False
    plot_flux = False
    plot_budget = False
    plot_histo = False
    plot_recharge = False
    tables = True
    # set universal figure properties:
    set_graph_specifications()
    set_map_specifications()
    pst = pyemu.Pst(os.path.join(m_d, f"{modnm}.pst"))
    obgnmes = pst.observation_data.obgnme.unique()
    partypes = sorted({g.split("_k:")[0] for g in obgnmes if "_k:" in g})
    
    if tables:
        write_summary_tables(m_d=m_d, pst_name='elk_2lay.pst', noptmax=4, max_fail=2)
    
    if plot_budget:
        obsdict = get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
        #plot_water_budget(m_d, obsdict, pie_year=2023)
        plot_selected_budget_timeseries_posterior_only(
        obsdict=obsdict,
        stress_period_table_path=os.path.join("tables", "monthly_stress_period_info.csv"),
        out_dir=os.path.join(m_d, "results", "budget_figs"),
        date_col="end_datetime",  
    )
        
    if plot_recharge:
        plot_total_rch(m_d = m_d, ylims = (0.0,12.0))
        #plot_total_pumping(m_d = m_d)
        
        

    if plot_histo:
        for partype in ['k33']:
            print(f'Plotting {type} histograms')
            plot_array_histo(m_d, modnm=modnm, noptmax=4, partype=partype,  logbool=False)
        
    if plot_ies:
        obsdict = get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
        itrmx = max(obsdict)
        # plot_phi_sequence(m_d, modnm=modnm)
        # plot_simple_par_histo(m_d, modnm=modnm)
        for partype in ['k', 'k33', 'ss', 'sy']:
            print(f'Plotting {type} histograms')
            plot_array_histo(m_d, modnm=modnm, noptmax=itrmx, partype=partype,  logbool=False)
            # plot_array_histo(m_d, modnm=modnm, noptmax=6, type='k', logbool=False)

        #for partype in ['k', 'k33', 'ss', 'sy']:
        #    xsection_comp_pr_pt(m_d, modnm, noptmax=itrmx, partype=partype)

        #plot_simple_1to1(m_d, modnm=modnm)
        # plot_1_to_1_ss(m_d, obsdict,modnm=modnm)
        # plot_1_to_1_sspmp(m_d, obsdict, modnm=modnm)
        #plot_simple_timeseries(m_d, modnm=modnm)
        plot_layer_one2one_wdepth(m_d,obsdict, modnm=modnm)
        #plot_layer_one2one_bywell(m_d, obsdict, modnm=modnm)
        plot_parm_violins(m_d, f'{modnm}.pst', itrmx)
        #plot_water_budget(m_d, obsdict,pie_year=2022)
        
        #pardict = get_ies_par_dict(m_d=m_d, pst=None)

        #plot_obs_v_sim_by_well(m_d, obsdict)
        
        #plot_obs_v_sim_flux(m_d, obsdict)   # delte this unless we have some stream gauge locs we want to plot

        #plot_wateruse(m_d=m_d, modnm=modnm, noptmax=itrmx, max_fail=0)

        #plot_simple_obs_v_sim(m_d)

        #plot_base_params_map(m_d,obsdict,False,partype='hk') # only hk being added to obs at the moment
        #plot_mfinput1(m_d, 'elk_2lay.pst', noptmax=4)
        # takes a long time

        plot_fancy_obs_v_sim(m_d,obsdict) # this should be replaced with Spencer's hydrograph code, but add ensembles
        #write_summary_tables(m_d=m_d, pst_name='elk_2lay.pst', noptmax=3, max_fail=2)

    if plot_fancy:
        obsdict = get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
        itrmx = max(obsdict)
        plot_fancy_obs_v_sim_base(m_d,obsdict)
    # Forward run with base parameter sets
    if plot_base:
        #base_posterior_param_forward_run(m_d0=m_d, noptmax=4)
        m_d_base = m_d + '_forward_run_base'
        init_ws = os.path.join("model_ws", "elk_2lay_monthly")
        #plot_hds(m_d_base,sim_name=modnm,kstpkper=(0, 20))
        plot_base_params_map_mel(m_d_base,sim_name=modnm,  partype='hk', logscale_hk =False)
        plot_base_params_map_mel(m_d_base,sim_name=modnm, partype='vk', logscale_hk = False)
        plot_base_params_map_mel(m_d_base,sim_name=modnm,partype='ss', logscale_hk = False)
        plot_base_params_map_mel(m_d_base,sim_name=modnm, partype='sy', logscale_hk = False)
        #plot_base_params_map_mel(m_d_base, partype='rech', logscale_hk = False)
        # #run_zb_by_layer(m_d)
        # plot_simple_obs_v_sim_base(m_d_base)
        #plot_base_water_table(m_d_base)
        # model_packages_to_shp_joined(m_d_base)
        # plot_rech_initial_vs_base_mel(
        #     model_ws_initial=os.path.join("model_ws", "elk_2lay_monthly"),
        #     model_ws_base=m_d_base,
        #     sim_name=modnm,
        #     out_dir="fig_rech_compare",
        #     max_kper=None,   # or 323 if you only want historical
        # )
        # plot_well_pumping_comparison_mel(
        #     init_ws=init_ws,
        #     post_ws=m_d_base,
        #     sim_name="elk_2lay",
        #     wel_pkg_name="wel",
        #     out_dir="fig_wel_pumping",
        # )
        
        # plot_total_well_pumping_comparison_mel(
        #     init_ws=init_ws,
        #     post_ws=m_d_base,
        #     sim_name= "elk_2lay",
        #     wel_pkg_name= "wel",
        #     out_dir= "fig_wel_pumping"
        # )

        # plot_rech_timeseries_initial_vs_base_mel(
        #     model_ws_initial=init_ws,
        #     model_ws_base=m_d_base,
        #     sim_name="elk_2lay",
        #     out_dir="fig_rech_compare_ts",
        #     dpi=300,
        # )
    
    if plot_flux:
        m_d_base = m_d + '_forward_run_base'
        init_ws = os.path.join("model_ws", "elk_2lay_monthly")
        analyze_drn_riv_fluxes_timeseries(
        sim_ws=m_d_base,
        model_name="elk_2lay",
        out_dir=os.path.join(m_d_base, "cbb_plots")
)

    # plot sensitivity results
    if plot_sensitivity:
        s_d = m_d + '_sen'
        #sensitivity_figs(s_d)
