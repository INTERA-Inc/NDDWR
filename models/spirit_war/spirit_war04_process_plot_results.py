
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import platform
print('Env path order:')
for path in sys.path:
    print(path)
import pyemu
import flopy
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
years10 = mdates.YearLocator(10)
years20 = mdates.YearLocator(20)
years1 = mdates.YearLocator()
import matplotlib.ticker as ticker
years_fmt = mdates.DateFormatter('%Y')
import numpy as np
import pandas as pd
# Set some pandas options
pd.set_option('expand_frame_repr', False)
import shutil
import geopandas as gpd
import matplotlib as mpl
#mpl.rcParams['axes.formatter.limits'] = (-10,20)
#from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
#from scipy import stats
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import seaborn as sns
from shapely import Point
import warnings
warnings.filterwarnings('ignore')
from typing import Sequence, Union
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import contextily as cx
import os
import pandas as pd
import calendar
from matplotlib.patches import Patch
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

def get_ies_obs_dict(m_d='master_ies', pst=None, modnm='wahp'):
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

    pst = pyemu.Pst(os.path.join(m_d,'wahp.pst'))

    jcbName = os.path.join(m_d,'wahp.0.obs.jcb')

    jcb = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=jcbName)
    obs_df_dict = jcb
    return obs_df_dict

def get_ies_par_dict(m_d='master_ies', pst=None):
    par_df_dict = {}

    pst = pyemu.Pst(os.path.join(m_d,'wahp.pst'))
    if pst.control_data.noptmax == -1:
        itrs = [0]
    else:
        itrs = range(pst.control_data.noptmax+1)

    for i in itrs:
        jcbName = os.path.join(m_d,f'wahp.{i}.par.jcb')
        if os.path.exists(jcbName):
            print(f'loading itr {i}')
            jcb = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=jcbName)
            par_df_dict[i] = jcb
    return par_df_dict


def plot_simple_obs_v_sim(m_d, modnm='wahp'):
    obsdict = get_ies_obs_dict(m_d=m_d, modnm=modnm)
    pst = pyemu.Pst(os.path.join(m_d,f'{modnm}.pst'))
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,'wahp.obs+noise.jcb'))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('hifreq|ext|elev'),:].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    aobs = pd.read_csv(os.path.join(m_d,'wahp.adjusted.obs_data.csv'),index_col=0)
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


def plot_phi_sequence(m_d, modnm='swww'):
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


def plot_inset(ax=[],
               wl_loc=None,
               grp=[],
               cpts=[],
               sw_extent=[],
               ww_extent=[],
               k_barrier=[],
               rch_windows=[],
               drains=[]):

    rch_windows.plot(ax=ax,color='lightblue',label='Recharge Windows',edgecolor='k')
    k_barrier.plot(ax=ax,color='k',label='Low-K Barrier')
    ww_extent.boundary.plot(ax=ax,edgecolor='tan',linewidth=1.25,label='Warwick Aquifer')
    sw_extent.boundary.plot(ax=ax,edgecolor='lightgreen',linewidth=1.25,label='Spiritwood Aquifer')
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
    lb4 = mlines.Line2D([], [], color='lightgreen', linestyle='-', label='Spiritwood Boundary')
    lb5 = mlines.Line2D([], [], color='tan', linestyle='-', label='Warwick Boundary')
    lb6 = mpatches.Patch(facecolor='lightblue',linewidth=1,edgecolor='black', label='Recharge Windows')
    lb7 = mpatches.Patch(facecolor='k',linewidth=1,edgecolor='black', label='SW Low-K Barrier')
    leg = ax.legend(handles=[lb1, lb2, lb3, lb4, lb5, lb6, lb7], loc='lower right', frameon=True,
                    fontsize=10)
    leg.get_frame().set_facecolor('grey')       # Background color
    leg.get_frame().set_alpha(0.5)              # Transparency
    leg.get_frame().set_edgecolor('black')      # Dark outline color
    leg.get_frame().set_linewidth(2.0)          # Thickness of outline
    leg.set_bbox_to_anchor((.4, -0.04))


def plot_vert_xsec(ax, m_d, m, wl_loc=None, mwl=0, cpts=[]):

    if wl_loc is None:
        print('No water level location provided for vertical cross section plot')
        return

    df = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    points = [Point(xy) for xy in zip(df['x_2265'], df['y_2265'])]
    sites = gpd.GeoDataFrame(data=df,
                             geometry=points
                             )
    sites = sites.set_crs(2265)#.to_crs(2265)
    grp_num = sites['group_number']
    k = sites['k']
    sites['id'] = 'grpid:'+grp_num.astype(str) + '_k:' + k.astype(str)
    wpt = sites.loc[sites.id == wl_loc]
    wpt['screen_top'] = wpt['top_screen'].values[0]
    wpt['screen_bot'] = wpt['bottom_screen'].values[0]

    top = m.dis.top.array
    botm = m.dis.botm.array
    nlay = m.dis.nlay.data

    layers = {
        0: 'WW',
        1: 'CU',
        2: 'SW',
    }

    wpt['i'] = wpt['row']-1
    wpt['j'] = wpt['column']-1
    lay = wpt['k'].values[0]
    lykey = layers[lay]
    i = wpt.i.astype(int).values[0]
    j = wpt.j.astype(int).values[0]
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

    # get layer top and bottom:
    lybot = df[lykey].values[0]
    # get layer name above current layer:
    cols = list(df.columns)
    ly_above = cols[cols.index(lykey)-1]
    lytop = df[ly_above].values[0]

    sns.barplot(x='index', y='WW', data=df, color='tan', label='Warwick', bottom=mod_top,
                ax=ax)
    top_bar = mod_top + df['WW'].values[0]
    sns.barplot(x='index', y='CU', data=df, color='lightblue', label='Confining Unit', bottom=top_bar,
                ax=ax)
    top_bar += df['CU'].values[0]
    sns.barplot(x='index', y='SW', data=df, color='lightgreen', label='Spiritwood', bottom=top_bar, ax=ax)

    # plot vertical line between screen top and bottom:
    if np.isnan(scr_top) or np.isnan(scr_bot):
        scr_top = np.abs(lytop)
        scr_bot = np.abs(lybot)
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
    ax.set_position([box.x0 + 0.04, box.y0 + 0.06, box.width, box.height])


# Fancy obs vsim plots with model structure and map panels for each target
def plot_fancy_obs_v_sim(m_d, obsdict, modnm='swww',plot_hdiff=False,plt_noise=True,plt_pr=True,itrmx=None):
    print('...PLotting fancy obs v sim plots')
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
    # Grab if not provided
    if not itrmx:
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
    cpts = pd.read_csv(os.path.join('data', 'analyzed', 'transient_well_targets_lookup.csv'))
    cpts['grpid'] = 'grpid:' + cpts['group_number'].astype(str) + '_k:' + cpts['k'].astype(str)
    # make geodataframe from geometry column:
    cpts = gpd.GeoDataFrame(data=cpts, geometry=gpd.points_from_xy(cpts.x_2265, cpts.y_2265), crs=2265)
    cpts = cpts.groupby(['grpid']).last().reset_index()
    cpts.columns = cpts.columns.str.lower()

    g_d = os.path.join('..', '..', 'gis', 'input_shps', 'sw_ww')
    g_d_out = os.path.join('..', '..', 'gis', 'output_shps', 'sw_ww')
    sw_extent = gpd.read_file(os.path.join(g_d,'sw_extent_SJ.shp')).to_crs(2265)
    ww_extent = gpd.read_file(os.path.join(g_d,'warwick_larger.shp')).to_crs(2265)
    k_barrier = gpd.read_file(os.path.join(g_d,'HFB_V7.shp')).to_crs(2265)
    rch_windows = gpd.read_file(os.path.join(g_d,'sw_recharge_window_large.shp')).to_crs(2265)
    modelgrid = gpd.read_file(os.path.join(g_d_out, 'sw_ww_modelgrid.shp'))
    modelgrid = modelgrid.set_crs(2265)
    drains = gpd.read_file(os.path.join(g_d, 'RIV_lines.shp'))

    wls = pd.read_csv(os.path.join('data', 'raw','swww_sites_final.csv'))
    wls['manually_corrected_lay'] = pd.to_numeric(wls['manually_corrected_lay'],errors='coerce')
    wls = wls.dropna(subset='manually_corrected_lay')
    wls['grp_id'] = 'grpid:' + wls['group number'].astype(str) + '_k:' + (wls['manually_corrected_lay'].astype(int)-1).astype(str)

    usites = usitedf['site'].values
    base_export_records = []
    with PdfPages(os.path.join(o_d, 'obs_v_sim_inset_vcross.pdf')) as pdf:
        for site in usites:
            # limit to just target wells:
            if site not in wls['grp_id'].values:
                continue
            uobs = gwobs.loc[gwobs.id == site, :].copy()
            uobs.sort_values(by='datetime', inplace=True)

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
            ax4 = fig.add_subplot(gs[2:, :])

            ax3.set_xticklabels('')
            ax3.set_yticklabels('')
            ax3.tick_params(axis='both', which='both', direction='in')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.tick_params(axis='both', which='both', length=0)

            plot_inset(ax=ax1,
                       wl_loc=site,
                       cpts=cpts,
                       sw_extent=sw_extent,
                       ww_extent=ww_extent,
                       k_barrier=k_barrier,
                       rch_windows=rch_windows,
                       drains=drains)
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

                # ---- Base realization series for this site (same values you plot in orange)
                base_vals = obsdict[itrmx].loc["base", uobs.obsnme].values  # (ntimes,)
                
                # ---- Ensemble matrix for percentiles (drop base so it doesn't affect p05/p95)
                ens_df = obsdict[itrmx].loc[:, uobs.obsnme]
                ens_df = ens_df.drop(index="base", errors="ignore")  # keep only ensemble members
                ens_vals = ens_df.values  # (nreal, ntimes)
                
                
                # ---- Observed values aligned to the same datetimes (NaN if not observed at that date)
                obs_map = (
                    oobs[["datetime", "obsval"]]
                    .dropna()
                    .drop_duplicates(subset=["datetime"])
                    .set_index("datetime")["obsval"]
                )
                
                # ---- Export obs in friendlier file
                for dt in pd.to_datetime(dts):
                    base_export_records.append(
                        {
                            "id": site,
                            "k": int(k),
                            "i": int(i),
                            "j": int(j),
                            "datetime": dt,
                            "obs": float(obs_map.get(dt, np.nan)),
                        }
                    )


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
                ult_mx = midpoint + 20
                #if midpoint - 30 < ult_mn:
                ult_mn = midpoint - 20

                ax4.set_ylim(ult_mn - 10, ult_mx + 10)
                ax4.yaxis.set_major_locator(ticker.MultipleLocator(50))  # Major ticks every 50
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
            # ax4.tick_params(axis='x', labelbottom=False)
            # comma formateed y-axis labels:
            ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))
            # add vertical line for predictive period:
            ax4.axvline(x=pd.to_datetime('2025-01-01'), color='grey', linestyle='-.', linewidth=1)
            ax4.text(pd.to_datetime('2025-01-01'), ax4.get_ylim()[1] - 10, '-> Predictive period', fontsize=10,
                     ha='left', va='bottom', color='grey')

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
            water_levels = pd.read_csv(os.path.join('data','raw','swww_waterlevels.csv'))

            # Filter
            water_levels['date_meas'] = pd.to_datetime(water_levels['date_meas'])
            water_levels = water_levels.set_index('date_meas')
            water_levels.loc[water_levels['gwe_navd88']>10000,'gwe_navd88'] = np.nan
            water_levels.loc[water_levels['gwe_navd88']<-1000,'gwe_navd88'] = np.nan
            water_levels = water_levels.dropna()

            grp_num = int(site.split('_')[0].split(':')[1])
            k = int(site.split('_')[1].split(':')[1])
            sites_grp = wls[wls['group number'] == int(grp_num)]
            grp_full = sites_grp.copy()
            sites_grp = sites_grp[sites_grp['manually_corrected_lay']-1 == int(k)]
            color_cnt = 0
            leg6 = []
            for site_id in sites_grp['loc_id'].unique():
                wl_data = water_levels.loc[water_levels['loc_id']==site_id,'gwe_navd88']

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

            aq_key = {0: "Warwick Aquifer",
                      1: "Confining Unit (clay/till)",
                      2: "Spiritwood Aquifer",
                      }
            current_aq = aq_key.get(k)

            # sort grp_full by manually_corrected_lay
            grp_full = grp_full.sort_values(by=['manually_corrected_lay'])

            ax3.text(0.5, 0.75,
                     f'{current_aq}\n Group: {grp_num}\n Layer: {k + 1}, \nRow: {i + 1}, Column: {j + 1}\nModel top: {t}\n\n',
                     fontsize=12, ha='center', va='center', color='blue', transform=ax3.transAxes)

            # add text that icludes grp_full [loc_id, assigned aquifer]:
            ax3.text(0.5, 0.27, 'All wells in group:\n' + '\n'.join(
                [f"{row['loc_id']} - {aq_key.get(row['manually_corrected_lay']-1)}" for idx, row in grp_full.iterrows()]),
                     fontsize=10, ha='center', va='center', color='black', transform=ax3.transAxes)

            pdf.savefig()
            plt.close(fig)

    pdf.close()

    # Export water levels for base realization
    if len(base_export_records) > 0:
        df_base = pd.DataFrame(base_export_records)
        df_base = df_base.sort_values(["id", "datetime"]).reset_index(drop=True)

        out_csv = os.path.join(o_d, "obs_longform.csv")
        df_base.to_csv(out_csv, index=False)
        print(f"...Wrote base+obs export: {out_csv}")
    else:
        print("...No base+obs records collected; CSV not written.")



def base_posterior_param_forward_run(m_d0, noptmax):

    m_d = m_d0 + '_forward_run_base_temp'

    print('copying dir {0} to {1}'.format(m_d0, m_d))
    shutil.copytree(m_d0, m_d, ignore=shutil.ignore_patterns('*.cbb', '*.hds', '*.log', '*.lst', '*.rec', '*.rei', '*_obs.csv','*.pdf'))


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


def plot_simple_obs_v_sim_base(m_d):

    pst = pyemu.Pst(os.path.join(m_d,'wahp.pst'))

    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('freq|ext|elev'),:].copy()
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)

    heads_file = 'wahp.hds'
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
    hobs = m.obs[2].continuous.data['wahp.head.obs.output.csv']

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


def plot_base_params_map_mel(
    model_ws: Union[str, os.PathLike],
    partype: str,
    sim_name: str = 'swwww',
    layers: Union[str, Sequence[int]] = 'all',
    kstpkper: tuple[int, int] = (0, 0),          # for recharge only
    logscale_hk: bool = True,
    cmap_overrides: dict[str, str] | None = None,
    out_dir: str = 'fig_params',
    dpi: int = 300):
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
                              'wahp', 'wahp_outline_full.shp')
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
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
        print('saved →', png)


def plot_simple_timeseries(m_d, modnm='swww',noptmax=None):
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
            #[ax.plot(dts,nvals[i,:],'r',alpha=0.3,lw=0.2) for i in range(nvals.shape[0])]

            dts = gobs.datetime.values
            ptvals = pst.ies.__getattr__('obsen{0}'.format(itr)).loc[:,gobs.obsnme].values
            # if any value in ptvals is >1500 or less than 2000 then drop the dataset
            # ptvals = np.where(ptvals>980,np.nan,ptvals)
            # ptvals = np.where(ptvals<-1,np.nan,ptvals)

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


def plot_simple_1to1(m_d, modnm='swww'):
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


def plot_simple_par_histo(m_d, modnm='swww'):
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


def plot_simple_par_maps(m_d,modnm,noptmax=None,log=False):
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
                if log:
                    prarr[kobs.i,kobs.j] = np.log10(pr.loc[:,kobs.obsnme].values).mean(axis=0)
                else:
                    prarr[kobs.i,kobs.j] = pr.loc[:,kobs.obsnme].values.mean(axis=0)
                ptarr = np.zeros((nrow,ncol))
                if log:
                    ptarr[kobs.i,kobs.j] = np.log10(pt.loc[:,kobs.obsnme].values).mean(axis=0)
                else:
                    ptarr[kobs.i,kobs.j] = pt.loc[:,kobs.obsnme].values.mean(axis=0)
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
                if log:
                    prarr[kobs.i,kobs.j] = np.log10(pr.loc[:,kobs.obsnme].values).std(axis=0)
                else:
                    prarr[kobs.i,kobs.j] = pr.loc[:,kobs.obsnme].values.std(axis=0)
                ptarr = np.zeros((nrow,ncol))
                if log:
                    ptarr[kobs.i,kobs.j] = np.log10(pt.loc[:,kobs.obsnme].values).std(axis=0)
                else:
                    ptarr[kobs.i,kobs.j] = pt.loc[:,kobs.obsnme].values.std(axis=0)

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
                    if log:
                        prarr[kobs.i,kobs.j] = np.log10(pr.loc[real,kobs.obsnme].values)
                    else:
                        prarr[kobs.i,kobs.j] = pr.loc[real,kobs.obsnme].values
                    ptarr = np.zeros((nrow,ncol))
                    if log:
                        ptarr[kobs.i,kobs.j] = np.log10(pt.loc[real,kobs.obsnme].values)
                    else:
                        ptarr[kobs.i,kobs.j] = pt.loc[real,kobs.obsnme].values

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


def plot_layer_one2one_wdepth(m_d,obsdict, modnm='swww'):
    o_d = os.path.join(m_d, 'results', 'figures', 'one2one_plots')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    #pwl = pd.read_csv(os.path.join(m_d, 'tables', 'processed_WL_timeseries.csv'))

    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6',load_only=['dis'])
    m = sim.get_model(modnm)
    nlay = m.dis.nlay.array
  
    layers = {
        0: 'WW',
        1: 'Till',
        2: 'SW',
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


def plot_1to1_all_head_targs(nopt_max,modnm,m_d):
    print("...Plotting 1-to-1 scatter for all observations")

    p = os.path.join(m_d,f"{modnm}.{nopt_max}.obs.jcb")
    obs_raw = pyemu.ObservationEnsemble.from_binary(None, p)
    obs = pd.DataFrame(obs_raw,
                       columns=obs_raw.keys(),
                       index=obs_raw.index)

    # Only plot the base run
    obs = obs.loc[obs.index=='base']

    # Parse out the transient head targets
    obs = obs[[x for x in obs.columns if 'trans' in x]]

    # Clean jcb into a useable DataFrame
    s = obs.iloc[0]
    s.name = "head"
    # Regex to parse the column name pieces
    pat = re.compile(
        r"transh_grpid:(?P<transh_grpid>[^_]+)"
        r"_k:(?P<k>-?\d+)"
        r"_i:(?P<i>-?\d+)"
        r"_j:(?P<j>-?\d+)"
        r"_datetime:(?P<datetime>[^_]+)$"
        )
    # Extract fields from column names
    parts = s.index.to_series().str.extract(pat)
    # Build the tidy DataFrame
    out = (
        parts
        .assign(head=s.values)
        .dropna(subset=["transh_grpid", "k", "i", "j", "datetime"])
        )
    # Clean types
    out["k"] = out["k"].astype(int)
    out["i"] = out["i"].astype(int)
    out["j"] = out["j"].astype(int)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    # If transh_grpid is numeric, make it int; otherwise leave as string
    out["transh_grpid"] = pd.to_numeric(out["transh_grpid"], errors="ignore")
    # Reorder columns
    out = out[["transh_grpid", "k", "i", "j", "datetime", "head"]]

    # Get the observations
    targs = pd.read_csv(os.path.join(m_d,f'{modnm}.obs_data.csv'))
    targs = targs.loc[targs['obsnme'].str.contains('trans')]

    # Define key index wells and well 48 which I want to skip for now...
    index_wells = ['3','5','11','13','29','48','59','60','71','76','88','93',
                   '99','100','112','116','118','121','122','123','127',
                   '137','138','168','169','219','220','221']

    bad_wells = ['187','47','48','215']

    # Track RMSE
    sse_total = 0.0
    n_total = 0
    sse_total_idx = 0.0
    n_total_idx = 0
    # MAE
    mae_total = 0.0
    mae_total_idx = 0.0
    # ME (bias)
    me_total = 0.0
    me_total_idx = 0.0

    # --- per-layer accumulators ---
    sse_by_k = defaultdict(float)
    n_by_k = defaultdict(int)
    mae_by_k = defaultdict(float)
    me_by_k = defaultdict(float)
    
    # (optional) per-layer for index wells too, if you want
    sse_by_k_idx = defaultdict(float)
    n_by_k_idx = defaultdict(int)
    mae_by_k_idx = defaultdict(float)
    me_by_k_idx = defaultdict(float)

    

    # Plot each target as a scatter on each
    residuals = pd.DataFrame()
    with PdfPages(os.path.join(m_d,'results','one_to_one_allTargs.pdf')) as pdf:
        max_er = 0
        max_er_targ = 'temp'
        fig,ax = plt.subplots(figsize=(7,7))
        for grp_id in out['transh_grpid'].unique():
            if str(int(grp_id)) in index_wells:
                index = True
            else:
                index = False

            sim = out.loc[out['transh_grpid'] == grp_id]
            obs = targs.loc[targs['grpid'] == grp_id]

            # Skip de-weighted targets
            if str(int(grp_id)) in bad_wells:
                continue

            for k in sim['k'].unique():
                # Skip confining unit targets
                if k == 1:
                    continue

                _sim = sim.loc[sim['k'] == k]
                _sim = _sim.set_index('datetime')

                # Obs
                _obs = obs.loc[obs['k'] == k, ['obsval','datetime']]
                _obs = _obs.set_index(pd.to_datetime(_obs['datetime']))

                # Obs with Sim
                dat = pd.concat([_sim,_obs],axis=1)
                dat = dat.loc[dat['obsval'] > 0]
                dat = dat.drop_duplicates(subset='datetime')

                # One special case
                if grp_id == '213':
                    dat = dat.loc[dat['datetime'] < pd.Timestamp("1994-01-01")]

                # Calc errors
                err = (dat['head'] - dat['obsval']).to_numpy()
                residuals = pd.concat([residuals,dat['head'] - dat['obsval']],axis=0)
                if len(err) > 0:
                    if max(abs(err)) > max_er:
                        max_er_targ = grp_id
                sse_total += float((err**2).sum())
                n_total += err.size
                mae_total += float(np.abs(err).sum())
                me_total += float(err.sum())
                if index:
                    sse_total_idx += float((err**2).sum())
                    n_total_idx += err.size
                    mae_total_idx += float(np.abs(err).sum())
                    me_total_idx += float(err.sum())
                    sse_by_k_idx[k] += float((err**2).sum())
                    n_by_k_idx[k] += err.size
                    mae_by_k_idx[k] += float(np.abs(err).sum())
                    me_by_k_idx[k] += float(err.sum())
                
                # --- per-layer totals ---
                sse_by_k[k] += float((err**2).sum())
                n_by_k[k] += err.size
                mae_by_k[k] += float(np.abs(err).sum())
                me_by_k[k] += float(err.sum())

                
                # Plot
                ax.plot(dat['obsval'].to_numpy(),
                        dat['head'].to_numpy(),
                        linestyle='None',
                        marker='o',
                        ms=3 if index else 1.5,
                        alpha=0.7,
                        # Color by aquifer
                        color=('tab:blue' if k == 2 else 'tab:orange'),
                        markeredgecolor='k' if index else None
                        )

                # Export a nice example
                # if grp_id == 71:
                #     dat.to_csv(os.path.join('data','analyzed','ex_sim_ts_SW_71.csv'))
                # if grp_id == 169:
                #     dat.to_csv(os.path.join('data','analyzed','ex_sim_ts_WW_169.csv'))

        print(f"group with max error = {max_er_targ}")
        overall_rmse = round(np.sqrt(sse_total / n_total),2)
        # overall_nrmse = round((overall_rmse / (targs.obsval.max() - targs.obsval.min())) * 100,2)
        # overall_mae = round(mae_total / n_total, 2)

        index_rmse = round(np.sqrt(sse_total_idx / n_total_idx),2)
        # # Need to filter to only index wells in the targets
        # index_nrmse = round((index_rmse / (targs.loc[targs['grpid'].astype(int).astype(str).isin(index_wells)].obsval.max() - targs.loc[targs['grpid'].astype(int).astype(str).isin(index_wells)].obsval.min())) * 100,2)
        # index_mae = round(mae_total_idx / n_total_idx, 2)

        # # Mean error
        # overall_me = round(me_total / n_total, 2)
        # index_me = round(me_total_idx / n_total_idx, 2)

        # -- Pring some info
        # All targets
        # print(f"RMSE all = {overall_rmse}")
        # print(f"Normalized RMSE all = {overall_nrmse}%")
        # print(f"MAE all = {overall_mae}")
        # print(f"ME all = {overall_me}\n")

        # # Index
        # print(f"RMSE index = {index_rmse}")
        # print(f"Normalized RMSE index = {index_nrmse}%")
        # print(f"MAE index = {index_mae}")
        # print(f"ME all = {index_me}\n")

        # --- Per-layer metrics (all targets) ---
        print("Per-layer metrics (all targets):")
        for k in sorted(n_by_k.keys()):
            if n_by_k[k] == 0:
                continue
        
            rmse_k = np.sqrt(sse_by_k[k] / n_by_k[k])
            # normalize by observed range within that layer's targets
            # Grab layer
            obs_k = targs.loc[targs["k"] == k]
            # Drop bad targs
            obs_k = obs_k.loc[~obs_k['grpid'].isin(bad_wells)]
            # Grab non zero vals
            obs_k = obs_k.loc[obs_k['obsval'] != 0,'obsval']
            
            rng_k = (obs_k.max() - obs_k.min())
            nrmse_k = np.nan if (pd.isna(rng_k) or rng_k == 0) else (rmse_k / rng_k) * 100
            mae_k = mae_by_k[k] / n_by_k[k]
            me_k = me_by_k[k] / n_by_k[k]
            print(f"  Layer k={k}: RMSE={rmse_k:.2f}, range={rng_k}, nRMSE={nrmse_k:.2f}%, MAE={mae_k:.2f}, ME={me_k:.2f}")
        
        # --- Per-layer metrics (index wells) [optional] ---
        print("\nPer-layer metrics (index wells):")
        targs_idx = targs.loc[targs["grpid"].astype(int).astype(str).isin(index_wells)]
        for k in sorted(n_by_k_idx.keys()):
            if n_by_k_idx[k] == 0:
                continue
        
            rmse_k = np.sqrt(sse_by_k_idx[k] / n_by_k_idx[k])
            # Grab layer
            obs_k = targs.loc[targs["k"] == k]
            # Drop bad targs
            obs_k = obs_k.loc[~obs_k['grpid'].isin(bad_wells)]
            # Grab non zero vals
            obs_k = obs_k.loc[obs_k['obsval'] != 0,'obsval']
            rng_k = (obs_k.max() - obs_k.min())
            nrmse_k = np.nan if (pd.isna(rng_k) or rng_k == 0) else (rmse_k / rng_k) * 100
            mae_k = mae_by_k_idx[k] / n_by_k_idx[k]
            me_k = me_by_k_idx[k] / n_by_k_idx[k]
            print(f"  Layer k={k}: RMSE={rmse_k:.2f}, range={rng_k}, nRMSE={nrmse_k:.2f}%, MAE={mae_k:.2f}, ME={me_k:.2f}")


        # One to One line
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        m = min(x0, y0)
        M = max(x1, y1)
        m = 1360
        ax.plot([m, M], [m, M], '--', color='k', lw=1, zorder=1)
        ax.set_ylim(m,M)
        ax.set_xlim(m,M)

        # Circle around SW-Sheyenne Targets
        ax.add_artist(Circle([1375,1375],
                             radius=11,
                             color='k',
                             fill=False
                             )
                      )
        ax.text(1366,1389,
                'SW-Sheyenne\nTargets:',
                fontsize=9
                )

        # Custom legend
        handles = [Line2D([],[],color='tab:blue',ls='',marker='o',label='Spiritwood',
                          markersize=6),
                   Line2D([],[],color='tab:orange',ls='',marker='o',label='Warwick',
                          markersize=6),
                   Line2D([],[],mfc=None,ls='',marker='o',label='Black Border\nindicates INDEX',
                          markersize=6,
                          mec='k'),
                   Line2D([],[],color='k',ls='--',label='1-to-1 Line')]
        ax.legend(handles=handles,
                  loc=4,
                  framealpha=1,
                  frameon=True)
        ax.set_title("Observed vs Simulated Heads - All Groups")
        ax.text(1362,1478,
                f"Overall RMSE = {overall_rmse} ft, N = {n_total}\nIndex Well RMSE = {index_rmse}, N = {n_total_idx}",
                fontsize=9,
                ha='left',
                va='top',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=0.8,
                    alpha=0.9
                    )
                )
        ax.set_xlabel("Observed (ft NAVD88)")
        ax.set_ylabel("Simulated (ft NAVD88)")
        ax.grid()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------
        # --- Plot residual histogram
        # ---------------------------
        fig,ax = plt.subplots(figsize=(7,7))
        residuals.plot.hist(ax=ax,
                            bins=40,
                            edgecolor='k',
                            legend=False)
        mean_res = residuals.mean().values[0]
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Groundwater-level residual (observed minus simulated)\nin feet above average sea level')
        ax.set_title(f'Groundwater-level Observation Residuals')

        ax.axvline(mean_res,color='k',ls='--',
                   label=f'Mean Residual = {round(mean_res,1)} feet')

        h,_ = ax.get_legend_handles_labels()
        h = [h[-1]]
        ax.legend(handles=h,
                  loc=1)
        ax.grid()

        pdf.savefig(fig)
        plt.close(fig)



def model_packages_to_shp(d='.'):
    sim = flopy.mf6.MFSimulation.load(sim_ws=d)
    mf = sim.get_model('swww')
    mf._modelgrid._angrot = 0.0
    mf._modelgrid._xoff = 2388853.4424208435
    mf._modelgrid._yoff = 260219.09632163405
    epsg = 2265

    # Create model package shapefile directory
    o_d = os.path.join(d,'output_shapefiles','model_packages')
    if not os.path.exists(o_d):
        os.makedirs(o_d)

    # list packages:
    mf.get_package_list()

    # dis
    mf.dis.top.export(os.path.join(o_d, 'top.shp'), epsg=epsg)
    mf.dis.botm.export(os.path.join(o_d, 'bottoms.shp'), epsg=epsg)

    # --- DRN files (3 total packages)
    # Surface drns
    mf.drn.stress_period_data.export(os.path.join(o_d, 'drn.shp'), epsg=epsg)
    drn = gpd.read_file(os.path.join(o_d,'drn.shp'))
    drn['elev'] = drn.elev11
    drn['cond'] = drn.cond11
    # Remove cells that don't have drains
    drn = drn[drn.elev!=0]
    # Drop unneccessary columns
    drn = drn.loc[:, ['node','row','column','elev','cond', 'geometry']]
    drn = drn.set_crs('epsg:2265', allow_override=True)
    drn.to_file(os.path.join(o_d, 'surface_drn.shp'))

    # Sheyenne Valley drns
    mf.drn_valley.stress_period_data.export(os.path.join(o_d, 'drn.shp'), epsg=epsg)
    drn = gpd.read_file(os.path.join(o_d,'drn.shp'))
    drn['elev'] = drn.elev11
    drn['cond'] = drn.cond11
    # Remove cells that don't have drains
    drn = drn[drn.elev!=0]
    # Drop unneccessary columns
    drn = drn.loc[:, ['node','row','column','elev','cond', 'geometry']]
    drn = drn.set_crs('epsg:2265', allow_override=True)
    drn.to_file(os.path.join(o_d, 'valley_drn.shp'))

    # Sheyenne River drn (SW-Sheyenne connection)
    mf.drn_riv.stress_period_data.export(os.path.join(o_d, 'drn.shp'), epsg=epsg)
    drn = gpd.read_file(os.path.join(o_d,'drn.shp'))
    drn['elev'] = drn.elev31
    drn['cond'] = drn.cond31
    # Remove cells that don't have drains
    drn = drn[drn.elev!=0]
    # Drop unneccessary columns
    drn = drn.loc[:, ['node','row','column','elev','cond', 'geometry']]
    drn = drn.set_crs('epsg:2265', allow_override=True)
    drn.to_file(os.path.join(o_d, 'sw_sheyenne_drn.shp'))

    # --- riv
    mf.riv.stress_period_data.export(os.path.join(o_d, 'riv.shp'), epsg=epsg)
    riv = gpd.read_file(os.path.join(o_d,'riv.shp'))
    riv['stage'] = riv.stag21
    riv['cond'] = riv.cond21
    riv['rbot'] = riv.rbot21
    # Remove cells that don't have rivs
    riv = riv[riv.stage != 0]
    # Drop unneccessary columns
    riv = riv.loc[:, ['node','row','column','stage','cond', 'rbot', 'geometry']]
    riv = riv.set_crs('epsg:2265', allow_override=True)
    riv.to_file(os.path.join(o_d, 'riv.shp'))

    # ghb edge cells
    mf.ghb.stress_period_data.export(os.path.join(o_d, 'ghb3.shp'), epsg=epsg)
    ghb_edge = gpd.read_file(os.path.join(o_d, 'ghb3.shp'))
    # Remove cells and columns  that don't have ghbs
    cols_active = [f for f in ghb_edge.columns[:-1] if ghb_edge[f].sum() !=0]
    cols_active.append('geometry')
    ghb_edge = ghb_edge.loc[:, cols_active]
    ghb_edge = ghb_edge.set_crs('epsg:2265', allow_override=True)
    ghb_edge = ghb_edge.loc[ghb_edge['bhea31']>0]
    ghb_edge.to_file(os.path.join(o_d, 'ghb_edge.shp'))

    # recharge
    # mf.rch.export(os.path.join(o_d,'rch.shp'),epsg=epsg)
    # rch = gpd.read_file(os.path.join(o_d, 'rch.shp'))
    # rch = rch.loc[:, [f for f in rch.columns if 'irch' not in f]]
    # rch = rch.set_crs('epsg:2265', allow_override=True)
    # rch.to_file(os.path.join(o_d, 'rch.shp'))

    # npf
    # # get list of npf_icelltype files in dir:
    # icell_files = [f for f in os.listdir(d) if f.startswith('npf_icelltype')]
    # for file in icell_files:
    #     # read in with numpy
    #     icelltype_data = np.loadtxt(os.path.join(d, file))
    #     # change everything to ints:
    #     icelltype_data = icelltype_data.astype(int)
    #     # write to a new file:
    #     np.savetxt(os.path.join(d, file), icelltype_data, fmt='%i')

    mf.npf.export(os.path.join(o_d,'npf.shp'), epsg=epsg)
    npf = gpd.read_file(os.path.join(o_d, 'npf.shp'))
    #npf = npf.loc[:, [f for f in npf.columns if 'icell' not in f]]
    npf = npf.set_crs('epsg:2265', allow_override=True)
    npf.to_file(os.path.join(o_d, 'npf.shp'))

    # sto
    mf.sto.export(os.path.join(o_d,'sto.shp'), epsg=epsg)

    # wels
    mf.wel.export(os.path.join(o_d,'wel.shp'), epsg=epsg)
    df = gpd.read_file(os.path.join(o_d,'wel.shp'))
    df = df.set_crs('epsg:2265', allow_override=True)
    df.to_file(os.path.join(o_d, 'wel.shp'))


def base_prior_hydParam_compare(base_dir,
                                plot=True):
    CRS = 2265
    # Load prior data
    prior_d = os.path.join('model_ws','swww_clean')
    sim = flopy.mf6.MFSimulation.load(sim_ws=prior_d)
    mf_pr = sim.get_model('swww')

    # Load base realization of posterior
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_dir)
    mf_post = sim.get_model('swww')

    # Calc thickness for trans
    top = mf_pr.dis.top.array
    botm = mf_pr.dis.botm.array
    thk = np.zeros(shape=(3,top.shape[0],top.shape[1]))
    thk[0,:,:] = top - botm[0,:,:]
    thk[1,:,:] = botm[0,:,:] - botm[1,:,:]
    thk[2,:,:] = botm[1,:,:] - botm[2,:,:]

    # gis dir
    gis_dir = os.path.join('..','..','gis')

    # Load the grid shapefile for plotting
    mg = gpd.read_file(os.path.join(gis_dir,'output_shps','sw_ww','sw_ww_modelgrid.shp')).set_crs(2265)
    mod_bnd = gpd.GeoDataFrame(geometry=[mg.union_all()])
    # Load relevant shapefiles
    # TODO Update this shapefile
    pumping_wells = gpd.read_file(os.path.join(gis_dir,'input_shps','sw_ww','spiritwood_warwick_wateruse_wells_combined.shp')).to_crs(2265)
    ww_shp = gpd.read_file(os.path.join(gis_dir,'input_shps','sw_ww','warwick_larger_noSwShey.shp')).to_crs(2265)
    sw_shp = gpd.read_file(os.path.join(gis_dir,'input_shps','sw_ww','sw_extent_SJ.shp')).to_crs(2265)
    sw_shp = sw_shp.clip(mod_bnd)
    flow_barrier = gpd.read_file(os.path.join(gis_dir,'input_shps','sw_ww','HFB_V7.shp')).to_crs(2265)
    # rch_windows = gpd.read_file(os.path.join(gis_dir,'input_shps','sw_ww','sw_recharge_window_large.shp')).to_crs(2265)
    shey_riv = gpd.read_file(os.path.join(gis_dir,'input_shps','sw_ww','RIV_lines.shp')).to_crs(2265)
    shey_riv = shey_riv.loc[shey_riv['gnis_name']=='Sheyenne River']
    shey_riv = shey_riv.clip(mod_bnd)

    # Load the grid with parameter values
    idom = mf_pr.dis.idomain.array
    for i in range(3):
        # --- Prior
        mg[f'hk_{i}_pr'] = np.where(idom[i,:,:].flatten() == 1, mf_pr.npf.k.array[i,:,:].flatten(), np.nan)        
        
        # Convert VK ratio to actual VK
        mg[f'vk_{i}_pr'] = np.where(idom[i,:,:].flatten() == 1, mf_pr.npf.k33.array[i,:,:].flatten() * mf_pr.npf.k.array[i,:,:].flatten(), np.nan)
        mg[f'sy_{i}_pr'] = np.where(idom[i,:,:].flatten() == 1, mf_pr.sto.sy.array[i,:,:].flatten(), np.nan)
        mg[f'ss_{i}_pr'] = np.where(idom[i,:,:].flatten() == 1, mf_pr.sto.ss.array[i,:,:].flatten(), np.nan)
        # Calc transmissivity
        mg[f't_{i}_pr'] = mg[f'hk_{i}_pr'] * thk[i,:,:].flatten()

        # Prior -> log scale
        mg[f'hk_{i}_pr_log'] = np.where(mg[f'hk_{i}_pr'] == 0, 0, np.log10(mg[f'hk_{i}_pr']))
        # Convert VK ratio to actual VK
        mg[f'vk_{i}_pr_log'] = np.where(mg[f'vk_{i}_pr'] == 0, 0, np.log10(mg[f'vk_{i}_pr'] * mg[f'hk_{i}_pr']))
        mg[f'sy_{i}_pr_log'] = np.where(mg[f'sy_{i}_pr'] == 0, 0, np.log10(mg[f'sy_{i}_pr']))
        mg[f'ss_{i}_pr_log'] = np.where(mg[f'ss_{i}_pr'] == 0, 0, np.log10(mg[f'ss_{i}_pr']))
        # Calc transmissivity
        mg[f't_{i}_pr_log'] = np.where(mg[f'hk_{i}_pr'] == 0, 0, np.log10(mg[f'hk_{i}_pr'] * thk[i,:,:].flatten()))

        # --- Posterior
        mg[f'hk_{i}_po'] = np.where(idom[i,:,:].flatten() == 1, mf_post.npf.k.array[i,:,:].flatten(), np.nan)
        
        # Print out HK stats
        hk = mg[f'hk_{i}_po'].values
        avg = np.nanmean(hk)
        ma = np.nanmax(hk)
        mi = np.nanmin(hk)
        q25 = np.nanquantile(hk,q=0.25)
        q75 = np.nanquantile(hk,q=0.75)
        std = np.nanstd(hk)
        print(f"\n------ Layer {i} HK (posterior) summary stats ------")
        print(f"Mean        : {avg: .4e}")
        print(f"Std Dev     : {std: .4e}")
        print(f"Min         : {mi: .4e}")
        print(f"25th pctile : {q25: .4e}")
        print(f"75th pctile : {q75: .4e}")
        print(f"Max         : {ma: .4e}")
        
        # Convert VK ratio to actual VK
        mg[f'vk_{i}_po'] = np.where(idom[i,:,:].flatten() == 1, mf_post.npf.k33.array[i,:,:].flatten() * mf_post.npf.k.array[i,:,:].flatten(), np.nan)
        
        # Print out VK stats
        vk = mg[f'vk_{i}_po'].values
        avg = np.nanmean(vk)
        ma = np.nanmax(vk)
        mi = np.nanmin(vk)
        q25 = np.nanquantile(vk,q=0.25)
        q75 = np.nanquantile(vk,q=0.75)
        std = np.nanstd(vk)
        print(f"\n------ Layer {i} VK (posterior) summary stats ------")
        print(f"Mean        : {avg: .4e}")
        print(f"Std Dev     : {std: .4e}")
        print(f"Min         : {mi: .4e}")
        print(f"25th pctile : {q25: .4e}")
        print(f"75th pctile : {q75: .4e}")
        print(f"Max         : {ma: .4e}")
        
        mg[f'sy_{i}_po'] = np.where(idom[i,:,:].flatten() == 1, mf_post.sto.sy.array[i,:,:].flatten(), np.nan)
        mg[f'ss_{i}_po'] = np.where(idom[i,:,:].flatten() == 1, mf_post.sto.ss.array[i,:,:].flatten(), np.nan)
        # Print out HK stats
        ss = mg[f'ss_{i}_po'].values
        avg = np.nanmean(ss)
        ma = np.nanmax(ss)
        mi = np.nanmin(ss)
        q25 = np.nanquantile(ss,q=0.25)
        q75 = np.nanquantile(ss,q=0.75)
        std = np.nanstd(ss)
        print(f"\n------ Layer {i} SS (posterior) summary stats ------")
        print(f"Mean        : {avg: .4e}")
        print(f"Std Dev     : {std: .4e}")
        print(f"Min         : {mi: .4e}")
        print(f"25th pctile : {q25: .4e}")
        print(f"75th pctile : {q75: .4e}")
        print(f"Max         : {ma: .4e}")
        input("holding....")

        
        mg[f't_{i}_po'] = mg[f'hk_{i}_po'] * thk[i,:,:].flatten()

        # Posterior -> log scale
        mg[f'hk_{i}_po_log'] = np.where(mg[f'hk_{i}_po'] == 0, 0, np.log10(mg[f'hk_{i}_po']))
        # Convert VK ratio to actual VK
        mg[f'vk_{i}_po_log'] = np.where(mg[f'vk_{i}_po'] == 0, 0, np.log10(mg[f'vk_{i}_po'] * mg[f'hk_{i}_po']))
        mg[f'sy_{i}_po_log'] = np.where(mg[f'sy_{i}_po'] == 0, 0, np.log10(mg[f'sy_{i}_po']))
        
        sy = mg[f'sy_{i}_po'].values
        avg = np.nanmean(sy)
        ma = np.nanmax(sy)
        mi = np.nanmin(sy)
        q25 = np.nanquantile(sy,q=0.25)
        q75 = np.nanquantile(sy,q=0.75)
        std = np.nanstd(sy)
        print(f"\n------ Layer {i} SY (posterior) summary stats ------")
        print(f"Mean        : {avg: .4e}")
        print(f"Std Dev     : {std: .4e}")
        print(f"Min         : {mi: .4e}")
        print(f"25th pctile : {q25: .4e}")
        print(f"75th pctile : {q75: .4e}")
        print(f"Max         : {ma: .4e}")
        input("holding....")
        
        mg[f'ss_{i}_po_log'] = np.where(mg[f'ss_{i}_po'] == 0, 0, np.log10(mg[f'ss_{i}_po']))
        mg[f't_{i}_po_log'] = np.where(mg[f'hk_{i}_po'] == 0, 0, np.log10(mg[f'hk_{i}_po'] * thk[i,:,:].flatten()))

    # Save the modelgrid with all the NPF/STO information
    mg.to_file(os.path.join('output_shapefiles','mg_with_props.shp'))

    # Load basemap tiles
    west, south, east, north = mg.to_crs(3857).total_bounds
    img, ext = cx.bounds2img(west,
                             south,
                             east,
                             north,
                             # source=cx.providers.USGS.USTopo
                             )
    img,ext = cx.warp_tiles(img,ext,t_crs=2265)
    if plot:
        # Plot parameter fields and differences -> Normal and log scale
        os.makedirs(os.path.join(base_dir,'results','par_maps'),exist_ok=True)
        with PdfPages(os.path.join(base_dir,'results','par_maps','parm_compare.pdf')) as pdf:
            for param in ['hk','vk','sy','ss','t']:
                print(f"...Plotting {param} compare")
                for i in range(3):
                    # Skip a few that do not matter
                    if param == 'sy' and i in [1,2]:
                        continue
                    if param == 'ss' and i in [0]:
                        continue
                    # Init figure for current param
                    fig,axes = plt.subplots(2,3,figsize=(18,12))

                    # Loop through and plot normal and log scale
                    for row in range(2):
                        # Calc vmin and vmax for the colorbars
                        vmin = np.nanmin(mg[f'{param}_{i}_po']) if row == 0 else np.nanmin(mg[f'{param}_{i}_po_log'])
                        vmax = np.nanmax(mg[f'{param}_{i}_po']) if row == 0 else np.nanmax(mg[f'{param}_{i}_po_log'])

                        if param == 'sy':
                            vmin = 0.07 if row == 0 else np.log10(0.07)
                            vmax = 0.18 if row == 0 else np.log10(0.18)

                        # Colormap
                        cmap = 'viridis'

                        # Plot prior
                        axes[row,0].set_title(f"{param}: Prior" if row == 0 else f"Log10 {param}: Prior")
                        mg.plot(ax=axes[row,0],
                                column=f'{param}_{i}_pr' if row == 0 else f'{param}_{i}_pr_log',
                                legend=True,
                                legend_kwds={'shrink':0.6},
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax
                                )
                        # Plot posterior
                        axes[row,1].set_title(f"{param}: Base of Posterior" if row == 0 else f"Log10 {param}: Base of Posterior")
                        mg.plot(ax=axes[row,1],
                                column=f'{param}_{i}_po' if row == 0 else f'{param}_{i}_po_log',
                                legend=True,
                                legend_kwds={'shrink':0.6},
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax
                                )

                        # Calc and plot difference
                        if row == 0:
                            mg['diff'] = mg[f'{param}_{i}_po'] - mg[f'{param}_{i}_pr']
                        else:
                            mg['diff'] = mg[f'{param}_{i}_po_log'] - mg[f'{param}_{i}_pr_log']
                        axes[row,2].set_title(f"{param}: Difference" if row == 0 else f"Log10 {param}: Difference")
                        mg.plot(ax=axes[row,2],
                                column='diff',
                                legend=True,
                                legend_kwds={'shrink':0.6},
                                cmap=cmap
                                )

                    # Format plot and add shapefile features
                    for _i,ax in enumerate(axes.ravel()):
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        # rch_windows.plot(ax=ax,
                        #                  alpha=0.3,
                        #                  color='cyan',
                        #                  label='Recharge Windows'
                        #                  )
                        mod_bnd.boundary.plot(ax=ax,
                                              color='k',
                                              label='Model Boundary')
                        ww_shp.boundary.plot(ax=ax,
                                             color='purple',
                                             label='Warwick Boundary')
                        sw_shp.boundary.plot(ax=ax,
                                             color='green',
                                             label='SW Boundary')
                        shey_riv.plot(ax=ax,
                                      color='blue',
                                      label='Sheyenne River')
                        # Pumping wells by layer
                        if i == 0:
                            pumping_wells.loc[pumping_wells['aquifer']=='Warwick'].plot(ax=ax,
                                                                                           color='k',
                                                                                           label='Pumping Wells',
                                                                                           markersize=10
                                                                                           )
                        elif i == 2:
                            pumping_wells.loc[pumping_wells['aquifer']=='Spiritwood'].plot(ax=ax,
                                                                                           color='k',
                                                                                           label='Pumping Wells',
                                                                                           markersize=10
                                                                                           )
                        # Flow barrier for layers 1 and 2
                        if i == 0:
                            flow_barrier.plot(ax=ax,
                                              color='darkgreen',
                                              label='Flow Barrier'
                                              )

                        # Add basemap tiles
                        ylims = ax.get_ylim()
                        xlims = ax.get_xlim()
                        ax.imshow(img,
                                  extent=ext,
                                  origin="upper",
                                  zorder=0,
                                  alpha=0.6)
                        ax.set_ylim(ylims)
                        ax.set_xlim(xlims)
                        # Show legend only on one panel
                        if _i == 0:
                            ax.legend(loc=3,
                                      framealpha=1,
                                      frameon=True)

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)


def plot_zone_histos(m_d, obsdict, logscale=False):
    import matplotlib.ticker as mticker
    fdir = os.path.join(m_d,'results','figures','zone_histos')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    iters = list(obsdict.keys())
    iters.sort()
    obs = obsdict[iters[-1]].pst.observation_data.copy()
    robs = obs.loc[obs.obsnme.str.contains("arr"),:]
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
    sim     = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name="mf6", load_only=['dis'])
    m       = sim.get_model("swww")
    idom    = m.dis.idomain.data      # shape (nlay, nrow, ncol)
    pr      = obsdict[0]._df          # prior DataFrame: index = real IDs, cols = obs names
    pt      = obsdict[iters[-1]]._df  # posterior DataFrame
    robs    = robs                      # your obs‐info DataFrame with ['obsnme','k','i','j'], index=obs names
    zon_arr = np.load('zone_array.npy')
    reals   = pr.index.values


    # --- begin plotting ---
    with PdfPages(os.path.join(fdir, "layer_zone_histos.pdf")) as pdf:
        for par in pars:
            short, longname = par_dict[par]
            # select only obs for this parameter
            parobs = robs[robs.obsnme.str.contains(f"{short}_k")]
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
                        title_zone = "(whole layer)"
                    else:
                        coords = lyobs[['i','j']].values
                        cols = [
                            col for col, (i,j) in zip(lyobs.index, coords)
                            if (zon_arr[k,i,j] == zone) and (idom[k,i,j] > 0)
                        ]
                        title_zone = f"(Zone {zone})"

                    if not cols:
                        continue

                    # extract prior/post values
                    pr_vals = pr.loc[reals, cols].to_numpy().ravel()
                    pt_vals = pt.reindex(index=reals, columns=cols).to_numpy().ravel()
                    pr_vals = pr_vals[~np.isnan(pr_vals)]
                    pt_vals = pt_vals[~np.isnan(pt_vals)]

                    # debug summary
                    print(f"Layer {k+1} {title_zone}: "
                        f"{len(pr_vals)} prior pts, {len(pt_vals)} post pts, "
                        f"prior [{pr_vals.min():.2e}, {pr_vals.max():.2e}], "
                        f"post [{pt_vals.min():.2e}, {pt_vals.max():.2e}]")

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
                            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
                        )

                    # common y‐axis formatting
                    ax.yaxis.set_major_formatter(
                        mticker.FuncFormatter(lambda y, _: f"{int(y):,}")
                    )

                    # plot histograms
                    ax.hist(pr_vals, bins=bins, alpha=0.75, color='0.5', label='Prior')
                    ax.hist(pt_vals, bins=bins, alpha=0.6, color='b',   label='Posterior')
                    ax.set_title(f"Layer {k+1} {title_zone}", fontsize=10)
                    ax.set_xlabel(longname)
                    ax.set_ylabel("Frequency")
                    ax.legend(loc='upper right', fontsize=8)

                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)


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

def run_zb_by_layer(w_d='',modnm='swww', plot=True):

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

def _wel_yearly_pumping_by_layer(sim, gwf, layers=(0, 2), start_date="1970-01-01"):

    AF_PER_FT3 = 1.0 / 43560.0

    # TDIS
    tdis = sim.get_package("tdis")
    pd_data = tdis.perioddata.get_data()
    perlen = np.array(pd_data["perlen"])
    nper = len(perlen)

    # WEL
    wel = gwf.get_package("wel")
    if wel is None:
        raise ValueError("No WEL package found in model.")

    spd_all = wel.stress_period_data.get_data()  # dict: kper -> recarray

    base_date = pd.Timestamp(start_date)
    dates = []
    # one list of volumes per layer
    vols_by_layer = {lay: [] for lay in layers}

    cum_days = 0.0
    for kper in range(nper):
        data = spd_all.get(kper, None)

        # sum q by layer
        qsum_by_layer = {lay: 0.0 for lay in layers}
        if data is not None:
            for rec in data:
                cellid = rec["cellid"]  # e.g., (k, i, j)
                q = rec["q"]
                # assume DIS-style cellid (k, i, j), 0-based layers
                k = cellid[0]
                if k in qsum_by_layer:
                    qsum_by_layer[k] += q

        dt = float(perlen[kper])  # days in this stress period
        date = base_date + pd.to_timedelta(cum_days, unit="D")
        dates.append(date)

        # convert to volume per layer
        for lay in layers:
            qsum = qsum_by_layer[lay]
            vol_ft3 = -qsum * dt  # make pumping positive
            vol_af = vol_ft3 * AF_PER_FT3
            vols_by_layer[lay].append(vol_af)

        cum_days += dt

    # build dataframe: time index = SP start date, columns = layers
    s_dict = {
        lay: pd.Series(vols_by_layer[lay], index=pd.to_datetime(dates))
        for lay in layers
    }

    df = pd.DataFrame(s_dict)

    # yearly sums (calendar years)
    yearly = df.resample("YS").sum()
    yearly.index = yearly.index.year
    yearly = yearly.loc[1970:2023]

    return yearly

def plot_wateruse(base_dir=".", modnm="swww"):
    # ----- Load prior (original) model -----
    prior_d = os.path.join("model_ws", "swww_clean")
    sim_pr = flopy.mf6.MFSimulation.load(sim_ws=prior_d,
                                         load_only=["dis", "tdis", "wel"])
    mf_pr = sim_pr.get_model(modnm)

    # ----- Load base realization of posterior -----
    sim_post = flopy.mf6.MFSimulation.load(sim_ws=base_dir,
                                           load_only=["dis", "tdis", "wel"])
    mf_post = sim_post.get_model(modnm)

    layers = (0, 2)

    # ----- Compute yearly pumping (acre-ft/yr) by layer -----
    pr_yearly = _wel_yearly_pumping_by_layer(sim_pr, mf_pr,
                                             layers=layers,
                                             start_date="1970-01-01")
    pr_yearly['total'] = pr_yearly.sum(axis=1)

    post_yearly = _wel_yearly_pumping_by_layer(sim_post, mf_post,
                                               layers=layers,
                                               start_date="1970-01-01")
    post_yearly['total'] = post_yearly.sum(axis=1)

    # ----- Plot subplots -----
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    colors = ["#1b9e77", "#d95f02"]
    # Warwick Pumping
    ww_df = pd.DataFrame({
            "Original model": pr_yearly[0],
            "Base posterior": post_yearly[0],
            }
        )
    ww_df.plot(kind="line",
               ax=axes[0],
               # width=0.8,
               lw=2,
               marker='o',
               color=colors)
    axes[0].set_ylabel("Pumping (acre-ft/yr)")
    axes[0].set_title("Warwick Pumping")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend(loc=2)

    # Spiritwood Pumping
    sw_df = pd.DataFrame({
            "Original model": pr_yearly[2],
            "Base posterior": post_yearly[2],
            }
        )
    sw_df.plot(kind="line",
               ax=axes[1],
               color=colors,
               lw=2,
               marker='o',
               # width=0.8
               )
    axes[1].set_ylabel("Pumping (acre-ft/yr)")
    axes[1].set_title("Spiritwood Pumping")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(loc=2)

    # Combined Pumping
    total_df = pd.DataFrame({
            "Original model": pr_yearly["total"],
            "Base posterior": post_yearly["total"],
            }
        )
    total_df.plot(kind="line",
                  ax=axes[2],
                  color=colors,
                  lw=2,
                  marker='o',
                  # width=0.8
                  )
    axes[2].set_ylabel("Pumping (acre-ft/yr)")
    axes[2].set_title("Model Total Pumping")
    axes[2].grid(True, alpha=0.4)
    axes[2].legend(loc=2)

    axes[-1].set_xlabel("Year")
    fig.suptitle("Total WEL Pumping by Year and Layer", y=0.98,
                 fontsize=11)
    plt.tight_layout()

    diff = ((total_df['Original model'] - total_df['Base posterior']) / total_df['Original model']).mean()
    print(f"]n ---- Average pumping difference = {diff} ----\n")

    with PdfPages(os.path.join(base_dir,'results','figures','wateruse.pdf')) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def _rch_yearly_total(sim, gwf, start_date="1970-01-01"):
    # --- Time discretization ---
    tdis = sim.get_package("tdis")
    pd_data = tdis.perioddata.get_data()
    perlen = np.array(pd_data["perlen"])
    nper = len(perlen)

    # --- Recharge package ---
    rch = gwf.get_package("rch")
    if rch is None:
        raise ValueError("No RCH package found in model.")

    rch_spd = rch.recharge  # flopy MF6 array
    rch_spd = rch_spd.get_data()
    base_date = pd.Timestamp(start_date)
    dates = []
    vols_in = []
    cum_days = 0.0
    for kper in range(nper):
        dt = float(perlen[kper])  # days in this stress period
        r_arr = rch_spd[kper]
        r_arr = np.array(r_arr)
        # Rch is ft/day, to mult by days to get feet
        vol_ft = np.nansum(r_arr) * dt
        vol_in = (vol_ft * 12) / (175 * 155)
        date = base_date + pd.to_timedelta(cum_days, unit="D")
        dates.append(date)
        vols_in.append(vol_in)
        cum_days += dt

    s = pd.Series(vols_in, index=pd.to_datetime(dates))
    yearly = s.resample("YS").sum()
    yearly.index = yearly.index.year

    # Clip to your model span if desired
    yearly = yearly.loc[1970:2023]

    return yearly


def plot_recharge(base_dir=".", modnm="swww"):
    """
    Plot yearly total recharge (acre-ft/yr) for the prior and base posterior models.
    """
    # ----- Load prior (original) model -----
    prior_d = os.path.join("model_ws", "swww_clean")
    sim_pr = flopy.mf6.MFSimulation.load(sim_ws=prior_d,
                                         load_only=["dis", "tdis", "rch"]
                                         )
    mf_pr = sim_pr.get_model(modnm)

    # ----- Load base realization of posterior -----
    sim_post = flopy.mf6.MFSimulation.load(sim_ws=base_dir,
                                           load_only=["dis", "tdis", "rch"]
                                           )
    mf_post = sim_post.get_model(modnm)

    # ----- Compute yearly total recharge (acre-ft/yr) -----
    pr_yearly_rch = _rch_yearly_total(sim_pr, mf_pr, start_date="1970-01-01")
    post_yearly_rch = _rch_yearly_total(sim_post, mf_post, start_date="1970-01-01")

    # ----- Plot -----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    colors = ["#1b9e77", "#d95f02"]

    rch_df = pd.DataFrame(
            {"Original input": pr_yearly_rch,
            "Base posterior": post_yearly_rch}
            )

    rch_df.plot(kind="line",
                ax=ax,
                color=colors,
                lw=2,
                marker='o',
                )

    ax.set_ylabel("Recharge (inches/year)")
    ax.set_title("Total Model Recharge by Year")
    ax.grid(True, alpha=0.4)
    ax.legend(loc=2)
    ax.set_xlabel("Year")

    fig.suptitle("Total Recharge by Year (Original vs Base Posterior)", y=0.97,
                 fontsize=10)
    plt.tight_layout()

    out_pdf = os.path.join(base_dir, "results", "figures", "recharge.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def plot_idx_residuals_base(nopt_max, modnm, m_d):
    print("...Plotting residuals for index wells")
    p = os.path.join(m_d,f"{modnm}.{nopt_max}.obs.jcb")
    obs_raw = pyemu.ObservationEnsemble.from_binary(None, p)
    obs = pd.DataFrame(obs_raw,
                       columns=obs_raw.keys(),
                       index=obs_raw.index)

    # Only plot the base run
    obs = obs.loc[obs.index=='base']

    # Parse out the transient head targets
    obs = obs[[x for x in obs.columns if 'trans' in x]]

    # Clean jcb into a useable DataFrame
    s = obs.iloc[0]
    s.name = "head"

    # Regex to parse the column name pieces
    pat = re.compile(
        r"transh_grpid:(?P<transh_grpid>[^_]+)"
        r"_k:(?P<k>-?\d+)"
        r"_i:(?P<i>-?\d+)"
        r"_j:(?P<j>-?\d+)"
        r"_datetime:(?P<datetime>[^_]+)$"
        )
    # Extract fields from column names
    parts = s.index.to_series().str.extract(pat)
    # Build the tidy DataFrame
    df = (
        parts
        .assign(head=s.values)
        .dropna(subset=["transh_grpid", "k", "i", "j", "datetime"])
        )
    # Clean types
    df["k"] = df["k"].astype(int)
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    # If transh_grpid is numeric, make it int; otherwise leave as string
    df["transh_grpid"] = pd.to_numeric(df["transh_grpid"], errors="ignore")
    # Reorder columns
    df = df[["transh_grpid", "k", "i", "j", "datetime", "head"]]

    # Get the observations
    targs = pd.read_csv(os.path.join(m_d,f'{modnm}.obs_data.csv'))
    targs = targs.loc[targs['obsnme'].str.contains('trans')]

    # Define key index wells and well 48 which I want to skip for now...
    index_wells = ['3','5','11','13','29','48','59','60','71','76','88','93',
                   '99','100','112','116','118','121','122','123','127',
                   '137','138','168','169','219','220','221']



    # Plot each target as a scatter on each
    with PdfPages(os.path.join(m_d,'results','index_residuals.pdf')) as pdf:
        for grp_id in df['transh_grpid'].unique():
            # Skip if not index well
            if str(int(grp_id)) not in index_wells:
                continue

            sim = df.loc[df['transh_grpid'] == grp_id]
            obs = targs.loc[targs['grpid'] == grp_id]

            for k in sim['k'].unique():
                fig,ax = plt.subplots(figsize=(8,5))
                # Skip confining unit targets
                if k == 1:
                    continue

                # Simulated
                _sim = sim.loc[sim['k'] == k]
                _sim = _sim.set_index('datetime')

                # Observed
                _obs = obs.loc[obs['k'] == k, ['obsval','datetime']]
                _obs = _obs.set_index(pd.to_datetime(_obs['datetime']))

                # Concat to one
                dat = pd.concat([_sim,_obs],axis=1)
                dat = dat.loc[dat['obsval'] > 0]
                dat = dat.drop_duplicates(subset='datetime')

                # Calc RMSE
                err = (dat['head'] - dat['obsval']).to_numpy()
                sse = float((err**2).sum())
                n = err.size
                rmse = round(np.sqrt(sse / n),2)

                # Plot observed
                dat['obsval'].plot(ax=ax,
                                   color='red',
                                   ls='',
                                   marker='o',
                                   markersize=2,
                                   label='Obs')
                # Plot simulated
                dat['head'].plot(ax=ax,
                                 color='blue',
                                 ls='--',
                                 label='Sim')

                ax.legend()
                ax.set_ylabel('Water Level (ft NAVD88)')
                ax.set_xlabel('Date')
                ax.set_title(f"Group {grp_id}, Layer {k+1}\nRMSE = {rmse}")
                ax.grid()

                # ---- NEW: expand y-limits by ±10 ft and set 5-ft ticks ----
                ymin, ymax = ax.get_ylim()
                ymin_new = ymin - 10
                ymax_new = ymax + 10
                ax.set_ylim(ymin_new, ymax_new)

                # Set y-ticks every 5 ft
                tick_min = 5 * np.floor(ymin_new / 5)
                tick_max = 5 * np.ceil(ymax_new / 5)
                ax.set_yticks(np.arange(tick_min, tick_max + 0.1, 5))
                # -----------------------------------------------------------

                pdf.savefig(fig)
                plt.close(fig)

def plot_idx_residuals_base_unified_sw(nopt_max, modnm, m_d,plot_others=False):
    name_map = pd.read_csv(os.path.join('data','raw','swww_sites_final.csv'))

    print("...Plotting residuals for index wells")
    p = os.path.join(m_d,f"{modnm}.{nopt_max}.obs.jcb")
    obs_raw = pyemu.ObservationEnsemble.from_binary(None, p)
    obs = pd.DataFrame(obs_raw,
                       columns=obs_raw.keys(),
                       index=obs_raw.index)

    # Only plot the base run
    obs = obs.loc[obs.index=='base']

    # Parse out the transient head targets
    obs = obs[[x for x in obs.columns if 'trans' in x]]

    # Clean jcb into a useable DataFrame
    s = obs.iloc[0]
    s.name = "head"

    # Regex to parse the column name pieces
    pat = re.compile(
        r"transh_grpid:(?P<transh_grpid>[^_]+)"
        r"_k:(?P<k>-?\d+)"
        r"_i:(?P<i>-?\d+)"
        r"_j:(?P<j>-?\d+)"
        r"_datetime:(?P<datetime>[^_]+)$"
        )
    # Extract fields from column names
    parts = s.index.to_series().str.extract(pat)
    # Build the tidy DataFrame
    df = (
        parts
        .assign(head=s.values)
        .dropna(subset=["transh_grpid", "k", "i", "j", "datetime"])
        )
    # Clean types
    df["k"] = df["k"].astype(int)
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    # If transh_grpid is numeric, make it int; otherwise leave as string
    df["transh_grpid"] = pd.to_numeric(df["transh_grpid"], errors="ignore")
    # Reorder columns
    df = df[["transh_grpid", "k", "i", "j", "datetime", "head"]]

    # Get the observations
    targs = pd.read_csv(os.path.join(m_d,f'{modnm}.obs_data.csv'))
    targs = targs.loc[targs['obsnme'].str.contains('trans')]

    # Index wells to plot for SW
    plotted_index_wells = ['99','13','138',
                           '127','122','118',
                           '100','220','93',
                           '123','116','71']

    if plot_others:
        index_wells = ['3','5','11','13','29','48','59','60','71','76','88','93',
                       '99','100','112','116','118','121','122','123','127',
                       '137','138','168','169','219','220','221']
        plotted_index_wells = [x for x in index_wells if x not in plotted_index_wells]


    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    axes_iter = iter(axes.ravel())
    resids = []
    # Plot each target as a scatter on each
    with PdfPages(os.path.join('.','index_residuals_unified.pdf')) as pdf:
        for grp_id in df['transh_grpid'].unique():
            # Skip if not index well
            if str(int(grp_id)) not in plotted_index_wells:
                continue

            well_name = name_map.loc[(name_map['group number'].astype(str) == str(int(grp_id))) &
                                     (name_map['manually_corrected_lay'] == '3'), 'loc_id']

            if len(well_name) > 0:
                well_name = well_name.values[0]
            else:
                continue

            # Fricken group 48...
            if well_name == "15006223BBB":
                continue

            sim = df.loc[df['transh_grpid'] == grp_id]
            obs = targs.loc[targs['grpid'] == grp_id]

            # Grab plotting axis
            ax = next(axes_iter)
            print(f"Plotting {grp_id}")
            for k in sim['k'].unique():
                # Skip confining unit targets
                if k != 2:
                    continue

                # Simulated
                _sim = sim.loc[sim['k'] == k]
                _sim = _sim.set_index('datetime')

                # Observed
                _obs = obs.loc[obs['k'] == k, ['obsval','datetime']]
                _obs = _obs.set_index(pd.to_datetime(_obs['datetime']))

                # Concat to one
                dat = pd.concat([_sim,_obs],axis=1)
                dat = dat.loc[dat['obsval'] > 0]
                dat = dat.drop_duplicates(subset='datetime')

                # Calc RMSE
                err = (dat['head'] - dat['obsval']).to_numpy()
                me = round(np.mean(err),1)
                resids.append(me)
                sse = float((err**2).sum())
                n = err.size
                rmse = round(np.sqrt(sse / n),2)

                # Plot observed
                dat['obsval'].plot(ax=ax,
                                   color='red',
                                   ls='',
                                   marker='o',
                                   markersize=2,
                                   label='Obs')
                # Plot simulated
                dat['head'].plot(ax=ax,
                                 color='blue',
                                 ls='--',
                                 label='Sim')

                # ax.set_xlabel('Date')
                ax.set_title(f"{well_name}\nMean res = {me}")
                ax.grid()

                # expand y-limits by ±10 ft and set 5-ft ticks
                ymin, ymax = ax.get_ylim()
                ymin_new = ymin - 10
                ymax_new = ymax + 10
                ax.set_ylim(ymin_new, ymax_new)

                # Enforce xlimits
                ax.set_xlim(pd.Timestamp('01-01-1969'),
                            pd.Timestamp('01-01-2025'))

                # Set y-ticks every 10 ft
                tick_increm = 20
                tick_min = tick_increm * np.floor(ymin_new / tick_increm)
                tick_max = tick_increm * np.ceil(ymax_new / tick_increm)
                ax.set_yticks(np.arange(tick_min, tick_max + 0.1, tick_increm))
                ax.set_xlabel('')

                # -----------------------------------------------------------

        axes[0,0].legend()
        for i in range(4):
            axes[i,0].set_ylabel("Water Level")

        for i in range(3):
            for j in range(3):
                axes[i,j].set_xticklabels([])

        if plot_others:
            axes[3,1].axis('off')
            axes[3,2].axis('off')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.savefig('index_res_uni.png',
                    bbox_inches='tight',
                    dpi=250)
        plt.close(fig)

        pd.DataFrame(data=resids,
                     columns=['mean_res']).to_csv('resid_vals_sw.csv')



# --- Quick one to make a plot this only works locally, not on cluster
def process_sw_index():
    name_map = pd.read_csv(os.path.join('data','raw','swww_sites_final.csv'))

    name_map = gpd.GeoDataFrame(data=name_map,
                                geometry=gpd.points_from_xy(name_map['x_2265'], name_map['y_2265'],
                                                            crs=2265)
                                )

    plotted_wells = [
     '15106215BBB','15106220DAD1','15106221BAA',
     '15106224AAA','15106224CCC','15106227AAA2',
     '15106236CCC','15006130ABB','15206207ACA1',
     '15106130AAA','15106213CBB','15106215AAA'
     ]

    # Filter
    name_map = name_map.loc[
        name_map['loc_id'].astype(str).isin(plotted_wells)
    ].copy()

    # Enforce order
    name_map['loc_id'] = name_map['loc_id'].astype(str)
    name_map['loc_id'] = pd.Categorical(
        name_map['loc_id'],
        categories=plotted_wells,
        ordered=True
    )

    # Sort
    name_map = name_map.sort_values('loc_id')
    resids = pd.read_csv('resid_vals_sw.csv')
    name_map['mean_res'] = resids['mean_res'].values
    name_map = name_map[['geometry','mean_res','loc_id']]
    name_map['is_neg'] = 0
    name_map.loc[name_map['mean_res']<0,'is_neg'] = 1
    name_map['label'] = name_map['loc_id'].astype(str) + ': ' + name_map['mean_res'].astype(str)
    name_map.to_file(os.path.join('..','..','gis','output_shps','sw_ww','sw_index_res.shp'))



# plot_sw_index()

def plot_idx_residuals_base_unified_ww(nopt_max, modnm, m_d):
    name_map = pd.read_csv(os.path.join('data','raw','swww_sites_final.csv'))

    print("...Plotting residuals for index wells")
    p = os.path.join(m_d,f"{modnm}.{nopt_max}.obs.jcb")
    obs_raw = pyemu.ObservationEnsemble.from_binary(None, p)
    obs = pd.DataFrame(obs_raw,
                       columns=obs_raw.keys(),
                       index=obs_raw.index)

    # Only plot the base run
    obs = obs.loc[obs.index=='base']

    # Parse out the transient head targets
    obs = obs[[x for x in obs.columns if 'trans' in x]]

    # Clean jcb into a useable DataFrame
    s = obs.iloc[0]
    s.name = "head"

    # Regex to parse the column name pieces
    pat = re.compile(
        r"transh_grpid:(?P<transh_grpid>[^_]+)"
        r"_k:(?P<k>-?\d+)"
        r"_i:(?P<i>-?\d+)"
        r"_j:(?P<j>-?\d+)"
        r"_datetime:(?P<datetime>[^_]+)$"
        )
    # Extract fields from column names
    parts = s.index.to_series().str.extract(pat)
    # Build the tidy DataFrame
    df = (
        parts
        .assign(head=s.values)
        .dropna(subset=["transh_grpid", "k", "i", "j", "datetime"])
        )
    # Clean types
    df["k"] = df["k"].astype(int)
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    # If transh_grpid is numeric, make it int; otherwise leave as string
    df["transh_grpid"] = pd.to_numeric(df["transh_grpid"], errors="ignore")
    # Reorder columns
    df = df[["transh_grpid", "k", "i", "j", "datetime", "head"]]

    # Get the observations
    targs = pd.read_csv(os.path.join(m_d,f'{modnm}.obs_data.csv'))
    targs = targs.loc[targs['obsnme'].str.contains('trans')]

    # Index wells to plot for SW
    plotted_index_wells = ['127','29','123',
                           '59','60','169',
                           '168','116','121',
                           ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes_iter = iter(axes.ravel())
    resids = []
    # Plot each target as a scatter on each
    with PdfPages(os.path.join('.','index_residuals_unified_ww.pdf')) as pdf:
        for grp_id in df['transh_grpid'].unique():
            # Skip if not index well
            if str(int(grp_id)) not in plotted_index_wells:
                continue

            well_name = name_map.loc[(name_map['group number'].astype(str) == str(int(grp_id))) &
                                     (name_map['manually_corrected_lay'] == '1'), 'loc_id'].values[0]

            sim = df.loc[df['transh_grpid'] == grp_id]
            obs = targs.loc[targs['grpid'] == grp_id]

            # Grab plotting axis
            ax = next(axes_iter)
            print(f"Plotting {grp_id}")
            for k in sim['k'].unique():
                # Skip confining unit targets
                if k != 0:
                    continue

                # Simulated
                _sim = sim.loc[sim['k'] == k]
                _sim = _sim.set_index('datetime')

                # Observed
                _obs = obs.loc[obs['k'] == k, ['obsval','datetime']]
                _obs = _obs.set_index(pd.to_datetime(_obs['datetime']))

                # Concat to one
                dat = pd.concat([_sim,_obs],axis=1)
                dat = dat.loc[dat['obsval'] > 0]
                dat = dat.drop_duplicates(subset='datetime')

                # Calc RMSE
                err = (dat['head'] - dat['obsval']).to_numpy()
                me = round(np.mean(err),1)
                resids.append(me)
                sse = float((err**2).sum())
                n = err.size
                rmse = round(np.sqrt(sse / n),2)

                # Plot observed
                dat['obsval'].plot(ax=ax,
                                   color='red',
                                   ls='',
                                   marker='o',
                                   markersize=2,
                                   label='Obs')
                # Plot simulated
                dat['head'].plot(ax=ax,
                                 color='blue',
                                 ls='--',
                                 label='Sim')

                # ax.set_xlabel('Date')
                ax.set_title(f"{well_name}\nMean res = {me}")
                ax.grid()

                # Enforce xlimits
                ax.set_xlim(pd.Timestamp('01-01-1969'),
                            pd.Timestamp('01-01-2025'))

                # expand y-limits by ±10 ft and set 5-ft ticks
                ymin, ymax = ax.get_ylim()
                ymin_new = ymin - 10
                ymax_new = ymax + 10
                ax.set_ylim(ymin_new, ymax_new)

                # Set y-ticks every 10 ft
                tick_increm = 20
                tick_min = tick_increm * np.floor(ymin_new / tick_increm)
                tick_max = tick_increm * np.ceil(ymax_new / tick_increm)
                ax.set_yticks(np.arange(tick_min, tick_max + 0.1, tick_increm))
                ax.set_xlabel('')

        axes[0,0].legend()
        for i in range(3):
            axes[i,0].set_ylabel("Water Level")

        for i in range(2):
            for j in range(3):
                axes[i,j].set_xticklabels([])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.savefig('index_res_uni_ww.png',
                    bbox_inches='tight',
                    dpi=250)
        plt.close(fig)

        pd.DataFrame(data=resids,
                     columns=['mean_res']).to_csv('resid_vals_ww.csv')


def process_ww_index():
    name_map = pd.read_csv(os.path.join('data','raw','swww_sites_final.csv'))

    name_map = gpd.GeoDataFrame(data=name_map,
                                geometry=gpd.points_from_xy(name_map['x_2265'], name_map['y_2265'],
                                                            crs=2265)
                                )

    plotted_wells = [
     '15106220DAD2','15106223ABB3','15106224CCC3',
     '15106227AAA1','15106322CBB','15106325AAB',
     '15006206BBC','15006313BBB','15006315BBB',
     ]

    # Filter
    name_map = name_map.loc[
        name_map['loc_id'].astype(str).isin(plotted_wells)
    ].copy()

    # Enforce order
    name_map['loc_id'] = name_map['loc_id'].astype(str)
    name_map['loc_id'] = pd.Categorical(
        name_map['loc_id'],
        categories=plotted_wells,
        ordered=True
    )

    # Sort
    name_map = name_map.sort_values('loc_id')
    resids = pd.read_csv('resid_vals_ww.csv')

    name_map['mean_res'] = resids['mean_res'].values

    name_map = name_map[['geometry','mean_res','loc_id']]
    name_map['is_neg'] = 0
    name_map.loc[name_map['mean_res']<0,'is_neg'] = 1

    name_map['label'] = name_map['loc_id'].astype(str) + ': ' + name_map['mean_res'].astype(str)

    name_map.to_file(os.path.join('..','..','gis','output_shps','sw_ww','ww_index_res.shp'))


def plot_array_histo(m_d, modnm='swww',
                     noptmax=None, 
                     logbool = True):
    # Zones from IES setup
    ref_zones = np.load('zone_array.npy')
    # Zone 7 is flow barrier, zones 8 and 9 are recharge windows
    
    # ------- Load or define all independent variables ------- #
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name='mf6', load_only=['dis','npf', 'sto', 'chd', 'obs'],
                                      verbosity_level=0,)
    m = flopy.mf6.MFSimulation().load(sim_ws=m_d, load_only=['npf', 'sto', 'chd', 'obs']).get_model()  # for base values
    idom = m.dis.idomain.array
    idom[idom<0] = 0

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data.copy()
    
    SAVEDIR = os.path.join(m_d, 'results', 'figures', 'param_distribs')
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
   
    
    pdf = PdfPages(os.path.join(SAVEDIR, f'app_G2_swww_parhistos.pdf'))
    
    for partype in ['k', 'k33', 'ss', 'sy']:
        groups = [group for group in obs.obgnme.unique() if group.startswith(f'{partype}_k:')]
    
        zon_arr = idom.copy()
        zon_arr[zon_arr==0] = -9999
        zon_arr[0,:,:] = np.where(zon_arr[0,:,:]>0,0,zon_arr[0,:,:])  # layer 1
        zon_arr[1,:,:] = np.where(zon_arr[1,:,:]>0,1,zon_arr[1,:,:])  # layer 2
        zon_arr[2,:,:] = np.where(zon_arr[2,:,:]>0,1,zon_arr[2,:,:])  # layer 3
    
        zon_dict = {0:'Warwick Aquifer',
                    1: 'Confining Unit (Glacial Till)',
                    2: 'Spiritwood Aquifer'
                    }
        par_dict = {'k33': ['Anisotropy Ratio', '[-]'],
                    'k': ['Horizontal Hydraulic Conductivity', '[ft/d]'],
                    'ss': ['Specific Storage', '[1/ft]'],
                    'sy': ['Specific Yield','[-]']}
    
        prior = pyemu.ParameterEnsemble.from_binary(pst=pst,
                                                        filename=os.path.join(m_d,f'{modnm}.0.obs.jcb'))._df
        if noptmax:
            posterior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(m_d,
                                                                                    f'{modnm}.{noptmax}.obs.jcb'))._df
    
        for idx, group in enumerate(groups):
            zones = zon_arr[idx]
            # reorder columns
            cols = []
            for i in range(m.dis.nrow.data):
                for j in range(m.dis.ncol.data):
                    cols.append(f'oname:{group}_otype:arr_i:{i}_j:{j}')
    
            pr_ = prior.loc[:, cols]
            pt_ = posterior.loc[:,cols] if noptmax else None
    
            try:
                pr_base = pr_.loc['base',:].values.reshape((m.dis.nrow.data, m.dis.ncol.data))*idom[idx]
                pt_base = pt_.loc['base',:].values.reshape((m.dis.nrow.data, m.dis.ncol.data))*idom[idx] if pt_ is not None else None
            except:
                pr_base = np.nan
                pt_base = np.nan
    
            pr_arr = pr_.values.reshape((pr_.shape[0], m.dis.nrow.data, m.dis.ncol.data))*idom[idx]
            pt_arr = pt_.values.reshape((pt_.shape[0], m.dis.nrow.data, m.dis.ncol.data))*idom[idx] if pt_ is not None else None
            
            for zone in np.unique(zones):
                if zone < -1:
                    continue
                zone_arr = np.where(zones == zone, 1, 0)
                pr_zn = pr_arr * zone_arr
                pt_zn = pt_arr * zone_arr if pt_arr is not None else None
    
                prb_zn = pr_base * zone_arr
                ptb_zn = pt_base * zone_arr if pt_arr is not None else None
    
    
                pr_zn[pr_zn == 0] = np.nan
                prb_zn[prb_zn == 0 ] = np.nan
                # set zeroes to NaN for histogram
                if pt_zn is not None:
                    pt_zn[pt_zn == 0] = np.nan
                    ptb_zn[ptb_zn == 0] = np.nan
    
                # Create histogram
                fig, ax = plt.subplots(dpi = 200)
                values = pr_zn.flatten() #values.ravel()
    
                # Plot
    
                ax.hist(values, color='k', edgecolor='k', bins=25, alpha=0.15,
                        weights=np.zeros_like(values) + 1. / values.size, label='prior')
    
                # create vertical line of base value
    
                ax.axvline(np.nanmean(prb_zn), color='k', linestyle='--', linewidth=1, label='base prior mean')
    
                if noptmax:
                    values = pt_zn.flatten()
                    # Plot
    
                    ax.hist(values, color='b', edgecolor='b', bins=25, alpha=0.5,
                            weights=np.zeros_like(values) + 1. / values.size, label='posterior'
                            )
                    ax.axvline(np.nanmean(ptb_zn), color='b', linestyle='--', linewidth=1, label='base posterior mean')
    
                if logbool:
                    ax.set_xscale('log')
                                    
                # stats in the title will be of the posterior if present else of prior
                ax.set_title(f'{par_dict[partype][0]} Layer: {idx + 1}, Zone: {zon_dict[zone]}\n' +
                             f'Prior Base Mean:{np.nanmean(prb_zn):.2e}\n' +
                             f'Posterior Base Mean:{np.nanmean(ptb_zn):.2e}\n', fontsize=10)
                # f'Min:{np.nanmin(values):.2e} Max:{np.nanmax(values):.2e}', fontsize=12)
                # ax.set_title('μ')
                ax.set_ylabel('Relative Frequency', fontsize=12)
    
                ax.set_xlabel(f'{par_dict[partype][0]} {par_dict[partype][1]}', fontsize=12)
    
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                # ax.set_ylim([0, 0.14])
                ax.grid()
                ax.legend(fontsize=10)
                plt.tight_layout()
                
                if logbool:
                    plt.savefig(os.path.join(SAVEDIR, f'{partype}_log10_lay{idx + 1}_zn{zone}.png'))
                    pdf.savefig()
                else:
                    plt.savefig(os.path.join(SAVEDIR, f'{partype}_lay{idx + 1}_zn{zone}.png'))
                    pdf.savefig()
                # plt.show()
                
                plt.close()
    # close pdf
    pdf.close()
    
    
def plot_array_histo_zones(m_d, modnm='swww',
                           noptmax=None,
                           logbool=True):

    # Zones from IES setup
    ref_zones = np.load('zone_array.npy')
    # Zone 7 is flow barrier, zones 8 and 9 are recharge windows

    # ------- Load or define all independent variables ------- #
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=m_d, exe_name='mf6',
        load_only=['dis', 'npf', 'sto', 'chd', 'obs'],
        verbosity_level=0,
        )
    m = flopy.mf6.MFSimulation().load(
        sim_ws=m_d, load_only=['npf', 'sto', 'chd', 'obs']
        ).get_model()

    idom = m.dis.idomain.array
    idom[idom < 0] = 0

    pst = pyemu.Pst(os.path.join(m_d, f'{modnm}.pst'))
    obs = pst.observation_data.copy()

    SAVEDIR = os.path.join(m_d, 'results', 'figures', 'param_distribs')
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)

    pdf = PdfPages(os.path.join(SAVEDIR, f'app_G2_swww_parhistos.pdf'))

    for partype in ['k', 'k33', 'ss', 'sy']:
        groups = [group for group in obs.obgnme.unique()
                  if group.startswith(f'{partype}_k:')]

        # your "zones" array (just used to pick which layer we are in)
        zon_arr = idom.copy()
        zon_arr[zon_arr == 0] = -9999
        zon_arr[0, :, :] = np.where(zon_arr[0, :, :] > 0, 0, zon_arr[0, :, :])  # layer 1
        zon_arr[1, :, :] = np.where(zon_arr[1, :, :] > 0, 1, zon_arr[1, :, :])  # layer 2
        zon_arr[2, :, :] = np.where(zon_arr[2, :, :] > 0, 1, zon_arr[2, :, :])  # layer 3

        # keep your dicts
        zon_dict = {0: 'Warwick Aquifer',
                    1: 'Confining Unit (Glacial Till)',
                    2: 'Spiritwood Aquifer'}
        par_dict = {'k33': ['Anisotropy Ratio', '[-]'],
                    'k': ['Horizontal Hydraulic Conductivity', '[ft/d]'],
                    'ss': ['Specific Storage', '[1/ft]'],
                    'sy': ['Specific Yield', '[-]']}

        prior = pyemu.ParameterEnsemble.from_binary(
            pst=pst,
            filename=os.path.join(m_d, f'{modnm}.0.obs.jcb')
        )._df
        posterior = None
        if noptmax:
            posterior = pyemu.ParameterEnsemble.from_binary(
                pst=pst,
                filename=os.path.join(m_d, f'{modnm}.{noptmax}.obs.jcb')
            )._df

        for idx, group in enumerate(groups):
            zones = zon_arr[idx]  # only used for layer 1 in the hacked logic

            # reorder columns
            cols = []
            for i in range(m.dis.nrow.data):
                for j in range(m.dis.ncol.data):
                    cols.append(f'oname:{group}_otype:arr_i:{i}_j:{j}')

            pr_ = prior.loc[:, cols]
            pt_ = posterior.loc[:, cols] if posterior is not None else None

            try:
                pr_base = pr_.loc['base', :].values.reshape(
                    (m.dis.nrow.data, m.dis.ncol.data)
                ) * idom[idx]
                pt_base = (pt_.loc['base', :].values.reshape(
                    (m.dis.nrow.data, m.dis.ncol.data)
                ) * idom[idx]) if pt_ is not None else None
            except Exception:
                pr_base = np.nan
                pt_base = np.nan

            pr_arr = pr_.values.reshape((pr_.shape[0], m.dis.nrow.data, m.dis.ncol.data)) * idom[idx]
            pt_arr = (pt_.values.reshape((pt_.shape[0], m.dis.nrow.data, m.dis.ncol.data)) * idom[idx]) if pt_ is not None else None

            # ------------------------------------------------------------
            # HACK: define "zones to plot" as two masks for layer 2 & 3
            # ------------------------------------------------------------
            plot_masks = []

            if idx == 1:
                # Layer 2: recharge windows (ref_zones == 8 or 9) vs everything else
                rw_mask = (idom[idx] > 0) & np.isin(ref_zones[idx, :, :], [8, 9])
                cu_mask = (idom[idx] > 0) & (~rw_mask)

                plot_masks.append(("Recharge Windows", rw_mask))
                plot_masks.append(("Confining Unit (Glacial Till)", cu_mask))

            elif idx == 2:
                # Layer 3: flow barrier (ref_zones == 7) vs everything else
                fb_mask = (idom[idx] > 0) & (ref_zones[idx, :, :] == 7)
                aq_mask = (idom[idx] > 0) & (~fb_mask)

                plot_masks.append(("Low Flow Barrier", fb_mask))
                plot_masks.append(("Spiritwood Aquifer", aq_mask))

            else:
                # Layer 1: keep existing behavior (unique zones)
                for zone in np.unique(zones):
                    if zone < -1:
                        continue
                    zmask = (idom[idx] > 0) & (zones == zone)
                    # use your zon_dict labeling
                    plot_masks.append((zon_dict.get(int(zone), f"Zone {zone}"), zmask))

            # ------------------------------------------------------------
            # Loop the "plot_masks" instead of looping numeric zones
            # ------------------------------------------------------------
            for zone_label, zmask in plot_masks:
                zone_arr = np.where(zmask, 1, 0)

                pr_zn = pr_arr * zone_arr
                pt_zn = pt_arr * zone_arr if pt_arr is not None else None

                prb_zn = pr_base * zone_arr
                ptb_zn = pt_base * zone_arr if pt_arr is not None else None

                # set zeroes to NaN for histogram
                pr_zn[pr_zn == 0] = np.nan
                prb_zn[prb_zn == 0] = np.nan
                if pt_zn is not None:
                    pt_zn[pt_zn == 0] = np.nan
                    ptb_zn[ptb_zn == 0] = np.nan

                # Create histogram
                fig, ax = plt.subplots(dpi=200)

                values = pr_zn.flatten()
                ax.hist(values,
                        color='k', edgecolor='k', bins=25, alpha=0.15,
                        weights=np.zeros_like(values) + 1. / values.size,
                        label='prior')

                ax.axvline(np.nanmean(prb_zn), color='k', linestyle='--',
                           linewidth=1, label='base prior mean')

                if noptmax and pt_zn is not None:
                    values = pt_zn.flatten()
                    ax.hist(values,
                            color='b', edgecolor='b', bins=25, alpha=0.5,
                            weights=np.zeros_like(values) + 1. / values.size,
                            label='posterior')
                    ax.axvline(np.nanmean(ptb_zn), color='b', linestyle='--',
                               linewidth=1, label='base posterior mean')

                if logbool:
                    ax.set_xscale('log')

                ax.set_title(
                    f'{par_dict[partype][0]} Layer: {idx + 1}, Zone: {zone_label}\n'
                    f'Prior Base Mean:{np.nanmean(prb_zn):.2e}\n'
                    f'Posterior Base Mean:{np.nanmean(ptb_zn):.2e}\n',
                    fontsize=10
                )

                ax.set_ylabel('Relative Frequency', fontsize=12)
                ax.set_xlabel(f'{par_dict[partype][0]} {par_dict[partype][1]}', fontsize=12)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.grid()
                ax.legend(fontsize=10)
                plt.tight_layout()

                # Print out summary information
                print(f"{partype}, layer {idx + 1}")
                print(f'Posterior Base Mean:{np.nanmean(ptb_zn):.2e}')
                print(f'Posterior Base std:{np.nanstd(ptb_zn):.2e}')
                print(f'Posterior Base min:{np.nanmin(ptb_zn):.2e}')
                print(f'Posterior Base 25th:{np.nanquantile(ptb_zn,q=0.25):.2e}')
                print(f'Posterior Base 75th:{np.nanquantile(ptb_zn,q=0.75):.2e}')
                print(f'Posterior Base max:{np.nanmax(ptb_zn):.2e}')
                
                input('Holding....')

                # filenames (safe-ish)
                safe_lbl = zone_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                if logbool:
                    plt.savefig(os.path.join(SAVEDIR, f'{partype}_log10_lay{idx + 1}_{safe_lbl}.png'))
                    pdf.savefig()
                else:
                    plt.savefig(os.path.join(SAVEDIR, f'{partype}_lay{idx + 1}_{safe_lbl}.png'))
                    pdf.savefig()

                plt.close()

    pdf.close()



def plot_total_pumping(
    m_d=".",
    modnm="swww",
    iter_num=4,
    obs_data_csv="model.obs_data.csv",
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
        od = od.loc[od["datetime"] < pd.to_datetime("2024-01-01")].copy()

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
    # Using q5% and q95%
    ax.fill_between(x, pt_ts.quantile(q=0.05,axis=0).values, pt_ts.quantile(q=0.95,axis=0).values, color='lightblue', alpha=0.5, zorder=3, label="Posterior ensemble range")

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
    
    plt.savefig('ensemble_pumping.png',
                dpi=250,
                bbox_inches='tight')
    
    plt.close(fig)


def plot_total_rch(
    m_d=".",
    modnm="swww",
    iter_num=4,
    obs_data_csv="swww.obs_data.csv",
    require_in_suffix=True,
    hide_forecast=True,
    fill_plot=True,
    take_abs=False,
    log_y=False,
    ylims=None,
    ):

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
        od = od.loc[od["datetime"] < pd.to_datetime("2024-01-01")].copy()
 
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
        pt_ts.quantile(q=0.05,axis=0).values,
        pt_ts.quantile(q=0.95,axis=0).values,
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
 
    ax.set_title("Warwick Recharge", loc="left")
    ax.set_ylabel("Recharge (in/yr)")
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left")
    plt.tight_layout()
 
    out_pdf = os.path.join("ensemble_rch_in_per_yr.pdf")
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(dpi=250, bbox_inches="tight")
    
    plt.savefig('ensemble_recharge.png',
                dpi=250,
                bbox_inches='tight')
        
    plt.close(fig)
 
    return out_pdf


def export_obs_ensemble_p05_p95(
    m_d,
    obsdict,
    modnm="swww",
    itrmx=None,
    out_name="targets_p05_p95_longform.csv",
    include_base=True,
    ):
    """
    Export p05/p95 envelopes for ALL obs through time.

    Parameters
    ----------
    obsdict : dict[int, pd.DataFrame]
        Each value is an ObservationEnsemble-like DF:
        index = realization names (e.g. 'base', 'real_0001', ...)
        columns = observation names (must align with pst.observation_data)
    itrmx : int
        Which iteration to export from. If None, uses max(obsdict).
    include_base : bool
        If True, includes 'base' in percentile calc (usually True).
    """
    pst = pyemu.Pst(os.path.join(m_d, f"{modnm}.pst"))
    obs = pst.observation_data.copy()

    # choose iteration
    if itrmx is None:
        itrmx = max(obsdict)

    ens = obsdict[itrmx].copy()

    # Drop base unless explicitly requested
    if not include_base:
        ens = ens.drop(index="base", errors="ignore")

    # Keep only transient targets we care about
    # If your transient obs group naming differs, adjust this filter.
    trans_obs = obs.loc[obs.obgnme.astype(str).str.contains("trans", na=False), :].copy()
    trans_obs["datetime"] = pd.to_datetime(trans_obs["datetime"], errors="coerce")

    # We only export obs that exist as columns in the ensemble
    keep_obsnme = trans_obs["obsnme"].isin(ens.columns)
    trans_obs = trans_obs.loc[keep_obsnme].copy()

    if trans_obs.empty:
        raise ValueError("No transient observations found that match ensemble columns.")

    # ---- Compute p05/p95 per observation column (across realizations)
    # Result: Series indexed by obsnme for each quantile
    p05 = ens.min(axis=0)
    p95 = ens.max(axis=0)

    # ---- Map each obsnme -> datetime using pst metadata
    # If you have duplicate obsnme w/ multiple datetimes (unlikely), handle accordingly.
    obs_time_map = trans_obs.set_index("obsnme")["datetime"]

    out = pd.DataFrame(
        {
            "obsnme": p05.index,
            "datetime": obs_time_map.reindex(p05.index).values,
            "p05": p05.values,
            "p95": p95.values,
        }
    )

    out = out.dropna(subset=["datetime"]).sort_values(["obsnme", "datetime"]).reset_index(drop=True)

    o_d = os.path.join(m_d, "results", "figures", "obs_vs_sim")
    os.makedirs(o_d, exist_ok=True)
    out_csv = os.path.join(o_d, out_name)
    out.to_csv(out_csv, index=False)
    print(f"...Wrote p05/p95 export: {out_csv}")

    return out_csv

# ---- Main
if __name__ == '__main__':

    m_d = 'master_flow_MasterRun3'
    modnm = 'swww'
    nopt_max = 4
    os.makedirs(os.path.join(m_d,'results','figures'),
                exist_ok=True)

    plot_ies = True
    plot_base = True

    # set universal figure properties:
    set_graph_specifications()
    set_map_specifications()

    if plot_ies:
        plot_total_pumping(
                    m_d=m_d,
                    modnm=modnm,
                    iter_num=nopt_max,
                    obs_data_csv=f"{modnm}.obs_data.csv",
                    require_in_suffix=True,             # only *_in terms
                    hide_forecast=True,
                    fill_plot=True,
                    take_abs=False,
                    )
        
        plot_total_rch(m_d=m_d,
                       modnm=modnm,
                       iter_num=4,
                       obs_data_csv=f"{modnm}.obs_data.csv",
                       require_in_suffix=True,
                       hide_forecast=True,
                       fill_plot=True,
                       take_abs=False,
                       log_y=False,
                       ylims=(0,12),
                       )
        
        
        plot_array_histo_zones(m_d, 
                               modnm='swww',
                               noptmax=nopt_max,
                               logbool=False
                               )
            
        plot_phi_sequence(m_d)

        # Simple param histrograms
        plot_simple_1to1(m_d=m_d)
        plot_simple_par_histo(m_d, modnm=modnm)

        # Load IES results
        obsdict = get_ies_obs_dict(m_d=m_d, pst=None, modnm=modnm)
        itrmx = nopt_max
        
        export_obs_ensemble_p05_p95(
            m_d=m_d,
            obsdict=obsdict,
            modnm=modnm,
            itrmx=nopt_max,
            out_name="targets_p05_p95_longform.csv",
            include_base=True,
        )
        
        # 1to1 to match other models
        plot_layer_one2one_wdepth(m_d,obsdict, modnm='swww')

        # 1-to-1 plot and calc RMSE for all head targs
        plot_1to1_all_head_targs(itrmx, modnm, m_d)

        # Plot individual hydrographs for all key wells as well as a
        # residual map
        plot_idx_residuals_base(itrmx, modnm, m_d)

        # --- Subset of index wells to plot residuals for Spiritwood
        # --> Too many to put all on a single plot
        plot_idx_residuals_base_unified_sw(itrmx, modnm,m_d, plot_others=False)

        # --- Subset of index wells to plot residuals for Warwick
        plot_idx_residuals_base_unified_ww(itrmx, modnm,m_d)

        # Obs v Sim with ensemble, noise, etc
        plot_fancy_obs_v_sim(m_d,obsdict,itrmx=itrmx,plt_pr=False,plt_noise=False)

        # Histogram of parameters by zone
        plot_zone_histos(m_d, obsdict)

    # Forward run with base parameter sets
    if plot_base:
        base_posterior_param_forward_run(m_d0=m_d, noptmax=nopt_max)
        m_d_base = m_d + '_forward_run_base_temp'

        # Compare parameter fields for all STO/NPF params
        base_prior_hydParam_compare(m_d_base,plot=True)

        # Zone budget by layer
        run_zb_by_layer(w_d=m_d_base,modnm='swww', plot=True)

        # Water use (pumping) and recharge comparisons
        plot_wateruse(base_dir=m_d_base)
        plot_recharge(base_dir=m_d_base)

        # Save the shapefiles from the base run
        model_packages_to_shp(m_d_base)
