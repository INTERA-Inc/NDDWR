import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import pyemu
import flopy
import platform
import pandas as pd
import shutil
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import random
import time

import warnings
warnings.filterwarnings("ignore")


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

    shutil.copytree(os.path.join('..','..','dependencies','flopy'), os.path.join(d,"flopy"))

    try:
        shutil.rmtree(os.path.join(d,"pyemu"))
    except:
        pass

    shutil.copytree(os.path.join('..','..','dependencies',"pyemu"), os.path.join(d,"pyemu"))


def budget_process():
    df = pd.read_csv("budget.csv",index_col=0)
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    # m = sim.get_model('swww')

    # change columns names that have "well" in them to "wel" to match the model:
    wcols = [c for c in df.columns if "WEL" in c]
    for c in wcols:
        nc = 'wel_'+c.split('(')[1].split(')')[0].lower()+c.split('(')[1].split(')')[1].lower()
        df = df.rename(columns={c:c.replace(c,nc)})

    wcols = [c for c in df.columns if "DRN" in c or "GHB" in c]
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


def init_budget_process(d):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = budget_process()
    os.chdir(b_d)
    return dfs


# Track heads as observations -> For emulator
def record_head_arrays(ws='.'):
    hds = flopy.utils.HeadFile(os.path.join(ws,'swww.hds'))
    heads = hds.get_alldata()
    fnames = []
    for kper in range(heads.shape[0]):
        # Only track layers 1 and 3 (Warwick and Spiritwood)
        for layer in [0,2]:
            fname = f"hds_layer{layer}_kper{kper}.txt"
            fnames.append(fname)
            np.savetxt(os.path.join(ws,fname),heads[kper,layer,:,:],fmt='%15.6e')
    return fnames


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


def process_listbudget_obs(mod_name='swww'):
    '''post processor to return volumetric flux and cumulative flux values from MODFLOW list file

    Args:
        mod_name (str): MODFLOW model name

    Returns:
        flx: Pandas DataFrame object of volumetric fluxes from listbudget output
        cum: Pandas DataFrame object of cumulative volumetric fluxes from listbudget output
    '''
    lst = flopy.utils.Mf6ListBudget('{0}.lst'.format(mod_name))
    flx, cum = lst.get_dataframes(diff=True, start_datetime='1979-12-31')
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
        #print(f'{pkg}')
        if pkg == 'dom':
            flx.loc[:, 'dom-reject'] = flx.loc[:, 'dom-uin'] - flx.loc[:, 'dom-simin']
        elif pkg == 'irr':
            flx.loc[:, 'irr-reject'] = flx.loc[:, 'irr-uin'] - flx.loc[:, 'irr-simin']
        elif pkg == 'stk':
            flx.loc[:, 'stk-reject'] = flx.loc[:, 'stk-uin'] - flx.loc[:, 'stk-simin']
        elif pkg == 'mfg':
            flx.loc[:, 'mfg-reject'] = flx.loc[:, 'mfg-uin'] - flx.loc[:, 'mfg-simin']
        elif pkg == 'min':
            flx.loc[:, 'min-reject'] = flx.loc[:, 'min-uin'] - flx.loc[:, 'min-simin']

    flx.to_csv('listbudget_flx_obs.csv')
    cum.to_csv('listbudget_cum_obs.csv')
    return flx,cum


def init_listbudget_obs(d='.', mod_name='swww'):
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


def process_mfinput_obs(mod_name='swww'):
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


def init_mfinput_obs(template_ws='template', mod_name='swww'):
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


def setup_pstpp(org_d,modnm,run_tag,template,flex_con=False,num_reals=96,
                run_base=True,
                high_dim=True):

    assert os.path.exists(org_d)

    temp_d = org_d + '_temp'
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(org_d,temp_d)

    # copy over head obs:
    shutil.copy2(os.path.join('data','analyzed','transient_well_head_diffs.csv'),os.path.join(temp_d,'transient_well_head_diffs.csv'))
    shutil.copy2(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'),os.path.join(temp_d,'transient_well_targets_lookup_shrt.csv'))
    # shutil.copy2(os.path.join('data','raw','water_lvl_targs_manual_ly_assign.csv'),os.path.join(temp_d,'water_lvl_targs_manual_ly_assign.csv'))
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

    if run_base:
        pyemu.os_utils.run('mf6',cwd=temp_d)

    # load flow model and model info:
    flow_dir = os.path.join(temp_d)
    sim = flopy.mf6.MFSimulation.load(sim_ws=flow_dir, exe_name='mf6')
    start_datetime = sim.tdis.start_date_time.array

    perlen = sim.tdis.perioddata.array['perlen']
    nper = len(perlen)
    dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(perlen),unit='d')

    m = sim.get_model(f'{modnm}')
    nlay = m.dis.nlay.data
    # id = m.dis.idomain.array
    # cellx = m.dis.delr.array[0]

    pkg_lst = m.get_package_list()
    pkg_lst = [p.lower() for p in pkg_lst]

    # instantiate PEST container
    pf = pyemu.utils.PstFrom(original_d=temp_d, 
                             new_d=template,
                             remove_existing=True,
                             longnames=True,
                             spatial_reference=m.modelgrid,
                             zero_based=False, 
                             start_datetime=start_datetime
                             )

    # ------------------------------------
    # load in geostats parms from run csv:
    # ------------------------------------
    pp_space = 8
    geostat = pd.read_csv(os.path.join('run_inputs',f'{modnm}_{run_tag}',f'{modnm}_geostructs.csv'))

    kgs = geostat.loc[geostat.pname == 'k_pp',:]
    k_pp = pyemu.geostats.ExpVario(contribution=kgs['contribution'].values[0],
                                   a=kgs['corr_range'].values[0],
                                   anisotropy=kgs['aniso'],
                                   bearing=kgs['bearing'].values[0])
    k_pp_gs = pyemu.geostats.GeoStruct(variograms=k_pp)
    # k_pp_space = kgs['pp_space'].values[0]

    # kgs = geostat.loc[geostat.pname == 'k_grd',:]
    # k_grd = pyemu.geostats.ExpVario(contribution=kgs['contribution'].values[0],
    #                                 a=kgs['corr_range'].values[0],
    #                                 anisotropy=kgs['aniso'],
    #                                 bearing=kgs['bearing'].values[0])
    # k_grd_gs = pyemu.geostats.GeoStruct(variograms=k_grd)

    ssgs = geostat.loc[geostat.pname == 'ss_pp',:]
    ss_pp = pyemu.geostats.ExpVario(contribution=ssgs['contribution'].values[0],
                                    a=ssgs['corr_range'].values[0],
                                    anisotropy=ssgs['aniso'],
                                    bearing=ssgs['bearing'].values[0])
    ss_pp_gs = pyemu.geostats.GeoStruct(variograms=ss_pp)
    # ss_pp_space = ssgs['pp_space'].values[0]

    sygs = geostat.loc[geostat.pname == 'sy_pp',:]
    sy_pp = pyemu.geostats.ExpVario(contribution=sygs['contribution'].values[0],
                                    a=sygs['corr_range'].values[0],
                                    anisotropy=sygs['aniso'],
                                    bearing=sygs['bearing'].values[0])
    sy_pp_gs = pyemu.geostats.GeoStruct(variograms=sy_pp)
    # sy_pp_space = sygs['pp_space'].values[0]

    rgs = geostat.loc[geostat.pname == 'rch_pp',:]
    r_pp = pyemu.geostats.ExpVario(contribution=rgs['contribution'].values[0],
                                   a=rgs['corr_range'].values[0],
                                   anisotropy=rgs['aniso'],
                                   bearing=rgs['bearing'].values[0])
    rch_pp_gs = pyemu.geostats.GeoStruct(variograms=r_pp)
    # rch_pp_space = rgs['pp_space'].values[0]

    # Variograms for ghb
    # ghb_gs = geostat.loc[geostat.pname == 'ghb_grd',:]
    # ghb_grd = pyemu.geostats.ExpVario(contribution=ghb_gs['contribution'].values[0],
    #                                   a=ghb_gs['corr_range'].values[0],
    #                                   anisotropy=ghb_gs['aniso'],
    #                                   bearing=ghb_gs['bearing'].values[0])
    # ghbb_grd_gs = pyemu.geostats.GeoStruct(variograms=ghb_grd, transform='log', name='ghb_grd_gs')

    # temporal geostats for rch and...:
    rcht_gs = geostat.loc[geostat.pname == 'rcht_cn',:]
    rcht_cn = pyemu.geostats.ExpVario(contribution=rcht_gs['contribution'].values[0],
                                      a=rcht_gs['corr_range'].values[0],
                                      anisotropy=rcht_gs['aniso'],
                                      bearing=rcht_gs['bearing'].values[0])
    rcht_gs = pyemu.geostats.GeoStruct(variograms=rcht_cn, transform='log', name='rcht_gs')

    rivgs = geostat.loc[geostat.pname == 'riv_cn',:]
    riv_cn = pyemu.geostats.ExpVario(contribution=rivgs['contribution'].values[0],
                                     a=rivgs['corr_range'].values[0],
                                     anisotropy=rivgs['aniso'],
                                     bearing=rivgs['bearing'].values[0])
    riv_cn_gs = pyemu.geostats.GeoStruct(variograms=riv_cn, transform='log', name='riv_gs')

    riv_grd_gs = geostat.loc[geostat.pname == 'riv_grd',:]
    riv_grd = pyemu.geostats.ExpVario(contribution=riv_grd_gs['contribution'].values[0],
                                      a=riv_grd_gs['corr_range'].values[0],
                                      anisotropy=riv_grd_gs['aniso'],
                                      bearing=riv_grd_gs['bearing'].values[0])
    riv_grd_gs = pyemu.geostats.GeoStruct(variograms=riv_grd, transform='log', name='riv_grd_gs')

    temporal_v = pyemu.geostats.ExpVario(contribution=1.0, a=365*3, name='temporal_v')
    temporal_gs = pyemu.geostats.GeoStruct(variograms=temporal_v, transform='log', name='temporal_gs')
    # temporal_gs.to_struct_file(os.path.join(template_d, 'temporal_gs.struct'))

    # import flopy as part of the forward run process
    pf.extra_py_imports.append('flopy')

    k_files = [f for f in os.listdir(template) if 'k_' in f and f.endswith('.txt')]
    k_files.sort()

    k33_files = [f for f in os.listdir(temp_d) if "k33" in f]
    k33_files.sort()

    ss_files = [f for f in os.listdir(template) if 'sto_ss_' in f and f.endswith('.txt')]
    ss_files.sort()

    sy_files = [f for f in os.listdir(template) if 'sto_sy_' in f and f.endswith('.txt')]
    sy_files.sort()

    # load in par bounds:
    par = pd.read_csv(os.path.join('run_inputs',f'{modnm}_{run_tag}',f'{modnm}_parm_controls.csv'))

    kpp = par.loc[par.parm=='k_pp']
    k_bounds_pp = {k:[kpp.lbound.values[0],kpp.ubound.values[0]] for k in range(nlay)}
    kcn = par.loc[par.parm=='k_cn']
    k_bounds_cn = {k:[kcn.lbound.values[0],kcn.ubound.values[0]] for k in range(nlay)}
    # kgrd = par.loc[par.parm=='k_grd']
    # k_bounds_grd = {k:[kgrd.lbound.values[0],kgrd.ubound.values[0]] for k in range(nlay)}

    k33pp = par.loc[par.parm=='aniso_pp']
    k33_bounds_pp = {k:[k33pp.lbound.values[0],k33pp.ubound.values[0]] for k in range(nlay)}
    k33cn = par.loc[par.parm=='aniso_cn']
    k33_bounds_cn = {k:[k33cn.lbound.values[0],k33cn.ubound.values[0]] for k in range(nlay)}
    #k33_bounds_grd = {k:[0.2,5.0] for k in range(nlay)}

    sspp = par.loc[par.parm=='ss_pp']
    ss_bounds_pp = {k:[sspp.lbound.values[0],sspp.ubound.values[0]] for k in range(nlay)}
    sscn = par.loc[par.parm=='ss_cn']
    ss_bounds_cn = {k:[sscn.lbound.values[0],sscn.ubound.values[0]] for k in range(nlay)}

    sypp = par.loc[par.parm=='sy_pp']
    sy_bounds_pp = {k:[sypp.lbound.values[0],sypp.ubound.values[0]] for k in range(nlay)}
    sycn = par.loc[par.parm=='sy_cn']
    sy_bounds_cn = {k:[sycn.lbound.values[0],sycn.ubound.values[0]] for k in range(nlay)}

    # load ultimate (hard) bounds:
    # load them from pp parms because those types will likely always be used, but be sure to check this
    k_ubounds = {k:[kpp.ult_lbound.values[0],kpp.ult_ubound.values[0]] for k in range(nlay)}
    k33_ubounds = {k:[k33pp.ult_lbound.values[0],k33pp.ult_ubound.values[0]] for k in range(nlay)}
    ss_ubounds = {k:[sspp.ult_lbound.values[0],sspp.ult_ubound.values[0]] for k in range(nlay)}
    sy_ubounds = {k:[sypp.ult_lbound.values[0],sypp.ult_ubound.values[0]] for k in range(nlay)}

    stacked_files = [k_files,k33_files,ss_files,sy_files]
    stacked_ubnds = [k_ubounds,k33_ubounds,ss_ubounds,sy_ubounds]

    # --- Load the zone array
    # ---> 3D numpy array generated in spirit_war02_model_build.py during build
    zone_array = np.load('zone_array.npy')

    # ---- NPF params
    for files, kubnds in zip(stacked_files,stacked_ubnds):
        assert len(files) > 0
        [print(x) for x in files]
        lays = [int(f.split('.')[1].split('_')[2].replace('layer',''))-1 for f in files]
        par_name_base = ''.join(files[0].split('_')[1]).replace("_","-") #.replace('.txt','').replace('layer','ly')
        par_name_base = par_name_base+"_k:"
        if flex_con and par_name_base != 'k_k:':
            continue
        # assert len(files) == nlay
        # Handles which layers the zones are applied to
        for k,f in zip(lays,files):
            ubnds = kubnds[k]
            if 'k33' in f:
                bnds_pp = k33_bounds_pp[k]
                bnds_cn = k33_bounds_cn[k]
                # No params on Spiritwood - IES doing weird things in SW-Sheyenne
                if k in [0,1]:
                    parType = 'zone'
                    zoneArray = zone_array[k]
                    pf.add_parameters(f, 
                                      par_type = parType, 
                                      upper_bound = 4, 
                                      lower_bound = 0.1,
                                      ult_ubound = 0.13, 
                                      ult_lbound = 1e-4 if k == 1 else 0.07,
                                      par_name_base='cn-' + par_name_base,
                                      pargp='cn-' + par_name_base + str(k).zfill(3),
                                      zone_array=zoneArray
                                      )
                    if high_dim:
                        pf.add_parameters(f, 
                                          par_type='pilotpoints', 
                                          upper_bound=4, 
                                          lower_bound=0.1,
                                          ult_ubound=0.13,
                                          ult_lbound=1e-4 if k == 1 else 0.07, 
                                          par_name_base='pp-' + par_name_base,
                                          pargp='pp-' + par_name_base + str(k).zfill(3),
                                          geostruct=k_pp_gs, 
                                          pp_space=pp_space,
                                          )
                else:
                    continue

            elif 'k' in f and 'k33' not in f:
                print(f'FILENAME IS {k}')
                bnds_pp = k_bounds_pp[k]
                bnds_cn = k_bounds_cn[k]
                if k in [0,1,2]:
                    parType = 'zone'
                    zoneArray = zone_array[k] #if k == 2 else id[k]
                    # Bounds
                    lbound = ubnds[0]
                    ubound = ubnds[1]
                    
                    # HK bounds for Warwick
                    # Prior HK is 140
                    if k == 0:
                        # ultimate bounds
                        lbound = 5
                        # Increase upper bound on Warwick to ~300 ft/day
                        ubound = 280
                        # bounds
                        # Thick about how to center these around the mean
                        low_bound = 0.1
                        up_bound = 1.8
                    
                    # Confining Unit
                    if k == 1:
                        # ultimate bounds
                        lbound = 1e-4
                        ubound = 50
                        # bounds
                        low_bound = 0.1
                        up_bound = 10
                        
                    # HK bounds for Spiritwood
                    # Prior HK is 160
                    if k == 2:
                        # ultimate bounds
                        lbound = 1e-4
                        ubound = 280
                        # bounds
                        low_bound = 0.3
                        up_bound = 2.2
                        
                    pf.add_parameters(f, 
                                      par_type=parType, 
                                      upper_bound=up_bound, 
                                      lower_bound=low_bound,
                                      ult_ubound=ubound, 
                                      ult_lbound=lbound, 
                                      par_name_base='cn-' + par_name_base,
                                      pargp='cn-' + par_name_base + str(k).zfill(3),
                                      zone_array=zoneArray
                                      )
                    if high_dim:
                        pf.add_parameters(f, 
                                          par_type='pilotpoints', 
                                          upper_bound=up_bound, 
                                          lower_bound=low_bound,
                                          ult_ubound=ubound, 
                                          ult_lbound=lbound, 
                                          par_name_base='pp-' + par_name_base,
                                          pargp='pp-' + par_name_base + str(k).zfill(3), 
                                          geostruct=k_pp_gs, 
                                          # Refining pp spacing for layers 1 and 2
                                          pp_space = pp_space,
                                          )
                else:
                    continue

            elif 'ss' in f:
                bnds_pp = ss_bounds_pp[k]
                bnds_cn = ss_bounds_cn[k]
                # Skipping layer 1 since we're using 1/thickness
                if k in [1,2]:
                    parType = 'zone'
                    zoneArray = zone_array[k]
                    pf.add_parameters(f, 
                                      par_type=parType, 
                                      upper_bound=bnds_cn[1],
                                      lower_bound=bnds_cn[0],
                                      ult_ubound=ubnds[1],
                                      ult_lbound=ubnds[0], 
                                      par_name_base='cn-' + par_name_base,
                                      pargp='cn-' + par_name_base + str(k).zfill(3),
                                      zone_array=zoneArray
                                      )
                    if high_dim:
                        pf.add_parameters(f, 
                                          par_type='pilotpoints', 
                                          upper_bound=bnds_pp[1], 
                                          lower_bound=bnds_pp[0],
                                          ult_ubound=ubnds[1], 
                                          ult_lbound=ubnds[0],
                                          par_name_base='pp-' + par_name_base,
                                          pargp='pp-' + par_name_base + str(k).zfill(3), 
                                          geostruct=ss_pp_gs, 
                                          pp_space=pp_space,
                                          )
                else:
                    continue

            # Specific Yield
            elif 'sy' in f:
                bnds_pp = sy_bounds_pp[k]
                bnds_cn = sy_bounds_cn[k]
                # Only including layer 1 (Warwick) -> Ult bounds of 8% - 16%
                if k in [0]:
                    parType = 'zone'
                    zoneArray = zone_array[k]
                    
                    # Tighter bounds so it doesnt smack the ult bounds immediately 
                    lower_bound = 0.7
                    upper_bound = 1.25
                    
                    # Enforcing tighter manual bounds on Sy here
                    pf.add_parameters(f, 
                                      par_type=parType, 
                                      upper_bound=upper_bound, 
                                      lower_bound=lower_bound,
                                      ult_ubound=0.16, 
                                      ult_lbound=0.08, 
                                      par_name_base='cn-' + par_name_base,
                                      pargp='cn-' + par_name_base + str(k).zfill(3),
                                      zone_array=zoneArray
                                      )
                    if high_dim:
                        # Even tighter on PP, and increase pp_space to reduce
                        # heterogeneity, congerence issues occur when adding 
                        # high dim specific yield
                        pf.add_parameters(f, 
                                          par_type='pilotpoints', 
                                          upper_bound=1.05, 
                                          lower_bound=0.95,
                                          ult_ubound=0.16, 
                                          ult_lbound=0.08, 
                                          par_name_base='pp-' + par_name_base,
                                          pargp='pp-' + par_name_base + str(k).zfill(3), 
                                          geostruct=sy_pp_gs, 
                                          # Higher pp spacing, not expecting/wanting a ton of heterogeneity in sy
                                          pp_space=12,
                                          )
                else:
                    continue

            pf.add_observations(f,obsgp=par_name_base + str(k).zfill(3),prefix=par_name_base + str(k).zfill(3))
    if flex_con:
        pf.mod_py_cmds.append("print('model')")
    else:
        # ---- RCH params
        rch_files = [f for f in os.listdir(template) if 'rcha_recharge' in f and f.endswith('.txt')]
        rch_files = sorted(rch_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        assert len(rch_files) == nper
        
        # Constant mults for each stress period
        # -> Temporally correlated by 3-years
        for i,rch_file in enumerate(rch_files):
            kper = int(rch_file.split('_')[2].split('.')[0])
            pf.add_parameters(rch_file,
                              par_type='constant',
                              pargp='rcht_cn',
                              par_name_base='rchtcn_kper:{0:03d}'.format(kper),
                              datetime=dts[i],
                              geostruct=temporal_gs,
                              upper_bound = 15,
                              lower_bound = 0.1,
                              )
        
        # Pilot points, large spacing (20 grid cells) -> one set per SP
        if high_dim:
            pp_space_rch = 16
            # PPs for each stress period
            # for i,rch_file in enumerate(rch_files):
            #     kper = int(rch_file.split('_')[2].split('.')[0])
            #     pf.add_parameters(
            #         rch_file,
            #         par_type="pilotpoints",
            #         upper_bound=10,
            #         lower_bound=0.1,
            #         par_name_base="rchtpp_{0:03d}".format(kper),
            #         pargp="rcht_pp",
            #         geostruct=rch_pp_gs,
            #         pp_space=pp_space_rch,
            #         )
            
            # Removing predictive period from these pilot points to help scenario setup
            rch_files = [x for x in rch_files if int(x.split('_')[-1].split('.txt')[0]) <= 319]
            
            # Constant PP's through time
            pf.add_parameters(rch_files, 
                              par_type='pilotpoints', 
                              upper_bound=15, 
                              lower_bound=0.1,
                              par_name_base='pp-rchcn',
                              pargp='pp-rchcn', 
                              geostruct=rch_pp_gs, 
                              # Recharge pp spacing
                              pp_space=pp_space_rch,
                              )
        
        # ---- RIV params
        riv_files = [f for f in os.listdir(template) if '-riv.riv_' in f and f.endswith('.txt')]
        riv_files = sorted(riv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(riv_files) == nper
                
        pf.add_parameters(riv_files,
                          par_type='constant',
                          index_cols=[0, 1, 2],
                          use_cols=[4],
                          pargp='rivcond-cn',
                          par_name_base='rivcond-cn',
                          geostruct=riv_cn_gs,
                          # Bounds
                          upper_bound=5,
                          lower_bound=0.2,
                          mfile_skip=0
                          )

        # ---- DRN params - Surface (layer 1)
        drn_files = [f for f in os.listdir(template) if 'drn_' in f and f.endswith('.txt') and f.startswith('swww.')]
        drn_files = sorted(drn_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_files) == nper

        pf.add_parameters(filenames=drn_files,
                          par_type='constant',
                          par_name_base='drn-cn',
                          geostruct=riv_cn_gs,
                          pargp='drn-cn',
                          index_cols=[0, 1, 2],
                          use_cols=[4],
                          # Bounds
                          upper_bound=3,
                          lower_bound=0.33,
                          ult_ubound=20000,
                          ult_lbound=0,
                          mfile_skip=0
                          )
        
        
        # ---- DRN params - Sheyenne valley seepage (layer 1)
        drn_files = [f for f in os.listdir(template) if 'drn_' in f and f.endswith('.txt') and f.startswith('swww_0')]
        drn_files = sorted(drn_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_files) == nper

        pf.add_parameters(filenames=drn_files,
                          par_type='constant',
                          par_name_base='drnValley-cn',
                          geostruct=riv_cn_gs,
                          pargp='drnValley-cn',
                          index_cols=[0, 1, 2],
                          use_cols=[4],
                          # Bounds
                          upper_bound=5,
                          lower_bound=0.2,
                          ult_ubound=20000,
                          ult_lbound=0,
                          mfile_skip=0
                          )
        
        
        # ---- DRN params - SW-Sheyenne connection (layer 3)
        drn_files = [f for f in os.listdir(template) if 'drn_' in f and f.endswith('.txt') and f.startswith('swww_1')]
        drn_files = sorted(drn_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_files) == nper
        
        pf.add_parameters(filenames=drn_files,
                          par_type='constant',
                          par_name_base='drnRiv-cn',
                          geostruct=riv_cn_gs,
                          pargp='drnRiv-cn',
                          index_cols=[0, 1, 2],
                          use_cols=[4],
                          # Bounds
                          upper_bound=10,
                          lower_bound=0.1,
                          ult_ubound=20000,
                          ult_lbound=0,
                          mfile_skip=0
                          )
        
        # ---- GHB params
        ghb_files = [f for f in os.listdir(template) if 'ghb_' in f and f.endswith('.txt')]
        ghb_files.sort()
                
        # Constant mult for GHB conductance
        pf.add_parameters(filenames=ghb_files,
                          par_type='constant',
                          par_name_base='ghbcond-cn',
                          # geostruct=ghbb_grd_gs,
                          pargp='ghbcond-cn',
                          index_cols=[0, 1, 2],
                          use_cols=[4],
                          # Bounds
                          upper_bound=1.8,
                          lower_bound=0.2,
                          # It doesnt seem to matter past like ~200
                          ult_lbound=10,
                          ult_ubound=500,
                          mfile_skip=0
                          )

        # GHB Head --> constant that can adjust +/- 10-ft
        pf.add_parameters(filenames=ghb_files,
                          par_type='constant',
                          # par_name_base='ghbhd-cn',
                          pargp='ghbhd-cn',
                          index_cols=[0, 1, 2],
                          use_cols=[3],
                          upper_bound=10,
                          lower_bound=-10,
                          mfile_skip=0,
                          # Change from multiplier type
                          par_style='a',
                          )
        
        # ---- WEL params
        wel_files = [f for f in os.listdir(template) if 'wel_' in f and f.endswith('.txt')]
        wel_files.sort()
                
        # +/- 15% constant on pumping per Stress Period
        for i,wel_file in enumerate(wel_files):
           kper = int(wel_file.split('_')[4].split('.')[0])
           # if kper <= 319:
           pf.add_parameters(wel_file,
                              par_type='constant',
                              pargp='welt',
                              index_cols=[0, 1, 2],
                              use_cols=[3],
                              par_name_base='welt_kper:{0:03d}'.format(kper),
                              datetime=dts[i],
                              upper_bound=1.15,
                              lower_bound=0.85,
                              geostruct=temporal_gs,
                              )
        
        # hard coded the df return order
        dfs = init_budget_process(template)

        pf.add_observations('budget.csv',
                            index_cols=['datetime'],
                            use_cols=dfs[0].columns.to_list(),
                            obsgp='bud',
                            ofile_sep=',',
                            prefix='bud'
                            )

        hdf = init_head_targets_process(template)
        
        # Add steady state observations
        # pf.add_observations('swww.ss_head.obs.output.csv',
        #                     index_cols=['datetime'],
        #                     use_cols=hdf[0].columns.to_list(),
        #                     obsgp='sshds',
        #                     ofile_sep=',',
        #                     prefix='sshds'
        #                     )
        
        # Add transient observations
        pf.add_observations('swww.trans_head.obs.output.csv',
                            index_cols=['datetime'],
                            use_cols=hdf[1].columns.to_list(),
                            obsgp='transhds',
                            ofile_sep=',',
                            prefix='transhds'
                            )

        pf.mod_sys_cmds.append('mf6')
        pf.add_py_function('spirit_war03_setup_pst.py','budget_process()',is_pre_cmd=False)

        # Listbudget Obs
        # process model output
        flx, cum = init_listbudget_obs(template, 'swww')
        
        # add post process function to forward run script
        pf.add_py_function('spirit_war03_setup_pst.py', 'process_listbudget_obs()', is_pre_cmd=None)
        
        # add call to processing script to pst forward run
        pf.post_py_cmds.append("process_listbudget_obs('{0}')".format('swww'))
        
        # add obs via PstFrom
        ignore_cols = ['datetime', 'in-out', 'total', 'wel-in']
        cols = [c for c in flx.columns if c not in ignore_cols]
        
    pf.parfile_relations.to_csv(os.path.join(pf.new_d, 'mult2model_info.csv'))
    curdir = os.getcwd()
    os.chdir(pf.new_d)
    df = pyemu.helpers.calc_array_par_summary_stats()
    os.chdir(curdir)
    pf.post_py_cmds.append('pyemu.helpers.calc_array_par_summary_stats()')
    pf.add_observations('arr_par_summary.csv',
                        index_cols=['model_file'],
                        use_cols=df.columns.tolist(),
                        obsgp='arrparsum',
                        prefix='arrparsum',
                        ofile_sep=','
                        )

    # MODFLOW input value observations
    # summary statistic observations of modflow inputs resulting from multiplier application
    df = init_mfinput_obs(template, 'swww')

    # add post process function to forward run script
    pf.add_py_function('spirit_war03_setup_pst.py',
                       'process_mfinput_obs()',
                       is_pre_cmd=None
                       )

    # add call to processing script to pst forward run
    pf.post_py_cmds.append("process_mfinput_obs('{0}')".format('swww'))

    # add obs via PstFrom
    cols = ['upper_bound', 'lower_bound', 'min', 'qnt25', 'qnt50', 'qnt75', 'max', 'near_lbnd', 'near_ubnd']

    pf.add_observations('mfinput_obs.csv',
                        insfile='mfinput_obs.csv.ins',
                        index_cols=['input'],
                        use_cols=cols,
                        prefix='mfin'
                        )

    # build pest control file
    pst = pf.build_pst(version=None)

    pst.control_data.noptmax = 0
    pst.pestpp_options['additional_ins_delimiters'] = ','
    pst.write(os.path.join(template,'swww.pst'), version=2)
    pyemu.os_utils.run('pestpp-ies swww.pst', cwd=template)
    pst.set_res(os.path.join(template, 'swww.base.rei'))
    print('phi',pst.phi)
    assert pst.phi < 1e-4

    rei = pst.res
    # sort rei by residual magnitude:
    rei = rei.sort_values(by='residual')
    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=False)
    pe.to_binary(os.path.join(template, 'prior.jcb'))
    pst.pestpp_options['ies_par_en'] = 'prior.jcb'
    pst.pestpp_options['save_binary'] = True

    # write the updated pest control file
    pst.write(os.path.join(pf.new_d, 'swww.pst'),version=2)

    shutil.copy(os.path.join(pf.new_d, 'swww.obs_data.csv'),
                os.path.join(pf.new_d, 'swww.obs_data_orig.csv'))

    return template # return the template directory name


def set_obsvals_and_weights(template_d,
                            timediff_tol=185,
                            flow_weight_scheme='basic',
                            include_vertheads=False,
                            set_less_than_obs=False,
                            phi_factor_dict=None):

    pst = pyemu.Pst(os.path.join(template_d,'swww.pst'))

    # now set obsvals and weights
    obs = pst.observation_data
    obs.loc[:,'weight'] = 0
    obs.loc[:,'observed'] = False
    obs.loc[:,'count'] = 0
    obs.loc[:,'standard_deviation'] = 0
    obs.loc[:,'obsval'] = 0.0

    if flow_weight_scheme is not None:
        # set water level obs targets:
        # h_df = pd.read_csv(os.path.join('data','analyzed','SS_target_heads.csv'),
        #                                 index_col=['start_dt'], parse_dates=True)
        # # h_df = h_df.dropna()
        # h_df.loc[:, "datetime"] = pd.to_datetime(h_df.index, format="%Y-%m-%d")
        # h_df.loc[:,'k'] = h_df.k.astype(int)
        # h_df.loc[:,'i'] = h_df.i.astype(int)
        # h_df.loc[:,'j'] = h_df.j.astype(int)
        # h_df.loc[:,'obsprefix'] = h_df.obsprefix.apply(lambda x: x.replace('.','-'))
        # uprefixes = h_df.obsprefix.unique()
        # uprefixes.sort()
        # print(uprefixes)

        # # ---- Add Steady State head targets
        # oname_obsval_dict = {}
        # for prefix in uprefixes:
        #     print("...",prefix)
        #     if 'ssh_id:' not in prefix:
        #         continue
        #     uh_df = h_df.loc[h_df.obsprefix==prefix,:].copy()
        #     uk = uh_df.k.unique()
        #     uk.sort()

        #     pobs = obs.loc[obs.obsnme.apply(lambda x: prefix in x),:].copy()
        #     if pobs.shape[0] == 0:
        #         print('empty obs for prefix:{0}'.format(prefix))
        #         continue
        #     pobs.loc[:,'k'] = uh_df.k.values[0].astype(int)
        #     pobs.loc[:,'i'] = uh_df.i.values[0].astype(int)
        #     pobs.loc[:,'j'] = uh_df.j.values[0].astype(int)

        #     for k in uk:
        #         kuh_df = uh_df.loc[uh_df.k==k,:].copy()
        #         if kuh_df.shape[0] == 0:
        #             print('empty layer df for k:{0},prefix:{1}'.format(k,prefix))
        #             continue
        #         ukobs = pobs.loc[pobs.k==k,:].copy()
        #         if ukobs.shape[0] == 0:
        #             print('empty ukobs for k:{0},prefix:{1}'.format(k,prefix))
        #             continue
        #         for head,dt in zip(kuh_df.loc[:,'gwe_ft'],kuh_df.index):
        #             #print(head, dt)
        #             if dt < pd.to_datetime('1970-01-01'):
        #                 mn_oname = ukobs.iloc[0,:].obsnme
        #                 oname_obsval_dict.setdefault(mn_oname, []).append(head)

        # print('\n\n\n  ---  found {0} SS gw level obs  ---  \n\n\n'.format(len(oname_obsval_dict)))
        # assert len(oname_obsval_dict) > 0

        # process and then set transient water level obs targets:
        t_df_loc = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup_shrt.csv'))
        t_df = pd.read_csv(os.path.join('data','analyzed','transient_well_targets.csv'))
        unq_prefix = t_df_loc['obsprefix'].unique()

        # ---- Add Transient head targets
        oname_obsval_dict_trans = {}
        for prefix in unq_prefix:
            if 'transh_grpid:' not in prefix:
                continue

            # Grab data for specific index
            uh_df = t_df.loc[:,['start_datetime',prefix]].copy()

            # check if any data outside of nan values in uh_df, if all nans continue
            if uh_df[prefix].isnull().all():
                print('no data for prefix:{0}'.format(prefix))
                continue

            uh_df = uh_df.loc[uh_df[prefix].notnull(),:].copy()

            pobs = obs.loc[obs.obsnme.apply(lambda x: prefix in x),:].copy()
            # get pobs that include start_datetime string in uh_df['start_datetime']
            dt_vals = uh_df['start_datetime'].unique()
            dt_vals.sort()
            pobs_w_meas = pobs.loc[pobs.obsnme.apply(lambda x: any(dt in x for dt in dt_vals)),:].copy()

            if pobs_w_meas.shape[0] == 0:
                print('empty obs for prefix:{0}'.format(prefix))
                continue

            for idx, row in pobs_w_meas.iterrows():
                oname = row.obsnme
                sim_date = oname.split(':')[-1]
                val = uh_df.loc[uh_df['start_datetime'] == sim_date, prefix].values[0]
                if np.isnan(val):
                    assert False, 'nan value for {0}, something went wrong...qa needed'.format(oname)
                oname_obsval_dict_trans.setdefault(oname, []).append(val)

        print('\n\n\n  ---  found {0} transient gw level obs  ---  \n\n\n'.format(len(oname_obsval_dict_trans)))
        assert len(oname_obsval_dict_trans) > 0


        if flow_weight_scheme == 'all_wl_meas':
            # ---- parameterize Steady State head targets
            # for oname,vals in oname_obsval_dict.items():
            #     vals = np.array(vals)
            #     # obspre = oname.split(':')[3].replace('_k','')
            #     # oinfo = h_df.loc[h_df.obsprefix.apply(lambda x: obspre in x),:].copy()
            #     obs.loc[oname,'obsval'] = vals.mean()
            #     obs.loc[oname,'observed'] = True
            #     obs.loc[oname,'standard_deviation'] = 2
            #     obs.loc[:,'count'] = len(vals)
            #     obs.loc[oname,'weight'] = 1

            #     # Name obs based on layer
            #     if int(obs.loc[oname,'k']) == 2:
            #         obs.loc[oname,'obgnme'] = 'swsshds'
            #     else:
            #         obs.loc[oname,'obgnme'] = 'wwsshds'
            
            # ---- parameterize Transient head targets
            for oname,vals in oname_obsval_dict_trans.items():
                # Wells in Spiritwood Sheyenne
                SW_sheyenne_targs = ['13','11','51','7','77','6','2','187','75']
                
                # Observations that are flooding, weighting and changin the std
                # to address this
                flooded_targs = ['29', '123', '127']
                
                reduce_noise = ['124','126','18','17','171']
                
                # Key index wells
                index_wells = ['3','5','11','13','29','48','59','60','71','76','88','93',
                               '99','100','112','116','118','121','122','123','127',
                               '137','138','168','169','219','220','221']
                
                # 187: Bad data - head is like 100-ft higher than everything else
                # 47: Model struggles to his this and its a very low-frequency target
                # Model REALLY struggles with 48, and its biasing the area around it too low
                # ^^ Its in the S-W corner of spiritwood, right up against the flow barrier
                drop_targs = ['187', '47', '48']

                vals = np.array(vals)
                year = int(oname.split(':')[8].split('-')[0])
                obs.loc[oname,'obsval'] = vals
                obs.loc[oname,'observed'] = True
                obs.loc[oname,'standard_deviation'] = 1
                obs.loc[:,'count'] = len(vals)
                obs.loc[oname,'weight'] = 1
                
                # Grab layer of target
                k = int(obs.loc[oname,'k'])
                
                # WW vs SW targets (by layer)
                # NOTE - this will be lumping in confining unit (layer 2) targs w/ Warwick
                if k == 2:
                    obs.loc[oname,'obgnme'] = 'swtranshds'
                else:
                    obs.loc[oname,'obgnme'] = 'wwtranshds'
                
                # Assign group name and increase standard deviation on pre2k targets
                # Since they're yearly medians, there is more uncertainty
                if year < 2000:
                    obs.loc[oname,'obgnme'] = 'pre2ktranshds'
                    obs.loc[oname,'standard_deviation'] = 2

                # Spirwood-Sheyenne targets
                grpid = oname.split('transh_grpid:')[-1].split('_')[0]
                if grpid in SW_sheyenne_targs:
                    obs.loc[oname,'obgnme'] = 'swsheyennetranshds'
                    
                # Higher weight to index wells within each group
                # NOTE: This is only being appied in the transient period since not weighting SS
                if grpid in index_wells:
                    obs.loc[oname,'weight'] = 2

                # De-weight a few targets enitrely 
                if grpid in drop_targs:
                    obs.loc[oname,'weight'] = 0

                # De-weighting a period of well 213 post 1994
                # Suspect data, drawdown pattern suddenly stops and then well is plugged -> See hydrograph
                if grpid == '213' and year > 1993:
                    obs.loc[oname,'weight'] = 0
                    
                # Changing to higher weight here to hit them
                if grpid in flooded_targs and k == 0:
                    # Lower std on these seems to help prevent flooding
                    obs.loc[oname,'standard_deviation'] = 0.2
                    obs.loc[oname,'weight'] = 4
                    
                # Tracking but really low weight on confining unit targets
                if k == 1:
                    obs.loc[oname,'weight'] = 0.01           
                
                # Edit to 121 and 60 to try and hit them better
                if grpid in ['121','60'] and k == 0:
                    obs.loc[oname,'weight'] = 3
                    obs.loc[oname,'standard_deviation'] = 1
                    
                if grpid in reduce_noise and k == 0:
                    obs.loc[oname,'standard_deviation'] = 0.5
                    obs.loc[oname,'weight'] = 2
                
            # ---- Inequality targets
            # Forcing a higher K zone in Spiritwood
            kobs_greater_than = True
            if kobs_greater_than:
                # Cells along suspected high-K channel in Spiritwood
                ij_pairs = [(4,33),(23,43),(30,67),(51,77),(68,87),(82,92),(97,97)]
                obsnmes = []
                for ij in ij_pairs:
                    sw_kh = obs.loc[obs.obsnme.str.contains('k_k:002')]
                    sw_pts = sw_kh.loc[(sw_kh.i==str(ij[0])) & (sw_kh.j==str(ij[1])) ,'obsnme'].values[0]
                    obsnmes.append(sw_pts)
                
                assert len(obsnmes) == len(ij_pairs), 'kobs locs were not found, investigate'
                
                obs.loc[obs.obsnme.isin(obsnmes), 'weight'] = 1.0
                obs.loc[obs.obsnme.isin(obsnmes), 'obgnme'] = "greater_than_sw_kh"
                obs.loc[obs.obsnme.isin(obsnmes), 'obsval'] = 180.0
            
            # Prevent high K zone from appearing in SW-Sheyenne
            kobs_less_than = True
            if kobs_less_than:
                # Cells in SW-Sheyenne
                ij_pairs = [(123,93),(99,118),(122,110),(143,110),(123,132)]
                obsnmes = []
                for ij in ij_pairs:
                    sw_kh = obs.loc[obs.obsnme.str.contains('k_k:002')]
                    sw_pts = sw_kh.loc[(sw_kh.i==str(ij[0])) & (sw_kh.j==str(ij[1])) ,'obsnme'].values[0]
                    obsnmes.append(sw_pts)
                
                assert len(obsnmes) == len(ij_pairs), 'kobs locs were not found, investigate'
                
                obs.loc[obs.obsnme.isin(obsnmes), 'weight'] = 1.0
                obs.loc[obs.obsnme.isin(obsnmes), 'obgnme'] = "less_than_sw_kh"
                obs.loc[obs.obsnme.isin(obsnmes), 'obsval'] = 200
            
    assert pst.nnz_obs > 0

    nzobs = obs.loc[obs.weight>0,:]
    vc = nzobs.obgnme.value_counts()
    for gname,c in zip(vc.index,vc.values):
        print('group ',gname,' has ',c,' nzobs')

    if phi_factor_dict is not None:
        with open(os.path.join(template_d,'phi_facs.csv'),'w') as f:
            keys = list(phi_factor_dict.keys())
            keys.sort()
            for key in keys:
                f.write('{0},{1}\n'.format(key,phi_factor_dict[key]))
        pst.pestpp_options['ies_phi_factor_file'] = 'phi_facs.csv'

    # check that the mean par values will run
    pst.write(os.path.join(template_d, 'swww.pst'), version=2)
    pyemu.os_utils.run('pestpp-ies swww.pst', cwd=template_d)

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


def run_ies(template_ws='template_d', 
            m_d=None, 
            num_workers=12, 
            noptmax=-1, 
            num_reals=None,
            init_lam=None, 
            drop_conflicts=False, 
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
    pst.pestpp_options['ies_drop_conflicts'] = drop_conflicts
    
    # Factor of base runtime where it gives up
    pst.pestpp_options['overdue_giveup_fac'] = 4
    pst.pestpp_options['ies_bad_phi_sigma'] = 1.5
    
    # Lowering this from 1e+20 t0 1e+8
    pst.pestpp_options['ies_bad_phi'] = 1e+8
    
    # Option to reinflate
    # pst.pestpp_options["ies_n_iter_reinflate"] = [-2,999]
    
    # See if commenting out this line increases variance 
    pst.pestpp_options["ies_multimodal_alpha"] = 0.99

    #pst.pestpp_options['panther_agent_freeze_on_fail'] = True

    pst.pestpp_options['save_binary'] = True
    if num_reals is not None:
        pst.pestpp_options['ies_num_reals'] = num_reals

    if init_lam is not None:
        pst.pestpp_options['ies_initial_lambda'] = init_lam
    pst.pestpp_options['ies_subset_size'] = -10
    for k,v in kwargs.items():
        pst.pestpp_options[k] = v
    # intit run log file:
    f = open(os.path.join(template_ws, 'swwwpst_run.log'), 'w')
    f.close()

    # obs sainty check:
    pobs = pst.observation_data
    pobsmax = pobs.weight.max()
    if pobsmax <= 0:
        raise Exception('setting weighted obs failed!!!')
    pst.write(os.path.join(template_ws, 'swww.pst'), version=2)

    prep_worker(template_ws, template_ws + '_clean')

    master_p = None

    if hostname is None:
        pyemu.os_utils.start_workers(template_ws, 'pestpp-ies', 'swww.pst',
                                     num_workers=num_workers, worker_root='.',
                                     master_dir=m_d, local=local,port=4269)

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

def draw_noise_reals(t_d,modnm):
    print("...Drawing Noise Reals")
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

    # First the tr head groups
    trobs = obs.loc[(obs.oname=="transhds") & (obs.weight>0),:]
    print(trobs.standard_deviation.unique())
    assert trobs.shape[0] > 0
    trobs["datetime"] = pd.to_datetime(trobs.datetime)
    trobs['id'] = trobs.apply(lambda x: (x.grpid,x.k),axis=1)
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
        v = pyemu.geostats.ExpVario(contribution=1.0,a=3650*3)
        gs = pyemu.geostats.GeoStruct(variograms=v,name=uid)
        struct_dict[gs] = uobs.obsnme.tolist()

    np.random.seed(1123556564)
    oe = pyemu.helpers.autocorrelated_draw(pst,struct_dict,num_reals=num_reals)
    
    # Only filter top for Warwick targets, arbitrary 100,000 ft top for other targes
    # so that they are never filtered as WL > top
    trobs_top = pd.Series(
        [top[i, j] if k == 0 else 100000 for k, i, j in zip(trobs.k.values, trobs.i.values, trobs.j.values)],
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
  
    # now for ss obs
    # ss_groups = ["sshds"]
    # ssobs = obs.loc[(obs.weight>0)&(obs.obgnme.str.contains("ss")),:]
    # ssobs["standard_deviation"] = ssobs["standard_deviation"].astype(float)
    # assert len(ssobs) > 0
    # for group in ss_groups:
    #     sobs = ssobs.loc[ssobs.obgnme==group,:]
    #     print(group,sobs.shape)
    #     assert sobs.shape[0] > 0
    #     ovals = sobs.obsval
    #     std = sobs.standard_deviation.max()
    #     reals = np.array([ovals+(d*std) for d in draws])
    #     print(reals.shape)
    #     oe.loc[:,sobs.obsnme] = reals

    oe.to_binary(os.path.join(t_d,"noise.jcb"))
    pst.pestpp_options["ies_obs_en"] = "noise.jcb"
    pst.control_data.noptmax = -2
    pst.write(os.path.join(t_d,f"{modnm}.pst"),version=2)
    #pyemu.os_utils.run(f"pestpp-ies {modnm}.pst",cwd=t_d)

    with PdfPages(os.path.join(t_d,"noise_draws.pdf")) as pdf:

        # for group in ss_groups:
        #     sobs = ssobs.loc[ssobs.obgnme==group,:]
        #     fig,ax = plt.subplots(1,1,figsize=(10,10))
        #     ovals = sobs.obsval
        #     vals = oe.loc[:,sobs.obsnme].values
        #     [ax.plot(ovals,vals[i,:],'r-') for i in range(vals.shape[0])]
        #     mn = min(ax.get_xlim()[0],ax.get_ylim()[0])
        #     mx = max(ax.get_xlim()[1],ax.get_ylim()[1])
        #     ax.set_xlim(mn,mx)
        #     ax.set_ylim(mn,mx)
        #     ax.grid()
        #     ax.set_title(group,loc="left")
        #     pdf.savefig()
        #     plt.close(fig)
            
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
            if tp_series[0] < 100000:
                ax.plot(dts, tp_series, "k--", lw=2.0, zorder=5, label="top elevation ft")

            ax.set_title(uobs.obgnme.iloc[0] + " " + uid+ ' top:' + str(tp_series[0]), loc="left")
            ax.grid()
            ax.grid(which="minor", color="k", alpha=0.1, linestyle=":")
            ax.set_xlim(trobs.datetime.min(), trobs.datetime.max())
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

# ---- Main
if __name__ == "__main__":
    #simple_obs_vs_sim_plots("master_flow_barrier_rivCell_zones")
    #exit()
    print("Running setup_pst.py")
    print('Env path order:')
    for path in sys.path:
        print(path)

    modnm = 'swww'
    run_tag ='MasterRun3'
    org_d = os.path.join('model_ws', modnm+'_clean')
    
    # !!! Check sums!
    # Phi factor weighting dict
    phi_factor_dict = {#'swsshds':1e-20,               # Spiritwood SS -> Zero weight
                       #'wwsshds':1e-20,               # Warwick SS -> Zero weight
                       'swtranshds':0.18,             # Spiritwood transient, post 2000
                       'pre2ktranshds':0.40,          # All pre-2000 transient targets
                       'wwtranshds':0.30,             # Warwick transient, post 2000
                       'swsheyennetranshds':0.04,     # Spiritwood-Sheyenne, all years -> Low relative weight
                       'greater_than_sw_kh':0.04,     # Keeping HK higher in central channel of Spiriwood
                       'less_than_sw_kh':0.04,        # Try and get K in SW-Sheyenne Lower than main Spiritwood
                       # 'less_than_sw_k33':0.03,       # Control K33 in SW_sheyenne
                       }
    
    # Check if run inputs exisit:
    if not os.path.exists(os.path.join('run_inputs',f'{modnm}_{run_tag}')):
        raise FileNotFoundError(f'Run inputs for {modnm}_{run_tag} do not exist, please create them')

    # dir locations
    m_d_flow = "master_flow_" + run_tag
    t_d_flow = "template_flow_" + run_tag

    template = t_d_flow

    # prep the flow template
    prep_flow = True

    # High dim flag (True adds PPs on NPF and STO parameters)
    high_dim = True

    # Run the copied over base model
    run_base = True

    # run ies for the flow template
    run_flow = True
    
    # Init lambda (-100 suggested by Ryan)
    init_lam = -100

    # CONDOR FLAG
    use_condor = True
    print(f' use condor: {use_condor}')

    if use_condor:
        num_reals_flow = 240
        num_workers_flow = 120
        hostname = '10.99.10.30'
        pid = os.getpid()
        current_time = int(time.time())
        seed_value = (pid + current_time) % 65536
        np.random.seed(seed_value)
        port = random.randint(4001, 4999)
        print(f'port #: {port}')
    else:
        num_reals_flow = 60
        num_workers_flow = 12
        hostname = None
        port = None

    # How many iters to use
    noptmax_flow = 4

    local = True

    if prep_flow:
        print('{0}\n\npreparing flow-IES\n\n{1}'.format('*'*17,'*'*17))

        setup_pstpp(org_d,
                    modnm,
                    run_tag,
                    t_d_flow,
                    flex_con=False,
                    num_reals=num_reals_flow,
                    run_base=run_base,
                    high_dim=high_dim
                    )

        print(f'------- flow-ies has been setup in {t_d_flow} ----------')

        obs = set_obsvals_and_weights(t_d_flow,
                                      flow_weight_scheme='all_wl_meas',
                                      phi_factor_dict=phi_factor_dict
                                      )
        
        draw_noise_reals(t_d_flow, modnm)

    if run_flow:
       print('*** running flow-ies to get posterior ***')
       run_ies(t_d_flow,
               m_d=m_d_flow,
               num_workers=num_workers_flow,
               noptmax=noptmax_flow,
               init_lam=init_lam,
               local=local,
               use_condor=use_condor,
               hostname=hostname,
               port=port)

    print('All Done!, congrats we did a thing :)')
