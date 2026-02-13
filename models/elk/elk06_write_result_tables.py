import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import flopy
import pyemu
import pandas as pd
import numpy as np
import warnings  
import geopandas as gpd   
import matplotlib.pyplot as plt       
warnings.filterwarnings("ignore")

import elk04_process_plot_results as wpp
wpp.set_graph_specifications()
wpp.set_map_specifications()

import elk02_model_build as whap_build

# ============================================
# parameter and observation summary tables
# ============================================

def write_summary_tables(d='.', pst_name="elk_2lay.pst", outdir='', noptmax=0, max_fail=2):
    """writes parameter and observation summary tables using pyemu methods
    Args:
        d (str): relative path to master directory
        pst_name (str): pest control file name
        noptmax (int): pest parameter estimation iteration to plot
        max_fail (int): maximum threshold of convergence failures allowed before dropping realization from plots
    """ 
    o_d = outdir
    if os.path.exists(o_d) == False:
        os.makedirs(o_d)
    pst = pyemu.Pst(os.path.join(d, pst_name))
    pst.write_par_summary_table(filename=os.path.join(o_d, "par_summary.xlsx"))
    pst.write_obs_summary_table(filename=os.path.join(o_d, "obs_summary.xlsx"))

def write_basic_mf6_info_table(d='.',outdir='',modnm='elk_2lay'):
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=d, sim_name=modnm)
    gwf = sim.get_model(modnm)

    # print list of packages
    print(gwf.get_package_list())
    
    nlay = gwf.dis.nlay.array
    nrow = gwf.dis.nrow.array
    ncol = gwf.dis.ncol.array
    delr = gwf.dis.delr.array
    delc = gwf.dis.delc.array
    idom = gwf.dis.idomain.array
    
    info_df = pd.DataFrame(columns=['Layer'])
    info_df['Layer'] = np.arange(1,nlay+1)
    
    for layer in range(nlay):
        # get number of active cells
        n_active = np.sum(idom[layer,:,:]==1)
        info_df.loc[layer, 'Active Cells'] = n_active
    
    riv = gwf.riv.stress_period_data.get_data()[0]
    riv = pd.DataFrame(riv, columns=riv.dtype.names)
    riv["layer"] = riv["cellid"].apply(lambda x: x[0] if isinstance(x, tuple) and len(x)==3 else None)
    riv["row"]   = riv["cellid"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x)==3 else None)
    riv["col"]   = riv["cellid"].apply(lambda x: x[2] if isinstance(x, tuple) and len(x)==3 else None)
    riv['layer'] = riv['layer'] + 1  # convert to 1-based indexing
    
    riv_counts = riv.groupby('layer').size().reset_index(name='Number of River Cells')
    for _, row in riv_counts.iterrows():
        info_df.loc[info_df['Layer'] == row['layer'], 'Number of River Cells'] = row['Number of River Cells']
        
    otriv = gwf.otriv.stress_period_data.get_data()[0]
    otriv = pd.DataFrame(otriv, columns=otriv.dtype.names)
    otriv["layer"] = otriv["cellid"].apply(lambda x: x[0] if isinstance(x, tuple) and len(x)==3 else None)
    otriv["row"]   = otriv["cellid"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x)==3 else None)
    otriv["col"]   = otriv["cellid"].apply(lambda x: x[2] if isinstance(x, tuple) and len(x)==3 else None)
    otriv['layer'] = otriv['layer'] + 1  # convert to
    
    otriv_counts = otriv.groupby('layer').size().reset_index(name='Number of Otter River Cells')
    for _, row in otriv_counts.iterrows():
        info_df.loc[info_df['Layer'] == row['layer'], 'Number of Otter River Cells'] = row['Number of Otter River Cells']
    
    # get drain cell counts:
    drn = gwf.drn.stress_period_data.get_data()[0]
    drn = pd.DataFrame(drn, columns=drn.dtype.names)
    drn["layer"] = drn["cellid"].apply(lambda x: x[0] if isinstance(x, tuple) and len(x)==3 else None)
    drn["row"]   = drn["cellid"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x)==3 else None)
    drn["col"]   = drn["cellid"].apply(lambda x: x[2] if isinstance(x, tuple) and len(x)==3 else None)
    drn['layer'] = drn['layer'] + 1  # convert to 1-based indexing
    drn_counts = drn.groupby('layer').size().reset_index(name='Number of Drain Cells')
    for _, row in drn_counts.iterrows():
        info_df.loc[info_df['Layer'] == row['layer'], 'Number of Drain Cells'] = row['Number of Drain Cells']
        
    # get ghb cell counts:
    edg_ghb = gwf.ghb.stress_period_data.get_data()[0]
    edg_ghb = pd.DataFrame(edg_ghb, columns=edg_ghb.dtype.names)
    edg_ghb["layer"] = edg_ghb["cellid"].apply(lambda x: x[0] if isinstance(x, tuple) and len(x)==3 else None)
    edg_ghb["row"]   = edg_ghb["cellid"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x)==3 else None)
    edg_ghb["col"]   = edg_ghb["cellid"].apply(lambda x: x[2] if isinstance(x, tuple) and len(x)==3 else None)
    edg_ghb['layer'] = edg_ghb['layer'] + 1  # convert to 1-based indexing
    ghb_counts = edg_ghb.groupby('layer').size().reset_index(name='Number of GHB Cells')
    for _, row in ghb_counts.iterrows():
        info_df.loc[info_df['Layer'] == row['layer'], 'Number of GHB Cells'] = row['Number of GHB Cells']
        
    wbv_ghb = gwf.ghb_wbv.stress_period_data.get_data()[0]
    wbv_ghb = pd.DataFrame(wbv_ghb, columns=wbv_ghb.dtype.names)
    wbv_ghb["layer"] = wbv_ghb["cellid"].apply(lambda x: x[0] if isinstance(x, tuple) and len(x)==3 else None)
    wbv_ghb["row"]   = wbv_ghb["cellid"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x)==3 else None)
    wbv_ghb["col"]   = wbv_ghb["cellid"].apply(lambda x: x[2] if isinstance(x, tuple) and len(x)==3 else None)
    wbv_ghb['layer'] = wbv_ghb['layer'] + 1  # convert to 1-based indexing
    wbv_ghb_counts = wbv_ghb.groupby('layer').size().reset_index(name='Number of GHB WBV Cells')
    for _, row in wbv_ghb_counts.iterrows():
        info_df.loc[info_df['Layer'] == row['layer'], 'Number of GHB WBV Cells'] = row['Number of GHB WBV Cells']
        
    lst_well_pkgs = ['CAR', 'MALT', 'COW', 'MINN', 'COB']
    for well_pkg in lst_well_pkgs:
        well = getattr(gwf, well_pkg.lower())
        well_data = well.stress_period_data.get_data()
        rows = []
        for key, recarr in well_data.items():
            for cellid, q in recarr:
                rows.append({
                    "stress_period": key,
                    "layer": cellid[0],
                    "row": cellid[1],
                    "col": cellid[2],
                    "q": q
                })
        df = pd.DataFrame(rows)
        # get unique layer row col combinations
        well_data = df[['layer','row','col']].drop_duplicates()
        well_data['layer'] = well_data['layer'] + 1  # convert to 1-based indexing
        well_counts = well_data.groupby('layer').size().reset_index(name=f'Number of {well_pkg} Wells')
        for _, row in well_counts.iterrows():
            info_df.loc[info_df['Layer'] == row['layer'], f'Number of {well_pkg} Wells'] = row[f'Number of {well_pkg} Wells']    
    
    
    info_df.to_csv(os.path.join(outdir,'basic_mf6_model_info_table.csv'),index=False)

def write_avg_thickness_table(d='.',outdir='',modnm='elk_2lay'):
    sim = flopy.mf6.MFSimulation.load(sim_ws=d, sim_name=modnm,load_only=['DIS'])
    gwf = sim.get_model(modnm)

    top = gwf.dis.top.array
    botm = gwf.dis.botm.array
    idom = gwf.dis.idomain.array
    
    thk_stats = pd.DataFrame(index=np.arange(0,len(botm)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
   
    for k in range(len(botm)):
        col = f'thk_{k}'
        if k == 0:
            thk = top - botm[k]
            thk = np.where(idom[k,:,:]<1, np.nan, thk)
            thk_stats.loc[k,:] = pd.DataFrame(thk.reshape(thk.shape[0]*thk.shape[1])).describe()[0].values
        else:
            thk = botm[k-1] - botm[k]
            thk = np.where(idom[k,:,:]<1, np.nan, thk)
            thk_stats.loc[k,:] = pd.DataFrame(thk.reshape(thk.shape[0]*thk.shape[1])).describe()[0].values
    
    thk_stats.index.name = 'Layer'
    thk_stats = thk_stats.reset_index()
    thk_stats.to_csv(os.path.join(outdir,'layer_thickness_stats_table.csv'),index=False)
        
def stress_period_table(d='.',outdir='',annual_only=False):
    fpath = os.path.join('tables', 'monthly_stress_period_info.csv')
    # Try to load the file, otherwise generate it
    try:
        df = pd.read_csv(fpath)
    except FileNotFoundError:
        whap_build.stress_period_df_gen(d, 1965, annual_only=annual_only)
        if annual_only == False:
            fpath = os.path.join(d, 'tables', 'monthly_stress_period_info.csv')
        df = pd.read_csv(fpath)
        
    df = df.reset_index()
    df = df[['stress_period', 'perlen', 'cum_days', 'start_datetime', 'end_datetime', 'steady_state']]
    df = df.rename(columns={'stress_period':'Stress Period','start_datetime':'Stress Period Begins','end_datetime':'Stress Period Ends', 'perlen':'Stress Period Length (days)','steady_state':'SS or TR','cum_days':'Cumulative Days'})
    df.loc[df['SS or TR']==True,'SS or TR'] = 'SS'
    df.loc[df['SS or TR']==False,'SS or TR'] = 'TR' 
    df.to_csv(os.path.join(outdir,'stress_period_data_table_docformat.csv'),index=False)
    
def recharge_table_ann_only(d='.',outdir='.'):
    # reads in recharge files from working directory, associates values with counties, and determines the average
    # recharge rate for each county in each annual stress period. For comparision purposes, this function also
    # pulls down the precipitation rasters from PRISM and calculates the average annual precipitation for each county.
    print('Building recharge table...')
    
    rch_files = [f for f in os.listdir(d) if f.startswith('rch_') and f.endswith('.txt')]
    
    spdf = pd.read_csv(os.path.join('tables','monthly_stress_period_info.csv'))
    spdf['start_datetime'] = pd.to_datetime(spdf['start_datetime'])
    spdf['year'] = spdf['start_datetime'].dt.year
    
    all_avg_df = pd.DataFrame(columns=['Stress Period','Year'])
    
    for f in rch_files:
        df_raw = pd.read_csv(
            os.path.join(d, f),
            index_col=False,
            delim_whitespace=True,
            header=None
        )

        arr = df_raw.values

        df = (
            pd.DataFrame(arr)
            .stack()
            .reset_index()
            .rename(columns={'level_0': 'row', 'level_1': 'col', 0: 'rate'})
        )
        
        df['Stress Period'] = int(f.split('_')[-1].split('.')[0])
        df['year'] = spdf.loc[spdf['stress_period']==df['Stress Period'].values[0],'year'].values[0]
        df = df.drop(['row','col'],axis=1)
        ftd_2_inyr = 12*365.25
        df['Recharge Rate (in/yr)'] = df['rate']*ftd_2_inyr
        df = df.rename(columns={'year':'Year'})
        df = df.drop('rate',axis=1)

        
        avg_df = df.groupby(['Stress Period']).mean().reset_index()
        all_avg_df = pd.concat([all_avg_df,avg_df],ignore_index=True)
    # sort by stress period
    all_avg_df = all_avg_df.sort_values(by='Stress Period')
  
    all_avg_df.to_csv(os.path.join(outdir,'recharge_table_docformat.csv'),index=False)
    
def ghb_table(d='.',outdir='.'):
    # reads in ghb files from working directory, and writes out head and conductance values for each layer/row/col 
    # with a ghb boundary condition. 
    print('Building GHB table...')
    
    ghb_files = [f for f in os.listdir(d) if f.startswith('ghb') and f.endswith('.txt')]
    wahp_ghb_files = [f for f in os.listdir(d) if f.startswith('wahp7ly_wbv.ghb') and f.endswith('.txt')]
    
    f = ghb_files[0]
    
    ghb_df = pd.DataFrame(columns=['Stress Period','Layer','Row','Column','Head','Conductance'])
    
    
    df = pd.read_csv(os.path.join(d,f),index_col=False,delim_whitespace=True,header=None)
    df.columns = ['#k','i','j','bhead','cond']
    df['Stress Period'] = int(f.split('.')[0].split('_')[-1])+1
    df = df.rename(columns={'#k':'Layer','i':'Row','j':'Column','bhead':'Head','cond':'Conductance'})
    ghb_df = pd.concat([ghb_df,df],ignore_index=True)
    ghb_df['GHB Type'] = 'Edge GHB'
    
    f = wahp_ghb_files[0]
    df_wbv = pd.read_csv(os.path.join(d,f),index_col=False,delim_whitespace=True,header=None)
    df_wbv.columns = ['#k','i','j','bhead','cond']
    df_wbv['Stress Period'] = int(f.split('.')[-2].split('_')[-1])+1
    df_wbv = df_wbv.rename(columns={'#k':'Layer','i':'Row','j':'Column','bhead':'Head','cond':'Conductance'})
    df_wbv['GHB Type'] = 'WBV GHB'
    ghb_df = pd.concat([ghb_df,df_wbv],ignore_index=True)
    
    ghb_df.to_csv(os.path.join(outdir,'ghb_table_docformat.csv'),index=False)
               
def well_table(d='.',outdir='.'):
    # reads in well files from working directory, and writes out the total pumping in each county, each use type, and for each stress period.
    # pumping also gets broken down into Layer 1 (Seymour and Trinity) and all lower layers (Cross Timbers)
    print('Building well table...')
    use_typp_acronyms = ['dom','irr','stk','min','mfg'] #NOTE: add municipial later

    for ut in use_typp_acronyms:
        well_files = [f for f in os.listdir(d) if f.startswith(f'ctgam_{ut}') and f.endswith('.txt') and ut in f]
        well_df = pd.DataFrame(columns=['Stress Period','County','Pumping (cfd)','Cross Timbers Pumping'])
        for f in well_files:
            df = pd.read_csv(os.path.join(d,f),header=None,index_col=False,delim_whitespace=True)
            df.columns = ['Layer','Row','Column','Pumping Rate (cfd)','Boundname']
            df['Use Type'] = df['Boundname'].apply(lambda x: x.split('.')[0])
            df['Stress Period'] = int(f.split('.')[0].split('_')[-1])+1
            df = df.rename(columns={'i':'Row','j':'Column','q':'Pumping Rate'})
            df['County'] = df['Boundname'].apply(lambda x: x.split('.')[1])
            df['County'] = df.County.str.title()
            well_df = pd.concat([well_df,df],ignore_index=True)
        well_df.to_csv(os.path.join(outdir,f'ctgam_{ut}_table_docformat.csv'),index=False)
    
def drains_table(d='.',outdir='.'):
    # reads in drain files from working directory, and writes out the head and conductance values for each layer/row/col with a drain boundary condition.
    print('Building drains table...')
    drn_files = [f for f in os.listdir(d) if f.startswith('drn') and f.endswith('.dat')]
    
    drn_df = pd.DataFrame(columns=['Stress Period','Layer','Row','Column','Elevation','Conductance','Boundname'])
    for f in drn_files:
        df = pd.read_csv(os.path.join(d,f),index_col=False,delim_whitespace=True,header=None)
        df.columns = ['Layer','Row','Column','Elevation','Conductance','Boundname']
        df['Stress Period'] = int(f.split('.')[0].split('_')[1])+1
        drn_df = pd.concat([drn_df,df],ignore_index=True)
    
    drn_df.to_csv(os.path.join(outdir,f'drn_table_docformat.csv'),index=False)   
    
def river_table(d='.',outdir='.'):
    # reads in river files from working directory, and writes out the stage and conductance values for each layer/row/col with a river boundary condition.
    print('Building river table...')
    
    riv_files = [f for f in os.listdir(d) if f.startswith('riv') and f.endswith('.txt')]
    otter_files = [f for f in os.listdir(d) if f.startswith('otriv') and f.endswith('.txt')]
    
    riv_df = pd.DataFrame(columns=['Stress Period','Layer','Row','Column','Stage','Conductance','Boundname'])
    for f in riv_files:
        df = pd.read_csv(os.path.join(d,f),index_col=False,delim_whitespace=True,header=None)
        df.columns = ['Layer','Row','Column','Stage','Conductance','River Bottom Elevation (ft-abv m.s.l.)']
        df['Stress Period'] = int(f.split('.')[0].split('_')[-1])
        df['Boundname'] = 'Red and other rivers'
        riv_df = pd.concat([riv_df,df],ignore_index=True)
        
        otf = f.replace('riv','otriv')
        df_otter = pd.read_csv(os.path.join(d,otf),index_col=False,delim_whitespace=True,header=None)
        df_otter.columns = ['Layer','Row','Column','Stage','Conductance','River Bottom Elevation (ft-abv m.s.l.)']
        df_otter['Stress Period'] = int(otf.split('.')[0].split('_')[-1])
        df_otter['Boundname'] = 'Otter river'
        riv_df = pd.concat([riv_df,df_otter],ignore_index=True)
        
    riv_df.to_csv(os.path.join(outdir,'riv_table_docformat.csv'),index=False)

# ============================================
# package stats spreadsheets:
# ============================================

def create_K_stats_tables(d='.'):
    outdir =os.path.join(d,'for_documentation','tables','package_tables')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    kh_files = [f for f in os.listdir(d) if f.startswith('npf_k_') and f.endswith('.txt')]
    aniso_files =  [f for f in os.listdir(d) if f.startswith('npf_k33_') and f.endswith('.txt')]
    idom_files = [f for f in os.listdir(d) if f.startswith('dis_idom') and f.endswith('.txt')]

    assert len(kh_files) > 0, "No kh files found in directory."
    assert len(aniso_files) > 0, "No anisotropy files found in directory."
    assert len(idom_files) > 0, "No idomain files found in directory."
    
    # sort files:
    kh_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))
    aniso_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))
    idom_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))
    
    kh_stats = pd.DataFrame(index=np.arange(0,len(kh_files)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
    for lay in range(len(kh_files)):
        arr = np.loadtxt(os.path.join(d,kh_files[lay]))
        idom = np.loadtxt(os.path.join(d, idom_files[lay]))
        idom[idom<1] = np.nan
        arr = arr * idom
        arr_df = pd.DataFrame(arr.reshape(arr.shape[0]*arr.shape[1])).describe()
        kh_stats.loc[lay,:] = arr_df[0].values
    kh_stats.to_csv(os.path.join(outdir, 'kh_stats.csv'))

    aniso_stats = pd.DataFrame(index=np.arange(0,len(aniso_files)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
    for lay in range(len(aniso_files)):
        arr = np.loadtxt(os.path.join(d,aniso_files[lay]))
        idom = np.loadtxt(os.path.join(d, idom_files[lay]))
        idom[idom<1] = np.nan
        arr = arr * idom
        arr_df = pd.DataFrame(arr.reshape(arr.shape[0]*arr.shape[1])).describe()
        aniso_stats.loc[lay,:] = arr_df[0].values
    aniso_stats.to_csv((os.path.join(outdir, 'aniso_stats.csv')))
    
    kv_stats = pd.DataFrame(index=np.arange(0,len(kh_files)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
    for lay in range(len(kh_files)):
        kh_arr = np.loadtxt(os.path.join(d,kh_files[lay]))
        aniso_arr = np.loadtxt(os.path.join(d,aniso_files[lay]))
        idom = np.loadtxt(os.path.join(d, idom_files[lay]))
        idom[idom<1] = np.nan
        kv_arr = kh_arr * aniso_arr
        kv_arr = kv_arr * idom
        arr_df = pd.DataFrame(kv_arr.reshape(kv_arr.shape[0]*kv_arr.shape[1])).describe()
        kv_stats.loc[lay,:] = arr_df[0].values
    kv_stats.to_csv((os.path.join(outdir, 'kv_stats.csv')))

def create_sto_stats_tables(d='.'):
    outdir =os.path.join(d,'for_documentation','tables','package_tables')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    ss_files = [f for f in os.listdir(d) if f.startswith('sto_ss_') and f.endswith('.txt')]
    sy_files =  [f for f in os.listdir(d) if f.startswith('sto_sy_') and f.endswith('.txt')]
    idom_files = [f for f in os.listdir(d) if f.startswith('dis_idom') and f.endswith('.txt')]
    
    assert len(ss_files) > 0, "No specific storage files found in directory."
    assert len(sy_files) > 0, "No specific yield files found in directory."
    assert len(idom_files) > 0, "No idomain files found in directory."
    
    idom_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))
    sy_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))
    ss_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('layer','')))

    ss_stats = pd.DataFrame(index=np.arange(0,len(ss_files)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
    for lay in range(len(ss_files)):
        arr = np.loadtxt(os.path.join(d,ss_files[lay]))
        idom = np.loadtxt(os.path.join(d, idom_files[lay]))
        idom[idom<1] = np.nan
        arr = arr * idom
        arr_df = pd.DataFrame(arr.reshape(arr.shape[0]*arr.shape[1])).describe()
        ss_stats.loc[lay,:] = arr_df[0].values
    ss_stats.to_csv(os.path.join(outdir, 'ss_stats.csv'))

    sy_stats = pd.DataFrame(index=np.arange(0,len(sy_files)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
    for lay in range(len(sy_files)):
        arr = np.loadtxt(os.path.join(d,sy_files[lay]))
        idom = np.loadtxt(os.path.join(d, idom_files[lay]))
        idom[idom<1] = np.nan
        arr = arr * idom
        arr_df = pd.DataFrame(arr.reshape(arr.shape[0]*arr.shape[1])).describe()
        sy_stats.loc[lay,:] = arr_df[0].values
    sy_stats.to_csv((os.path.join(outdir, 'sy_stats.csv')))


def recharge_stats_table(d='.'):
    outdir =os.path.join(d,'for_documentation','tables','package_tables')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    rch_files = [f for f in os.listdir(d) if f.startswith('rch_') and f.endswith('.txt')]
    assert len(rch_files) > 0, "No recharge files found in directory."
    
    #sort:
    rch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    rch_stats = pd.DataFrame(index=np.arange(0,len(rch_files)), columns = ['count', 'mean', 'std', 'min', '25%','50%', '75%', 'max'])
    for sp in range(len(rch_files)):
        arr = np.loadtxt(os.path.join(d,rch_files[sp]))
        arr_df = pd.DataFrame(arr.reshape(arr.shape[0]*arr.shape[1])).describe()
        rch_stats.loc[sp,:] = arr_df[0].values
    rch_stats.index.name = 'Stress Period'
    rch_stats = rch_stats.reset_index()
    rch_stats['Stress Period'] = rch_stats['Stress Period'] + 1  # convert to 1-based indexing
    
    # add one last row that is the overall stats:
    overall_stats = pd.DataFrame(columns=rch_stats.columns)
    overall_stats.loc[0,'Stress Period'] = 'Overall'
    overall_stats.loc[0,'count'] = rch_stats['count'].mean()
    overall_stats.loc[0,'mean'] = rch_stats['mean'].mean()
    overall_stats.loc[0,'std'] = rch_stats['std'].mean()
    overall_stats.loc[0,'min'] = rch_stats['min'].mean()
    overall_stats.loc[0,'25%'] = rch_stats['25%'].mean()
    overall_stats.loc[0,'50%'] = rch_stats['50%'].mean()
    overall_stats.loc[0,'75%'] = rch_stats['75%'].mean()
    overall_stats.loc[0,'max'] = rch_stats['max'].mean()
    
    rch_stats = pd.concat([rch_stats, overall_stats], ignore_index=True)
    
    # now conver to inches per year from feet per day:
    ftd_2_inyr = 12*365.25
    rch_stats.loc[:, 'mean':'max'] = rch_stats.loc[:, 'mean':'max'] * ftd_2_inyr
    
    rch_stats.to_csv(os.path.join(outdir, 'recharge_stats_inyr.csv'))


def make_ghb_drn_summary_spreadsheets(d='.'):
    outdir =os.path.join(d,'for_documentation','tables', 'package_tables')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    # os.makedirs(outdir, exist_ok=True)

    ghb_files = [f for f in os.listdir(d) if f.startswith('ghb') and f.endswith('.txt')]
    wahp_ghb_files = [f for f in os.listdir(d) if f.startswith('wahp7ly_wbv.ghb') and f.endswith('.txt')]
    
    f = ghb_files[0]

    data = pd.read_csv(os.path.join(d, f), index_col=False, delim_whitespace=True, header=None)
    data.columns = ['layer', 'row', 'column', 'bhead', 'cond']
    
    data = data.drop_duplicates(subset=['layer', 'row', 'column'], keep='last')

    lay_min = data['layer'].min()
    lay_max = data['layer'].max()

    df_stats_stage = pd.DataFrame(columns=['mean', 'min', 'max', 'median', 'std'], index=np.arange(lay_min, lay_max+1))
    df_stats_cond = pd.DataFrame(columns=['mean', 'min', 'max', 'median', 'std'], index=np.arange(lay_min, lay_max+1))

    for lay in range(lay_min, lay_max+1):
        data_lay = data[data['layer']==lay]
        if data_lay.shape[0] == 0:
            continue
        else:
            df_stats_stage.loc[lay, :] = [np.mean(data_lay['bhead'].values), np.min(data_lay['bhead'].values),
                                            np.max(data_lay['bhead'].values),
                                        np.median(data_lay['bhead'].values), np.std(data_lay['bhead'].values)]

            df_stats_cond.loc[lay, :] = [np.mean(data_lay['cond'].values), np.min(data_lay['cond'].values),
                                            np.max(data_lay['cond'].values),
                                        np.median(data_lay['cond'].values), np.std(data_lay['cond'].values)]

    df_stats_stage.to_csv(os.path.join(d,'package_tables', f'edg_stage_stats.csv'))
    df_stats_cond.to_csv(os.path.join(d,'package_tables', f'edg_cond_stats.csv'))
    
    f = wahp_ghb_files[0]
    
    data = pd.read_csv(os.path.join(d, f), index_col=False, delim_whitespace=True, header=None)
    data.columns = ['layer', 'row', 'column', 'bhead', 'cond']
    data = data.drop_duplicates(subset=['layer', 'row', 'column'], keep='last')
    lay_min = data['layer'].min()
    lay_max = data['layer'].max()
    
    df_stats_stage = pd.DataFrame(columns=['mean', 'min', 'max', 'median', 'std'], index=np.arange(lay_min, lay_max+1))
    df_stats_cond = pd.DataFrame(columns=['mean', 'min', 'max', 'median', 'std'], index=np.arange(lay_min, lay_max+1))
    
    for lay in range(lay_min, lay_max+1):
        data_lay = data[data['layer']==lay]
        if data_lay.shape[0] == 0:
            continue
        else:
            df_stats_stage.loc[lay, :] = [np.mean(data_lay['bhead'].values), np.min(data_lay['bhead'].values),
                                            np.max(data_lay['bhead'].values),
                                        np.median(data_lay['bhead'].values), np.std(data_lay['bhead'].values)]

            df_stats_cond.loc[lay, :] = [np.mean(data_lay['cond'].values), np.min(data_lay['cond'].values),
                                            np.max(data_lay['cond'].values),
                                        np.median(data_lay['cond'].values), np.std(data_lay['cond'].values)]
    
    df_stats_stage.to_csv(os.path.join(d,'package_tables', f'wbv_stage_stats.csv'))
    df_stats_cond.to_csv(os.path.join(d,'package_tables', f'wbv_cond_stats.csv'))

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

def calc_transient_range_scaled_residuals(
    m_d="master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base",
    modnm="elk_2lay",
    obs_csv=None,
    include_group_regex=r"transhds|transearlyhds",   # <-- only true transient heads
    exclude_group_regex=r"deriv|dhd|dhdt|diff|change|ddtrgs|sshds",  # <-- drop derivative/drawdown/etc.
    time_col_candidates=("time", "totim", "tottime", "datetime", "date", "kper"),
    dry_head=-9999.0,
    drop_dry=True,
    drop_negative_obs=False,   # set True if you REALLY want to force positive observed heads
    drop_negative_sim=False,   # same for sim
):
    """
    Compute unweighted transient residuals (obs - sim) and "scaled" residuals where scaling
    is by the observed head range (max(obs) - min(obs)), using only selected transient head groups.

    Scaling definition:
        resid_scaled = (obsval - sim) / (max_obs - min_obs)

    Returns:
        mrg: long-form dataframe with residuals
        obs_range: range used for scaling
        scaled_rms: RMS(resid_scaled)
        by_time: scaled RMS by time
    """

    # ---- load pst obs (obsval + groups)
    pst_path = os.path.join(m_d, f"{modnm}.pst")
    pst = pyemu.Pst(pst_path)
    obs = pst.observation_data.copy()

    # ---- filter: include only transient head groups, exclude derivative-ish groups
    sel = obs.obgnme.str.contains(include_group_regex, case=False, regex=True, na=False)

    if exclude_group_regex:
        sel &= ~obs.obgnme.str.contains(exclude_group_regex, case=False, regex=True, na=False)

    gwobs = obs.loc[sel, :].copy()

    # keep observed obs and numeric obsval
    gwobs = gwobs.loc[gwobs.observed == True, :].copy()
    gwobs["obsval"] = pd.to_numeric(gwobs["obsval"], errors="coerce")
    gwobs = gwobs.loc[gwobs["obsval"].notnull(), :].copy()

    if gwobs.empty:
        raise ValueError(
            "No observations left after group filtering.\n"
            f"include_group_regex='{include_group_regex}', exclude_group_regex='{exclude_group_regex}'"
        )

    # ---- drop dry observed values before computing range (and optionally drop negatives)
    gwobs_nd = gwobs.copy()
    if drop_dry:
        gwobs_nd = gwobs_nd.loc[gwobs_nd["obsval"] != dry_head, :].copy()
    if drop_negative_obs:
        gwobs_nd = gwobs_nd.loc[gwobs_nd["obsval"] >= 0.0, :].copy()

    if gwobs_nd.empty:
        raise ValueError(
            "All selected observed values were removed (dry/negative filtering). "
            f"dry_head={dry_head}, drop_negative_obs={drop_negative_obs}"
        )

    obs_min = float(gwobs_nd["obsval"].min())
    obs_max = float(gwobs_nd["obsval"].max())
    obs_range = obs_max - obs_min
    if not np.isfinite(obs_range) or obs_range <= 0:
        raise ValueError(f"Invalid obs range: min={obs_min}, max={obs_max}, range={obs_range}")

    # ---- find MF6 obs CSV if not provided
    if obs_csv is None:
        candidates = []
        candidates += glob.glob(os.path.join(m_d, "**", "*.obs.csv"), recursive=True)
        candidates += glob.glob(os.path.join(m_d, "**", "*obs*.csv"), recursive=True)
        candidates = [c for c in candidates if os.path.isfile(c) and os.path.getsize(c) > 0]
        if not candidates:
            raise FileNotFoundError(
                f"Couldn't find an MF6 obs CSV under {m_d}. Pass obs_csv=... explicitly."
            )
        obs_csv = max(candidates, key=os.path.getmtime)

    simdf = pd.read_csv(obs_csv)
    if simdf.empty:
        raise ValueError(f"MF6 obs CSV is empty: {obs_csv}")

    # ---- detect time column
    time_col = None
    for c in time_col_candidates:
        if c in simdf.columns:
            time_col = c
            break

    # ---- convert MF6 wide CSV -> long
    if time_col is not None:
        obs_cols = [c for c in simdf.columns if c != time_col]
        long = simdf.melt(id_vars=[time_col], value_vars=obs_cols, var_name="obsnme", value_name="sim")
        long.rename(columns={time_col: "time"}, inplace=True)
    else:
        long = simdf.melt(var_name="obsnme", value_name="sim")
        long["time"] = np.nan

    long["obsnme"] = long["obsnme"].astype(str)
    long["sim"] = pd.to_numeric(long["sim"], errors="coerce")

    # ---- build merge key robustly
    gwobs2 = gwobs.copy()
    if "obsnme" in gwobs2.columns:
        gwobs2["obsnme_key"] = gwobs2["obsnme"].astype(str)
    else:
        gwobs2["obsnme_key"] = gwobs2.index.astype(str)

    long["obsnme_key"] = long["obsnme"].astype(str)

    mrg = long.merge(
        gwobs2[["obsnme_key", "obsval", "obgnme"]],
        on="obsnme_key",
        how="inner",
    )
    mrg = mrg.loc[mrg["sim"].notnull(), :].copy()

    if mrg.empty:
        raise ValueError(
            f"No overlap between MF6 obs names and pst obsnme.\n"
            f"MF6 obs example: {long.obsnme.dropna().unique()[:10]}\n"
            f"PST obs example: {gwobs2.obsnme_key.values[:10]}\n"
            f"MF6 obs csv used: {obs_csv}"
        )

    # ---- drop dry/no-data heads from residuals (both obs and sim), optionally drop negatives
    if drop_dry:
        mrg = mrg.loc[(mrg["obsval"] != dry_head) & (mrg["sim"] != dry_head), :].copy()
    if drop_negative_obs:
        mrg = mrg.loc[mrg["obsval"] >= 0.0, :].copy()
    if drop_negative_sim:
        mrg = mrg.loc[mrg["sim"] >= 0.0, :].copy()

    if mrg.empty:
        raise ValueError(
            "After filtering dry/negative values, no rows remain. "
            f"dry_head={dry_head}, drop_negative_obs={drop_negative_obs}, drop_negative_sim={drop_negative_sim}"
        )

    # ---- residuals and range-scaled residuals
    mrg["resid"] = mrg["obsval"] - mrg["sim"]
    mrg["resid_scaled"] = mrg["resid"] / obs_range

    # ---- summaries
    rms = float(np.sqrt(np.mean(mrg["resid"] ** 2)))
    scaled_rms = float(np.sqrt(np.mean(mrg["resid_scaled"] ** 2)))

    # ---- scaled RMS by time
    by_time = (
        mrg.groupby("time", dropna=False)["resid_scaled"]
        .apply(lambda s: float(np.sqrt(np.mean(s.values ** 2))))
        .reset_index()
        .rename(columns={"resid_scaled": "scaled_rms"})
        .sort_values("time")
    )

    # keep nice output
    mrg["obsnme"] = mrg["obsnme_key"]
    mrg = mrg.drop(columns=["obsnme_key"])

    # ---- prints
    print(f"MF6 obs csv: {obs_csv}")
    print(f"Selected obs count in pst (after group filter): {len(gwobs):,}")
    if drop_dry:
        print(f"Dry obs removed (pst): {(gwobs['obsval'] == dry_head).sum():,}")
    print(f"Matched simulated rows (post-filter): {len(mrg):,}")
    print(f"Observed range used for scaling: {obs_min:.3f} to {obs_max:.3f} (range={obs_range:.3f})")
    print(f"RMS (head units): {rms:.4f}")
    print(f"Scaled RMS (RMS / obs_range): {scaled_rms:.6f}")

    return mrg, obs_range, scaled_rms, by_time

def calc_transient_scaled_rms_from_mf6_obs(
    m_d="master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base",
    modnm="elk_2lay",
    obs_csv=None,
    group_regex=r"transhds|transearlyhds|sshds|ddtrgs",
    time_col_candidates=("time", "totim", "tottime", "datetime", "date", "kper"),
    sim_col_candidates=("sim", "simulated", "value", "obsval", "calculated"),
):
    """
    Computes transient scaled RMS using:
      - observed values + weights from PEST pst (obsval, weight, obgnme)
      - simulated values from MF6 OBS output CSV
    """

    # ---- load pst obs (source of obsval, weight, groups)
    pst_path = os.path.join(m_d, f"{modnm}.pst")
    pst = pyemu.Pst(pst_path)
    obs = pst.observation_data.copy()

    # filter to transient groups (same intent as before)
    gwobs = obs.loc[obs.obgnme.str.contains(group_regex, case=False, regex=True, na=False), :].copy()
    gwobs = gwobs.loc[(gwobs.weight > 0) & (gwobs.observed == True), :].copy()

    if gwobs.empty:
        raise ValueError(f"No weighted observed obs found in pst matching group_regex='{group_regex}'")

    # ---- find MF6 obs CSV if not provided
    if obs_csv is None:
        # common patterns: *.obs.csv, *obs*.csv, *.csv in model folder
        candidates = []
        candidates += glob.glob(os.path.join(m_d, "**", "*.obs.csv"), recursive=True)
        candidates += glob.glob(os.path.join(m_d, "**", "*obs*.csv"), recursive=True)

        # keep only reasonably sized csv files
        candidates = [c for c in candidates if os.path.isfile(c) and os.path.getsize(c) > 0]

        if not candidates:
            raise FileNotFoundError(
                f"Couldn't find an MF6 obs CSV under {m_d}. "
                f"Pass obs_csv=... explicitly."
            )

        # pick the newest
        obs_csv = max(candidates, key=os.path.getmtime)

    simdf = pd.read_csv(obs_csv)
    if simdf.empty:
        raise ValueError(f"MF6 obs CSV is empty: {obs_csv}")

    # ---- detect time column (optional; we can still compute overall without it)
    time_col = None
    for c in time_col_candidates:
        if c in simdf.columns:
            time_col = c
            break

    # ---- MF6 obs CSV is usually wide: time column + one col per obsnme
    # We want long format: (time, obsnme, sim)
    wide_cols = list(simdf.columns)
    if time_col is not None:
        obs_cols = [c for c in wide_cols if c != time_col]
        long = simdf.melt(id_vars=[time_col], value_vars=obs_cols, var_name="obsnme", value_name="sim")
        long.rename(columns={time_col: "time"}, inplace=True)
    else:
        # no time col; treat as single snapshot (rare but possible)
        long = simdf.melt(var_name="obsnme", value_name="sim")
        long["time"] = np.nan

    # ---- match simulated obs names to pst obsnme
    # MF6 sometimes writes names with prefixes; usually it's exact though.
    long["obsnme"] = long["obsnme"].astype(str)
    gwobs["obsnme"] = gwobs.index.astype(str)

    mrg = long.merge(
        gwobs[["obsval", "weight", "obgnme"]],
        on="obsnme",
        how="inner"
    )

    if mrg.empty:
        # show some debugging info
        raise ValueError(
            f"No overlap between MF6 obs names and pst obsnme.\n"
            f"MF6 obs example: {long.obsnme.dropna().unique()[:10]}\n"
            f"PST obs example: {gwobs.index.values[:10]}\n"
            f"MF6 obs csv used: {obs_csv}"
        )

    # ---- residuals & scaled residuals
    mrg["resid"] = mrg["obsval"] - mrg["sim"]
    mrg["scaled_resid"] = mrg["resid"] * np.sqrt(mrg["weight"].astype(float))

    # ---- overall scaled RMS (across all times/obs)
    overall_scaled_rms = float(np.sqrt(np.mean(mrg["scaled_resid"] ** 2)))

    # ---- scaled RMS by time (transient)
    by_time = (
        mrg.groupby("time", dropna=False)["scaled_resid"]
        .apply(lambda s: float(np.sqrt(np.mean(s.values ** 2))))
        .reset_index()
        .rename(columns={"scaled_resid": "scaled_rms"})
        .sort_values("time")
    )

    # ---- scaled RMS by group (and by time if you want)
    by_group = (
        mrg.groupby("obgnme")["scaled_resid"]
        .apply(lambda s: float(np.sqrt(np.mean(s.values ** 2))))
        .reset_index()
        .rename(columns={"scaled_resid": "scaled_rms"})
        .sort_values("scaled_rms", ascending=False)
    )

    print(f"MF6 obs csv: {obs_csv}")
    print(f"N matched rows: {len(mrg):,}")
    print(f"Overall transient scaled RMS: {overall_scaled_rms:.4f}")

    return overall_scaled_rms, by_time, by_group, mrg

def resid_stats(m_d='master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base',outdir='', modnm='elk_2lay'):
    o_d = outdir
    layers = {
        0: 'Layer 1',
        1: 'Elk Valley Aquifer',
        }
    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, exe_name="mf6",load_only=["dis"])
    m = sim.get_model(modnm)
    # load cnty grid csv:
    #cnty_arr = pd.read_csv('model_grid_with_counties.csv')
    obsdict = get_ies_obs_dict(m_d=m_d,modnm=modnm)
    pst = pyemu.Pst(os.path.join(m_d,f"{modnm}.pst"))
    obs = pst.observation_data
    gwobs = obs.loc[obs.obgnme.str.contains('transhds|transearlyhds|sshds|ddtrgs'), :].copy()  # *** NOTE - this is a hard-coded filter
    gwobs['datetime'] = pd.to_datetime(gwobs.datetime.values)
    # add ids to trans obs:
    gwobs.loc[gwobs.obgnme.str.contains('trans'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('trans'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])
    #gwobs.loc[gwobs.obgnme.str.contains('idx_well'),'id'] = gwobs.loc[gwobs.obgnme.str.contains('idx_well'),'obsnme'].apply(lambda x: x.split('transh_')[1].split('_i')[0])
    #gwobs.loc[gwobs.obgnme.str.contains('idx_well'),'oname'] = 'idx_well'
    aobs = pd.read_csv(os.path.join(m_d, f'{modnm}.adjusted.obs_data.csv'), index_col=0)
    conflicts = set(gwobs.loc[gwobs.weight > 0, 'obsnme'].tolist()) - set(aobs.loc[aobs.weight > 0, :].index.tolist())
    print(conflicts)
    for col in ['k', 'i', 'j']:
        gwobs[col] = gwobs[col].astype(int)
    top = m.dis.top.array
    botm = m.dis.botm.array
    itrmx = 4

    # drop missing ids (likely NaN) and force to string to keep type consistent
    usites = gwobs["id"].dropna().astype(str).unique()

    # build weight table (no need to sort usites here; you'll sort usitedf by weight below)
    usitedf = pd.DataFrame({"site": usites}, index=usites)
    usitedf['weight'] = 0
    for usite in usites:
        uobs = gwobs.loc[gwobs.id == usite, :]
        usitedf.loc[usite, 'weight'] = uobs.weight.sum()
    usitedf.sort_values(by=['weight', 'site'], ascending=False, inplace=True)
    usites = usitedf['site'].values
    gwobs = gwobs.loc[gwobs.id.isin(usites), :]
    # merege some groups:
    #gwobs.loc[gwobs.oname=='idx_well','oname'] = 'transhds'
    #gwobs.loc[(gwobs.oname=='transhds') & (gwobs.k.isin([0,1,2])),'k'] = 2
    # only want stats on transhds:
    gwobs = gwobs.loc[gwobs.oname == 'transhds', :]
    #gwo_grps = gwobs.oname.unique()
    gwo_grps = gwobs.groupby('k')
    # print keys:
    print(gwo_grps.groups.keys())
    master_df = pd.DataFrame() 
    for nm,grp in gwo_grps:
        df = pd.DataFrame()
        layer = nm
        print(layer)
        lynm = layers[layer].replace('_',' ')
        oobs = grp.loc[grp.observed == True, :]
        wobs = oobs.loc[oobs.weight > 0, :]
        wobs = wobs.reset_index(drop=True)
        if len(wobs) == 0:
            print('No weighted observations for layer:',lynm)
            continue
        vals = obsdict[itrmx].loc[:,obsdict[itrmx].columns.isin(grp.obsnme.values)].T
        base = vals.loc[:,"base"].reset_index()
        base = base.rename(columns={"index":"obsnme"})
        base_mrg = pd.merge(base,wobs,on='obsnme',how='left')
        base = base_mrg.loc[base_mrg.obsval.notnull(),:]
        base['resid'] = base.obsval - base.base
        base = base[['obsnme','obgnme','base','obsval','resid']]
        #base = base.loc[base.obgnme.str.contains('hi|ext'),:]
        # count by obgnme:
        df['layer'] = [lynm]
        df['count'] = base['obgnme'].count()
        df['resid_mean'] = base['resid'].mean()
        df['resid_median'] = base['resid'].median()
        df['resid_rmse'] = np.sqrt(np.mean(base['resid']**2))
        df['resid_mae'] = np.mean(np.abs(base['resid']))
        df['obsmin'] = base['obsval'].min()
        df['obsmax'] = base['obsval'].max()
        df['range'] = df['obsmax'] - df['obsmin']
        master_df = pd.concat([master_df,df])
    master_df.to_csv(os.path.join(o_d,f'{modnm}_residual_stats_wext.csv'),index=False)
 
    
if __name__ == '__main__':
    # model workspace:
    d = os.path.join('master_flow_08_highdim_restrict_bcs_flood_full_final_rch_forward_run_base')
    #d = os.path.join('model_ws','wahp7ly_gwv_sspmp')
    
    table_output = os.path.join(d,'for_documentation','tables')
    if not os.path.exists(table_output):
        os.makedirs(table_output)
    
    stress_period_table(d,table_output)
    write_summary_tables(d, outdir=table_output)
    #write_basic_mf6_info_table(d, outdir=table_output) # by layer counts of model pkgs
    write_avg_thickness_table(d, outdir=table_output)

        
    # boundary conditions tables:
    #river_table(d,table_output)
    #ghb_table(d,table_output)
    #recharge_table_ann_only(d,table_output) # writes average annual recharge by stress period
    #well_table(d,table_output)
    #drains_table(d,table_output)
    
    # pkg stats tables:
    create_K_stats_tables(d)
    create_sto_stats_tables(d)
    recharge_stats_table(d)
    make_ghb_drn_summary_spreadsheets(d)