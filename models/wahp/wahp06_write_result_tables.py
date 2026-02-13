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

import wahp04_process_plot_results as wpp
wpp.set_graph_specifications()
wpp.set_map_specifications()

import wahp02_model_build_conf as whap_build

# Script to build output tables for documentation

def write_summary_tables(d='.', pst_name="wahp7ly.pst", outdir='', noptmax=0, max_fail=2):
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

def write_basic_mf6_info_table(d='.',outdir='',modnm='wahp7ly'):
    
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
        

    
    
def stress_period_table(d='.',outdir='',annual_only=True):
    fpath = os.path.join(d, 'tables', 'annual_stress_period_info.csv')
    # Try to load the file, otherwise generate it
    try:
        df = pd.read_csv(fpath)
    except FileNotFoundError:
        whap_build.stress_period_df_gen(d, 1970, annual_only=annual_only)
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
    
    spdf = pd.read_csv(os.path.join(d,'tables','annual_stress_period_info.csv'))
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
    
def water_table_obs(d='.',outdir='.'):
    # reads in water table observation files from working directory, and writes out the number of observed head values by:
        # - year
        # - month
        # - layer
    print('Building water table observation table...')
 
    
if __name__ == '__main__':
    # model workspace:
    d = os.path.join('master_flow_gwv_sspmp_highdim_noWR_final_wss_reweight_forward_run_base')
    
    table_output = os.path.join(d,'for_documentation','tables')
    if not os.path.exists(table_output):
        os.makedirs(table_output)
    
    stress_period_table(d,table_output)
    write_summary_tables(d, outdir=table_output)
    write_basic_mf6_info_table(d, outdir=table_output) # by layer counts of model pkgs
    
    # boundary conditions tables:
    river_table(d,table_output)
    ghb_table(d,table_output)
    recharge_table_ann_only(d,table_output) # writes average annual recharge by stress period
    #well_table(d,table_output)
    drains_table(d,table_output)