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


import warnings            
warnings.filterwarnings("ignore")

# set fig formats:
import wahp04_process_plot_results as wpp
wpp.set_graph_specifications()
wpp.set_map_specifications()


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


def init_gw_vista_ss_mods(org_d,mwd):
    """initializes the gw_vista_ss_mods directory with the necessary files for the gw_vista_ss_mods model"""
    d = os.path.join('model_ws',mwd)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(org_d,d)
    
    # nest = os.path.join(d,'ss_with_pmp')
    # if os.path.exists(nest):
    #     shutil.rmtree(nest)
    #     shutil.copytree(org_d, nest)
    
    # know there is two copies of mf6 clean in new directory.
    
    # now I need to load gv vista files and take upf convert to npf, take zones, and for the pumping
    # case take the pumping file and get everything in modflow6 and then makes sure things run:
    
    # load ss:
    gws = os.path.join('gwvistas','final_ss_models_ONLY_riv','no_pumping')
    m = flopy.modflow.Modflow.load(
        'gv9.nam', model_ws=gws, check=False, verbose=True
    )
    upw = m.upw
    hk_array = upw.hk.array
    vka_array = upw.vka.array

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(hk_array[3,:,:], cmap='viridis')
    # # colorbar
    # cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical')
    # cbar.set_label('Hydraulic Conductivity (m/s)', rotation=270, labelpad=20)
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(vka_array[3,:,:], cmap='viridis')
    # # colorbar
    # cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical')
    # cbar.set_label('Vert Conductivity (m/s)', rotation=270, labelpad=20)
    
    aniso_array = vka_array / hk_array
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(aniso_array[3,:,:], cmap='viridis')
    # # colorbar
    # cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical')
    # cbar.set_label('Anisotropy (vka/hk)', rotation=270, labelpad=20)
    
    # remove recharge and ghb package as we are ignoring them for now:
    # remove GHB6 and RCH6 lines in-place
    with open(os.path.join(d,"wahp.nam"), "r") as f:
        lines = f.readlines()
    keep = [ln for ln in lines if not ln.lstrip().startswith(("GHB6", "RCH6"))]
    with open(os.path.join(d,"wahp.nam"), "w") as f:
        f.writelines(keep)

    def upf_to_npf(w_d,hk_array,aniso_array):
        """convert upf to npf files for mf6"""
        # npf
        kh_files = [f for f in os.listdir(w_d) if f.startswith('npf_k_')]
        kh_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0].replace('layer','')))
        init_kh = {}
        for file in kh_files:
            kh = np.loadtxt(os.path.join(w_d, file))
            ly = int(file.split('_')[2].split('.')[0].replace('layer',''))
            hk = hk_array[ly-1,:,:]
            np.savetxt(os.path.join(w_d, file), hk, fmt='%.8f')


        aniso_files = [f for f in os.listdir(w_d) if f.startswith('npf_k33')]
        # sort files by layer number:
        aniso_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0].replace('layer','')))
        init_aniso = {}
        for file in aniso_files:
            # read in with numpy
            aniso = np.loadtxt(os.path.join(w_d, file))
            med_ansio = np.median(aniso)    
            ly = int(file.split('_')[2].split('.')[0].replace('layer',''))
            # if any aniso_array is > 1, replace with 1e-3
            if np.any(aniso_array[ly-1,:,:] > 1):
                aniso_array[ly-1,:,:] = np.where(aniso_array[ly-1,:,:] > 1, 1e-3, aniso_array[ly-1,:,:])
            vka = aniso_array[ly-1,:,:]
            print(f"Layer {ly} max aniso: {np.max(aniso_array[ly-1,:,:])}, median: {med_ansio}")
            np.savetxt(os.path.join(w_d, file), vka, fmt='%.8f')
    
    upf_to_npf(d,hk_array,aniso_array)
    
    # # load ss with pumping:
    # gws = os.path.join('gwvistas','final_ss_models_ONLY_riv','pumping')
    # m = flopy.modflow.Modflow.load(
    #     'gv9.nam', model_ws=gws, check=False, verbose=True
    # )
    # upw = m.upw
    # hk_array = upw.hk.array
    # vka_array = upw.vka.array
    
    # upf_to_npf(nest,hk_array,vka_array)
    
    prep_deps(d)
 

    # parm the basse with zones
    # run the bse in forward
    # copy the base to nest, ignore larg output files
    # will need to modify orginal obs pkg so that new targets are tracked
    # copy over just the pumping file and mod the nam run and then collect obs. 
    
    # copy in the model files:
    shutil.copy2(os.path.join('gwvistas','final_ss_models_ONLY_riv','zones.dat'), d)
    shutil.copy2(os.path.join('gwvistas','final_ss_models_ONLY_riv','targets_nopumping.csv'), d)
    shutil.copy2(os.path.join('gwvistas','final_ss_models_ONLY_riv','targets_pumping.csv'), d)
    
    # modify obs package to include the new targets:

    odf = pd.read_csv(os.path.join(d,'wahp.obs_continuous_wahp.ss_head.obs.output.txt'),delim_whitespace=True,header=None)
    odf.columns = ['obsnme','otype','ly','row','col']
    targs_nopmp = pd.read_csv(os.path.join('gwvistas','final_ss_models_ONLY_riv','targets_nopumping.csv'))
    targs_nopmp.columns = targs_nopmp.columns.str.lower()
    targs_pmp = pd.read_csv(os.path.join('gwvistas','final_ss_models_ONLY_riv','targets_pumping.csv'))
    targs_pmp.columns = targs_pmp.columns.str.lower()
    
    # get a dataframe of layer, row, col locations that are not already in odf:
    apnd_df = pd.DataFrame(columns=['obsnme','otype','ly','row','col'])
    for _, row in targs_nopmp.iterrows():
        ly = row['layer']
        row_num = row['row']
        col_num = row['column']
        id = row['name']
        if not ((odf.ly == ly) & (odf.row == row_num) & (odf.col == col_num)).any():
            temp = pd.DataFrame({'obsnme':f'ssh_id:{int(id)}_k:{int(ly)-1}_i:{row_num-1}_j:{int(col_num)-1}',
                                      'otype':'HEAD',
                                      'ly':int(ly),
                                      'row':int(row_num),
                                      'col':int(col_num)}, index=[0])
            apnd_df = pd.concat([apnd_df, temp], ignore_index=True)
                
                
    for _, row in targs_pmp.iterrows():
        ly = row['layer']
        row_num = row['row']
        col_num = row['column']
        id = row['name']
        temp = pd.DataFrame({'obsnme':f'sspmp_id:{int(id)}_k:{int(ly)-1}_i:{row_num-1}_j:{int(col_num)-1}',
                                        'otype':'HEAD',
                                        'ly':int(ly),
                                        'row':int(row_num),
                                        'col':int(col_num)}, index=[0])
        apnd_df = pd.concat([apnd_df, temp], ignore_index=True)
    # append the new targets to the odf:
    odf = pd.concat([odf, apnd_df], ignore_index=True)
    # write the new obs package file:
    odf.to_csv(os.path.join(d,'wahp.obs_continuous_wahp.ss_head.obs.output.txt'), sep=' ', index=False, header=False)
    
    # make copy of fodler:
    od = os.path.join('model_ws',mwd,'org_mws')
    if not os.path.exists(od):
        os.makedirs(od)
    #shutil.copytree(d,od)
    mfiles = ['wahp.oc', 'wahp.obs', 'wahp.nam', 'wahp.tdis']
    # replace the modified files with the original files:
    for f in mfiles:
        shutil.copy2(os.path.join(d, f), os.path.join(od, f))
    # remove .cbb and .hds and lst:
    for f in os.listdir(od):
        if f.endswith('.cbb'):
            os.remove(os.path.join(od,f))
        elif f.endswith('.hds'):
            os.remove(os.path.join(od,f))
        elif f.endswith('.lst'):
            os.remove(os.path.join(od,f))
    

def run_2x_mf6():
    import re
    from datetime import datetime
    
    # run the model with the original files:
    pyemu.os_utils.run(r'mf6')
    
    # run the SS pmp model, but we need to change relvant outputs, so we do not mess up
    # targets later on:
    '''
      1. need to change output file names in .oc contorol file:
      2. need to change name of obs out files in whap.obs
      3. need to change .nam file to add the pumping stress period data file
      4. modify tdis to just be ss period
    '''
    
    # modify output control
    file = "wahp.oc"
    with open(file, "r") as f:
        txt = f.read()
    # insert _sspmp before .hds / .csv / .cbb  (case-insensitive)
    txt = re.sub(r'(\S+?)(\.(?:hds|csv|cbb))', r'\1_sspmp\2', txt, flags=re.IGNORECASE)
    # overwrite with the updated text
    with open(file, "w") as f:
        f.write(txt)

    # modify the wahp.obs file
    # replace every "wahp.ss_head" with "wahp.sspmp_head" in-place
    with open("wahp.obs", "r") as f:
        text = f.read()
    text = text.replace(" wahp.ss_head", " wahp.sspmp_head")
    with open("wahp.obs", "w") as f:
        f.write(text)
    
    fname = "wahp.obs"
    with open(fname, "r") as f:
        lines = f.readlines()
    # keep everything except the three lines that mention wahp.trans_head.obs.output
    keep = [ln for ln in lines if "wahp.trans_head.obs.output" not in ln]
    with open(fname, "w") as f:
        f.writelines(keep)
        
    # build the pumping file and add to .nam file:
    fnm = 'sspmp_stress_period_data_1.txt'
    rows = [
    "4       115        47    -6234.2832 ",
    "4       114        47    -40475.301 ",
    "4       117        48    -59543.594 ",
    "4       111        45    -47391.832 ",
    "4       131        47    -44424.121 ",
    ]
    with open(fnm, "w") as f:
        f.write("\n".join(rows))

    fname = "wahp.sspmp"
    now = datetime.now()
    header = f"# File generated by Flopy version 3.7.0.dev0 on {now:%m/%d/%Y} at {now:%H:%M:%S}.\n"

    content = (
        "BEGIN options\n"
        "  SAVE_FLOWS\n"
        "  AUTO_FLOW_REDUCE       0.10000000\n"
        "END options\n\n"
        "BEGIN dimensions\n"
        "  MAXBOUND  5 \n"
        "END dimensions\n\n"
        "BEGIN period  1\n"
        "    OPEN/CLOSE     'sspmp_stress_period_data_1.txt'\n"
        "END period  1\n"
    )

    with open(fname, "w") as f:
        f.write(header + content)
    print(f"{fname} written.")

    # now modify .nam to include new well pkg:
    fname = "wahp.nam"
    new_line = "  WEL6  wahp.sspmp  sspmp\n"

    with open(fname, "r") as f:
        lines = f.readlines()
    # insert the new line just before "END packages"
    for i, line in enumerate(lines):
        if line.lstrip().startswith("END packages"):
            lines.insert(i, new_line)
            break
    with open(fname, "w") as f:
        f.writelines(lines)
        
    with open(fname, "r") as f:
        lines = f.readlines()

    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("BEGIN options"):
            lines.insert(i + 1, "  LIST  wahp_sspmp.lst\n")
            break                     # done—only insert once
    with open(fname, "w") as f:
        f.writelines(lines)

    # modify tdis:
    # remove lines that begin (after spaces) with 365 or 366
    fname = "wahp.tdis"
    with open(fname, "r") as f:
        lines = f.readlines()

    new_lines = []
    for ln in lines:
        # skip any period-data line starting with 365 or 366
        if ln.lstrip().startswith(("365", "366")):
            continue
        # change “NPER  75” (or any other number) to “NPER  1”
        if ln.lstrip().startswith("NPER"):
            ln = "  NPER  1\n"
        new_lines.append(ln)

    with open(fname, "w") as f:
        f.writelines(new_lines)
        
    # run the model with all chnages:
    pyemu.os_utils.run(r'mf6')
    
    # list of modified files names:
    mfiles = ['wahp.oc', 'wahp.obs', 'wahp.nam', 'wahp.tdis']
    # replace the modified files with the original files:
    for f in mfiles:
        shutil.copy2(os.path.join('org_mws', f), os.path.join('.', f))
    
    # remove wel file:
    wel_file = 'wahp.sspmp'
    if os.path.exists(wel_file):
        os.remove(wel_file)
        os.remove('sspmp_stress_period_data_1.txt')
         

def init_2x_mf6(d):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = run_2x_mf6()
    os.chdir(b_d)
    
    
def process_resistivity_data(mws='.',to_epsg=2265):
    ows = os.path.join('data','processed','resistivity','shps','points')
    if not os.path.exists(ows):
        os.makedirs(ows)
    
    grd = gpd.read_file(os.path.join('..','..','gis','output_shps','wahp','cell_size_660ft_epsg2265.grid.shp'))
        
    flow_dir = os.path.join(mws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=flow_dir, exe_name='mf6',load_only=['dis'])
    m = sim.get_model(f'{modnm}')
    nlay = m.dis.nlay.data
    id = m.dis.idomain.array
    mtop = m.dis.top.array
    mbots = m.dis.botm.array
    
    
    df = pd.read_csv(os.path.join('data','raw','Wahpeton_SCI_Inverted.xyz'))
    df.columns = df.columns.str.lower()
    nlys = 29 # there are 29 resitivity layers in inversion from AGF
    epsg = 2266 # State Plane North Dakota South used in AGF study
    # make shapefile for each resitivity layer:
    for ly in np.arange(0,nlys):
        x = df.x_ft.values
        y = df.y_ft.values
        rho = df[f'rho[{ly}]'].values
        rho_std = df[f'rho_std[{ly}]'].values
        dem = df[f'dem_ft'].values
        top = df[f'dep_top_ft[{ly}]'].values
        bot = df[f'dep_bot_ft[{ly}]'].values
        midpoint = (top + bot) / 2.0
        midpoint_elev = dem + midpoint # note midpoints are negative values, so add to dem to get elevation
        thk  = df[f'thk_ft[{ly}]'].values
        doi = df['doi_upper_ft'].values # their conservative doi estimate
        
        temp = pd.DataFrame({'x':x,'y':y,'rho':rho,'rho_std':rho_std,
                             'dem':dem,'top':top,'bot':bot,
                             'midpoint_dpth':midpoint, 'midpoint_elev':midpoint_elev,
                             'thick':thk,'doi':doi})
        gdf = gpd.GeoDataFrame(temp, geometry=gpd.points_from_xy(x, y), crs=f'epsg:{epsg}')
        # convert to new epsg:
        gdf = gdf.to_crs(epsg=to_epsg)
        # get x and y of new epsg:
        gdf.x = gdf.geometry.x
        gdf.y = gdf.geometry.y
        
        # saptial join with grd:
        gdf = gpd.sjoin(gdf, grd, how='left')
        # get where node is not nan:
        gdf = gdf[gdf['node'].notna()]


        # make node, i, j, row, col columns int type:
        gdf['node'] = gdf['node'].astype(int)
        gdf['i'] = gdf['i'].astype(int)
        gdf['j'] = gdf['j'].astype(int)
        gdf['row'] = gdf['row'].astype(int)
        gdf['col'] = gdf['col'].astype(int)
        
        # add model top and bots:
        gdf['mtop'] = mtop[gdf['i'],gdf['j']]
        mbot_ij = mbots[:,gdf['i'],gdf['j']]
        cols = ['bot_{0}'.format(k) for k in range(nlay)]
        for c in cols:
            gdf[c] = mbot_ij[int(c.split('_')[1])]
        
        gdf = gdf[['x', 'y', 'row', 'col', 'i', 'j', 'dem', 'top', 'bot', 'midpoint_dpth',
                    'midpoint_elev', 'mtop', 'bot_0', 'bot_1', 'bot_2', 'bot_3',
                    'rho', 'rho_std',  'thick', 'doi', 'node',
                    'geometry']]
        
        # filter out 0 rho vals:
        gdf = gdf[gdf['rho'] > 0]     
        gdf.to_file(os.path.join(ows,f'layer_{ly}.shp'))
    
    records = []

    for ly in range(nlys):  # 29 resistivity layers
        gdf = gpd.read_file(os.path.join(ows, f'layer_{ly}.shp'))

        for _, row in gdf.iterrows():
            i = int(row['i'])
            j = int(row['j'])
            midpoint = row['midpoint_e']
            rho = row['rho']
            rho_std = row['rho_std']

            # model top and all bot elevations at this cell
            top_elev = mtop[i, j]
            bot_elevs = mbots[:, i, j]
            bottom_of_model = bot_elevs[-1]

            # 1. above the top of the model
            if midpoint > top_elev:
                records.append({
                    'resistivity_layer':   ly,
                    'model_layer':         1,           
                    'model_top_elev':      top_elev,
                    'model_bot_elev':      bot_elevs[0], # top of layer 1 bottom
                    'i':                   i,
                    'j':                   j,
                    'rho':                 rho,
                    'rho_std':             rho_std,
                    'midpoint':            midpoint
                })
                continue
            # 2. below the bottom of the model
            if midpoint < bottom_of_model:
                continue
            # 3: somewhere within the model column
            for k in range(nlay):
                layer_top = top_elev if k == 0 else bot_elevs[k-1]
                layer_bot = bot_elevs[k]

                if layer_top >= midpoint >= layer_bot:
                    records.append({
                        'resistivity_layer':   ly,
                        'model_layer':         k + 1,       # 1‑based layer index
                        'model_top_elev':      layer_top,
                        'model_bot_elev':      layer_bot,
                        'i':                   i,
                        'j':                   j,
                        'rho':                 rho,
                        'rho_std':             rho_std,
                        'midpoint':            midpoint
                    })
                    break  # stop once we've found the right model layer

    df_model = pd.DataFrame(records)

    # Summarize per model layer
    summary = df_model.groupby(['model_layer','i','j']).agg({
        'rho': ['mean','median', 'min', 'max'],
        'rho_std': ['mean','median', 'min', 'max'],
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['k','i','j','rho_mean','rho_median', 'rho_min', 'rho_max',
                    'rho_std_mean','rho_std_median', 'rho_std_min', 'rho_std_max']
    
    # add geometry column:
    summary = summary.merge(grd[['i','j','geometry']], on=['i','j'], how='left')
    gdf = gpd.GeoDataFrame(summary, geometry='geometry', crs=f'epsg:{to_epsg}')
    
    outws = os.path.join('data','processed','resistivity','shps','model_layers')
    if not os.path.exists(outws):
        os.makedirs(outws)
    
    unq_ly = gdf['k'].unique()
    for ly in unq_ly:
        lgdf = gdf[gdf['k'] == ly]
        lgdf.to_file(os.path.join(outws,f'layer_{ly}_res_stats.shp'))         
    
                       
def incorp_res_data():
    df = pd.read_csv(os.path.join('data','raw','Wahpeton_SCI_Inverted.xyz'))
    
    from sklearn.decomposition import PCA

    def fit_local_variogram_with_pca(grid, row, col, win_size=21, use_variogram=True):
        half = win_size // 2
        subgrid = grid[row-half:row+half+1, col-half:col+half+1]

        # Get coordinates of valid values
        X, Y = np.meshgrid(np.arange(subgrid.shape[1]), np.arange(subgrid.shape[0]))
        vals = subgrid.flatten()
        coords = np.column_stack((X.flatten(), Y.flatten()))
        
        # Filter out NaNs
        mask = ~np.isnan(vals)
        coords, vals = coords[mask], vals[mask]
        
        if len(vals) < 10:
            return None  # Not enough data to estimate

        # Center coordinates before PCA
        coords_centered = coords - coords.mean(axis=0)

        # Run PCA
        pca = PCA(n_components=2)
        pca.fit(coords_centered)

        # PCA-based anisotropy
        eigenvalues = pca.explained_variance_
        anisotropy = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else np.nan

        # Bearing (angle of 1st principal component)
        pc1 = pca.components_[0]
        bearing_rad = np.arctan2(pc1[1], pc1[0])  # Y, X
        bearing_deg = np.degrees(bearing_rad) % 180  # range 0–180°

        results = {
            'anisotropy_pca': anisotropy,
            'bearing_pca': bearing_deg,
        }


def riv_flux_process():
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis','oc'])
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model('wahp')
    
    df = pd.read_csv(
        "otriv_stress_period_data_1.txt",
        header=None,
        names=["lay", "row", "col", "stage", "cond","botm"],
        delim_whitespace=True,
    )
    
    df.sort_values(by=["row","col"],inplace=True)
    
    cbc = flopy.utils.CellBudgetFile(m.oc.budget_filerecord.array[0][0], precision="double")
    times=cbc.times
    dates=pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    flow_riv=cbc.get_data(full3D=True,text='RIV')
    
    riv_flux= pd.DataFrame(times, columns=["time"])
    for idx,t in enumerate(times):
        for r, c in zip(df.row, df.col):
            riv_flux.loc[idx,f"{r}_{c}"]=flow_riv[idx][0,r-1,c-1]
    
    riv_flux.index = pd.to_datetime(start_datetime) + pd.to_timedelta(riv_flux.time.values,unit='d') 
    riv_flux.index.name = "datetime"
    riv_flux.columns = riv_flux.columns.map(lambda x: x.lower().replace(" ", "_"))
    riv_flux=riv_flux[riv_flux.columns.values[1:]]
    riv_flux.to_csv("riv_flx_south.csv")
    
    return riv_flux

    
def init_riv_process(d):
    b_d = os.getcwd()
    os.chdir(d)
    dfs = riv_flux_process()
    os.chdir(b_d)
    return dfs

    
def budget_process():
    df = pd.read_csv("budget.csv",index_col=0)
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model('wahp')
    
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


def head_targets_process():
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=".",load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    m = sim.get_model('wahp')
     
    df = pd.read_csv("wahp.ss_head.obs.output",index_col=0)
    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    df.columns = [c.lower().replace(".","-") for c in df.columns]
    df.index.name = "datetime"
    dfs = [df]
    
    df.to_csv("wahp.ss_head.obs.output.csv")
    
    df = pd.read_csv("wahp.trans_head.obs.output",index_col=0)
    df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    df.columns = [c.lower().replace(".","-") for c in df.columns]
    df.index.name = "datetime"
    
    dfs.append(df)
    
    df.to_csv('wahp.trans_head.obs.output.csv')

    # df = pd.read_csv("wahp.sspmp_head.obs.output",index_col=0)
    # df.index = pd.to_datetime(start_datetime) + pd.to_timedelta(df.index.values,unit='d')
    # df.columns = [c.lower().replace(".","-") for c in df.columns]
    # df.index.name = "datetime"
    
    # df.to_csv('wahp.sspmp_head.obs.output.csv')
    
    dfs.append(df)
    
    return dfs


def init_head_targets_process(d):
    b_d = os.getcwd()
    os.chdir(d)
    ssdf =head_targets_process()
    os.chdir(b_d)
    return ssdf


def process_listbudget_obs(mod_name='wahp'):
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


def init_listbudget_obs(d='.', mod_name='wahp'):
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


def process_mfinput_obs(mod_name='wahp'):
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


def init_mfinput_obs(template_ws='template', mod_name='wahp'):
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


def init_regional_ghbs(d='.', mod_name='wahp'):
    sim = flopy.mf6.MFSimulation.load(sim_ws=d,load_only=['dis','tdis'])
    start_datetime = sim.tdis.start_date_time.array
    nper = sim.tdis.nper.data
    m = sim.get_model('wahp')
    nrow = m.modelgrid.nrow
    ncol = m.modelgrid.ncol
    delr = m.modelgrid.delr[0]
    id = m.dis.idomain.array
    
    # now the ghb
    df = pd.DataFrame(
        columns=["sp", "x", "y", "p1", "p2", "p3", "p4"],
        index=["ghb" + str(x) for x in np.arange(0, nper)],
        )
    df.index.set_names(["ghb_nme"], inplace=True)
    (
        df.loc[:, "p1"],
        df.loc[:, "p2"],
        df.loc[:, "p3"],
        df.loc[:, "p4"],
        df.loc[:, "x"],
        df.loc[:, "y"],
        df.loc[:, "sp"],
    ) = (0.0, 0.0, 1.0, 1.0, 0, 0, np.arange(0, nper))
    df.loc[:, "y"] = (
        pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"][0:]), unit="D")
    ).days.values

    # get ghb file sin the model working directory:
    ghb_files = [f for f in os.listdir(d) if f.startswith('ghb_stress_period_data_') and f.endswith('.txt')]
    ghb_file = ghb_files[0]
    ghb_df = pd.read_csv(os.path.join(d, ghb_file), header=None, delim_whitespace=True)
    # get layers with ghbs:
    ghb_layers = ghb_df[0].unique()
    
    for lay in ghb_layers:
        df.to_csv(os.path.join(d, f"pp_ghb_lay{lay}.csv"))

    pp_ghb_files = [f for f in os.listdir(d) if "pp_ghb_" in f and f.endswith(".csv")]
    
    # get corner PPs:
    id_fk = id[0,:,:].copy()* 0
    id_fk[0,0] = 1
    id_fk[-1,0] = 1
    id_fk[0,-1] = 1
    id_fk[-1,-1] = 1

    ppdf = ppu.get_2d_pp_info_structured_grid(
        1,
        os.path.join(d, "wahp.dis.grb"),
        array_dict={"zone": id_fk},
    )
    #ppdf.head()
    
    # make zone_arr where we will interp:
    zone_arr = id[0,:,:].copy()* 0
    # set edge cells to 1, so 0 and -1 col/row
    zone_arr[0,:] = 1
    zone_arr[-1,:] = 1
    zone_arr[:,0] = 1
    zone_arr[:,-1] = 1
    
    easting = m.modelgrid.xcellcenters
    northing = m.modelgrid.ycellcenters
    easting = easting[: nrow * ncol]
    northing = northing[: nrow * ncol]

    fac_file = os.path.join(d, "factors_ghb.bin")

    lib = PestUtilsLib()
    
    zone_pp = np.ones_like(ppdf.x.values, dtype=int)
    
    ipts = lib.calc_kriging_factors_2d(
        ppdf.x.values,
        ppdf.y.values,
        zone_pp,
        easting.flatten(),
        northing.flatten(),
        zone_arr.flatten(),
        "exp",
        "ordinary",
        [delr * len(northing)/2] * len(northing.flatten()), # vario range
        1,  # anisotropy,
        0,  # bearing
        1e30, # search distance
        50, # max_pts
        1, # min_pts
        fac_file,
        "binary",
    )
    
    
    return pp_ghb_files


def interp_engine_ghb():
    fac_file = os.path.join(".", "factors_ghb.bin")
    sim = flopy.mf6.MFSimulation.load(sim_ws=".", load_only=["dis"])
    gwf = sim.get_model("wahp")
    botm = gwf.dis.botm.array
    easting = gwf.modelgrid.xcellcenters
    northing = gwf.modelgrid.ycellcenters
    nrow = gwf.modelgrid.nrow
    ncol = gwf.modelgrid.ncol
    
    # get ghb_files in dir:
    pp_ghb_files = [f for f in os.listdir(".") if f.startswith("pp_ghb_lay") and f.endswith(".csv")]
    
    lay3 = pd.read_csv(pp_ghb_files[2])
    lay5 = pd.read_csv(pp_ghb_files[3])
    lay6 = pd.read_csv(pp_ghb_files[4])
    
    for sp in lay3.sp:
        shutil.copy2(
            os.path.join("org", f"ghb_stress_period_data_{sp+1}.txt"),
            os.path.join(".", f"ghb_stress_period_data_{sp+1}.txt"),
        )
        
        df = pd.read_csv(
            f"ghb_stress_period_data_{sp+1}.txt",
            header=None,
            names=["l", "row", "col", "stage", "cond"],
            delim_whitespace=True,
        )

        # layer 1 - (sub layers 1-3)
        [p1, p2, p3, p4] = list(
            lay3.loc[lay3.sp == sp, ["p1", "p2", "p3", "p4"]].values[0]
        )
        
        p4 = p4 * p1
        p3 = p3 * p2
        from pypestutils.pestutilslib import PestUtilsLib

        lib = PestUtilsLib()
        result = lib.krige_using_file(
            fac_file,
            "binary",
            easting.flatten().shape[0],
            "ordinary",
            "none",
            np.array([p1, p2, p3, p4]),
            np.zeros(easting.flatten().shape[0]),
            0,
            ).copy()
        rr = result["targval"].copy()
        rr = rr.reshape(nrow, ncol)
        # rrr = rr.reshape(nrow, ncol)
        # fig, axes = plt.subplots(1, 1)
        # ax = axes
        # ax.set_aspect("equal")
        # ax.set_title("pp interpolated array")
        # ax.pcolormesh(easting, northing, rrr) 
        # # add colorbar
        # cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        # cbar.set_label("pp interpolated array values")
        
        new_ghb = pd.DataFrame(columns=df.columns)
        counter = 0
        for idx, row in df.iterrows():
            lay = int(row["l"])
            if lay <= 3:
                r = int(row['row'])
                c = int(row['col'])
                k = int(row['l'])-1
                i = r - 1
                j = c - 1
                cond = row["cond"]
                stage = np.round(row["stage"] + rr[i,j], 6)
                if stage > botm[k, i, j]:
                    new_ghb.loc[counter] = [lay, r, c, stage, cond]
                    counter += 1
                
        # now lay 5 deep clay
        del rr, lib, PestUtilsLib
        [p1, p2, p3, p4] = list(
            lay5.loc[lay5.sp == sp, ["p1", "p2", "p3", "p4"]].values[0]
        )
        p4 = p4 * p1
        p3 = p3 * p2
        from pypestutils.pestutilslib import PestUtilsLib

        lib = PestUtilsLib()
        result = lib.krige_using_file(
            fac_file,
            "binary",
            easting.flatten().shape[0],
            "ordinary",
            "none",
            np.array([p1, p2, p3, p4]),
            np.zeros(easting.flatten().shape[0]),
            0,
            ).copy()
        rr = result["targval"].copy()
        rr = rr.reshape(nrow, ncol)
        for idx, row in df.iterrows():
            lay = int(row["l"])
            if lay == 5:
                r = int(row['row'])
                c = int(row['col'])
                k = int(row['l'])-1
                i = r - 1
                j = c - 1
                cond = row["cond"]
                stage = np.round(row["stage"] + rr[i,j], 6)
                if stage > botm[k, i, j]:
                    new_ghb.loc[counter] = [lay, r, c, stage, cond]
                    counter += 1
        
        # now lay 6
        del rr, lib, PestUtilsLib
        [p1, p2, p3, p4] = list(
            lay6.loc[lay6.sp == sp, ["p1", "p2", "p3", "p4"]].values[0]
        )
        p4 = p4 * p1
        p3 = p3 * p2
        from pypestutils.pestutilslib import PestUtilsLib
        lib = PestUtilsLib()
        result = lib.krige_using_file(
            fac_file,
            "binary",
            easting.flatten().shape[0],
            "ordinary",
            "none",
            np.array([p1, p2, p3, p4]),
            np.zeros(easting.flatten().shape[0]),
            0,
            ).copy()
        rr = result["targval"].copy()
        rr = rr.reshape(nrow, ncol)
        for idx, row in df.iterrows():
            lay = int(row["l"])
            if lay == 6:
                r = int(row['row'])
                c = int(row['col'])
                k = int(row['l'])-1
                i = r - 1
                j = c - 1
                cond = row["cond"]
                stage = np.round(row["stage"] + rr[i,j], 6)
                if stage > botm[k, i, j]:
                    new_ghb.loc[counter] = [lay, r, c, stage, cond]
                    counter += 1        
        new_ghb['l'] = new_ghb['l'].astype(int)
        new_ghb['row'] = new_ghb['row'].astype(int)
        new_ghb['col'] = new_ghb['col'].astype(int)        
        new_ghb.to_csv(
            f"ghb_stress_period_data_{sp+1}.txt",
            header=False,
            index=False,
            sep=" ",
            lineterminator="\n",
        )


def riv_drn_bot_chk(model_ws='.'):
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, exe_name="mf6",load_only=['dis'])
    m = sim.get_model()
    botm = m.dis.botm.array
    
    riv_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"riv_stress")]
    otriv_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"otriv_stress")]
    drn_stress_files = [os.path.join(model_ws, f) for f in os.listdir(model_ws) if f.startswith(f"drn_stress")]
    for drn_file in drn_stress_files:
        df = pd.read_csv(drn_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(drn_file, sep=' ', index=False, header=False)
    for riv_file in riv_stress_files:
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
    for otriv_file in otriv_stress_files:
        df = pd.read_csv(otriv_file, delim_whitespace=True,header=None)
        df.columns = ['ly','row','col','stage','cond','rbot']
        bot = botm[df['ly'].values-1,df['row'].values-1,df['col'].values-1]
        df['mbot'] = bot
        df['diff'] = df['stage'] - df['mbot']
        # case where stage is 
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
        df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1
        # case where rbot is below model bottom:
        df['diff'] = df['rbot'] - df['mbot']
        df.loc[df['diff'] < 0,'rbot'] = df['mbot'] + 0.1
        df.loc[df['diff'] < 0,'stage'] = df['mbot'] + 3.0
        # case where stage is below rbot:
        df['diff'] = df['stage'] - df['rbot']
        df.loc[df['diff'] < 0,'stage'] = df['rbot'] + 0.1
        df = df.drop(columns=['mbot','diff'])
        df.to_csv(otriv_file, sep=' ', index=False, header=False)


def setup_pstpp(org_d,modnm,run_tag,template,flex_con=False,num_reals=96):
    
    assert os.path.exists(org_d)
    
    # make the 4-PP regional ghb field:
    #pp_ghb_files = init_regional_ghbs(d=org_d, mod_name='wahp')
    
    temp_d = org_d + '_temp'
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(org_d,temp_d)
    
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
                         zero_based=False, start_datetime=start_datetime)
    

    # copy over kobs data an dor resitivity data targets if we go this route:
    #shutil.copy2(os.path.join('data','kobs.csv'),os.path.join(template_d,'kobs.csv'))
    
    # ------------------------------------    
    # load in geostats parms from run csv:
    # ------------------------------------
    
    k_files = [f for f in os.listdir(template) if 'k_' in f and f.endswith('.txt')]
    k_files.sort()

    k33_files = [f for f in os.listdir(template) if "k33" in f]
    k33_files.sort()

    ss_files = [f for f in os.listdir(template) if 'sto_ss_' in f and f.endswith('.txt')]
    ss_files.sort()

    sy_files = [f for f in os.listdir(template) if 'sto_sy_' in f and f.endswith('.txt')]
    sy_files.sort()

    # load in par bounds:
    par = pd.read_csv(os.path.join('run_inputs',f'{modnm}{run_tag}',f'{modnm}_parm_controls.csv'))
    
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

    stacked_files = [k_files,k33_files,ss_files,sy_files]
    stacked_ubnds = [k_ubounds,k33_ubounds,ss_ubounds,sy_ubounds]

    # if using zone array load here:
    zone = pd.read_csv(os.path.join(temp_d,'zones.dat'),delim_whitespace=True,header=None)
    zone.columns = ['row','col','ly','zone']
    zon_arr = id.copy()*0
    for ly,r,c,z in zip(zone.ly,zone.row,zone.col,zone.zone):
        if z == 2 and ly<=3:
           zon_arr[ly-1,r-1,c-1] = z
        elif z == 3 and ly==3:
            zon_arr[ly-1,r-1,c-1] = z
        elif z == 4 and ly==5:
            zon_arr[ly-1,r-1,c-1] = z
        elif z == 5 and ly==5:
            zon_arr[ly-1,r-1,c-1] = z
        elif z == 7 and ly==6:
            zon_arr[ly-1,r-1,c-1] = z
        elif z == 8 and ly == 5:
            zon_arr[ly-1,r-1,c-1] = z
    # no zones in wbv so just replace with idomain:
    zon_arr[3,:,:] = np.where(id[3,:,:]>0, 10, id[3,:,:])
    zon_arr[-1,:,:] = np.where(id[-1,:,:]>0, 11, id[-1,:,:])
    
    # # # imshow for each lay:
    # for k in range(nlay):
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(zon_arr[k,:,:], cmap='viridis')
    #     plt.colorbar(label='Zone')
    #     plt.title(f'Zone Array for Layer {k+1}')
    #     plt.xlabel('Column Index')
    #     plt.ylabel('Row Index')

    
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
                #bnds_grd = k33_bounds_grd[k]
                pf.add_parameters(f, par_type='zone', upper_bound=bnds_cn[1], lower_bound=bnds_cn[0],
                                    ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base,
                                    pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=zon_arr[k])

            elif 'k' in f and 'k33' not in f:
                print(f'FILENAME IS {k}')
                bnds_cn = k_bounds_cn[k]
                pf.add_parameters(f, par_type='zone', upper_bound=bnds_cn[1], lower_bound=bnds_cn[0],
                                    ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base,
                                    pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=zon_arr[k])

            elif 'ss' in f:
                bnds_cn = ss_bounds_cn[k]
                pf.add_parameters(f, par_type='zone', upper_bound=bnds_cn[1], lower_bound=bnds_cn[0],
                                    ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base,
                                    pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k])
                
            elif 'sy' in f:
                bnds_cn = sy_bounds_cn[k]
                pf.add_parameters(f, par_type='zone', upper_bound=bnds_cn[1], lower_bound=bnds_cn[0],
                                    ult_ubound=ubnds[1], ult_lbound=ubnds[0], par_name_base='cn-' + par_name_base,
                                    pargp='cn-' + par_name_base + str(k).zfill(3), zone_array=id[k])
 
            pf.add_observations(f,obsgp=par_name_base + str(k).zfill(3),prefix=par_name_base + str(k).zfill(3))             
         
    
    if flex_con:  
        pf.mod_py_cmds.append("print('model')")


    else:
       
        # --- riv paramterization ---
        
        # for pilot points we want to have the defined over a buffer around rivers rather than placed uniformly
        # across grid, this is a thought for now, but we will need to see how this works out in the end
        
        rivcond_cn = par.loc[par.parm=='rivcond_cn']
        rivstg_cn = par.loc[par.parm=='rivstg_cn']
        riv_files = [f for f in os.listdir(template) if f.startswith('riv_') and f.endswith('.txt')]
        riv_files = sorted(riv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(riv_files) == nper

        pf.add_parameters(riv_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[4], pargp='rivcond-cn',
                            par_name_base='rivcond-cn',
                            upper_bound=rivcond_cn['ubound'].values[0],
                            lower_bound=rivcond_cn['lbound'].values[0],
                            mfile_skip=0)

        pf.add_parameters(riv_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='rivstg-cn',
                            par_name_base='rivstg-cn', par_style="a",
                            upper_bound=rivstg_cn['ubound'].values[0],
                            lower_bound=rivstg_cn['lbound'].values[0],
                            mfile_skip=0)

        otcond_cn = par.loc[par.parm=='otcond_cn']
        otstg_cn = par.loc[par.parm=='otstg_cn']
        otter_files = [f for f in os.listdir(template) if f.startswith('otriv') and f.endswith('.txt')]
        otter_files = sorted(otter_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(otter_files) == nper
        
        pf.add_parameters(otter_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[4], pargp='otcond-cn',
                            par_name_base='otcond-cn',
                            upper_bound=otcond_cn['ubound'].values[0],
                            lower_bound=otcond_cn['lbound'].values[0],
                            mfile_skip=0)
        pf.add_parameters(otter_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='otstg-cn',
                            par_name_base='otstg-cn', par_style="a",
                            upper_bound=otstg_cn['ubound'].values[0],
                            lower_bound=otstg_cn['lbound'].values[0],
                            mfile_skip=0)
        
    
        drncond_cn = par.loc[par.parm=='drncond_cn']
        drnstg_cn = par.loc[par.parm=='drnstg_cn']
        drn_files = [f for f in os.listdir(template) if f.startswith('drn_') and f.endswith('.txt')]
        drn_files = sorted(drn_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        assert len(drn_files) == nper
        
        pf.add_parameters(filenames=drn_files, par_type='constant',
                            par_name_base='drncond-cn', 
                            pargp='drncond-cn', index_cols=[0, 1, 2], use_cols=[4],
                            upper_bound=drncond_cn['ubound'].values[0], lower_bound=drncond_cn['lbound'].values[0],
                            mfile_skip=0)
        
        pf.add_parameters(drn_files, par_type='constant', index_cols=[0, 1, 2], use_cols=[3], pargp='drnstg-cn',
                            par_name_base='drnstg-cn', par_style="a",
                            upper_bound=drnstg_cn['ubound'].values[0],
                            lower_bound=drnstg_cn['lbound'].values[0],
                            mfile_skip=0)                          
                
        # stress period mults correlated in time, constant in space
        # domestic water use
        # wel_packages = [f for f in os.listdir(template_d) if f.lower().endswith('.dom')]
        # for wel_package in wel_packages:
        #     wel_package = wel_package.replace('.','_')
        #     list_files = [f for f in os.listdir(template_d) if f.startswith(wel_package) and f.endswith('.txt')]
        #     list_files.sort()
        #     for list_file in list_files:
        #         sp = int(list_file.split('.')[0].split('_')[-1])
        #         kper = int(sp) - 1
        #         # add spatially constant, but temporally correlated wel flux pars
        #         pf.add_parameters(filenames=list_file, par_type='constant',
        #                           par_name_base='twel_mlt_dom_{0:04d}'.format(sp),
        #                           pargp='twel_mlt_dom', index_cols=[0, 1, 2], use_cols=[3],
        #                           upper_bound=1.1, lower_bound=0.9, initial_value=1.0, datetime=dts[kper], geostruct=dom_temporal_gs)
        
            # grid scale mults correlated in space, constant in time
            #pf.add_parameters(filenames=list_files, par_type='grid',
            #         par_name_base='dom_grid',
            #         pargp='dom_grid', index_cols=[0, 1, 2], use_cols=[3],
            #        upper_bound=1.33, lower_bound=0.66, initial_value=1.0)
            

        # irrigation water use
        # wel_packages = [f for f in os.listdir(template_d) if f.lower().endswith('.irr')]
        # for wel_package in wel_packages:
        #     wel_package = wel_package.replace('.','_')
        #     list_files = [f for f in os.listdir(template_d) if f.startswith(wel_package) and f.endswith('.txt')]
        #     list_files.sort()
        #     for list_file in list_files:
        #         sp = int(list_file.split('.')[0].split('_')[-1])
        #         kper = sp - 1
        #         # add spatially constant, but temporally correlated wel flux pars
        #         pf.add_parameters(filenames=list_file, par_type='constant',
        #                           par_name_base='twel_mlt_irr_{0:04d}'.format(sp),
        #                           pargp='twel_mlt_irr', index_cols=[0, 1, 2], use_cols=[3],
        #                           upper_bound=1.001, lower_bound=0.1, initial_value=1.0, datetime=dts[kper], geostruct=dom_temporal_gs)

        #     #add temporally indep and spatially uncorrelated wel flux pars
        #     pf.add_parameters(filenames=list_files, par_type='grid',
        #                      par_name_base='irr_grid',
        #                      pargp='irr_grid', index_cols=[0, 1, 2], use_cols=[3],
        #                      upper_bound=1.001, lower_bound=0.1, initial_value=1.0)


    
                    
        # ghb_files = [f for f in os.listdir(template) if f.startswith('ghb_') and f.endswith('.dat')]
        # ghb_files.sort()

        # ghb_edge_files = [f for f in ghb_files if 'edge' in f]
        # assert len(ghb_edge_files) == nper


        # # Create arrays for ghbs edge 
        # pf.add_parameters(filenames=ghb_edge_files, par_type="grid",
        #                   par_name_base='ghbstage_edge_' + "gr",
        #                   pargp='ghbstage_edge_' + "gr", index_cols=[0, 1, 2],
        #                   upper_bound=100, lower_bound=-100,initial_value = 0.0,
        #                   geostruct=gr_edg_ghb, use_cols=[3],
        #                   par_style="a", mfile_skip = 1,
        #                   transform="none")

        # pf.add_parameters(filenames=ghb_edge_files, par_type="grid",
        #                   par_name_base='ghbcond_edge_gr',
        #                   pargp='ghbcond_edge_gr',
        #                   index_cols=[0, 1, 2], use_cols=[4],
        #                   upper_bound=10.0, lower_bound=0.1,
        #                   ult_ubound=100, ult_lbound=0.00001,
        #                   geostruct=gr_edg_ghb, mfile_skip = 1,) #edge conductance so low

        
        # temporal_v = pyemu.geostats.ExpVario(contribution=1.0, a=365 * 3)
        # temporal_gs_ghb = pyemu.geostats.GeoStruct(variograms=temporal_v, transform="none")
        # pp_ghb_files = pp_ghb_files[2:] # layers 1 and 2 not needed
        # for file in pp_ghb_files:
        #     base = file.replace("_", "").split(".")[0]
        #     df = pf.add_parameters(
        #         file,
        #         par_type="grid",
        #         index_cols={"ghb_nme": "ghb_nme", "x": "x", "y": "y"},
        #         use_cols=["p1", "p2", "p3", "p4"],
        #         par_name_base=[base + "p1", base + "p2", base + "p3", base + "p4"],
        #         pargp=[base + "p1", base + "p2", base + "p3", base + "p4"],
        #         upper_bound=[15, 10, 1, 1],
        #         lower_bound=[-75, -85, 0, 0],
        #         par_style="d",
        #         geostruct=temporal_gs_ghb,
        #         transform="none",
        #     )  

        # hard coded the df return order
        dfs = init_budget_process(template)
        
        pf.add_observations('budget.csv',index_cols=['datetime'],use_cols=dfs[0].columns.to_list(),obsgp='bud',ofile_sep=',',prefix='bud')
        
        #adding riv obs
        dfs = init_riv_process(template)
        #pf.add_observations("riv_flx_south.csv", index_cols=["datetime"], use_cols=dfs.columns.values.tolist(), ofile_sep=",",
        #                    obsgp=["greater_rivflow"]*len(dfs.columns.values.tolist()), prefix="g_rivflow")
                
        # pf.add_observations('wahp.riv.obs.output.csv', index_cols=['datetime'], use_cols=dfs[2].columns.to_list(), obsgp='riv', ofile_sep=',',
        #                     prefix='riv')
        
        # pf.add_observations('wahp.wel.obs.output.csv', index_cols=['datetime'], use_cols=dfs[3].columns.to_list(), obsgp='wel', ofile_sep=',',
        #                     prefix='wel')
        # Listbudget Obs
        # process model output
        flx, cum = init_listbudget_obs(template, 'wahp')
        
        pf.mod_sys_cmds.append('mf6')
        
        # add post process function to forward run script
        pf.add_py_function('wahp03_setup_pst_cond.py', 'process_listbudget_obs()', is_pre_cmd=None)
        # add call to processing script to pst forward run
        pf.post_py_cmds.append("process_listbudget_obs('{0}')".format('wahp'))
        
        # init sspmp run:
        #init_2x_mf6(template)
        
        hdf = init_head_targets_process(template)
        pf.add_observations('wahp.ss_head.obs.output.csv', index_cols=['datetime'], use_cols=hdf[0].columns.to_list(), obsgp='sshds', ofile_sep=',',
                             prefix='sshds')

        pf.add_observations('wahp.trans_head.obs.output.csv', index_cols=['datetime'], use_cols=hdf[1].columns.to_list(), obsgp='transhds', ofile_sep=',',
                             prefix='transhds')
        
        #pf.add_observations('wahp.sspmp_head.obs.output.csv', index_cols=['datetime'], use_cols=hdf[2].columns.to_list(), obsgp='ssphds', ofile_sep=',',
        #                     prefix='sspmphds')
        
        
        #pf.post_py_cmds.append('head_targets_process()')

        # import flopy as part of the forward run process
        pf.extra_py_imports.append('flopy')    
        pf.extra_py_imports.append("shutil")  
        pf.extra_py_imports.append("pathlib")  
        
        
        #pf.mod_sys_cmds.append('mf6')
        #pf.add_py_function('wahp03_setup_pst_sspmp.py', 'interp_engine_ghb()', is_pre_cmd=True)
        pf.post_py_cmds.append('head_targets_process()')
        pf.post_py_cmds.append('budget_process()')
        #pf.post_py_cmds.append('riv_flux_process()')
        
        pf.add_py_function('wahp03_setup_pst_cond.py','budget_process()',is_pre_cmd=None)
        #pf.add_py_function('wahp03_setup_pst_sspmp.py','riv_flux_process()',is_pre_cmd=None)
        pf.add_py_function('wahp03_setup_pst_cond.py', 'head_targets_process()', is_pre_cmd=None)
        
        pf.add_py_function('wahp03_setup_pst_cond.py', 'riv_drn_bot_chk()', is_pre_cmd=True)
        # add call to processing script to pst forward run
    
        # add obs via PstFrom
        #ignore_cols = ['datetime', 'in-out', 'total', 'wel-in']
        #cols = [c for c in flx.columns if c not in ignore_cols]
        #pf.add_observations('listbudget_flx_obs.csv', insfile='listbudget_flx_obs.csv.ins',
        #                            index_cols=['datetime'], use_cols=cols, prefix='flx')
        # ^^^ RH add this back in once pumping is added
        
        

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
    df = init_mfinput_obs(template, 'wahp')

    # add post process function to forward run script
    pf.add_py_function('wahp03_setup_pst_cond.py', 'process_mfinput_obs()', is_pre_cmd=None)
    # add call to processing script to pst forward run
    pf.post_py_cmds.append("process_mfinput_obs('{0}')".format('wahp'))
    
    #pf.add_py_function('wahp03_setup_pst_sspmp.py','run_2x_mf6()',is_pre_cmd=True)
    
    # add obs via PstFrom
    cols = ['upper_bound', 'lower_bound', 'min', 'qnt25', 'qnt50', 'qnt75', 'max', 'near_lbnd', 'near_ubnd']
    
    pf.add_observations('mfinput_obs.csv', insfile='mfinput_obs.csv.ins',
                                index_cols=['input'], use_cols=cols, prefix='mfin')
  
    # build pest control file
    pst = pf.build_pst(version=None)        
    
    # for i in range(1, nper+1):
    #     shutil.copy2(
    #         os.path.join(org_d, f"ghb_stress_period_data_{i}.txt"),
    #         os.path.join(pf.new_d, "org", f"ghb_stress_period_data_{i}.txt"),
    #     )

    pst.control_data.noptmax = 0
    pst.pestpp_options['additional_ins_delimiters'] = ','
    pst.write(os.path.join(template,'wahp.pst'),version=2)
    pyemu.os_utils.run('pestpp-ies wahp.pst',cwd=template)
    pst.set_res(os.path.join(template,'wahp.base.rei'))
    print('phi',pst.phi)
    
    print('phi is greater than 1e-4, returning rei, investigate')
    rei = pst.res
    # sort rei by residual magnitude:
    rei = rei.sort_values(by='residual')
    bad_rei = rei.loc[rei.residual.abs() > 1e-6,:]
    # drop any name with 'datetime' in it
    bad_rei = bad_rei.loc[~bad_rei.name.str.contains('datetime'),:]
    # get bad_rei where 'k_k:0 is in group column:
    bad_k_rei = bad_rei.loc[bad_rei.group.str.contains('k_k:006'),:]
    
    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=True)
    pe.to_binary(os.path.join(template, 'prior.jcb'))
    pst.pestpp_options['ies_par_en'] = 'prior.jcb'
    pst.pestpp_options['save_binary'] = True

    # write the updated pest control file
    pst.write(os.path.join(pf.new_d, 'wahp.pst'),version=2)
    
    shutil.copy(os.path.join(pf.new_d, 'wahp.obs_data.csv'),
                os.path.join(pf.new_d, 'wahp.obs_data_orig.csv'))


    return template # return the template directory name


def set_obsvals_and_weights(template_d,
                            flow_weight_scheme='basic',
                            include_vertheads=False,
                            set_less_than_obs=False,
                            phi_factor_dict=None):

    pst = pyemu.Pst(os.path.join(template_d,'wahp.pst'))
    
    # now set obsvals and weights
    obs = pst.observation_data
    obs.loc[:,'weight'] = 0
    obs.loc[:,'observed'] = False
    obs.loc[:,'count'] = 0
    obs.loc[:,'standard_deviation'] = 0
    obs.loc[:,'obsval'] = 0.0  
    
    if flow_weight_scheme is not None:

        # process and then set ss water level obs targets:
        h_df = pd.read_csv(os.path.join('data','analyzed','processed_ss_head_targs.csv'),
                                        index_col=['start_dt'], parse_dates=True)
        h_df.loc[:, "datetime"] = pd.to_datetime(h_df.index, format="%Y-%m-%d")
        h_df.loc[:,'k'] = h_df.k.astype(int)
        h_df.loc[:,'i'] = h_df.i.astype(int)
        h_df.loc[:,'j'] = h_df.j.astype(int)
        h_df.loc[:,'obsprefix'] = h_df.obsprefix.apply(lambda x: x.replace('.','-'))
        uprefixes = h_df.obsprefix.unique()
        uprefixes.sort()
        print(uprefixes)

        oname_obsval_dict = {}
        for prefix in uprefixes:
            if 'ssh_id:' not in prefix:
                continue
            uh_df = h_df.loc[h_df.obsprefix==prefix,:].copy()
            uk = uh_df.k.unique()
            uk.sort()
            
            pobs = obs.loc[obs.obsnme.apply(lambda x: prefix in x),:].copy()
            if pobs.shape[0] == 0:
                print('empty obs for prefix:{0}'.format(prefix))
                continue
            pobs.loc[:,'k'] = uh_df.k.values[0].astype(int)
            pobs.loc[:,'i'] = uh_df.i.values[0].astype(int)
            pobs.loc[:,'j'] = uh_df.j.values[0].astype(int)

            for k in uk:
                kuh_df = uh_df.loc[uh_df.k==k,:].copy()
                if kuh_df.shape[0] == 0:
                    print('empty layer df for k:{0},prefix:{1}'.format(k,prefix))
                    continue
                ukobs = pobs.loc[pobs.k==k,:].copy()
                if ukobs.shape[0] == 0:
                    print('empty ukobs for k:{0},prefix:{1}'.format(k,prefix))
                    continue
                for head,dt in zip(kuh_df.loc[:,'gwe_ft'],kuh_df.index):
                    #print(head, dt)
                    if dt < pd.to_datetime('1970-01-01'):
                        mn_oname = ukobs.iloc[0,:].obsnme
                        oname_obsval_dict.setdefault(mn_oname, []).append(head)
        
        # process and then set transient water level obs targets:
        t_df_loc = pd.read_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
        t_df = pd.read_csv(os.path.join('data','analyzed','transient_well_targets.csv'))
        unq_prefix = t_df_loc['obsprefix'].unique()

        oname_obsval_dict_trans = {}
        for prefix in unq_prefix:
            if 'transh_id:' not in prefix:
                continue
            # isolate well id:
            wid = prefix.split(':')[1].split('_')[0]
            uh_df = t_df.loc[:,['start_datetime',wid]].copy()
            # chceck if any data outside of nan values in uh_df, if all nans continue
            if uh_df[wid].isnull().all():
                print('no data for prefix:{0}'.format(prefix))
                continue
            
            uh_df = uh_df.loc[uh_df[wid].notnull(),:].copy()
            
            pobs = obs.loc[obs.obsnme.apply(lambda x: prefix in x),:].copy()
            # get pobs that include start_datetime string in uh_df['start_datetime']
            dt_vals = uh_df['start_datetime'].unique()
            dt_vals.sort()
            pobs_w_meas = pobs.loc[pobs.obsnme.apply(lambda x: any(dt in x for dt in dt_vals)),:].copy()
      
            if pobs_w_meas.shape[0] == 0:
                print('empty obs for prefix:{0}'.format(prefix))
                continue
            # pobs.loc[:,'k'] = uh_df.k.values[0].astype(int)
            # pobs.loc[:,'i'] = uh_df.i.values[0].astype(int)
            # pobs.loc[:,'j'] = uh_df.j.values[0].astype(int)
     
            for idx, row in pobs_w_meas.iterrows():
                oname = row.obsnme
                sim_date = oname.split(':')[-1]
                val = uh_df.loc[uh_df['start_datetime'] == sim_date, wid].values[0]
                if np.isnan(val):
                    assert False, 'nan value for {0}, something went wrong...qa needed'.format(oname)
                oname_obsval_dict_trans.setdefault(oname, []).append(val)
        
        
        print('\n\n\n  ---  found {0} gw level obs  ---  \n\n\n'.format(len(oname_obsval_dict)))
        assert len(oname_obsval_dict) > 0

        if flow_weight_scheme == 'ss_only':
            for oname,vals in oname_obsval_dict.items():
                vals = np.array(vals)
                obspre = oname.split(':')[3].replace('_k','')
                oinfo = h_df.loc[h_df.obsprefix.apply(lambda x: obspre in x),:].copy()
                obs.loc[oname,'obsval'] = vals.mean()
                obs.loc[oname,'observed'] = True
                obs.loc[oname,'standard_deviation'] = 5
                obs.loc[:,'count'] = len(vals)
                obs.loc[oname,'weight'] = 1.0
            obs.loc[(obs.obgnme=="greater_rivflow") & (obs.obsnme.str.contains("1970-01-01")),'weight']=1.0

        if flow_weight_scheme == 'all_wl_meas':
            for oname,vals in oname_obsval_dict.items():
                vals = np.array(vals)
                obspre = oname.split(':')[3].replace('_k','')
                oinfo = h_df.loc[h_df.obsprefix.apply(lambda x: obspre in x),:].copy()
                obs.loc[oname,'obsval'] = vals.mean()
                obs.loc[oname,'observed'] = True
                obs.loc[oname,'standard_deviation'] = 5
                obs.loc[:,'count'] = len(vals)
                obs.loc[oname,'weight'] = 1.0
                if int(obs.loc[oname,'k']) == 3:
                    obs.loc[oname,'obgnme'] = 'wbvsshds'
                else:
                    obs.loc[oname,'obgnme'] = 'wsssshds'
                
            for oname,vals in oname_obsval_dict_trans.items():
                vals = np.array(vals)
                obspre = oname.split(':')[3].replace('_k','')
                oinfo = t_df_loc.loc[t_df_loc.obsprefix.apply(lambda x: obspre in x),:].copy()
                obs.loc[oname,'obsval'] = vals
                obs.loc[oname,'observed'] = True
                obs.loc[oname,'standard_deviation'] = 5
                obs.loc[:,'count'] = len(vals)
                obs.loc[oname,'weight'] = 1.0
                obs.loc[oname,'obgnme'] = 'transhds'
             
            #obs.loc[(obs.obgnme=="greater_rivflow") & (obs.obsnme.str.contains("1970-01-01")),'weight']=1.0
                        
        
        if include_vertheads:
            print('do a thing...')
            # vhd_df = pd.read_csv(os.path.join(template_d,'vert_hds_diff.csv'))
            # vhd_df['weight'] = 1
            # vhd_df = vhd_df.rename(columns={'idx':'obsnme'})
            # vhd_df.loc[:,'obsnme'] = vhd_df.obsnme.apply(lambda x: f'oname:vert_hds_diff.csv_otype:lst_usecol:obsval_idx:{x}')
            # vhd_obsval_dict = dict(zip(vhd_df.obsnme, vhd_df.obsval))
            # vhd_wt_dict = dict(zip(vhd_df.obsnme, vhd_df.weight))

            # obs['obsval'] = obs['obsnme'].map(vhd_obsval_dict).fillna(obs['obsval'])
            # obs['weight'] = obs['obsnme'].map(vhd_wt_dict).fillna(obs['weight'])

    
        if set_less_than_obs:
            print('do that prevent flood thing...')
            # # Set less than obs to make sure sim heads stay below top surface
            # obs.loc[obs.obgnme.str.contains("lessthan"),'weight'] = 1.0
            # obs.loc[obs.obgnme.str.contains("lessthan"), 'obgnme'] = "less_than_top"

            # # Update obsval to top - 10 ft for each less_than_top
            # sim = flopy.mf6.MFSimulation.load('mfsim.nam', sim_ws=template_d, load_only=['dis'])
            # m = sim.get_model("wahp")
            # top = m.dis.top.array

            # lt_rows = obs.loc[obs.obgnme == 'less_than_top', 'i'].values.astype(int)
            # lt_cols = obs.loc[obs.obgnme == 'less_than_top', 'j'].values.astype(int)

            # obs.loc[obs.obgnme=='less_than_top', 'obsval'] = top[lt_rows, lt_cols]


        # reset obs groups names if needed:
        #obs.loc[obs.obgnme.str.contains("sshid") , 'obgnme'] = 'ss_hds'
        
    assert pst.nnz_obs > 0  

    nzobs = obs.loc[obs.weight>0,:]
    vc = nzobs.obgnme.value_counts()
    for gname,c in zip(vc.index,vc.values):
        print('group ',gname,' has ',c,' nzobs')
    
    # drop_list = pd.read_csv(os.path.join('data','wl_obs_to_drop.csv'))
    # drop_list.columns = drop_list.columns.str.lower()
    # drop_list.columns = drop_list.columns.str.strip()
    # # drop duplicates by swn:
    # drop_list = drop_list.drop_duplicates(subset=['swn'])
    # drop_list['reason'] = drop_list['reason'].astype(str)
    # drop_list['reason'] = drop_list['reason'].apply(lambda x: x.lower())
    
    # for idx, well in drop_list.iterrows():
    #     if well['drop'] == True:
    #         # check if well id is in obgnme
    #         swn = well['swn']
    #         if 'flood' in well['reason']:
    #             rea = 'flooded_obs'
    #         elif 'uncof' in well['reason']:
    #             rea = 'unconfined_in_confined'
    #         elif 'quest' in well['reason']:
    #             rea = 'questionable_data'
    #         obs.loc[obs.obgnme.apply(lambda x: str(swn) in x),'obgnme'] = rea
    #         obs.loc[obs.obsnme.apply(lambda x: str(swn) in x),'weight'] = 0.0

    if phi_factor_dict is not None:
        with open(os.path.join(template_d,'phi_facs.csv'),'w') as f:
            keys = list(phi_factor_dict.keys())
            keys.sort()
            for key in keys:
                f.write('{0},{1}\n'.format(key,phi_factor_dict[key]))
        pst.pestpp_options['ies_phi_factor_file'] = 'phi_facs.csv'

    #check that the mean par values will run
    #pst.control_data.noptmax = -2
    #pst.write(os.path.join(template_d, 'wahp.pst'), version=2)
    #pyemu.os_utils.run('pestpp-ies wahp.pst', cwd=template_d)
    #pst.set_res(os.path.join(template_d,'wahp.mean.rei'))
    #print('mean par values run phi:',pst.phi)
    #assert pst.phi > 1.0


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


def run_ies(template_ws='template_d', m_d=None, num_workers=12, noptmax=-1, num_reals=None,
              init_lam=None, drop_conflicts=False, local=True, hostname=None, port=4263,
               use_condor=False,**kwargs):
    
    if m_d is None:
        m_d = template_ws.replace('template', 'master')

    pst = pyemu.Pst(os.path.join(template_ws, 'wahp.pst'))

    # Set control file options:
    pst.control_data.noptmax = noptmax
    pst.pestpp_options['ies_drop_conflicts'] = drop_conflicts
    pst.pestpp_options['overdue_giveup_fac'] = 5
    pst.pestpp_options['ies_bad_phi_sigma'] = 1.5
    pst.pestpp_options['ies_bad_phi'] = 1e+20
    #pst.pestpp_options["ies_n_iter_reinflate"] = [-2,999]
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
    pst.write(os.path.join(template_ws, 'wahp.pst'), version=2)
    
    prep_worker(template_ws, template_ws + '_clean')
    
    master_p = None

    if hostname is None:
        pyemu.os_utils.start_workers(template_ws, 'pestpp-ies', 'wahp.pst',
                                 num_workers=num_workers, worker_root='.',
                                 master_dir=m_d, local=local,port=4269)

    elif use_condor:
        check_port_number(port)
        
        jobid = condor_submit(template_ws=template_ws + '_clean', pstfile='wahp.pst', conda_zip_pth='nddwrpy311.tar.gz',
                              subfile='wahp.sub',
                              workerfile='worker.sh', executables=['mf6', 'pestpp-ies','mp7'], request_memory=5000,
                              request_disk='22g', port=port,
                              num_workers=num_workers)

        # jwhite - commented this out so not starting local workers on the condor submit machine # no -ross
        pyemu.os_utils.start_workers(template_ws + '_clean', 'pestpp-ies', 'wahp.pst', num_workers=0, worker_root='.',
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
            if f != 'cond_post.jcb' and f != 'autocorr_noise.jcb' and f != 'prior.jcb':  # need prior.jcb to run ies
                os.remove(os.path.join(new_d, f))
    mlt_dir = os.path.join(new_d, 'mult')
    for f in os.listdir(mlt_dir)[1:]:
        os.remove(os.path.join(mlt_dir, f))
    tpst = os.path.join(new_d, 'temp.pst')
    if os.path.exists(tpst):
        os.remove(tpst)


def condor_submit(template_ws, pstfile, conda_zip_pth='nddwrpy311.tar.gz', subfile='condor.sub', workerfile='worker.sh',
                  executables=[], request_memory=4000, request_disk='10g', port=4200, num_workers=71):
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


if __name__ == "__main__":

    import wahp02_model_build
    wahp02_model_build.main()
    
    print("Running setup_pst.py")
    print('Env path order:')
    for path in sys.path:
        print(path)
        
    modnm = 'wahp'
    run_tag ='_gwv_ss'
    org_d = os.path.join('model_ws', modnm+run_tag)
    
    # init gw vista model of ss:
    init_gw_vista_ss_mods(os.path.join('model_ws','wahp_clean'),mwd=modnm+run_tag)
    
    # assure run inputs exisit:
    if not os.path.exists(os.path.join('run_inputs',f'{modnm}{run_tag}')):
        raise FileNotFoundError(f'Run inputs for {modnm}{run_tag} do not exist, please create them')
    
    #dir locations
    m_d_flow = "master_flow"+run_tag
    t_d_flow = "template_flow"+run_tag
    
    template = t_d_flow
    
    # prep the flow template
    prep_flow = True
    # run ies for the flow template
    run_flow = True
    # run sensitivity
    run_sensitivity = False
    
    use_condor = False
    print(f' use condor: {use_condor}')
    
    if use_condor:
        num_reals_flow = 192
        num_workers_flow = 96
        hostname = '10.99.10.30'
        pid = os.getpid()
        current_time = int(time.time())
        seed_value = (pid + current_time) % 65536
        np.random.seed(seed_value)
        port = random.randint(4001, 4999)
        print(f'port #: {port}')
    else:
        num_reals_flow = 24
        num_workers_flow = 8
        hostname = None
        port = None
    
  
    # how many iters to use
    noptmax_flow = 2

    local = True

    if prep_flow:
        print('{0}\n\npreparing flow-IES\n\n{1}'.format('*'*17,'*'*17))
         
        temp_dir = setup_pstpp(org_d,modnm,run_tag,t_d_flow,flex_con=False,num_reals=num_reals_flow)

        print(f'------- flow-ies has been setup in {t_d_flow} ----------')

        set_obsvals_and_weights(t_d_flow,flow_weight_scheme= 'all_wl_meas', phi_factor_dict={ "wbvsshds":0.5,"transhds":0.2,'wsssshds':0.2},)

        #print('draw noise reals')
        #draw_noise_reals(t_d_flow,num_reals=num_reals_flow)


    if run_flow:
       print('*** running flow-ies to get posterior ***')
       run_ies(t_d_flow,m_d=m_d_flow,num_workers=num_workers_flow,noptmax=noptmax_flow,
               init_lam=None, local=local,
               use_condor=use_condor, hostname=hostname,port=port)


    #if run_sensitivity:
        # update pst control file to tie parameters
        # prepare_sen(m_d=m_d_flow, pst_name='wahp.pst', )

        # run_sen(m_d_flow, m_d=f'{m_d_flow}_sen', pst_name='wahp_sen.pst',
        #         num_workers=num_workers_flow,local=local,
        #         use_condor=use_condor, hostname=hostname, port=port)

    print('All Done!, congrats we did a thing')  