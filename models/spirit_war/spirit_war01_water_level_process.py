import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','pyemu')))
import flopy
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import Point
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import matplotlib.dates as mdates
import re


# -------------------
# --- Load Input Data
# -------------------
def load_input_data(model_name='swww'):
    # --- Load Water Level site reference
    sites = pd.read_csv(os.path.join('data','raw','swww_sites_final.csv'))
    
	# # Concat with PRESEN sites
    # sites_presens = pd.read_csv(os.path.join("data","raw","water_levels","sites_presens.txt"),
    #                             sep="\t")
    # sites_presens = sites_presens.loc[~sites_presens['Site_Index'].isin(sites['Site_Index'])]
    # sites = pd.concat([sites,sites_presens],axis=0)
    
    # Only want wells in spiritwood or ww aquifers
    # sites = sites.loc[sites['Aquifer'].isin(["Spiritwood","warwick","Warwick"])]
    # Make into gdf
    points = [Point(xy) for xy in zip(sites['x_2265'], sites['y_2265'])]
    sites = gpd.GeoDataFrame(data=sites,
                             geometry=points)
    sites = sites.set_crs(2265)#.to_crs(2265)
    
    sites.columns = [re.sub(r'\s+','_',col.strip()) for col in sites.columns]
    sites.columns = sites.columns.str.lower()

    # drop columns that will be recreated or not used later in this function:
    sites = sites.drop(columns=['top_lay1', 'bot_lay1', 'bot_lay2',
       'bot_lay3', 'bot_lay4', 'row', 'col']) 
    
    # if site drop_flag is True, drop the site
    print(f"Dropping {len(sites[sites['drop_flag'] == True])} sites with drop_flag set to True")
    sites = sites.loc[sites['drop_flag'] != True]
    
    # --- Load the modelgrid
    modelgrid = gpd.read_file(os.path.join('..','..','gis','output_shps','sw_ww',"sw_ww_modelgrid.shp"))
    modelgrid = modelgrid.set_crs(2265)

    # --- Load the model to get structure
    model_ws = os.path.join("model_ws",model_name)
    sim = flopy.mf6.MFSimulation.load(
        sim_name=f"{model_name}.nam",
        version="mf6",
        exe_name="mf6.exe",
        sim_ws=model_ws,
        load_only=["dis"],
        )
    gwf = sim.get_model(model_name)
    
    top = gwf.dis.top
    bots = gwf.dis.botm
    
    # --- Only need data within modelgrid
    sites = sites.clip(modelgrid)
    
    # --- Order sites by aquifer 
    priority_order = ['warwick', 'Warwick', 'Spiritwood']
    
    # Convert to string to handle non-str values (like NaNs) safely
    aquifer_str = sites['assigned_aquifer'].astype(str)
    
    # Get unique aquifer names not in priority list
    other_aquifers = sorted(set(aquifer_str) - set(priority_order))
    
    # Create full ordered list
    full_order = priority_order + other_aquifers
    
    # Assign sorting key
    sites['aquifer_sort'] = pd.Categorical(
        aquifer_str,
        categories=full_order,
        ordered=True
    )
    
    # --- Load water level data
    water_levels = pd.read_csv(os.path.join('data','raw','swww_waterlevels.csv'))
    
    # Filter
    water_levels = water_levels.loc[water_levels['loc_id'].isin(sites['loc_id'].values)]
    water_levels['date_meas'] = pd.to_datetime(water_levels['date_meas'])
    water_levels = water_levels.set_index('date_meas')
    water_levels.loc[water_levels['gwe_navd88']>10000,'gwe_navd88'] = np.nan
    water_levels.loc[water_levels['gwe_navd88']<-1000,'gwe_navd88'] = np.nan
    water_levels = water_levels.dropna()
    
    # Filter out sites with greater than X measurements
    meas_thresh = 15
    water_levels = water_levels[water_levels['loc_id'].map(water_levels['loc_id'].value_counts()) > meas_thresh]

    # Drop sites that have less than X measurements
    sites = sites.loc[sites['loc_id'].isin(water_levels['loc_id'].unique())]

    # --- SORT SITES by 1. aquifer and 2. length of record
    water_levels['datetime'] = water_levels.index
    record_spans = (
            water_levels.groupby('loc_id')['datetime']
            .agg(lambda x: (x.max() - x.min()).days)
        )
    sites = sites.merge(record_spans.rename('record_span_days'), left_on='loc_id', right_on='loc_id',how='left')
    sites['record_span_days'] = sites['record_span_days'].fillna(0).astype(int)
    sites = sites.sort_values(['record_span_days'], ascending=[False])
    
    # --- Get r/c indices for sites
    sites = gpd.sjoin(sites, modelgrid[['geometry', 'row', 'column']], how='left', predicate='intersects')
    
    # --- Create lithology reference for each site
    layer_data = []
    nlay = gwf.dis.nlay.data
    top_array = top.array
    botm_array = np.array([b.array for b in bots])  # shape (nlay, nrow, ncol)
    
    # --- Do a bit of processing on total depths and then sort by TD
    drop_idxs = []
    for idx, row in sites.iterrows():
        td = row['total_depth']
        bs = row['bottom_screen']
        if (td == 0) and (bs != 0):
            sites.at[idx, 'total_depth'] = bs
        if (td != 0) and (bs == 0):
            sites.at[idx, 'bottom_screen'] = td
        elif (td == 0) and (bs == 0):
            drop_idxs.append(idx)
    print(f"Dropping {len(drop_idxs)} sites with no total depth or bottom screen depth")
    sites = sites.drop(index=drop_idxs)
    
    # --- Find which layer the site is within
    layer_list = []
    for _, row in sites.iterrows():
        i, j = row['row']-1, row['column']-1
        depth = row['bottom_screen']
        surface_elev = row['land_surface_elev_navd88']
        depth_elev = surface_elev - depth
        # find first layer where depth elevation is above bottom
        assigned = False
        for k in range(nlay):
            top_k = top_array[i, j] if k == 0 else botm_array[k - 1, i, j]
            bot_k = botm_array[k, i, j]
            if top_k >= depth_elev >= bot_k:
                layer_list.append(k)
                assigned = True
                break
        if not assigned:
            if depth_elev > top_array[i, j]:
                layer_list.append(0)
            elif depth_elev < botm_array[-1, i, j]:
                layer_list.append(nlay - 1)
            else:
                layer_list.append(np.nan)  # should not happen unless DEM is weird
    # Add result to sites
    sites['model_layer'] = layer_list # zero-based layer index
    # where nan fill with 9999
    sites.loc[sites['model_layer'].isna(), 'model_layer'] = 9999
    sites['model_layer'] = sites['model_layer'].astype(int)+1 # convert to one-based layer index
    sites = sites.sort_values(by='model_layer')
    
    sites['k'] = np.nan
    # where model_layer is not equal to sites['manually_corrected_lay'], k value gets 'manually_corrected_lay' use np.where
    sites['k'] = np.where(sites['model_layer'] != sites['manually_corrected_lay'],
                          sites['manually_corrected_lay'],
                          sites['model_layer'])
    sites['k'] = sites['k'].astype(int)-1

    return sites,water_levels,modelgrid

# -------------------------
# --- Create lithology plot
# -------------------------
def plot_lith_multiwell(sites_grp,master_df,mnm,gwf,ax):
    
    top = gwf.dis.top
    bots = gwf.dis.botm.array
    layer_data = []
    nlay = gwf.dis.nlay.data
    top_array = top.array
    botm_array = bots

    # --- Save info when sites have the same row and column
    r_c_dict = {}
    for _, row in sites_grp.iterrows():
        r, c = int(row['row']), int(row['column'])
        aq = row['assigned_aquifer']
        r_c_dict[row['loc_id']] = [r,c,aq]
        for k in range(nlay):
            ztop = top_array[r-1, c-1] if k == 0 else botm_array[k-1, r-1, c-1]
            zbot = botm_array[k, r-1, c-1]
            layer_data.append({
                        'loc_id': row['loc_id'],
                        'layer': k,
                        'ztop': ztop,
                        'zbot': zbot
                        }
                )
    layer_df = pd.DataFrame(layer_data)
    
    # --- Create map for sites that share a row/column/aquifer index
    location_to_ids = defaultdict(list)
    for site_id, (r, c, aq) in r_c_dict.items():
        location_to_ids[(r, c, aq)].append(site_id)
    shared_location_dict = {
                site_id: location_to_ids[(r, c, aq)]
                for site_id, (r, c, aq) in r_c_dict.items()
                }
    
    # Colors for the aquifer layers
    colors = {0:"#4682B4", 
              1:"#FFE999", 
              2:"darkblue",
              3:"maroon",
              4:"orange",
              5:"#75D98F",
              6:"#A65628"
              }

    layers = layer_df[layer_df['loc_id'] == site_id]
    
    wellcnt = len(sites_grp)
    def compute_left_positions(n, bar_width=0.4, gap=0.0, x_min=0.0, x_max=1.0):
        span = x_max - x_min
        total = n*bar_width + (n-1)*gap
        if total > span:
            raise ValueError(f"Bars won’t fit: total={total} > span={span}")
        start = x_min + (span - total)/2.0  # center the block
        return start + np.arange(n)*(bar_width + gap)

    # example
    wellcnt = len(sites_grp)
    bar_width = 0.05
    gap = 0.0  # set >0 if you want spacing
    wpos = compute_left_positions(wellcnt, bar_width, gap)  # array of lefts, length = wellcnt
    
    # Plot lithology layers as wide bars

    for _, lyr in layers.iterrows():
        ax.barh(
            y=(lyr['ztop'] + lyr['zbot']) / 2,
            width=0.4,
            height=abs(lyr['ztop'] - lyr['zbot']),
            left=0.3,
            color=colors[lyr['layer']],
            edgecolor='k'
        )
    
    # Overlay well
    cnt = 0
    for idx,well in sites_grp.iterrows():
        mpe = well['land_surface_elev_navd88']
        td = well['total_depth']
        ts = well['top_screen']
        bs = well['bottom_screen']

        if td == 0  and ts != 0:
            td = ts
        
        if td == 0 and ts == 0:
            td = mpe - (layers.loc[layers['layer']==well['k'],'ztop'].values+layers.loc[layers['layer']==well['k'],'zbot'].values)/2
        
        if pd.notnull(mpe) and pd.notnull(td):
            well_top = mpe
            well_bot = mpe - td
            ax.barh(
                y=(well_top + well_bot) / 2,
                width=bar_width,
                height=abs(well_top - well_bot),
                left=wpos[cnt],
                color='grey'
            )
            
        # Overlay screen
        if pd.notnull(ts) and pd.notnull(bs):
            top_screen = mpe - ts
            bot_screen = mpe - bs
            ax.barh(
                y=(top_screen + bot_screen) / 2,
                width=0.05,
                height=abs(top_screen - bot_screen),
                left=wpos[cnt],
                color='white',
                hatch='-----',
            )
        

        # id to link to master_df
        grp_num = well['group_number']
        id = f'grp.{grp_num}'+f'_k.{well["k"]}'
        mwl = master_df[[id]].mean(axis=0)
        # --- Plot average WL
        ax.plot(wpos[cnt]+bar_width/2, mwl, 
                marker='x', 
                color='blue', 
                markersize=10)
        
        cnt += 1


# --------------------
# --- Load perlen data
# --------------------
def load_perlen(annual_flag=False):
    if annual_flag:
        perlen = pd.read_csv(os.path.join("tables","annual_stress_period_info.csv"))
    else:
        perlen = pd.read_csv(os.path.join("tables","monthly_stress_period_info.csv"))
    
    return perlen


# ------------------------------
# --- Smooth and process WL data
# ------------------------------
def process_wls(df,subset,annual_flag):
    # --- Create perlen and cumulative datetime 
    perlen = load_perlen(annual_flag=False)
    perlen = perlen.tail(len(perlen)-1)
    model_datetime = pd.to_datetime(perlen['start_datetime'])
    
    # --- Interpolate subset of obs WL's to daily
    ts = subset['gwe_navd88'].sort_index()
    ts = ts.groupby(level=0).median()     
    ts = ts.dropna()    
    
    # --- Split into pre-2000 and post-2000, cause dealing with different SP discretization 
    split_date = pd.Timestamp('2000-01-01')
    pre2000 = ts[ts.index < split_date]
    post2000 = ts[ts.index >= split_date]
    
    # --- Apply smoothing based on yearly/monthly SP period
    pre2000 = pre2000.resample('YE').median()
    
    # For post-2000, using a ~2-month rolling average
    post2000 = (
            ts[ts.index >= split_date]
            .resample('D').interpolate()
            .rolling("60D", center=True).mean()
            )
    
    # --- Recombine to a single time series
    smooth_ts = pd.concat([pre2000, post2000]).sort_index()
        
    # --- Align to model sim times (cum_datetime) to give apples:apples comparison w/ model results
    simtime_ts = smooth_ts.reindex(model_datetime, method="nearest", tolerance=pd.to_timedelta(181, unit="d"))  # --> Tolerance of 181 days
    simtime_ts = simtime_ts.groupby(simtime_ts.index).mean()  # group duplicates to avoid 2 meas on same date
    simtime_ts = simtime_ts.dropna()
    
    return simtime_ts


# -------------------------------------------------------------------
# --- Load and process PRISM precip data to color the plot by wet/dry
# -------------------------------------------------------------------
def load_precip_data(quantile_method=False):
    precip = pd.read_csv(os.path.join("data","raw","PRISM_precip","PRISM_ppt_provisional_4km_197001_202501_47.8746_-98.5477.csv"),
                         skiprows=10,
                         index_col=0,
                         parse_dates=True)
    
    # Calc 3-year rolling average and color by wet/dry
    rolling = precip.rolling('1095D',center=True).mean().resample('AS-JAN').first()
    
    # Use quantiles to assign wet vs dry
    if quantile_method:
        # Calculate thresholds based on percentiles
        wet_thresh = rolling['ppt (inches)'].quantile(0.75)
        dry_thresh = rolling['ppt (inches)'].quantile(0.25)
        
        # Assign color
        rolling['color'] = np.zeros(len(rolling))
        rolling.loc[rolling['ppt (inches)'] > wet_thresh, 'color'] = 'blue'
        rolling.loc[rolling['ppt (inches)'] < dry_thresh, 'color'] = 'red'
        rolling.loc[rolling['color']==0, 'color'] = 'grey'
    
    # Use standard deviation to assign wet vs dry
    else:
        rolling['color'] = np.zeros(len(rolling))
        rolling.loc[rolling['ppt (inches)']>(rolling['ppt (inches)'].mean() + (rolling['ppt (inches)'].std())),'color']='blue'
        rolling.loc[rolling['ppt (inches)']<(rolling['ppt (inches)'].mean() - (rolling['ppt (inches)'].std())),'color']='red'
        rolling.loc[rolling['color']==0,'color'] = 'grey'
        
    return rolling
    

# -----------------------------------------
# --- Save processed WL data w/out plotting
# -----------------------------------------
def save_processed_WL_data(sites,water_levels,annual_flag):
    # Process data for each site
    for idx,site_id in enumerate(sites['loc_id'].unique()):
        wl_data = water_levels.loc[water_levels['loc_id']==site_id,'gwe_navd88']
        # Calc smoothed WL data
        smoothed_wl = process_wls(water_levels,pd.DataFrame(wl_data),annual_flag)
        t = smoothed_wl.rename(site_id).to_frame()
        t.index = pd.to_datetime(t.index)
        t.columns = [site_id]
        if idx == 0:
            main_df = t
        else:
            main_df = pd.concat([main_df, t], axis=1)
    
    return main_df


# ---------------------------
# --- Create main PDF figures
# ---------------------------
def nested_sites(main_df,sites,water_levels,modelgrid,mnm,gen_plots=True):

    model_ws = os.path.join("model_ws",mnm)
    model_name = mnm
    sim = flopy.mf6.MFSimulation.load(
        sim_name=f"{model_name}.nam",
        version="mf6",
        exe_name="mf6.exe",
        sim_ws=model_ws,
        load_only=["dis"],
        )
    gwf = sim.get_model(model_name)

    for col in main_df.columns:
        site_info = sites[sites['loc_id'] == col]
        new_header = f'lid.{col}_i.{site_info["row"].values[0]-1}_j.{site_info["column"].values[0]-1}_k.{site_info["k"].values[0]}_grp.{site_info["group_number"].values[0]}'
        main_df.rename(columns={col: new_header}, inplace=True)
    
    def parse_column(col):

        """
        Extract (group, layer) as integers from a column like:
        'lid.13304720ABBA3_i.112_j.46_k.1_grp.15'
        """
        m = re.search(r"k\.(\d+)_grp\.(\d+)", col)
        if m:
            layer = int(m.group(1))
            group = int(m.group(2))
            return (group, layer)
        else:
            return (float("inf"), float("inf"))  # fallback if no match
    
    sorted_cols = sorted(main_df.columns, key=parse_column)
    main_df = main_df[sorted_cols]  # reindex with sorted columns   

    # get list of unique group_numbers in columns
    group_nums = main_df.columns.str.extract(r'_grp\.(\d+)')[0].unique()
    master_df = pd.DataFrame(index=main_df.index)
    for grp_num in group_nums:
        # Get columns that match the group_number
        grp_cols = main_df.columns[main_df.columns.str.endswith(f'_grp.{grp_num}')]
      
        if len(grp_cols) > 1:
            # see if we need to combine timeseries if they have the same k value
            k_values = grp_cols.str.extract(r'_k\.(\d+)')[0]
            dup_k_values = []
            # see if there are duplicate k values
            if k_values.duplicated().any():
                # get the duplicated k values
                dup_k_values = k_values[k_values.duplicated()].unique()
                for k_val in dup_k_values:
                    # get column names with this k value
                    dup_cols = [col for col in grp_cols if f'_k.{k_val}' in col]
                    ndf = main_df[dup_cols].median(axis=1,skipna=True)
                    master_df[f'grp.{grp_num}_k.{k_val}'] = ndf
                
                # get the non-duplicated columns
                # these are the columns that do not have the k value in their nam
                # remove dup_k_values from k_values:
                not_duplicated = k_values[~k_values.isin(dup_k_values)].tolist()
                if len(not_duplicated) > 0:
                    for k in not_duplicated:
                        col = [col for col in grp_cols if f'_k.{k}' in col]
                        ndf = main_df[col]  # there should only be one column with this k value
                        # add the non-duplicated columns to master_df
                        master_df[f'grp.{grp_num}_k.{k}'] = main_df[col]
            else:
                for col in grp_cols:
                    k_val = col.split('_k.')[1].split('_')[0]
                    master_df[f'grp.{grp_num}_k.{k_val}'] = main_df[col]
        else:
            for col in grp_cols:
                k_val = col.split('_k.')[1].split('_')[0]
                master_df[f'grp.{grp_num}_k.{k_val}'] = main_df[col]

    def parse_col(col):
        m = re.match(r"grp\.(\d+)_k\.(\d+)", col)
        if m:
            return int(m.group(1)), int(m.group(2))
        else:
            return float("inf"), float("inf")  # fallback if something doesn't match
    sorted_cols = sorted(master_df.columns, key=parse_col)
    master_df = master_df[sorted_cols]  # reindex with sorted columns
    
    # --- Now create dataframe of head differnce obs:
    k_aq = {0:'WW',1:'CS',2:'SW',3:'BED'}

    # Desired ordered aquifer pairs (A - B)
    PAIRS = [('WW', 'CS'),
            ('WW', 'SW'),
            ('CS', 'SW')]

    # Parse (grp, k) from column names
    def parse_grp_k(col):
        m = re.match(r'^grp\.(\d+)_k\.(\d+)$', col)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    # Build a helper structure: by group -> {k: column_name}
    by_grp = {}
    for col in master_df.columns:
        g, k = parse_grp_k(col)
        if g is None:
            continue
        by_grp.setdefault(g, {})[k] = col
    # filter dictionary down to groups with at least 2 layers
    by_grp_multi = {g: kmap for g, kmap in by_grp.items() if len(kmap) > 1}
    
    # Create output DF
    head_diff_df = pd.DataFrame(index=master_df.index)

    for grp, k_to_col in by_grp_multi.items():
        # Map aquifer -> list of (k, col)
        aq_to_layers = {}
        for k, col in k_to_col.items():
            aq = k_aq.get(k)
            if aq is None:
                continue
            aq_to_layers.setdefault(aq, []).append((k, col))

        for aqA, aqB in PAIRS:
            listA = aq_to_layers.get(aqA, [])
            listB = aq_to_layers.get(aqB, [])
            if not listA or not listB:
                continue

            for kA, colA in listA:
                for kB, colB in listB:
                    # Difference: A - B (respects pair order)
                    diff = master_df[colA] - master_df[colB]
                    # skip if the entire column is NaN
                    if diff.notna().sum() == 0:
                        continue
                    out_col = f'grp.{grp}_k.{kA}-{kB}_{aqA}-{aqB}'
                    head_diff_df[out_col] = diff

    if gen_plots:
        # Create PDF file
        pdf_path = os.path.join("data","analyzed",f"well_summary_plots_grouped.pdf")
        
        grp_nums = master_df.columns.str.extract(r'grp\.(\d+)')[0].unique()
        with PdfPages(pdf_path) as pdf:
            
            for grp in grp_nums:
                # if str(int(grp)) != '71':
                #     continue
                # get master_df columns that have the group_number
                grp_cols = master_df.columns[master_df.columns.str.startswith(f'grp.{grp}_')]
                
                # --- Load the site to plot
                print(f"\nMaking figure for group: {grp}\n")
               
                fig = plt.figure(figsize=(12, 6))
                gs = GridSpec(
                        nrows=2,
                        ncols=3,
                        height_ratios=[1, 1],
                        width_ratios=[1.01, 0.5, 1.2],  # make map column wider
                        figure=fig
                    )
                shared_colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628",]

                # --- Bottom Row: Hydrograph Time Series
                ax_hydro = fig.add_subplot(gs[1, :])

                # get sites with this group_number
                sites_grp = sites[sites['group_number'] == int(grp)]
                color_cnt = 0
                for site_id in sites_grp['loc_id'].unique():
                    wl_data = water_levels.loc[water_levels['loc_id']==site_id,'gwe_navd88']
                
                    # Daily resample on high-frequency transducer data
                    if len(wl_data) > 1200:
                        wl_data_resamp = wl_data.resample('D').mean()
                        wl_data_resamp = wl_data_resamp.dropna()
                        ax_hydro.plot(wl_data_resamp,
                                    marker='o',
                                    ms=4,
                                    color=shared_colors[color_cnt],
                                    label=site_id,
                                    alpha=0.5)
                        color_cnt += 1
                    else:
                        ax_hydro.plot(wl_data,
                                    marker='o',
                                    ms=4,
                                    color=shared_colors[color_cnt],
                                    label=site_id,
                                    alpha=0.5)
                        color_cnt += 1

                # now plot master_df for this group:
                for col in grp_cols:
                    smoothed_wl = master_df[col]
                    k = int(col.split('_k.')[1].split('_')[0])
                    aq = ''
                    if k == 0:
                        aq = 'WW'
                        aq_color = "#4F4C81"
                        marker = 'x'
                    elif k == 1:
                        aq = 'CS'
                        aq_color = "#8B4513"
                        marker = '^'
                    elif k == 2:
                        aq = 'SW'
                        aq_color = "#5A3921"
                        marker = 's'
                    elif k == 3:
                        aq = 'BED'
                        aq_color = "#413E3C"
                        marker = 'o'

                    ax_hydro.plot(smoothed_wl,
                                  color=aq_color,
                                  ls='-',
                                  marker=marker,
                                  ms=5,
                                  linewidth=1.5,
                                  markerfacecolor=None,
                                  label=aq)
          
                # Load and plot example simulated timeseries for a specific group...
                # if str(int(grp)) == "71":
                #     sim = pd.read_csv(os.path.join('data','analyzed','ex_sim_ts_SW_71.csv'),
                #                       index_col=0,
                #                       parse_dates=True)
                #     ax_hydro.plot(sim['head'],
                #                   color='royalblue',
                #                   ls='-',
                #                   ms=5,
                #                   linewidth=3,
                #                   markerfacecolor=None,
                #                   label='Simulated'
                #                   )
            
                ax_hydro.set_ylabel("Water Level\n(feet above mean sea level)")
                ax_hydro.set_xlabel("Year")
                ax_hydro.set_xlim(right=pd.Timestamp('2025-05-01'))
                # h,l = ax_hydro.get_legend_handles_labels()
                ax_hydro.grid()
                
                # add secondary y-axis on right side for head difference if available
                head_diff_cols = head_diff_df.columns[head_diff_df.columns.str.startswith(f'grp.{grp}_')]
                h_diff_colors = ['orange','red','purple','brown']
                cnt=0
                if len(head_diff_cols) > 0:
                    ax_hydro2 = ax_hydro.twinx()
                    ax_hydro2.set_ylabel("Head Difference\n(feet)")
                    mxdiff = 0
                    for col in head_diff_cols:
                        aq_pair = col.split('_')[-1]
                        ax_hydro2.bar(head_diff_df.index, head_diff_df[col],
                                      width=366,alpha=0.3,color=h_diff_colors[cnt],label=f'Head difference {aq_pair}')
                        mxdiff = max(mxdiff, head_diff_df[col].abs().max())
                        cnt += 1
                    ax_hydro2.set_ylim(0, 3*mxdiff)
                    ax_hydro2.legend(loc='lower left')
                ax_hydro.legend()
                
                # --- Color the plot by wet and dry periods
                # for i in range(len(wet_dry)-1):
                #     color = wet_dry.iloc[i].color
                #     if color == 'grey':
                #         continue
                #     d1 = wet_dry.iloc[i].name
                #     d2 = wet_dry.iloc[i+1].name - pd.Timedelta(days=10)
                #     ax_hydro.axvspan(d1,d2,color=color,
                #                      alpha=0.2,
                #                      edgecolor=None)
                
                # --- Metadata Text Block (Left)
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_meta.axis("off")
                swn = sites_grp['loc_id'].unique()
                swn_str = '\n'.join(swn)
                ax_meta.text(0.3, 0.5, f"Group #: {grp}\nState Well Numbers:\n {swn_str}\n", 
                    va="top", fontsize=12,
                    ha='center')
                
                # # Get well information
                # num_meas = len(wl_data)
                # if num_meas == 0:  # --> Don't plot if no WL data
                #     print(f"****SKIPPING {site_id}****")
                #     continue
                # aquifer = site['Aquifer']
                # lay_num = site['model_layer'] + 1
                # use = site['Purpose']
                # ts = site['Top_Screen']
                # bs = site['Bottom_Screen']
                # mse = site['Measuring_Point_Elev_NAVD88']
                # if mse < 100:
                #     mse = site['Land_Surface_Elev_NAVD88']
                # screen_midpoint = round(mse - (ts+bs)/2,1)
                # td = site['Total_Depth']
                # avg_wl = round(wl_data.mean(),1)
                # avg_dtw = round(mse - avg_wl,1)
                # record_len = round((wl_data.index.max() - wl_data.index.min()).days / 365.25,1)
                # deltas = wl_data.index.to_series().diff().dropna()
                # # Median frequency
                # freq_days = deltas.median().days
                # if freq_days > 300:
                #     freq_label = "Annual"
                # elif freq_days > 25:
                #     freq_label = "Monthly"
                # elif freq_days > 1:
                #     freq_label = "Weekly"
                # else:
                #     freq_label = "Daily"
                # Label figure with info

                # if site_id == 130509:
                #     ax_meta.text(0.3, 1,"**WL Adjusted to Land Surface Elevation")
                
                
                # --- B: Lithology Stack Bar
                ax_lith = fig.add_subplot(gs[0, 1])
                plot_lith_multiwell(sites_grp,master_df,'swww',gwf,ax_lith)
                
                ax_lith.set_xlim(0.4, 0.6)
                for spine in ['top', 'right', 'bottom','left']:
                    ax_lith.spines[spine].set_visible(False)
                ax_lith.tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)
                ax_lith.set_ylabel("Elevation (feet ASL)")
                
        
                # --- C: Map View
                ax_map = fig.add_subplot(gs[0, 2])
                ww_aquifer = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww',"warwick_only.shp"))
                sw_aquifer = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww',"spiritwood_aquifers.shp"))
                # make sure crss match
                ww_aquifer = ww_aquifer.to_crs(sites.crs)
                sw_aquifer = sw_aquifer.to_crs(sites.crs)

                ww_aquifer.boundary.plot(ax=ax_map,
                                    color='blue',alpha=0.5)
                sw_aquifer.boundary.plot(ax=ax_map,
                                    color='lightblue',alpha=0.5)
                sites.plot(ax=ax_map,
                           color='grey',
                           edgecolor='k',
                           markersize=8)
                
                # Plot selected site
                sites_grp.plot(ax=ax_map,
                                    color='orange',
                                    edgecolor='k',
                                    markersize=35)
                modelgrid.dissolve().boundary.plot(ax=ax_map,
                                                color='k')
                
                # --- Custom map legend
                legend_elements = [
                            Patch(edgecolor='k', facecolor='none', linewidth=1.5, label='Model Boundary'),
                            Patch(edgecolor='blue', facecolor='none', linewidth=1.5, label='Warwick Aquifer'),
                            Patch(edgecolor='lightblue', facecolor='none', linewidth=1.5, label='Spiritwood Aquifer'),
                            Line2D([0], [0], marker='o', color='grey', markeredgecolor='k',
                                markersize=6, linestyle='None', label=f'Monitoring Wells'),
                            Line2D([0], [0], marker='o', color='orange', markeredgecolor='k',
                                markersize=10, linestyle='None', label='Selected Well'),
                            ]
                # Add legend to map axis
                ax_map.legend(handles=legend_elements,
                            loc='center left', 
                            fontsize=8, 
                            frameon=True,
                            bbox_to_anchor=(0.8, 0.08),
                            framealpha=1,
                            edgecolor='black'
                            )
                ax_map.axis('off')
                
                # --- Custom lith legend
                ax_legend_lith = fig.add_axes([0.58, 0.82, 0.1, 0.1])  # [left, bottom, width, height]
                ax_legend_lith.axis('off')
                
                {0: "soils", 1: "sand", 2: "clay", 3: "clay", 4: "sand", 5: "clay", 6: "sand"}
                colors = {0:"#4682B4", 
                        1:"#FFE999", 
                        2:"darkblue",
                        3:"maroon",
                        4:"orange",
                        5:"#75D98F",
                        }
                # get columns in modelgrid that start with idom_
                icols = modelgrid.columns[modelgrid.columns.str.startswith('idom_')]
                lycnt = len(icols)

                lith_legend_elements = [
                            Patch(facecolor="#4682B4", edgecolor='k', label="Warwick"),
                            Patch(facecolor="#FFE999", edgecolor='k', label="Clay/Silt"),
                            Patch(facecolor="darkblue", edgecolor='k', label="Spiritwood"),
                            Patch(facecolor="maroon", edgecolor='k', label="Bedrock"),
                            Patch(facecolor='none', edgecolor='k', hatch='-----', label='Screen Interval'),
                            Line2D([0], [0], marker='x', color='blue', linestyle='None', markersize=8, label='Mean Water Level')
                            ]
                    
                # Add the lithology legend
                ax_legend_lith.legend(
                            handles=lith_legend_elements,
                            loc='upper left',
                            fontsize=8,
                            frameon=True,
                            framealpha=1,
                            edgecolor='black'
                        )
                
                fig.tight_layout()
                pdf.savefig(fig,bbox_inches='tight' )
                plt.show()
                plt.close(fig)
    return master_df, head_diff_df



# --------
# --- Main
# --------
def wl_process_main(mnm='swww',annual_flag=False,gen_plots=False):
    # Load the Input data
    sites,water_levels,modelgrid = load_input_data(model_name=mnm)
    main_df = save_processed_WL_data(sites,water_levels,annual_flag)
    wls, head_diff = nested_sites(main_df,sites,water_levels,modelgrid,mnm,gen_plots=gen_plots)    

    sites = sites.sort_values(by='group_number')

    sites.to_csv(os.path.join('data','analyzed','transient_well_targets_lookup.csv'))
    wls.to_csv(os.path.join('data','analyzed','transient_well_targets.csv'))
    head_diff.to_csv(os.path.join('data','analyzed','transient_well_head_diffs.csv'))

    return wls, head_diff


if __name__ == "__main__":
     print('processing water level timeseries')
     wl_process_main() 













