import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','dependencies','flopy_worker')))

import flopy 
import pyproj 
import spatialpy
from scipy.interpolate import griddata
import seaborn as sns
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling       
from rasterio.mask import mask
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

import numpy as np
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dateutil.relativedelta import relativedelta

import warnings            
warnings.filterwarnings("ignore")
# Set some pandas options
pd.set_option('expand_frame_repr', False)


def process_resistivity_rasters(in_raster_dir='')


def trim_reproj_rasters(in_raster_dir='',toepsg=2276):
    """_ reads in all raster files in ../input_rasters/Structural_Surfaces and replaces 
    zero values with a new no data value 1e30, replaces values greater than 3,850 feet 
    below mean sea level to no data, and also reprojects to rasters to specficied epsg

    Returns:
        _type_: adjusted rasters into ../input_rasters/Structural_Surfaces_Clean__
    """  

    # gam projection:
    #gam_crs = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'
    
    # read in all tifs in the Structural_Surfaces folder:
    tifs = [tif for tif in os.listdir(in_raster_dir) if tif.endswith('.tif')]


    for tif in tifs:
        outpth = redefine_projection(os.path.join(in_raster_dir,tif),toepsg=toepsg)
    
    in_raster_dir = outpth
    tifs = [tif for tif in os.listdir(in_raster_dir) if tif.endswith('.tif')]
    
    # make a new folder to store the cleaned rasters:
    out_raster_dir = os.path.join('gis','input_rasters','Structural_Surfaces_Clean')
    if not os.path.exists(out_raster_dir):
        os.makedirs(out_raster_dir)

    # iterate over rasters replacing 0 with 1e30:
    for tif in tifs:
        ras = rasterio.open(os.path.join(in_raster_dir,tif))
        crs = ras.crs
        arr = ras.read(1)
        arr[arr<-1e10] = 1e30
        arr[arr == 0] = 1e30
        arr[arr>1e10] = 1e30
        #arr[(arr<-3850.0)&(arr>-4100)] = 3850.0
        arr[(arr<-3850.0)] = -3850.0
        arr[np.isnan(arr)] = 1e30


        transform = ras.transform
        new_dataset = rasterio.open(os.path.join(out_raster_dir,tif), 'w', driver='GTiff',
                                    height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
                                    crs=crs, transform=transform, nodata=1e30)
        new_dataset.write(arr, 1)
        new_dataset.close()
        print(f'exported {os.path.join(out_raster_dir,tif)}')
        ras.close()


def merge_alluvium_layers():
    """_summary_
        merge the triniry and seymour alluvium layers into one layer
    """
    from rasterio.merge import merge

    # first get the seymour
    seymour_tif = os.path.join('gis','input_rasters','Structural_Surfaces_Clean','BaseElev_Layer1_a_Seymour.tif')
    seymour_obj = rasterio.open(seymour_tif)
    seymour_arr = seymour_obj.read(1)

    # now get the trinity bottom
    trinity_tif = os.path.join('gis','input_rasters','Structural_Surfaces_Clean','bhos_30ftmin.tif')
    trinity_obj = rasterio.open(trinity_tif)
    trinity_arr = trinity_obj.read(1)

    # Specify no data value if known (e.g., -9999)
    nodata_value = 1e30

    # Merge the rasters
    merged_raster, output_transform = merge([seymour_obj, trinity_obj], nodata=nodata_value)

    # Create a new raster file with the merged data
    with rasterio.open(os.path.join('gis','input_rasters','Structural_Surfaces_Clean','BaseElev_Layer1_Seymour_Trinity.tif'), 'w', 
                    driver='GTiff',
                    height=merged_raster.shape[1],
                    width=merged_raster.shape[2],
                    count=merged_raster.shape[0],
                    dtype=merged_raster.dtype,
                    crs=seymour_obj.crs,
                    transform=output_transform,
                    nodata=1e30) as dst:
        dst.write(merged_raster)

    seymour_obj.close()
    trinity_obj.close()


def generate_constant_surficial_botm(thk=150,in_surf_ras_pth='',tif_out_nm='surf_realtive_cnst_ft_thk_',res=(1320,1320)):
    """_summary_
        Make a generic raster with constant thickness relative to land surface
        
    Parameters
    ----------
    thk : int, required
        _description_, value to subtract from surface elevation to get surficial bottom elevation
    in_surf_ras_pth : str, required
        _description_, path to surface elevation raster
    tif_out_nm : str, required
        _description_, name of output tif file    
    """    
    left, bottom, right, top = 1468894.0, 6327767.0, 2313694.0, 7489367.0  
    resolution = (1320, 1320)  # 1 mile by 1 mile in feet
    #resolution = (5280.0,5280.0)
    width = int((right - left) / resolution[0])
    height = int((top - bottom) / resolution[1])
    
    # Calculate the transform and dimensions for the new resolution
    new_transform, new_width, new_height = calculate_default_transform(
        'EPSG:2276', 'EPSG:2276', width, height, 
        left=left, bottom=bottom, right=right, top=top, 
        resolution=resolution)
    
    # read in surface elevation
    dem_ras = rasterio.open(in_surf_ras_pth)
    dem_array = dem_ras.read(1)

    # read in layer 1 and create a mask for calc of constant thickness under surficial layer:
    in_ly_pth = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(res[0])}_ft','BaseElev_Layer1_Seymour_Trinity_1mi.tif')
    ly1 = rasterio.open(in_ly_pth)
    ly1_arr = ly1.read(1)
    ly1_arr = np.where(ly1_arr < 1e30,ly1_arr,np.nan)
    
    # subtract thk to get new bottom surfaces:
    ly1_rel_botm = ly1_arr-thk
    sur_botm = dem_array-thk
    
    # replace values in sur_botm with ly1_rel_botm where ly1_arr is not no data:
    newbot = np.where(np.isnan(ly1_rel_botm),sur_botm,ly1_rel_botm)
    newbot[newbot>1e10] = np.nan

    transform = dem_ras.transform
    new_dataset = rasterio.open(os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(res[0])}_ft',f'primary_aq_{int(thk)}ft_thick_1mi.tif'), 'w', driver='GTiff',
                                height=newbot.shape[0], width=newbot.shape[1], count=1, dtype=newbot.dtype,
                                crs=dem_ras.crs, transform=transform, nodata=1e30)
    new_dataset.write(newbot, 1)
    new_dataset.close()
    dem_ras.close()
    ly1.close()


def convert_raster_bottoms2points(indir='',thk=200):
    """_summary_
        take input raster directory and convert all rasters to point shapefiles for interpolation scheme
            
    Parameters
    ----------
    indir : str, required
        _description_, path to directory with input rasters, note all rasters must also be included in 
        nam2tif dictionary below
    thk : int, required
        _description_, thickness of the surficial layer, used for naming scheme only
    """   
    #
    nam2tif = {'GeomorphologyDEM1': os.path.join(indir,'txgam_dem.tif'),
        'BaseElev_Layer1_Seymour_Trinity': os.path.join(indir,'BaseElev_Layer1_Seymour_Trinity.tif'), # alluvium and seymour
        'GeomorphologyDEM1_200ft': os.path.join(indir,f'constant_thick_{thk}ft.tif'), # bottom of surficial layer
        'BaseElev_Layer2_TopClearFork': os.path.join(indir,'BaseElev_Layer2_TopClearFork.tif'),
        'BaseElev_Layer3_TopLeuders': os.path.join(indir,'BaseElev_Layer3_TopLeuders.tif'),
        'BaseElev_Layer4_BaseColemanJunction':os.path.join(indir,'baseelev_ly4.tif'),
        'BaseElev_Layer5_TopBreckenridge': os.path.join(indir,'baseelev_ly5.tif'),
        'BaseElev_Layer6_TopHomeCreek': os.path.join(indir,'baseelev_ly6.tif'),
        'BaseElev_Layer7_TopPaloPinto': os.path.join(indir,'baseelev_ly7.tif'),
        'BaseElev_Reef': os.path.join(indir,'BaseElev_Reef.tif'),
        'BaseElev_Layer8_TopDogBend': os.path.join(indir,'baseelev_ly8.tif'),
        'BaseElev_Layer9_TopMarbleFalls': os.path.join(indir,'baseelev_ly9.tif')}

    out_dir = os.path.join('gis','input_shapefiles','raster_botms_pts')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for nam in nam2tif.keys():
        # tif = lay2tif[lay]
        raster = nam2tif[nam]
        rasobj = rasterio.open(raster)

        gdf = spatialpy.utils.raster2pts(rasobj, column='botm')
        gdf = gdf.loc[gdf['botm'] < 1e30]
        #gdf.to_crs(gam_crs,inplace=True)

        gdf.to_file(os.path.join(out_dir,f'{nam}.pts.shp'))
        print(f'----- wrote {nam} to shapefile -----')


def write_swb_recharge_rasters():
    r_d = os.path.join("data","rch_arrays_ft_d")
    grd = gpd.read_file(os.path.join('gis','input_shapefiles','ctgam.gridsizes','mile_2276.grid.shp')) # load grid shapefile
    grd = grd[['node', 'row', 'column','geometry']] 

    out_dir = os.path.join('gis','input_rasters','interped_mile_rasters')
    
    rch_avg = np.zeros_like(np.loadtxt(os.path.join(r_d,os.listdir(r_d)[0])))
    for file in os.listdir(r_d):
        if file.endswith(('.asc')):
            print(file)
            rgrd = grd.copy()
            arr = np.loadtxt(os.path.join(r_d,file))
            rch_avg = np.add(rch_avg,arr)
            rgrd.loc[:,file.split('.')[0]] = np.reshape(arr,
                                                        (grd.row.max()*grd.column.max()))
            
            xmin, ymin, xmax, ymax = grd.total_bounds
            rgrd.geometry = grd.geometry.centroid

            ulc = (xmin,ymax)
            lrc = (xmax,ymin)

            ml = spatialpy.interp2d(rgrd,file.split('.')[0],res=5280,ulc=ulc,lrc=lrc) # 5280 ft = 1 mile, THIS IS HARD CODED NEED TO CHANGE FOR OTHER RESOLUTIONS IF WE GO THERE
            ras = ml.interpolate_2D()
            
            
            raspth = os.path.join(out_dir,file.split('.')[0]+'_1mi.tif')
            ml.write_raster(ras,path=raspth)
            print(f'*** Finshed interpolation, writing {file.split(".")[0]} to new grid.')
    rch_avg /= len(os.listdir(r_d))
    rgrd = grd.copy()
    rgrd.loc[:,file.split('.')[0]] = np.reshape(rch_avg,
                                                (grd.row.max()*grd.column.max()))
    
    xmin, ymin, xmax, ymax = grd.total_bounds
    rgrd.geometry = grd.geometry.centroid

    ulc = (xmin,ymax)
    lrc = (xmax,ymin)

    ml = spatialpy.interp2d(rgrd,file.split('.')[0],res=5280,ulc=ulc,lrc=lrc) # 5280 ft = 1 mile, THIS IS HARD CODED NEED TO CHANGE FOR OTHER RESOLUTIONS IF WE GO THERE
    ras = ml.interpolate_2D()
    
    
    raspth = os.path.join(out_dir,'rch_mean_1mi.tif')
    ml.write_raster(ras,path=raspth)
    print(f'*** Finshed interpolation of mean recharge, writing to new grid.')

    
def interp_ras_bots2grid_resolution():
    
    # make a directory to store the interpolated rasters:
    out_ras = os.path.join('gis','input_rasters','interped_mile_rasters')
    if not os.path.exists(out_ras):
        os.mkdir(out_ras)
    
    # make directory for grid shapefiles:
    out_shp = os.path.join('gis','input_shapefiles','mile_grd_shps')
    if not os.path.exists(out_shp):
        os.makedirs(out_shp)
       
    # directory where rasters have been converted to point data:
    indir =  os.path.join('gis','input_shapefiles','raster_botms_pts')
    
    # load grid shapefile:
    #fpth = os.path.join('gis','input_shapefiles','ctgam.gridsizes','mile_2276.grid.shp')
    #redefine_projection(fpth=fpth,toepsg=2276)
    grd = gpd.read_file(os.path.join('gis','input_shapefiles','ctgam.gridsizes','mile_2276.grid.shp')) # load grid shapefile
    grd = grd[['node', 'row', 'column','geometry']]
    if "nodenumber" not in grd.columns:
        if "node" in grd.columns:
            grd.loc[:,"nodenumber"] = grd.pop("node")
        else:
            raise Exception("'nodenumber' missing")
 
    # loop through each point dataset and interpolate to grid resolution:
    for file in os.listdir(indir):
        if file.endswith(('.shp')):
            print(file)
            pts = gpd.read_file(os.path.join(indir,file))
            # reset pts crs:
            pts = pts.to_crs(grd.crs)
            j = gpd.sjoin(grd,pts,how='left')
            j = j.sort_values('nodenumber')
            j = j.drop(columns='geometry')
            df = j.groupby(['nodenumber','row','column']).median()
            df = df.reset_index()
            df = df.drop(columns='index_right')
            ngrd = grd.merge(df,on='nodenumber',how='left')
            ngrd.botm.loc[ngrd.botm.isna()] = 1e30
            ngrd.to_file(os.path.join('gis','input_shapefiles','mile_grd_shps',file[:-8]+'.grd.shp'))
            npts = ngrd.copy()
            npts['geometry'] = ngrd.centroid
            npts = npts[['botm','geometry']]
            xmin, ymin, xmax, ymax = ngrd.total_bounds
            ulc = (xmin,ymax)
            lrc = (xmax,ymin)

            ml = spatialpy.interp2d(npts,'botm',res=5280,ulc=ulc,lrc=lrc) # 5280 ft = 1 mile, THIS IS HARD CODED NEED TO CHANGE FOR OTHER RESOLUTIONS IF WE GO THERE
            ras = ml.interpolate_2D()
            
            ras[ras>1e10] = np.nan
            
            raspth = os.path.join(out_ras,file[:-8]+'_1mi.tif')
            ml.write_raster(ras,path=raspth)
            print(f'*** Finshed interpolation, writing {file[:-8]} to new grid.')


def resample_rasters_2_gridsize(thk=200,resolution=(5280,5280)):
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    
    # make a directory to store the interpolated rasters:
    out_ras = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(resolution[0])}_ft')
    if not os.path.exists(out_ras):
        os.mkdir(out_ras)
    
    in_ras_dir = os.path.join('gis','input_rasters','Structural_Surfaces_Clean')
    
    nam2tif = {'dem': os.path.join(in_ras_dir,'surfelev_mosaic.tif'),
        'BaseElev_Layer1_Seymour_Trinity': os.path.join(in_ras_dir,'BaseElev_Layer1_Seymour_Trinity.tif'), # alluvium and seymour
        #f'primary_aq_{thk}ft_thick': os.path.join(in_ras_dir,f'surf_realtive_cnst_ft_thk_{thk}ft.tif'), # bottom of surficial layer
        'BaseElev_Layer3_ClearFork': os.path.join(in_ras_dir,'BaseElev_Layer2_TopClearFork.tif'),
        'BaseElev_Layer4_WichitaAlbany': os.path.join(in_ras_dir,'BaseElev_Layer3_TopLeuders.tif'),
        'BaseElev_Layer5_UpperCisco':os.path.join(in_ras_dir,'baseelev_ly4.tif'),
        'BaseElev_Layer6_LowerCisco': os.path.join(in_ras_dir,'baseelev_ly5.tif'),
        'BaseElev_Layer7_CanyonGroup': os.path.join(in_ras_dir,'baseelev_ly6.tif'),
        'BaseElev_Layer8_PaloPinto': os.path.join(in_ras_dir,'baseelev_ly7.tif'),
        'BaseElev_Layer9_Reef': os.path.join(in_ras_dir,'BaseElev_Reef.tif'),
        'BaseElev_Layer10_StrawnAtoka': os.path.join(in_ras_dir,'baseelev_ly8.tif'),
        'BaseElev_Layer11_MarbleFalls': os.path.join(in_ras_dir,'baseelev_ly9.tif')}
    
    
    files = [f for f in os.listdir(in_ras_dir) if f.endswith('.tif')]
    
    # filter files to needed rasters only:
    remove_lst = ['BaseElev_Layer1_a_Seymour.tif',
                'BaseElev_Layer1_b_Trinity.tif',
                'botelev_hosston.tif',
                'botelev_seymour.tif',
                'seymour_resamp.tif',
                ]
    # if file in remove list, remove it from files list:

    
    files = [f for f in files if f not in remove_lst]
    
    # Determine the common extent (left, bottom, right, top) and the resolution
    # For example:
    left, bottom, right, top = 1468894.0, 6327767.0, 2313694.0, 7489367.0  
    width = int((right - left) / resolution[0])
    height = int((top - bottom) / resolution[1])
    
    # Calculate the transform and dimensions for the new resolution
    new_transform, new_width, new_height = calculate_default_transform(
        'EPSG:2276', 'EPSG:2276', width, height, 
        left=left, bottom=bottom, right=right, top=top, 
        resolution=resolution)
    seymo_trin_transform, seymo_trin_width, seymo_trin_height = calculate_default_transform(
        'EPSG:2276', 'EPSG:2276', width, height,
        left=left, bottom=bottom, right=right, top=top,
        resolution=(5280.0,5280.0))
    # Resample each raster
    for key in nam2tif.keys():
        file = nam2tif[key]
        fnm = key+'_1mi.tif'
        inpth = file
        outpth = os.path.join(out_ras,fnm)
        # if key.startswith('BaseElev_Layer1_Seymour'):
        #     resample_raster(inpth, outpth, seymo_trin_transform, width, height)
        # else:
        resample_raster(inpth, outpth, new_transform, width, height)

    # # rch_files = [f for f in os.listdir(out_ras) if f.endswith('.tif') and f.startswith('rch')]
    # # for file in rch_files:
    # #     inpth = os.path.join(out_ras,file)
    # #     outpth = os.path.join(out_ras,'upd_'+file)
    # #     resample_raster(inpth, outpth, new_transform, new_width, new_height)
    #
    # # load model area shapefile:
    # fpth = os.path.join('gis','input_shapefiles','ctgam_model_cel_extent.shp')
    # model_area = gpd.read_file(fpth)
    # model_area = model_area.to_crs('EPSG:2276')
    # model_area.to_file(os.path.join('gis','input_shapefiles','ctgam_model_cel_extent.shp'))
    #
    # out_clp_dir = os.path.join('gis','input_rasters','clipped_mile_rasters')
    # if not os.path.exists(out_clp_dir):
    #     os.makedirs(out_clp_dir)
    #
    # def create_mask_from_raster(mask_raster_path):
    #     with rasterio.open(mask_raster_path) as mask_raster:
    #         mask_data = mask_raster.read(1)  # Read the first band
    #         mask_data = np.where(mask_data<1e10,1,np.nan)
    #
    #         # Create a mask where 1s are True and 0s are False
    #         return mask_data
    #
    # def apply_mask_to_raster(raster_path, mask, output_raster_path):
    #     with rasterio.open(raster_path) as src:
    #         # Read the raster data
    #         data = src.read()
    #         # Apply the mask
    #         masked_data = data * mask
    #         # Write the masked raster to a new file
    #         with rasterio.open(output_raster_path, "w", **src.meta) as dest:
    #             dest.write(masked_data)
    #
    # # Path to the raster that will be used as a mask
    # mask_raster_path = os.path.join(out_ras,'dem_1mi.tif')
    #
    # # Generate the mask
    # mask = create_mask_from_raster(mask_raster_path)
    #
    # # Apply the mask to each raster and save the result
    # ras_files = [f for f in os.listdir(out_ras) if f.endswith('.tif')]
    # ras_files = [f for f in ras_files if f != 'dem_1mi.tif']
    #
    # for file in ras_files:
    #     raster_path = os.path.join(out_ras, file)
    #     output_raster_path = os.path.join(out_clp_dir, file)
    #     try:
    #         apply_mask_to_raster(raster_path, mask, output_raster_path)
    #     except:
    #         print(f'Error in file: {file}')
    # # copy dem to clipped folder:
    # shutil.copy(mask_raster_path, os.path.join(out_clp_dir,'dem_1mi.tif'))
 

def extrapolate_rasters(raster_path,out_ras):
    from scipy.interpolate import Rbf
    
    fnm = os.path.basename(raster_path)
    
    with rasterio.open(raster_path) as src:
            data = src.read(1)  # Read the first band
            nodata = src.nodata  # Get the no-data value
            
            if np.isnan(nodata):
                mask = ~np.isnan(data)
            else:
                mask = data < 1e29

            finite_mask = np.isfinite(data) & mask
            
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
            x_known = x[finite_mask]
            y_known = y[finite_mask]
            z_known = data[finite_mask]
            
            if np.any(np.isnan(z_known)) or np.any(np.isinf(z_known)):
                raise ValueError("Data contains NaNs or Infs, which are not allowed.")

            # Using Radial Basis Function to extrapolate
            rbf = Rbf(x_known, y_known, z_known, function='linear')  # Try different functions like 'multiquadric'
            data_extrapolated = rbf(x, y)

            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(data, cmap='viridis')
            # ax[0].set_title("Original Data")
            # ax[1].imshow(data_extrapolated, cmap='viridis')
            # ax[1].set_title("Extrapolated Data")
        
            data[~mask] = data_extrapolated[~mask]

            new_meta = src.meta.copy()
            new_meta.update({
                'nodata': None  # or an appropriate no-data value
            })

            # Write the interpolated data to a new raster file
            with rasterio.open(os.path.join(out_ras,fnm), 'w', **new_meta) as dst:
                dst.write(data, 1)

   
def find_and_adjust_elevation_errors(resolution=(1320.0,1320.0)):

    # input raster directory:
    #rasdir = os.path.join('gis', 'input_rasters','clipped_mile_rasters')
    rasdir = os.path.join('gis', 'input_rasters', f'rescaled_rasters_resolution_{int(resolution[0])}_ft')
    
    # 1. assure that all layer bottoms are below ground surface:
    
    # read in surface elevation:
    #surf = os.path.join('gis','input_rasters','clipped_mile_rasters',f'dem_1mi.tif')
    surf = os.path.join('gis', 'input_rasters',  f'rescaled_rasters_resolution_{int(resolution[0])}_ft', f'dem_1mi.tif')
    raster = os.path.join(surf)
    surfras = rasterio.open(raster)
    # store raster values in array:
    gs = surfras.read(1)
    gs[gs > 100000] = np.nan
    gs[gs < -100000] = np.nan
    
    itrtif = {
            1: 'BaseElev_Layer1_Seymour_Trinity_1mi.tif',
            2: 'BaseElev_Layer3_ClearFork_1mi.tif',
            3: 'BaseElev_Layer4_WichitaAlbany_1mi.tif',
            4: 'BaseElev_Layer5_UpperCisco_1mi.tif',
            5: 'BaseElev_Layer6_LowerCisco_1mi.tif',
            6: 'BaseElev_Layer7_CanyonGroup_1mi.tif',
            7: 'BaseElev_Layer8_PaloPinto_1mi.tif',
            8: 'BaseElev_Layer9_Reef_1mi.tif',
            9: 'BaseElev_Layer10_StrawnAtoka_1mi.tif',
            10: 'BaseElev_Layer11_MarbleFalls_1mi.tif'
            }
    
    for lay in itrtif.keys():
        # read in raster layer:
        tif = itrtif[lay]
        raster = os.path.join(rasdir,tif)
        rasobj = rasterio.open(raster)
        # store raster values in array:
        array = rasobj.read(1)
        array[array > 100000] = np.nan
        array[array < -100000] = np.nan
        
        raster_path = os.path.join(surf)
        with rasterio.open(raster_path, 'r+') as src:
            # store raster values in array:
            gs = src.read(1)
        
            # check if there are locations where array is greater than gs:
            # if there are, then set array value to gs value:
            depth_mod = 1
            if np.any(array >= gs):
                print(f'*** Found {np.sum(array>=gs)} cells in {tif} that are above ground surface. ***')
                #array[array>=gs] = gs[array>=gs] - depth_mod
                if lay == 1:
                    gs[array>=gs] = array[array>=gs] + 10
                else:
                    gs[array>=gs] = array[array>=gs] + 1
                depth_mod += 1
            
            # replace nans in array with 1e30:
            array = np.where(np.isnan(array), 1e30, array)
            gs = np.where(np.isnan(gs), 1e30, gs)
            
            # transform = rasobj.transform
            # new_dataset = rasterio.open(os.path.join(out_ras, tif), 'w',
            #                             driver='GTiff',
            #                             height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
            #                             crs=rasobj.crs, transform=transform, nodata=1e30)
            # new_dataset.write(array, 1)
            # new_dataset.close()
            
            src.write(gs,1)
            src.close()

            print(f'*** Finished exporting new {tif} -account for surface elevations ***')
            rasobj.close()
    
    # 2. Check to make sure all layer elevations are above model bottom:
    out_bots = os.path.join('gis','input_rasters','model_bottom_corrected_mile_rasters')
    if not os.path.exists(out_bots):
        os.mkdir(out_bots)   
    
    # 2.1 make sure min thickness of seymour and trinity is 20 ft:
    sytrin = os.path.join(rasdir,'BaseElev_Layer1_Seymour_Trinity_1mi.tif')
    dem = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(resolution[0])}_ft','dem_1mi.tif')
    sytrin_ras = rasterio.open(sytrin)
    dem_ras = rasterio.open(dem)
    sytrin_arr = sytrin_ras.read(1)
    dem_arr = dem_ras.read(1)
    dem_arr[dem_arr > 100000] = np.nan
    sytrin_arr[sytrin_arr > 100000] = np.nan
    
    #diff_elev = dem_arr - sytrin_arr
    #sytrin_arr = np.where(diff_elev < 20, dem_arr - 20, sytrin_arr)
    # replace nans with 1e30:
    sytrin_arr = np.where(np.isnan(sytrin_arr), 1e30, sytrin_arr)
    dem_ras.close()
    new_dataset = rasterio.open(os.path.join('gis','input_rasters','model_bottom_corrected_mile_rasters', 'BaseElev_Layer1_Seymour_Trinity_1mi.tif'), 'w',
                                driver='GTiff',
                                height=sytrin_arr.shape[0], width=sytrin_arr.shape[1], count=1, dtype=sytrin_arr.dtype,
                                crs=sytrin_ras.crs, transform=sytrin_ras.transform, nodata=1e30)
    new_dataset.write(sytrin_arr, 1)
    new_dataset.close()
    sytrin_ras.close()
    
    # 2.2
    itrtif = {
            2: 'BaseElev_Layer3_ClearFork_1mi.tif',
            3: 'BaseElev_Layer4_WichitaAlbany_1mi.tif',
            4: 'BaseElev_Layer5_UpperCisco_1mi.tif',
            5: 'BaseElev_Layer6_LowerCisco_1mi.tif',
            6: 'BaseElev_Layer7_CanyonGroup_1mi.tif',
            7: 'BaseElev_Layer8_PaloPinto_1mi.tif',
            8: 'BaseElev_Layer9_Reef_1mi.tif',
            9: 'BaseElev_Layer10_StrawnAtoka_1mi.tif',
            }
    
    marble = os.path.join(rasdir,'BaseElev_Layer11_MarbleFalls_1mi.tif')
    raster = os.path.join(marble)
    msurf = rasterio.open(raster)
    model_bot = msurf.read(1)
    model_bot[model_bot > 100000] = np.nan
    model_bot[model_bot < -100000] = np.nan
    
    for lay in itrtif.keys():
        # read in raster layer:
        tif = itrtif[lay]
        raster = os.path.join(rasdir,tif)
        rasobj = rasterio.open(raster)
        # store raster values in array:
        array = rasobj.read(1)
        array[array > 100000] = np.nan
        array[array < -100000] = np.nan
        
        # check if there are locations where model_bot is greater than array:
        # if there are, then set array value to model_bot value:
        depth_mod = 10
        if np.any(array <= model_bot):
            print(f'*** Found {np.sum(array<=model_bot)} cells in {tif} that are below model bottom. ***')
            array[array<=model_bot] = model_bot[array<=model_bot] + depth_mod
            depth_mod -= 1
            # plot array and locations where array is less than model_bot:
            #fig = plt.figure(figsize=(10,10))
            #plt.imshow(np.where(array<=model_bot,1,0))

            # replace nans in array with 1e30:
            array = np.where(np.isnan(array), 1e30, array)

            transform = rasobj.transform
            new_dataset = rasterio.open(os.path.join(out_bots, tif), 'w',
                                        driver='GTiff',
                                        height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
                                        crs=rasobj.crs, transform=transform, nodata=1e30)
            new_dataset.write(array, 1)
            new_dataset.close()
            print(f'*** Finished exporting new {tif} -accounted for model bottom***')
        else:
            shutil.copy(os.path.join(rasdir,tif), os.path.join(out_bots,tif))   
            print(f'*** Exporting new {tif} - no issue with model bottoms***') 
        rasobj.close()
        
    # 3. Where top of reef is greater than top home creek and top of palo pinto, set home creek and palo pinto to 1e30:
    out_reef = os.path.join('gis','input_rasters','reef_corrected_mile_rasters')
    if not os.path.exists(out_reef):
        os.mkdir(out_reef)      
    
    itrtif = {
            5: 'BaseElev_Layer7_CanyonGroup_1mi.tif',
            6: 'BaseElev_Layer8_PaloPinto_1mi.tif'
            }
    
    reef = os.path.join(out_bots,'BaseElev_Layer9_Reef_1mi.tif')
    raster = os.path.join(reef)
    rsurf = rasterio.open(raster)
    reef_top = rsurf.read(1)
    reef_top[reef_top > 100000] = np.nan
    reef_top[reef_top < -100000] = np.nan
    
    for lay in itrtif.keys():
        # read in raster layer:
        tif = itrtif[lay]
        raster = os.path.join(out_bots,tif)
        rasobj = rasterio.open(raster)
        # store raster values in array:
        array = rasobj.read(1)
        array[array > 100000] = np.nan
        array[array < -100000] = np.nan
        
        # check if there are locations where top of reef is greater than elevations of palo pinto and home creek:
        # if there are, then set array value 1e30:
        
        if np.any(array <= reef_top):
            print(f'*** Found {np.sum(array<=reef_top)} cells in {tif} that are below top of reef formation. ***')
            array[array<=reef_top] = 1e30
            # plot array and locations where array is less than model_bot:
            #fig = plt.figure(figsize=(10,10))
            #plt.imshow(np.where(array<=model_bot,1,0))

            # replace nans in array with 1e30:
            array = np.where(np.isnan(array), 1e30, array)

            transform = rasobj.transform
            new_dataset = rasterio.open(os.path.join(out_reef, tif), 'w',
                                        driver='GTiff',
                                        height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
                                        crs=rasobj.crs, transform=transform, nodata=1e30)
            new_dataset.write(array, 1)
            new_dataset.close()
            print(f'*** Finished exporting new {tif} ***')
        rasobj.close()
    

def copy_processed_rasters_to_input_rasters(resolution=(1320.0,1320.0)):
    """
    copy processed rasters to input_rasters directory
    :return:
    """
    # copy processed rasters to input_rasters directory:
    in_ras = os.path.join('gis','input_rasters','processed_mile_rasters')
    if not os.path.exists(in_ras):
        os.mkdir(in_ras)
    
    # copy processed bottom rasters to processed ras directory:
    bot_ras = os.path.join('gis','input_rasters','model_bottom_corrected_mile_rasters')
    for f in os.listdir(bot_ras):
        if f.endswith('.tif'):
            shutil.copy(os.path.join(bot_ras,f), os.path.join(in_ras,f))
    
    # copy processed reef rasters to processed ras directory:
    reef_ras = os.path.join('gis','input_rasters','reef_corrected_mile_rasters')
    for f in os.listdir(reef_ras):
        if f.endswith('.tif'):
            shutil.copy(os.path.join(reef_ras,f), os.path.join(in_ras,f))   
            
    # copy surface elevation to in_ras:
    #surf = os.path.join('gis', 'input_rasters', 'clipped_mile_rasters', 'dem_1mi.tif')
    surf = os.path.join('gis','input_rasters', f'rescaled_rasters_resolution_{int(resolution[0])}_ft','dem_1mi.tif')
    shutil.copy(surf, os.path.join(in_ras,'dem_1mi.tif'))

    #mod_bot = os.path.join('gis', 'input_rasters', 'clipped_mile_rasters', 'BaseElev_Layer11_MarbleFalls_1mi.tif')
    mod_bot = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(resolution[0])}_ft','BaseElev_Layer11_MarbleFalls_1mi.tif')
    shutil.copy(mod_bot, os.path.join(in_ras,'BaseElev_Layer11_MarbleFalls_1mi.tif'))


def adjust_layers_intersect_cnstthkly(thk=150,resolution=(1320.0,1320.0)):
    """
    make a surface for the bottom of the surficial layer
    :param thk: thickness beneath top elevation
    :return:
    """
    # make a directory to store the deep bottom rasters:
    deep_ras_dir = os.path.join('gis', 'input_rasters', 'Structural_Surfaces_Clean_1mi_final')
    if not os.path.exists(deep_ras_dir):
        os.mkdir(deep_ras_dir)
    
    rasdir = os.path.join('gis','input_rasters','processed_mile_rasters')
    
    # copy constant thickness raster into rasdir:
    #con_thk_ras = os.path.join('gis','input_rasters','clipped_mile_rasters',f'primary_aq_{thk}ft_thick_1mi.tif')
    con_thk_ras = os.path.join('gis', 'input_rasters', f'rescaled_rasters_resolution_{int(resolution[0])}_ft', f'primary_aq_{thk}ft_thick_1mi.tif')
    shutil.copy(con_thk_ras, os.path.join(rasdir,f'primary_aq_{thk}ft_thick_1mi.tif'))

    # read in ground surface raster:
    raster = os.path.join(rasdir, 'dem_1mi.tif')
    rasobj = rasterio.open(raster)
    dem = rasobj.read(1)
    dem[dem > 100000] = np.nan
    dem[dem < -100000] = np.nan


    # read in constat thickness raster:
    raster_cs = os.path.join(rasdir, f'primary_aq_{thk}ft_thick_1mi.tif')
    rasobj = rasterio.open(raster_cs)
    cs_array = rasobj.read(1)
    cs_array[cs_array > 100000] = np.nan
    cs_array[cs_array < -100000] = np.nan
    
    # if constant thickness raster elevation is greater than seymour/trinity bottom, set to seymour/trinity bottom:
    # SRM - not understanding statement above,
    #  where trin/sey is present you are setting primary aquifer base to be 200 ft below trinity/seymopur base
    raster = os.path.join(rasdir,'BaseElev_Layer1_Seymour_Trinity_1mi.tif')
    rasobj = rasterio.open(raster)
    sts_array = rasobj.read(1)
    sts_array[sts_array > 100000] = np.nan
    sts_array[sts_array < -100000] = np.nan
    print(f'*** Found {np.sum(cs_array>=sts_array)} cells in {raster_cs} that are above seymour/trinity bottom. ***') # seymour/trinity below constan

    cs_array = np.where(np.abs(sts_array)>0,sts_array-thk,cs_array) # thk unit under all trintiy seymour

    # calculate basic stats on cs_array:
    thk_chk = sts_array-cs_array
    
    # where thk_chk is less than 50, add the difference to the constant thickness raster:
    #cs_array[thk_chk<50] = cs_array[thk_chk<50] - (50-thk_chk[thk_chk<50])
    
    # write new array to raster:
    cs_array = np.where(np.isnan(cs_array), 1e30, cs_array)

    transform = rasobj.transform
    new_dataset = rasterio.open(os.path.join(deep_ras_dir, f'primary_aq_{thk}ft_thick_1mi.tif'), 'w',
                                driver='GTiff',
                                height=cs_array.shape[0], width=cs_array.shape[1], count=1, dtype=cs_array.dtype,
                                crs=rasobj.crs, transform=transform, nodata=1e30)
    new_dataset.write(cs_array, 1)
    new_dataset.close()

    # layers that intersect the constant thicknes layer:
    lay2tif = {
            2: 'BaseElev_Layer3_ClearFork_1mi.tif',
            3: 'BaseElev_Layer4_WichitaAlbany_1mi.tif',
            4: 'BaseElev_Layer5_UpperCisco_1mi.tif',
            5: 'BaseElev_Layer6_LowerCisco_1mi.tif',
            6: 'BaseElev_Layer7_CanyonGroup_1mi.tif',
            7: 'BaseElev_Layer8_PaloPinto_1mi.tif',
            8: 'BaseElev_Layer9_Reef_1mi.tif',
            9: 'BaseElev_Layer10_StrawnAtoka_1mi.tif',
            10: 'BaseElev_Layer11_MarbleFalls_1mi.tif'
        }
    
    # read in seymour/trintiy adjust constant thick layer:
    cnst_thk = os.path.join(deep_ras_dir, f'primary_aq_{thk}ft_thick_1mi.tif')
    rasobj = rasterio.open(cnst_thk)
    conthk = rasobj.read(1)
    conthk[conthk > 100000] = np.nan
    conthk[conthk < -100000] = np.nan
        
    depth_mod = 0.05
    #keyz = list(lay2tif.keys())[::-1]
    for lay in lay2tif.keys():
        # read in raster layer:
        tif = lay2tif[lay]
        raster = os.path.join(rasdir,tif)
        rasobj = rasterio.open(raster)
        array = rasobj.read(1)
        array[array > 100000] = np.nan
        array[array < -100000] = np.nan
        print(lay)
        print(array.shape)

        # check if there are locations where model_bot is greater than the bottom elevation of constant thickness layer:
        # if there are, then set array value to model_bot value to something super thin:
        
        if np.any(array >= conthk):
            print(f'*** Found {np.sum(array>=conthk)} cells in {tif} that are above the constant thickness ly. ***')
            array = np.where(array>=conthk, conthk - depth_mod, array)
            #array[array>=conthk] = conthk[array>=conthk] - depth_mod
        
        # write new array to raster:
        array = np.where(np.isnan(array), 1e30, array)

        transform = rasobj.transform
        new_dataset = rasterio.open(os.path.join(deep_ras_dir, tif), 'w',
                            driver='GTiff',
                            height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
                            crs=rasobj.crs, transform=transform, nodata=1e30)
        new_dataset.write(array, 1)
        new_dataset.close()
    
    # copy seymour/trinty and land surface raster to deep_ras_dir:
    grnd_surf = os.path.join(rasdir,'dem_1mi.tif')
    seytrin = os.path.join(rasdir,'BaseElev_Layer1_Seymour_Trinity_1mi.tif')
    shutil.copyfile(grnd_surf, os.path.join(deep_ras_dir,'dem_1mi.tif'))
    shutil.copyfile(seytrin, os.path.join(deep_ras_dir,'BaseElev_Layer1_Seymour_Trinity_1mi.tif'))
    

def extrapolate_clip_rasters_to_model_area():
    """
    clip rasters to model area
    :return:
    """
    # make a directory to store the clipped rasters:
    clip_ras_dir = os.path.join('gis', 'input_rasters', 'Structural_Surfaces_Clean_1mi_final_clp')
    if not os.path.exists(clip_ras_dir):
        os.mkdir(clip_ras_dir)
    extrap_ras_dir = os.path.join('gis', 'input_rasters', 'Structural_Surfaces_Clean_1mi_final_extrap')
    if not os.path.exists(extrap_ras_dir):
        os.mkdir(extrap_ras_dir)
    
    final_dir = os.path.join('gis', 'input_rasters', 'Structural_Surfaces_4_model_build')
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
    
    # input raster directory:
    rasdir = os.path.join('gis', 'input_rasters','Structural_Surfaces_Clean_1mi_final')
    ras_files = [f for f in os.listdir(rasdir) if f.endswith('.tif')]
    
    # rasters to extrapolate:
    extrap = ras_files.copy()
    # remove seymour and trinity from list:
    extrap.remove('BaseElev_Layer1_Seymour_Trinity_1mi.tif')
    
    for ras in extrap:
        print(ras)
        extrapolate_rasters(os.path.join(rasdir,ras),extrap_ras_dir)
    
    lst_2_clp = [f for f in os.listdir(extrap_ras_dir) if f.endswith('.tif')]
    lst_2_clp = ['BaseElev_Layer3_ClearFork_1mi.tif',
                'BaseElev_Layer4_WichitaAlbany_1mi.tif',
                'BaseElev_Layer5_UpperCisco_1mi.tif',
                'BaseElev_Layer6_LowerCisco_1mi.tif',
                'BaseElev_Layer7_CanyonGroup_1mi.tif',
                'BaseElev_Layer8_PaloPinto_1mi.tif',
                'BaseElev_Layer9_Reef_1mi.tif',
                'BaseElev_Layer11_MarbleFalls_1mi.tif']
    for file in lst_2_clp:
        layer = file.split('_')[1].split('Layer')[1]
        shpext = os.path.join('gis','input_shapefiles','raster_layer_extents',f'lyext_{layer}.shp')
        shpext = gpd.read_file(shpext)
        
        ras_pth = os.path.join(extrap_ras_dir,file)
        rasobj = rasterio.open(ras_pth)
        
        outpth = os.path.join(clip_ras_dir,file)
        
        spatialpy.utils.clip_raster(rasobj, shpext, outpth)
        print(f'*** Finished clipping {file} to layer extent ***')
        rasobj.close()
        
        # read in clipped raster and reaplce zero values with 1e30:
        rasobj = rasterio.open(outpth)
        array = rasobj.read(1)
        array = np.where(array==0,1e30,array)
        transform = rasobj.transform
        basnm = os.path.basename(outpth)
        new_dataset = rasterio.open(os.path.join(final_dir,basnm), 'w',
                                    driver='GTiff',
                                    height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
                                    crs=rasobj.crs, transform=transform, nodata=1e30)
        new_dataset.write(array, 1)
        new_dataset.close()
        rasobj.close()
        
    # get list of tifs in extrap_ras_dir:
    ras_files = [f for f in os.listdir(extrap_ras_dir) if f.endswith('.tif')]
    fin_files = [f for f in os.listdir(final_dir) if f.endswith('.tif')]
    
    for file in ras_files:
        if file not in fin_files:
            shutil.copy(os.path.join(extrap_ras_dir,file), os.path.join(final_dir,file))
    # copy seymour/trinty and land surface raster to final_dir:
    shutil.copy(os.path.join(rasdir,'BaseElev_Layer1_Seymour_Trinity_1mi.tif'), os.path.join(final_dir,'BaseElev_Layer1_Seymour_Trinity_1mi.tif'))
        

def adjust_dem_elevation_errors(resolution=(5280.0,5280.0)):
    
    # elev path:
    surf = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(resolution[0])}_ft','dem_1mi.tif')
    
    ras_dir = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(resolution[0])}_ft')
    rasfiles = [f for f in os.listdir(ras_dir) if f.endswith('.tif')]
    
    # remove files with dem in name:
    rasfiles = [f for f in rasfiles if 'dem' not in f]
    
    for tif in rasfiles:
        # read in raster layer:
        raster = os.path.join(ras_dir,tif)
        rasobj = rasterio.open(raster)
        # store raster values in array:
        array = rasobj.read(1)
        array[array > 100000] = np.nan
        array[array < -100000] = np.nan
        
        raster_path = os.path.join(surf)
        with rasterio.open(raster_path, 'r+') as src:
            # store raster values in array:
            gs = src.read(1)
        
            # check if there are locations where array is greater than gs:
            # if there are, then set array value to gs value:
            depth_mod = 1
            if np.any(array >= gs):
                print(f'*** Found {np.sum(array>=gs)} cells in {tif} that are above ground surface. ***')
                #array[array>=gs] = gs[array>=gs] - depth_mod
                if tif == 'BaseElev_Layer1_Seymour_Trinity_1mi.tif':
                    gs[array>=gs] = array[array>=gs] + 10
                else:
                    gs[array>=gs] = array[array>=gs] + 1
                depth_mod += 1
            
            # replace nans in array with 1e30:
            array = np.where(np.isnan(array), 1e30, array)
            gs = np.where(np.isnan(gs), 1e30, gs)
            
            src.write(gs,1)
            src.close()

            print(f'*** Finished exporting new {tif} -account for surface elevations ***')
            rasobj.close()
            
          
def prep_all_raster_bottoms(thk=200, res=(5280.0,5280.0)):
    
    # 1. remove 0 values from all rasters and reproject:
    in_raster_dir = os.path.join('gis','input_rasters','intera_updated_tifs')
    trim_reproj_rasters(in_raster_dir=in_raster_dir)
    # 2. merge alluvium layers (Trinity and Seymour) into one layer:
    merge_alluvium_layers()
    resample_rasters_2_gridsize(thk=200,resolution=res)
    adjust_dem_elevation_errors(resolution=res) # instances where layer bottoms are higher than DEM (top)
    # 3. create constant thickness aquifer layer:
    in_surf_ras_pth = os.path.join('gis','input_rasters',f'rescaled_rasters_resolution_{int(res[0])}_ft','dem_1mi.tif')
    generate_constant_surficial_botm(thk=thk,in_surf_ras_pth=in_surf_ras_pth,res=res)
    # 4. Adjust elevation errors:
    find_and_adjust_elevation_errors(resolution=res)
    # 7. copy corrected rasters to new raster workspace:
    copy_processed_rasters_to_input_rasters(resolution=res)
    # 8. Modifications to constant thickness aquifer layer and layers that intersect it:
    adjust_layers_intersect_cnstthkly(thk=thk,resolution=res)
    # 9. linearly extrapolate to solve issues around edges
    extrapolate_clip_rasters_to_model_area()

