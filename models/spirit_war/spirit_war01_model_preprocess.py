# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:39:53 2025

@author: shjordan
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import rasterize
import os

   
# --------------------------------------------------
# Load and process the Leapfrog files to .tif format
# --------------------------------------------------
def process_grd_files():
    for file in grd_files:
        file_path = os.path.join('..','..','gis','input_ras','sw_ww',f'{file}')
        raster_name = os.path.basename(file_path).replace(" ", "_")
        print(f"Converting {raster_name} asc")
        # Open Raster
        with rasterio.open(file_path) as src:
            raster_data = src.read(1)
            profile = src.profile
            # Set to known CRS
            if profile.get('crs') is None:
                profile.update(crs='epsg:2265')
            
        
        # Save the cleaned raster to processed data folder
        output_path = os.path.join('..','..','gis','output_ras','sw_ww',f"{raster_name.replace('.asc','.tif')}")
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            # dst.write(cleaned_raster_smoothed.astype(profile['dtype']), 1)
            dst.write(raster_data.astype(profile['dtype']), 1)
        print(f'Successfully converted to .tif: {output_path}')
    
    
# -------------------------------------
# turn SW polygon into a boolean raster
# -------------------------------------
def turn_extent_to_array():
    # Load Spirit wood aquifer polygon
    gdf = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww',"sw_extent_SJ.shp")).to_crs(2265)
    
    # Open a template raster for correct sizing
    with rasterio.open(os.path.join('..','..','gis','output_ras','sw_ww', 'Hydro_Model__01_-_Bedrock_younger.tif')) as src:
        out_shape = src.shape
        transform = src.transform
    
    # Rasterize to binary polygon
    binary_array = rasterize(
        [(gdf.geometry.iloc[0], 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    return binary_array


# ---------------------------------------------------------
# Create continuous top and bottom surfaces for model input
# ---------------------------------------------------------
def create_model_surfaces():
    # Dictionaries to store raster arrays and their profiles
    tif_dict = {}
    profile_dict = {}
    
    # Loop through .grd files
    for file in grd_files:
        raster_name = file.replace(" ", "_").replace('.asc','.tif')
        file_path = os.path.join('..','..','gis','output_ras','sw_ww', raster_name)
        with rasterio.open(file_path) as src:
            raster_data = src.read(1)
            profile = src.profile
            nodata = profile.get('nodata')
        # Replace nodata values with NaN
        raster_data[raster_data == nodata] = np.nan
        base_name = os.path.basename(file_path)
        tif_dict[base_name] = raster_data
        profile_dict[base_name] = profile
        
    # --- Compute model_bottom ---
    # If Basal Clay exists (not NaN) use it; otherwise use Bedrock
    model_bottom = np.where(
        np.isnan(tif_dict['Hydro_Model__02_-_Basal_Clay_younger.tif']),
        tif_dict['Hydro_Model__01_-_Bedrock_younger.tif'],
        tif_dict['Hydro_Model__02_-_Basal_Clay_younger.tif']
        )
    
    # --- Compute model_top ---
    # Work from the top layer down, filling NaNs with subsequent layers
    model_top = np.where(
        np.isnan(tif_dict['Hydro_Model__08_-_Top_Clay_younger.tif']),
        tif_dict['Hydro_Model__07_-_Upper_Aquifer_younger.tif'],
        tif_dict['Hydro_Model__08_-_Top_Clay_younger.tif']
        )
    model_top = np.where(
        np.isnan(model_top),
        tif_dict['Hydro_Model__06_-_Upper_Clay_younger.tif'],
        model_top
        )
    model_top = np.where(
        np.isnan(model_top),
        tif_dict['Hydro_Model__05_-_Middle_Aquifer_younger.tif'],
        model_top
        )
    model_top = np.where(
        np.isnan(model_top),
        tif_dict['Hydro_Model__04_-_Middle_Clay_younger.tif'],
        model_top
        )
    model_top = np.where(
        np.isnan(model_top),
        tif_dict['Hydro_Model__03_-_Lower_Aquifer_younger.tif'],
        model_top
        )
    model_top = np.where(
        np.isnan(model_top),
        model_bottom,
        model_top
        )
    
    # --- Save model_top and model_bottom
    template_profile = profile_dict['Hydro_Model__03_-_Lower_Aquifer_younger.tif'].copy()
    template_profile.update({
            'dtype': 'float32',
            'count': 1
            }
        )
    
    # Enforce dtypes
    model_top = model_top.astype('float32')
    model_bottom = model_bottom.astype('float32')
    
    # Define output file paths
    model_top_path = os.path.join('..','..','gis','output_ras','sw_ww', 'model_top.tif')
    model_bottom_path = os.path.join('..','..','gis','output_ras','sw_ww', 'model_bottom.tif')
    
    # Write model_top raster
    with rasterio.open(model_top_path, 'w', **template_profile) as dst:
        dst.write(model_top, 1)
    
    # Write model_bottom raster
    with rasterio.open(model_bottom_path, 'w', **template_profile) as dst:
        dst.write(model_bottom, 1)
    
    print("Model surfaces have been created and saved successfully.")


# ----------------------------
# Create the top of SW Aquifer
# ----------------------------
def create_SW_layer():
    with rasterio.open(os.path.join('..','..','gis','output_ras','sw_ww','Hydro_Model__03_-_Lower_Aquifer_younger.tif')) as rst:
        raster_data = rst.read(1)
        nodata = rst.nodata
        profile = rst.profile
        
    sw_extent = turn_extent_to_array()
    
    raster_data = np.where(sw_extent==0,nodata,raster_data)
    plt.imshow(raster_data)
    
    # Write top of SW layer to new file
    template_profile = profile.copy()
    template_profile.update({
            'dtype': 'float32',
            'count': 1
            }
        )
    output_path = os.path.join('..','..','gis','output_ras','sw_ww', 'top_of_SW.tif')
    with rasterio.open(output_path, 'w', **template_profile) as rst:
        rst.write(raster_data, 1)


# ----------------------------------------
# Create confining layer between SW and WW
# ----------------------------------------
def create_confining_layer():
    # Read in middle clay layer
    with rasterio.open(os.path.join('..','..','gis','output_ras','sw_ww', 'Hydro_Model__04_-_Middle_Clay_younger.tif')) as rst:
        middle_clay = rst.read(1)
        nodata = rst.nodata
        profile = rst.profile    
        
    # Read in Bottom layer
    with rasterio.open(os.path.join('..','..','gis','output_ras','sw_ww','model_bottom.tif')) as rst:
        model_bot = rst.read(1)

    # Combine middle clay with middle aquifer
    clay_layer = middle_clay.copy()
    # clay_layer = np.where(middle_clay==nodata,middle_aq,middle_clay)
    clay_layer = np.where(clay_layer==nodata,model_bot,clay_layer)

    # Save layer
    template_profile = profile.copy()
    template_profile.update({
            'dtype': 'float32',
            'count': 1
            }
        )
    output_path = os.path.join('..','..','gis','output_ras','sw_ww', 'confining_unit.tif')
    with rasterio.open(output_path, 'w', **template_profile) as rst:
        rst.write(clay_layer, 1)
        

# -------------
# Main function
# -------------
def main_preprocess():
    global grd_files
    # grd files with specified sigma value for smoothing
    grd_files = [
        'Hydro Model_ 01 - Bedrock_younger.asc',
        'Hydro Model_ 02 - Basal Clay_younger.asc',
        'Hydro Model_ 03 - Lower Aquifer_younger.asc',
        'Hydro Model_ 04 - Middle Clay_younger.asc',
        'Hydro Model_ 05 - Middle Aquifer_younger.asc',
        'Hydro Model_ 06 - Upper Clay_younger.asc',
        'Hydro Model_ 07 - Upper Aquifer_younger.asc',
        'Hydro Model_ 08 - Top Clay_younger.asc'
        ]
    
    # Smooth grd files and save as .tif
    process_grd_files()
    
    
    # Create surfaces to use in model
    create_model_surfaces()
    create_SW_layer()
    create_confining_layer()
    
if __name__ == "__main__":
    main_preprocess()














