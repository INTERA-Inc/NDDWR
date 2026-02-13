#!/usr/bin/env python

import os
import numpy as np
import rasterio
import shapefile  # Using PyShp instead of Fiona
from rasterio.windows import Window
from rasterio.transform import rowcol, Affine
from rasterio.io import MemoryFile
from rasterio.mask import mask
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter


##############################################################################
# Load Shapefile
##############################################################################
def load_shapefile_geometries(shapefile_path):
    """
    Loads polygons from a shapefile using PyShp and returns them as
    GeoJSON-like geometry dicts for rasterio's mask function.
    """
    sf = shapefile.Reader(shapefile_path)
    shapes = []
    for shape_rec in sf.shapeRecords():
        geom = shape_rec.shape.__geo_interface__
        shapes.append(geom)
    return shapes


##############################################################################
# Remove and Fill Values Above Threshold
##############################################################################
def remove_and_fill_with_median(dem_array, threshold, fill_size=3, max_iterations=5):
    """
    Removes cells above a given threshold by setting them to NaN, then iteratively
    fills those holes via local median filtering, up to a maximum number of iterations.
    """
    arr = dem_array.astype(float)
    arr[arr == 0] = np.nan  # If 0 is our nodata, treat it as NaN
    arr[arr > threshold] = np.nan

    for _ in range(max_iterations):
        filled_once = generic_filter(
            arr, function=np.nanmedian, size=fill_size, mode="constant", cval=np.nan
        )
        newly_filled = np.where(np.isnan(arr), filled_once, arr)

        changed_count = np.sum(np.isnan(arr) & ~np.isnan(newly_filled))
        arr = newly_filled

        if changed_count == 0:
            break

    return arr


##############################################################################
# Clip Raster and Determine Exact Window for Mosaicking
##############################################################################
def clip_raster_and_find_window(raster_path, shapefile_path):
    """
    Clips the input raster to the bounding box of a shapefile (crop=True).
    Returns:
      - clipped_array: the DEM subset,
      - clipped_profile: the updated raster profile,
      - sub_window: the pixel-level Window in the original DEM where this subset belongs.
    """
    geoms = load_shapefile_geometries(shapefile_path)

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True, nodata=0)
        clipped_profile = src.profile.copy()

        clipped_array = out_image[0, :, :]
        clipped_profile.update(
            {
                "height": clipped_array.shape[0],
                "width": clipped_array.shape[1],
                "transform": out_transform,
            }
        )

        # Figure out the pixel coordinates in the clipped transform
        left_px, top_px = (0, 0)
        right_px, bottom_px = (clipped_array.shape[1], clipped_array.shape[0])

        # Convert those pixel coords into map coords (in the clipped subset)
        top_left_mapx, top_left_mapy = out_transform * (left_px, top_px)
        bottom_right_mapx, bottom_right_mapy = out_transform * (right_px, bottom_px)

        # Convert map coords to row/col offsets in the original raster
        r0, c0 = rowcol(src.transform, top_left_mapx, top_left_mapy, op=round)
        r1, c1 = rowcol(src.transform, bottom_right_mapx, bottom_right_mapy, op=round)

        row_off = min(r0, r1)
        col_off = min(c0, c1)
        height = abs(r1 - r0)
        width = abs(c1 - c0)

        sub_window = Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        )

        return clipped_array, clipped_profile, sub_window


##############################################################################
# Re-clip to Polygon Boundary
##############################################################################
def reclip_to_polygon(filled_array, filled_profile, shapefile_path):
    """
    Re-applies the shapefile geometry (with crop=False) to the filled array to ensure
    everything outside the polygon is set to NaN again. This prevents overwriting areas
    outside the polygon when mosaicking back into the full DEM.
    """
    geoms = load_shapefile_geometries(shapefile_path)

    with MemoryFile() as memfile:
        with memfile.open(**filled_profile) as ds:
            ds.write(filled_array, 1)

            # Re-mask, but keep same shape. Outside polygon => nodata=np.nan
            out_image, _ = mask(ds, geoms, crop=False, nodata=np.nan)

    # out_image has shape (1, rows, cols); return just the 2D array
    return out_image[0]


##############################################################################
# Mosaic the Updated Subarray Back In
##############################################################################
def mosaic_subarea_back(raster_path, updated_subarray, window, out_raster_path):
    """
    Reads the full DEM, then merges the updated subarray only where it's valid
    (non-NaN). Outside the polygon (where it's NaN), we keep the original raster.
    """
    with rasterio.open(raster_path) as src:
        full_data = src.read(1).astype(float)
        profile = src.profile.copy()

    if (window.height, window.width) != updated_subarray.shape:
        raise ValueError("Updated subarray shape doesn't match the clipping window.")

    row_off = int(window.row_off)
    col_off = int(window.col_off)

    existing_sub = full_data[
        row_off : row_off + window.height, col_off : col_off + window.width
    ]

    # Only overwrite where updated_subarray is valid
    mosaic_sub = np.where(~np.isnan(updated_subarray), updated_subarray, existing_sub)

    full_data[row_off : row_off + window.height, col_off : col_off + window.width] = (
        mosaic_sub
    )

    profile.update(dtype="float32")
    with rasterio.open(out_raster_path, "w", **profile) as dst:
        dst.write(full_data.astype("float32"), 1)


##############################################################################
# Subset Workflow for Removing & Filling
##############################################################################
def plot_two_dems_subarea(original_clip, filled, cmap="terrain"):
    """
    Quick side-by-side plot for comparing original subarea vs. final filled subarea.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    im0 = axs[0].imshow(original_clip, cmap=cmap)
    axs[0].set_title("Original Clipped DEM")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(filled, cmap=cmap)
    axs[1].set_title("Removed & Filled DEM (Reclipped)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def process_subset_raster(
    raster_path,
    shapefile_path,
    threshold,
    fill_size=3,
    max_fill_iterations=5,
    out_raster_path="mosaicked_output.tif",
    plot_results=True,
):
    """
    1. Clip the DEM by the shapefile to isolate subarea.
    2. Remove/fill cells > threshold.
    3. Re-clip so any expansions outside the polygon become NaN again.
    4. Mosaic back into the original DEM only where we have valid updates.
    """
    clipped_array, clipped_profile, sub_window = clip_raster_and_find_window(
        raster_path, shapefile_path
    )

    original_clipped = clipped_array.astype(float)
    original_clipped[original_clipped == 0] = np.nan

    filled_array = remove_and_fill_with_median(
        clipped_array,
        threshold=threshold,
        fill_size=fill_size,
        max_iterations=max_fill_iterations,
    )

    reclip_filled = reclip_to_polygon(filled_array, clipped_profile, shapefile_path)

    if plot_results:
        plot_two_dems_subarea(original_clipped, reclip_filled)

    mosaic_subarea_back(raster_path, reclip_filled, sub_window, out_raster_path)


##############################################################################
# Whole-raster Smoothing & Resampling
##############################################################################
def smooth_dem(dem_array, method="gaussian", sigma=2, size=3):
    """
    Smooth the DEM using either a Gaussian or median filter, treating 0 as nodata.
    """
    from scipy.ndimage import gaussian_filter, generic_filter

    data = dem_array.astype(float)
    data[data == 0] = np.nan

    if method == "gaussian":
        print(f"Applying Gaussian smoothing with sigma={sigma}")
        valid = np.where(~np.isnan(data), 1, 0)
        data_zeroed = np.where(np.isnan(data), 0, data)
        data_f = gaussian_filter(data_zeroed, sigma=sigma)
        weight_f = gaussian_filter(valid, sigma=sigma)
        result = np.divide(
            data_f, weight_f, out=np.full_like(data_f, np.nan), where=weight_f != 0
        )
        result[weight_f < 1e-3] = np.nan
        return result

    elif method == "median":
        print(f"Applying median smoothing with size={size}")
        result = generic_filter(
            data, function=np.nanmedian, size=size, mode="constant", cval=np.nan
        )
        return result

    else:
        print("Unknown smoothing method; returning array as-is.")
        return data


def block_average(array_2d, factor):
    """
    Downsamples a 2D array by averaging blocks of (factor x factor).
    Ignores any NaNs in the process.
    """
    m, n = array_2d.shape
    m_new = m // factor
    n_new = n // factor
    trimmed = array_2d[: m_new * factor, : n_new * factor]
    return np.nanmean(trimmed.reshape(m_new, factor, n_new, factor), axis=(1, 3))


def plot_three_dems(original, smoothed, resampled, cmap="terrain"):
    """
    Plot the original DEM, the smoothed DEM, and the resampled DEM side-by-side.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axs[0].imshow(original, cmap=cmap)
    axs[0].set_title("Original DEM")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(smoothed, cmap=cmap)
    axs[1].set_title("Smoothed DEM")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(resampled, cmap=cmap)
    axs[2].set_title("Resampled DEM")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


##############################################################################
# Main
##############################################################################
def main():
    """
    This script has two workflows:
      1) Whole-raster smoothing/resampling if process_whole_raster=True
      2) Subset workflow (clip, remove/fill, reclip, mosaic) if process_whole_raster=False
    """
    process_whole_raster = True

    dem_path = os.path.join(
        "..", "..", "gis", "input_ras", "wahp", "dem", "wahp_full_res_dem_100.tif"
    )
    shp_path = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "wahp_building_area.shp"
    )
    out_subset_path = os.path.join(
        "..", "..", "gis", "output_ras", "dem_no_buildings.tif"
    )
    out_whole_smoothed_path = os.path.join(
        "..", "..", "gis", "output_ras", "wahp_full_res_dem_100_smoothed.tif"
    )
    out_whole_resampled_path = os.path.join(
        "..", "..", "gis", "output_ras", "wahp_full_res_dem_100_resampled.tif"
    )

    # Check input
    if not os.path.exists(dem_path):
        print(f"DEM file not found: {dem_path}")
        return

    if process_whole_raster:
        # Workflow for the entire DEM
        import rasterio

        with rasterio.open(out_subset_path) as src:
            dem = src.read(1).astype(float)
            profile = src.profile.copy()
            transform = src.transform

        # Convert zeros to NaNs
        dem[dem == 0] = np.nan

        # Smooth the entire DEM
        smoothing_method = "median"  # or "gaussian"
        if smoothing_method == "gaussian":
            smoothed = smooth_dem(dem, method="gaussian", sigma=2)
        else:
            smoothed = smooth_dem(dem, method="median", size=3)

        # # Write the smoothed DEM to disk
        # profile.update(dtype="float32")
        # with rasterio.open(out_whole_smoothed_path, "w", **profile) as dst:
        #     dst.write(smoothed.astype("float32"), 1)

        # Block-average to downsample
        factor = 6
        resampled = block_average(smoothed, factor=factor)

        # We need a new transform that accounts for the larger pixel size
        new_transform = Affine(
            transform.a * factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e * factor,
            transform.f,
        )

        # Update profile for the resampled array
        profile.update(
            height=resampled.shape[0], width=resampled.shape[1], transform=new_transform
        )

        # Write the resampled DEM
        with rasterio.open(out_whole_resampled_path, "w", **profile) as dst:
            dst.write(resampled.astype("float32"), 1)

        # Plot the original, smoothed, and resampled DEM side-by-side
        plot_three_dems(dem, smoothed, resampled)

    else:
        # Subset workflow: removing/filling in the shapefile region
        threshold_val = 295
        process_subset_raster(
            raster_path=dem_path,
            shapefile_path=shp_path,
            threshold=threshold_val,
            fill_size=7,
            max_fill_iterations=5,
            out_raster_path=out_subset_path,
            plot_results=True,
        )


if __name__ == "__main__":
    main()
