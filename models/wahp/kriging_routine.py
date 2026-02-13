#!/usr/bin/env python3
"""
This script reads thickness CSV files (one per model unit) generated from borehole data,
computes the experimental variogram with adjustable parameters (including horizontal anisotropy),
fits a spherical variogram model, and then performs kriging interpolation
over a grid covering the extent defined by a shapefile.
It produces a PDF plot for each layer showing:
  - Left: Experimental variogram (points) and the fitted theoretical variogram (line)
  - Right: Kriging interpolation map with kriging parameters displayed in the title.

Grid spacing for kriging is fixed at 660 feet.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial.distance import pdist
from pykrige.ok import OrdinaryKriging


# -------------------------------
# Compute the experimental variogram.
# -------------------------------
def compute_experimental_variogram(
    x, y, values, anisotropy_factor=1.0, n_lags=15, max_range=None
):
    """
    Compute an experimental variogram.
    Optionally applies an anisotropy transformation to the x coordinates.

    Parameters:
      x, y: coordinate arrays.
      values: the property to interpolate (thickness).
      anisotropy_factor: scaling factor applied to the x coordinate (default 1.0 for isotropic).
      n_lags: number of bins for the variogram.
      max_range: maximum lag distance (if None, set to half the maximum computed distance).

    Returns:
      bin_centers: array of lag distances.
      exp_variogram: array of computed semivariance values.
    """
    # Apply anisotropy transformation (scaling x)
    x_trans = x / anisotropy_factor
    coords = np.column_stack((x_trans, y))

    # Compute pairwise distances and semivariances
    dists = pdist(coords)
    semivariances = pdist(values.reshape(-1, 1)) ** 2 / 2.0

    if max_range is None:
        max_range = dists.max() / 2.0

    bins = np.linspace(0, max_range, n_lags + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    exp_variogram = np.empty(n_lags)

    for i in range(n_lags):
        mask = (dists >= bins[i]) & (dists < bins[i + 1])
        if np.sum(mask) > 0:
            exp_variogram[i] = np.mean(semivariances[mask])
        else:
            exp_variogram[i] = np.nan

    return bin_centers, exp_variogram


# -------------------------------
# Define a spherical variogram model.
# -------------------------------
def spherical_model(h, nugget, sill, rang):
    """
    Spherical variogram model.

    Parameters:
      h: lag distance(s)
      nugget: nugget effect
      sill: sill (total variance)
      rang: range parameter

    Returns:
      semivariance at lag h.
    """
    h = np.array(h)
    gamma = np.where(
        h <= rang,
        nugget + (sill - nugget) * (1.5 * (h / rang) - 0.5 * (h / rang) ** 3),
        nugget + (sill - nugget),
    )
    return gamma


# -------------------------------
# Perform kriging interpolation on a single layer.
# -------------------------------
def process_layer(
    csv_file,
    gridx,
    gridy,
    anisotropy_scaling=1.0,
    anisotropy_angle=0.0,
    nugget=None,
    sill=None,
    rang=None,
    n_lags=15,
    save_raster=False,
):
    """
    For a given thickness CSV file:
      - Load and filter the data.
      - For WR, add control points (1000 ft grid) directly.
      - For WBV, compute the variogram using only the original data,
        then augment the dataset for interpolation with breakline points along a
        2500 ft buffered outline, and finally force interpolated values to 0
        outside that buffered area.
      - Additionally, for WR, the breakline is generated using a different shapefile:
            "../../gis/input_shps/wahp/wr_extent.shp"
      - Compute the experimental variogram and fit a spherical model.
      - Perform ordinary kriging interpolation.
      - Plot the results with a custom colormap (0 = grey) and overlay the outline and extent shapefiles.
      - Optionally, save the final interpolated raster as a GeoTIFF.
    """
    from shapely.geometry import Point

    # --- Load and filter the CSV ---
    df = pd.read_csv(csv_file)
    # Keep rows where either partial is False, or if partial is True, thickness is at least 100.
    df = df[(df["partial"] == False) | (df["thickness"] >= 100)]

    layer = os.path.basename(csv_file).replace("_thickness.csv", "").upper()

    # Save original data for variogram computation.
    x_orig = df["x"].values
    y_orig = df["y"].values
    values_orig = df["thickness"].values

    # --- For WR layer: add control points on a 1000-ft grid ---
    if layer == "WR":
        minx, maxx = gridx.min(), gridx.max()
        miny, maxy = gridy.min(), gridy.max()
        control_x = np.arange(minx, maxx + 10000, 10000)
        control_y = np.arange(miny, maxy + 10000, 10000)
        grid_control_X, grid_control_Y = np.meshgrid(control_x, control_y)
        cp_x = grid_control_X.flatten()
        cp_y = grid_control_Y.flatten()
        existing_points = np.column_stack((x_orig, y_orig))
        new_control_x = []
        new_control_y = []
        tol = 1.0  # tolerance in feet; adjust as needed
        for cx, cy in zip(cp_x, cp_y):
            dists = np.sqrt(
                (existing_points[:, 0] - cx) ** 2 + (existing_points[:, 1] - cy) ** 2
            )
            if np.all(dists > tol):
                new_control_x.append(cx)
                new_control_y.append(cy)
        if new_control_x:
            control_df = pd.DataFrame(
                {"x": new_control_x, "y": new_control_y, "thickness": 0.0}
            )
            x_orig = np.concatenate([x_orig, control_df["x"].values])
            y_orig = np.concatenate([y_orig, control_df["y"].values])
            values_orig = np.concatenate([values_orig, control_df["thickness"].values])
        print(f"Added {len(new_control_x)} control points to {layer} layer.")

    # --- Compute experimental variogram using original data (WBV/WR variogram excludes breakline points) ---
    bin_centers, exp_var = compute_experimental_variogram(
        x_orig, y_orig, values_orig, anisotropy_factor=anisotropy_scaling, n_lags=n_lags
    )
    # Set variogram parameters if not provided.
    if nugget is None:
        nugget = 0.0
    if sill is None:
        sill = np.nanmax(exp_var)
    if rang is None:
        rang = bin_centers[-1]
    theo_var = spherical_model(bin_centers, nugget, sill, rang)

    # --- Augment dataset for interpolation ---
    # For WBV and WR, add breakline points along the 2500-ft buffered outline.
    if layer in ["WBV", "WR"]:
        # Choose the shapefile based on the unit.
        if layer == "WBV":
            shp_outline_path = os.path.join(
                "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
            )
        elif layer == "WR":
            shp_outline_path = os.path.join(
                "..", "..", "gis", "input_shps", "wahp", "wr_extent.shp"
            )
        outline_gdf = gpd.read_file(shp_outline_path)
        # Union the geometries using the recommended method.
        union_geom = outline_gdf.geometry.union_all()
        # Buffer the union by 2500 feet.
        buffered_geom = union_geom.buffer(2500)
        # Extract the boundary of the buffered polygon.
        boundary = buffered_geom.boundary
        breakline_points = []
        spacing = 100  # spacing along the boundary in feet; adjust as desired
        if boundary.geom_type == "MultiLineString":
            for line in boundary.geoms:
                num_points = int(np.ceil(line.length / spacing))
                for i in range(num_points + 1):
                    pt = line.interpolate(i / num_points * line.length)
                    breakline_points.append(pt)
        else:
            num_points = int(np.ceil(boundary.length / spacing))
            for i in range(num_points + 1):
                pt = boundary.interpolate(i / num_points * boundary.length)
                breakline_points.append(pt)
        breakline_x = np.array([pt.x for pt in breakline_points])
        breakline_y = np.array([pt.y for pt in breakline_points])
        breakline_values = np.zeros_like(breakline_x)
        print(
            f"Added {len(breakline_points)} breakline points to {layer} layer for interpolation."
        )
        # Define the augmented dataset for interpolation.
        x_interp = np.concatenate([x_orig, breakline_x])
        y_interp = np.concatenate([y_orig, breakline_y])
        values_interp = np.concatenate([values_orig, breakline_values])
    else:
        x_interp = x_orig
        y_interp = y_orig
        values_interp = values_orig

    # --- Perform ordinary kriging interpolation using the augmented dataset ---
    OK = OrdinaryKriging(
        x_interp,
        y_interp,
        values_interp,
        variogram_model="spherical",
        variogram_parameters={"nugget": nugget, "sill": sill, "range": rang},
        anisotropy_scaling=anisotropy_scaling,
        anisotropy_angle=anisotropy_angle,
        verbose=False,
        enable_plotting=False,
    )
    z, ss = OK.execute("grid", gridx, gridy)

    # --- For WBV and WR, force interpolation to 0 outside the buffered area ---
    if layer in ["WBV", "WR"]:
        if layer == "WBV":
            shp_outline_path = os.path.join(
                "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
            )
        elif layer == "WR":
            shp_outline_path = os.path.join(
                "..", "..", "gis", "input_shps", "wahp", "wr_extent.shp"
            )
        outline_gdf = gpd.read_file(shp_outline_path)
        union_geom = outline_gdf.geometry.union_all()
        buffered_geom = union_geom.buffer(2500)
        grid_X, grid_Y = np.meshgrid(gridx, gridy)
        pts = [Point(x, y) for x, y in zip(grid_X.flatten(), grid_Y.flatten())]
        mask = np.array([buffered_geom.contains(pt) for pt in pts]).reshape(
            grid_X.shape
        )
        z[~mask] = 0

    # --- Create the plots ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Variogram plot.
    ax[0].plot(bin_centers, exp_var, "o", label="Experimental")
    ax[0].plot(bin_centers, theo_var, "-", label="Spherical Model")
    ax[0].set_title(f"{layer} Variogram")
    ax[0].set_xlabel("Lag Distance")
    ax[0].set_ylabel("Semivariance")
    ax[0].legend()

    # Kriging interpolation plot.
    grid_X, grid_Y = np.meshgrid(gridx, gridy)

    # Create a custom colormap based on viridis with 0 values shown in grey.
    import matplotlib as mpl

    viridis = plt.get_cmap("viridis", 256)
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = np.array([0.5, 0.5, 0.5, 1.0])
    new_cmap = mpl.colors.ListedColormap(colors)

    c = ax[1].pcolormesh(
        grid_X, grid_Y, z, shading="auto", cmap=new_cmap, vmin=0, vmax=np.nanmax(z)
    )
    fig.colorbar(c, ax=ax[1], label="Thickness")
    ax[1].set_title(
        f"{layer} Kriging Interpolation\n"
        f"Kriging parameters: nugget={nugget}, sill={sill:.2f}, range={rang:.2f},\n"
        f"anisotropy_scaling={anisotropy_scaling}, anisotropy_angle={anisotropy_angle}"
    )
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    # Overlay the outline shapefile.
    if layer == "WR":
        shp_outline = os.path.join(
            "..", "..", "gis", "input_shps", "wahp", "wr_extent.shp"
        )
    else:
        shp_outline = os.path.join(
            "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
        )
    outline_gdf = gpd.read_file(shp_outline)
    outline_gdf.plot(ax=ax[1], facecolor="none", edgecolor="blue", lw=1)

    # Overlay the extent shapefile.
    shp_extent = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "updated_wahp_model_extent.shp"
    )
    extent_gdf = gpd.read_file(shp_extent)
    extent_gdf.plot(ax=ax[1], facecolor="none", edgecolor="red", lw=1)

    plt.suptitle(f"Layer: {layer}", fontsize=16)

    # --- Save the figure ---
    output_dir = os.path.dirname(csv_file)
    out_pdf = os.path.join(output_dir, f"{layer.lower()}_kriging.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved kriging plot for {layer} to {out_pdf}")

    # --- Optionally save the final interpolated raster as a GeoTIFF ---
    if save_raster:
        import rasterio
        from rasterio.transform import from_origin

        # Flip the array vertically so that the first row corresponds to the top of the raster.
        z_flipped = np.flipud(z)

        out_dir_raster = os.path.join(
            "..", "..", "gis", "output_ras", "interpolation_ras"
        )
        os.makedirs(out_dir_raster, exist_ok=True)
        out_tiff = os.path.join(out_dir_raster, f"{layer.lower()}_kriging.tif")

        # Compute the transform: using the minimum X and maximum Y of the grid.
        pixel_size = gridx[1] - gridx[0]
        transform = from_origin(gridx.min(), gridy.max(), pixel_size, pixel_size)
        new_dtype = "float32"
        with rasterio.open(
            out_tiff,
            "w",
            driver="GTiff",
            height=z_flipped.shape[0],
            width=z_flipped.shape[1],
            count=1,
            dtype=new_dtype,
            crs="EPSG:2265",
            transform=transform,
        ) as dst:
            dst.write(z_flipped.astype(new_dtype), 1)
        print(f"Saved raster to {out_tiff}")


# -------------------------------
# Main routine.
# -------------------------------
def main():
    # Directory where the thickness CSVs are stored.
    output_dir = os.path.join("data", "analyzed", "model_unit_thickness_4_lay")
    csv_files = glob.glob(os.path.join(output_dir, "*_thickness.csv"))
    csv_files.sort()

    # For testing with WBV or WR only, you can select one.
    # For example, for testing WR:
    # csv_files = [os.path.join(output_dir, "wr_thickness.csv")]
    # csv_files = [os.path.join(output_dir, "wbv_thickness.csv")]
    # csv_files = [os.path.join(output_dir, "wss_thickness.csv")]
    csv_files = [os.path.join(output_dir, "dc_thickness.csv")]
    # Load the extent shapefile to define the interpolation region.
    shp_extent = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "updated_wahp_model_extent.shp"
    )
    extent_gdf = gpd.read_file(shp_extent)
    minx, miny, maxx, maxy = extent_gdf.total_bounds

    # Create grid with a spacing of 660 feet.
    grid_spacing = 660
    gridx = np.arange(minx, maxx + grid_spacing, grid_spacing)
    gridy = np.arange(miny, maxy + grid_spacing, grid_spacing)

    # Set parameters – for example, using an anisotropy angle and scaling.
    anisotropy_scaling = 1  # Adjust as needed
    anisotropy_angle = 130  # e.g., 125° from the x-axis
    # Optionally, adjust variogram parameters:
    nugget = 4000  # If None, computed automatically (default 0.0)
    sill = 6500  # If None, computed automatically (max(exp_var))
    rang = 15000  # If None, computed automatically (last bin center)
    n_lags = 15

    # Process the selected layer.
    for csv_file in csv_files:
        print(f"Processing layer file: {csv_file}")
        process_layer(
            csv_file,
            gridx,
            gridy,
            anisotropy_scaling,
            anisotropy_angle,
            nugget,
            sill,
            rang,
            n_lags,
            save_raster=True,  # Set to True to save the final raster as a GeoTIFF.
        )


if __name__ == "__main__":
    main()
