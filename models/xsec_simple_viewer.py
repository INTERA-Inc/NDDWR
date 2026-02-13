import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from flopy.plot import PlotCrossSection, PlotMapView
import geopandas as gpd
import os
import flopy
import tqdm


# --- Options are swww, wahp, elk
model_name = 'swww'


# --- Layer coloring scheme
layer_colors_sw = [
    "#4682B4",
    "#FFE999",
    "darkblue",
    ]

layer_colors_wahp = [
    "#B8B0A0",  # WSS
    "#B8B0A0",  # WSS
    "#B8B0A0",  # WSS
    "#9C8B6D",  # Upper Confining Unit
    "#darkblue",  # WBV (main aquifer)
    "#6E6259",  # Lower Confining Unit
    ]

layer_colors_elk = [
    "#FFE999",
    "darkblue",
    ]

#  --- Names for the layers
layer_names_sw = [
    "Warwick Aquifer",
    "Confining Unit",
    "Spiritwood Aquifer",
    ]

layer_names_wahp = [
    "WSS",
    "WSS",
    "WSS",
    "Upper Confining Unit",
    "WBV",
    "Lower Confining Unit",
    ]

layer_names_elk = [
    "Soils/Clay/Silt",
    "Elk Valley Aquifer",
    ]

# ---------------------------------------------------
# Plot model geologic cross section with SW shapefile
# ---------------------------------------------------
def plot_xsecs_to_pdf(gwf, layer_colors, layer_names, output_dir=".", vertical_exag=10):
    # Load model grid with correct spatial information
    gwf.modelgrid.set_coord_info(
        xoff=2388853.4424208435,
        yoff=260219.09632163405,
        angrot=0, 
        epsg=2265
        )
    modelgrid = gwf.modelgrid
    nlay, nrow, ncol = gwf.dis.nlay.data, modelgrid.nrow, modelgrid.ncol

    # --- Build array colored by layer number and mask idomain
    idomain = gwf.dis.idomain.array.copy()
    arr = np.full_like(idomain, np.nan, dtype=float)
    for k in range(nlay):
        arr[k, :, :] = np.where(idomain[k, :, :] <= 0, np.nan, k + 1)

    cmap = ListedColormap(layer_colors)
    bounds = np.arange(0.5, len(layer_colors) + 1, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # spiritwood_gdf = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww',"sw_extent_SJ.shp")).to_crs(2265)
    
    spiritwood_gdf = gpd.read_file(os.path.join('..','..','gis','input_shps','sw_ww','Aquifers.shp')).to_crs(2265)
    
    def make_plot(line_type, index):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
        ax_xsec, ax_map = axes
        
        # Create cross section based on current line
        line = {line_type: index}
        xsect = PlotCrossSection(model=gwf, ax=ax_xsec, line=line)
        cs = xsect.plot_array(arr, cmap=cmap, norm=norm, alpha=0.8)
        ax_xsec.set_title(f"{line_type.title()} {index} Cross Section")

        # Set axis labels
        if line_type == "row":
            ax_xsec.set_xlabel("Easting (ft)")
        else:
            ax_xsec.set_xlabel("Northing (ft)")
        ax_xsec.set_ylabel("Elevation (ft)")
        
        # Create custom cbar for the geologic units
        try:
            cbar = plt.colorbar(cs, ax=ax_xsec, ticks=np.arange(1, len(layer_names) + 1))
        except:
            return None
        cbar.ax.set_yticklabels(layer_names)
        cbar.set_label("Geologic Unit")
        cbar.ax.invert_yaxis()

        # Map view
        pmv = PlotMapView(model=gwf, ax=ax_map)
        pmv.plot_grid(linewidth=0.3, alpha=0.2)
        
        # Plot the current row or column as a line
        if line_type == "row":
            for col in range(ncol):
                verts = modelgrid.get_cell_vertices(index, col)
                xs, ys = zip(*verts)
                ax_map.fill(xs, ys, facecolor="red", edgecolor="k", linewidth=0.3)
            ax_map.set_title(f"Row {index}")
        else:
            for row in range(nrow):
                verts = modelgrid.get_cell_vertices(row, index)
                xs, ys = zip(*verts)
                ax_map.fill(xs, ys, facecolor="red", edgecolor="k", linewidth=0.3)
            ax_map.set_title(f"Column {index}")
        
        # Plot Spiritwood Shapefile
        spiritwood_gdf.plot(ax=ax_map, facecolor="blue", edgecolor="blue", linewidth=1.5, alpha=0.7)     
        
        # Plot locations where model bottom is the top elevation
        # bottom_is_top = np.load(os.path.join('data','analyzed','bottom_is_top.npy'))
        # pmv.plot_array(bottom_is_top)
        
        # Format
        ax_map.set_aspect("equal")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        plt.tight_layout()
        
        return fig

    # --- Save column cross-sections
    with PdfPages(f"{output_dir}/col_xsecs_final.pdf") as pdf:
        print("Plotting columns x-sections")
        for col in tqdm.tqdm(range(ncol)):
            fig = make_plot("column", col)
            if fig:
                pdf.savefig(fig)
                # plt.show()
                plt.close(fig)

    # --- Save row cross-sections
    with PdfPages(f"{output_dir}/row_xsecs_final.pdf") as pdf:
        print("Plotting row x-sections")
        for row in tqdm.tqdm(range(nrow)):
            fig = make_plot("row", row)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved cross section PDFs to {output_dir}")


# -------------
# Main function
# -------------
def main():
    # --- Colors for the geologic units
    layer_colors = [
        "#4682B4",
        "#FFE999",
        "darkblue",
        ]
    
    # --- Names for the layers
    layer_names = [
        "Warwick Aquifer",
        "Middle Clay",
        "Spiritwood Aquifer",
        ]
    
    # --- Load model
    model_name = "swww"
    model_ws = os.path.join('model_ws',"swww_clean")
    sim = flopy.mf6.MFSimulation.load(
        sim_name=f"{model_name}.nam",
        version="mf6",
        exe_name="mf6.exe",
        sim_ws=model_ws,
        load_only=["dis"],
        )
    gwf = sim.get_model(model_name)
    
    # --- Plot cross sections
    plot_xsecs_to_pdf(
        gwf, 
        layer_colors, 
        layer_names, 
        output_dir=os.path.join('figures')
        )
    
    
# --------
# Run main
# --------
if __name__ == "__main__":
    main()
    
    
    
    

