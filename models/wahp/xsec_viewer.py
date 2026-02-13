#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import matplotlib

matplotlib.use("TkAgg")  # Or 'Qt5Agg' if you prefer
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import flopy


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def getEquidistantPoints(p1, p2, parts):
    """
    Create points spaced evenly (parts + 1) between p1 and p2.
    """
    return zip(
        np.linspace(p1[0], p2[0], parts + 1),
        np.linspace(p1[1], p2[1], parts + 1),
    )


def ckdnearest(gdA, gdB):
    """
    Use cKDTree to find nearest neighbor in gdB for each geometry in gdA.
    Returns a combined DataFrame with 'dist' (the distance) and 'idx' (the index in gdB).
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    import pandas as pd

    gdf = pd.concat(
        [gdA.reset_index(drop=True), pd.DataFrame({"dist": dist, "idx": idx})],
        axis=1,
    )
    return gdf


def getXsec(numXsec, numSample):
    """
    Interactive function: user clicks two points on the map for each cross-section.
    Saves the resulting lines in a shapefile: 'xsec_line.shp'.
    """
    mbndry = gpd.read_file(
        os.path.join("..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp")
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(
        axis="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1, label="Title")

    mbndry.plot(ax=ax, color="None", edgecolor="blue", alpha=0.2)
    x0, y0, x1, y1 = mbndry.total_bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title("Pick starting and ending location of X-section", fontsize=20, pad=10)

    xsec_pts = []
    for s in range(numXsec):
        xy = plt.ginput(2, timeout=-1)
        if len(xy) < 2:
            print(
                "Not enough points selected. Please select two points for each x-section."
            )
            continue
        ax.plot([xy[0][0], xy[1][0]], [xy[0][1], xy[1][1]], "-k")
        xsec_pts.extend(xy)

    plt.show()
    plt.close()

    geometry = [
        LineString([xsec_pts[i], xsec_pts[i + 1]]) for i in range(0, len(xsec_pts), 2)
    ]
    xsec = gpd.GeoDataFrame(geometry=geometry, crs=mbndry.crs).reset_index(drop=True)
    xsec["pt0_x"] = xsec["geometry"].apply(lambda x: x.coords[0][0])
    xsec["pt0_y"] = xsec["geometry"].apply(lambda x: x.coords[0][1])
    xsec["pt1_x"] = xsec["geometry"].apply(lambda x: x.coords[1][0])
    xsec["pt1_y"] = xsec["geometry"].apply(lambda x: x.coords[1][1])
    import string

    alphabet_uppercase = list(string.ascii_uppercase)
    xsec["label0"] = alphabet_uppercase[:numXsec]
    xsec["label1"] = [x + "'" for x in alphabet_uppercase[:numXsec]]
    xsec = xsec[["pt0_x", "pt0_y", "pt1_x", "pt1_y", "label0", "label1", "geometry"]]

    output_path = os.path.join(
        "..", "..", "gis", "output_shps", "wahp", "prelim_xsec_views"
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    xsec.to_file(os.path.join(output_path, "xsec_line.shp"))
    print(f"Cross-section shapefile saved to {output_path}")
    return xsec


def get_row_segments(gwf, row):
    """
    Return a list of line segments for all cells in a given row.
    Each segment is a pair [(x1, y1), (x2, y2)].
    """
    mg = gwf.modelgrid
    ncol = mg.ncol
    segments = []
    for col in range(ncol):
        verts = mg.get_cell_vertices(row, col)
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            segments.append([(x1, y1), (x2, y2)])
    return segments


def get_col_segments(gwf, col):
    """
    Return a list of line segments for all cells in a given column.
    """
    mg = gwf.modelgrid
    nrow = mg.nrow
    segments = []
    for row in range(nrow):
        verts = mg.get_cell_vertices(row, col)
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            segments.append([(x1, y1), (x2, y2)])
    return segments


# ---------------------------------------------------------------------------
# PLOT FUNCTION WITH plot_k OPTION, DISCRETE LABELS, AND SHAPEFILE OVERLAYS
# ---------------------------------------------------------------------------
def plot_output(
    mod_dir=os.path.join("model_ws", "wahp_mf6"),
    ofpdir="history_plots",
    mod_name="wahp_mf6",
    sp=20,
    sp_info="stress_period_info.csv",
    sub=False,
    xsecs=None,
    make_rc_xsecs=False,
    plot_k=False,  # If True, plot hydraulic conductivity values as discrete unique categories.
):
    """
    Produce cross-section plots.

    If plot_k is True, uses the npf.k values for colors and classifies them into discrete categories.
    For K values:
      - 0.1: dark blue
      - 0.2: blue
      - 0.3: light blue
      - 100: dark orange
      - 150: orange
      - 200: dark orange
      - 250: red
    Cells with inactive ibound (ibound==0) are masked and not plotted.
    Otherwise, uses the idomain/layer-based colors, masking out cells where ibound == 0.

    Also overlays additional shapefiles (L5 and L7 sand polygons) on the map panels.
    """
    fout = os.path.join(mod_dir, "prelim_figs")
    if not os.path.exists(fout):
        os.makedirs(fout)

    # Load additional shapefiles for overlays:
    base_shp_path = os.path.join("..", "..", "gis", "input_shps", "wahp")
    l5_sand_path = os.path.join(base_shp_path, "l5_sand_polygons_wbv.shp")
    l7_sand_path = os.path.join(base_shp_path, "l7_sand_polygons_wr.shp")
    l5_sand = gpd.read_file(l5_sand_path)
    l7_sand = gpd.read_file(l7_sand_path)

    # Load simulation; if plotting K, load npf as well.
    if plot_k:
        sim = flopy.mf6.MFSimulation.load(
            sim_name=f"{mod_name}.nam",
            version="mf6",
            exe_name="mf6.exe",
            sim_ws=os.path.join(mod_dir),
            load_only=["tdis", "dis", "npf"],
        )
    else:
        sim = flopy.mf6.MFSimulation.load(
            sim_name=f"{mod_name}.nam",
            version="mf6",
            exe_name="mf6.exe",
            sim_ws=os.path.join(mod_dir),
            load_only=["tdis", "dis"],
        )
    gwf = sim.get_model(mod_name)
    mg = gwf.modelgrid
    xll = mg.xoffset
    yll = mg.yoffset
    rot = mg.angrot
    mg.set_coord_info(xoff=xll, yoff=yll, angrot=rot, epsg=2265)

    # Load model boundary
    mbndry = gpd.read_file(
        os.path.join("..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp")
    )
    mbndry = mbndry.to_crs(epsg=mg.epsg)

    # If xsecs not provided, try to load from file
    fpth = os.path.join(
        "..", "..", "gis", "output_shps", "wahp", "prelim_xsec_views", "xsec_line.shp"
    )
    if xsecs is None and os.path.isfile(fpth):
        xsecs = gpd.read_file(fpth)
    elif xsecs is None:
        xsecs = gpd.GeoDataFrame()

    # Define colormap and array based on plot_k flag
    if plot_k:
        arr_label = "K (ft/day)"
        # Get the K array from npf as a numpy array and mask out cells where ibound==0
        arr = gwf.npf.k.array.copy()
        idom = gwf.dis.idomain.array.copy()
        arr[idom == 0] = np.nan

        # Explicit mapping for K values:
        # 0.1 -> dark blue, 0.2 -> blue, 0.3 -> light blue,
        # 100 -> dark orange, 150 -> orange, 200 -> dark orange, 250 -> red.
        k_color_dict = {
            0.1: "#00008B",
            0.2: "#0000FF",
            0.3: "#ADD8E6",
            100: "#8B4500",  # dark orange (example hex)
            150: "#FFA500",  # orange
            200: "#8B4500",  # dark orange (same as 100)
            250: "#FF0000",  # red
        }
        unique_vals = np.array(
            sorted(
                [
                    val
                    for val in k_color_dict.keys()
                    if val in np.unique(arr[~np.isnan(arr)])
                ]
            )
        )
        colors_list = [k_color_dict[val] for val in unique_vals]
        cmap_used = plt.cm.colors.ListedColormap(colors_list)
        # Compute boundaries halfway between present unique values
        boundaries = []
        if len(unique_vals) > 1:
            boundaries.append(unique_vals[0] - (unique_vals[1] - unique_vals[0]) / 2.0)
            for i in range(len(unique_vals) - 1):
                boundaries.append((unique_vals[i] + unique_vals[i + 1]) / 2.0)
            boundaries.append(
                unique_vals[-1] + (unique_vals[-1] - unique_vals[-2]) / 2.0
            )
        else:
            boundaries = [unique_vals[0] - 0.5, unique_vals[0] + 0.5]
        boundaries = np.array(boundaries)
        norm_used = plt.cm.colors.BoundaryNorm(boundaries, cmap_used.N)
        # Set tick positions as midpoints between boundaries and tick labels as unique values
        tick_locs = (boundaries[:-1] + boundaries[1:]) / 2.0
        tick_labels = [f"{val:g}" for val in unique_vals]
    else:
        # For discrete layer colors: mask out inactive cells (ibound==0)
        colors = {
            0: "white",  # inactive
            1: "#A67C52",  # Brownish mix - shallow sand and clay
            2: "#4682B4",  # Steel blue - Upper Middle clay
            3: "#FFE999",  # Yellow - Wahpeton Shallow Sand
            4: "#3A6F9B",  # Darker blue - Lower Middle clay
            5: "#E6C200",  # Darker yellow - Wahpeton Buried Valley Sand
            6: "#2E5C82",  # Even darker blue - Deep Clay
            7: "#CCAD00",  # Darker yellow - Wild Rice Sand
        }
        color_names = [
            "Inactive",
            "Mixed sand\nand clay",
            "Upper\nMiddle clay",
            "Wahpeton\nShallow Sand",
            "Lower\nMiddle clay",
            "Wahpeton Buried\nValley Sand",
            "Deep Clay",
            "Wild Rice\nSand",
        ]
        cmap_used = plt.cm.colors.ListedColormap(list(colors.values()))
        bounds = np.linspace(0, len(colors), len(colors) + 1)
        norm_used = plt.cm.colors.BoundaryNorm(bounds, cmap_used.N)
        arr_label = "Layer ID"
        arr = gwf.dis.idomain.array.copy()
        # Replace 0 with nan so inactive cells do not show up:
        for i in range(gwf.dis.nlay.data):
            arr[i, :, :] = np.where(arr[i, :, :] > 0, i + 1, np.nan)

    # --- 1) Plot user-drawn cross-sections, if available ---
    if not xsecs.empty:
        lines = flopy.plot.plotutil.shapefile_get_vertices(fpth)
        xsecs["coords"] = [x.coords[0] for x in xsecs.geometry]
        xsecs["name"] = [x for x in xsecs.index.values]
        exag = 40
        depth = 600
        top_elev = 1010
        bottom_elev = top_elev - depth
        fig_width = 10
        outpdf = os.path.join(fout, f"00_xsec_lines_example_sp{sp}.pdf")
        with PdfPages(outpdf) as pp:
            for ct, line in enumerate(lines):
                x0, y0 = line[0]
                x1, y1 = line[-1]
                length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                h = depth * exag * fig_width / length
                fig, (ax_xsec, ax_map) = plt.subplots(
                    1,
                    2,
                    figsize=(fig_width + h, h),
                    gridspec_kw={"width_ratios": [fig_width, h]},
                    dpi=150,
                )
                xsect = flopy.plot.PlotCrossSection(
                    ax=ax_xsec,
                    model=gwf,
                    line={"line": line},
                    extent=(0, length, bottom_elev, top_elev),
                )
                xsect.plot_array(arr, cmap=cmap_used, norm=norm_used, alpha=0.6)
                xsect.plot_grid(linewidth=0.5)
                ax_xsec.annotate(
                    text=f"{exag}x vert. exag",
                    xy=(0.01, 0.01),
                    xycoords="axes fraction",
                    fontsize="x-small",
                )
                ax_xsec.set_xticks([])
                xsec_this = xsecs.loc[xsecs["name"] == ct]
                if not xsec_this.empty:
                    ax_xsec.set_title(
                        f"{xsec_this['label0'].values[0]} - {xsec_this['label1'].values[0]}",
                        fontsize=9,
                    )
                mbndry.plot(ax=ax_map, color="None", edgecolor="blue")
                xsecs.plot(ax=ax_map, color="0.5", linewidth=1.5)
                xsecs.loc[xsecs["name"] == ct].plot(ax=ax_map, color="k", linewidth=1.5)
                # Overlay sand polygons on map panel:
                l5_sand.plot(
                    ax=ax_map, facecolor="none", edgecolor="magenta", linewidth=1.5
                )
                l7_sand.plot(
                    ax=ax_map, facecolor="none", edgecolor="cyan", linewidth=1.5
                )
                for idx, rowX in xsecs.iterrows():
                    pt0 = Point(rowX["pt0_x"], rowX["pt0_y"])
                    pt1 = Point(rowX["pt1_x"], rowX["pt1_y"])
                    ax_map.annotate(
                        text=rowX["label0"], xy=(pt0.x, pt0.y), fontsize="small"
                    )
                    ax_map.annotate(
                        text=rowX["label1"], xy=(pt1.x, pt1.y), fontsize="small"
                    )
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                ax_map.axis("off")
                plt.tight_layout()
                cbar_ax = fig.add_axes([0.5, 0.27, 0.2, 0.05])
                cb = plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap_used, norm=norm_used),
                    cax=cbar_ax,
                    orientation="horizontal",
                )
                if plot_k:
                    cb.set_ticks(tick_locs)
                    cb.set_ticklabels(tick_labels)
                    cb.set_label("K (ft/day)", fontsize=9)
                else:
                    cb.set_ticklabels(list(color_names), rotation=90, fontsize=9)
                cbar_ax.set_facecolor((0.5, 0.5, 0.5, 0.3))
                cbar_ax.patch.set_edgecolor("black")
                cbar_ax.patch.set_linewidth(1.5)
                pp.savefig(fig)
                plt.close(fig)
        print(f"Saved cross-section PDF (user lines) to {outpdf}")

    # --- 2) Plot cross-sections for each row & column, if requested ---
    if make_rc_xsecs:
        nrow_val = gwf.dis.nrow.array
        ncol_val = gwf.dis.ncol.array
        outpdf_rows = os.path.join(fout, "all_rows.pdf")
        outpdf_cols = os.path.join(fout, "all_cols.pdf")
        exag = 20
        top_elev = 1000
        bottom_elev = 400
        fig_width = 10
        with PdfPages(outpdf_rows) as pdf_rows:
            for row in range(nrow_val):
                fig_height = 6
                fig, (ax_cs, ax_map) = plt.subplots(
                    1,
                    2,
                    figsize=(fig_width * 1.5, fig_height),
                    gridspec_kw={"width_ratios": [2.5, 1]},
                )
                xsect = flopy.plot.PlotCrossSection(
                    model=gwf,
                    ax=ax_cs,
                    line={"row": row},
                    extent=(0, ncol_val * mg.delr[0], bottom_elev, top_elev),
                )
                xsect.plot_array(arr, cmap=cmap_used, norm=norm_used)
                xsect.plot_grid(linewidth=0.5, color="black", alpha=0.2)
                ax_cs.set_title(f"Row {row} Cross-Section", fontsize=10)
                ax_cs.annotate(
                    text=f"Exag={exag}",
                    xy=(0.01, 0.01),
                    xycoords="axes fraction",
                    fontsize=7,
                )
                mbndry.plot(ax=ax_map, facecolor="none", edgecolor="k", lw=1)
                from flopy.plot import PlotMapView

                pmv = PlotMapView(model=gwf, ax=ax_map)
                pmv.plot_grid(alpha=0.1)
                segments = get_row_segments(gwf, row)
                lc = LineCollection(segments, color="red", linewidths=1.5)
                ax_map.add_collection(lc)
                # Overlay sand polygons on map panel:
                l5_sand.plot(
                    ax=ax_map, facecolor="none", edgecolor="magenta", linewidth=1.5
                )
                l7_sand.plot(
                    ax=ax_map, facecolor="none", edgecolor="cyan", linewidth=1.5
                )
                ax_map.set_aspect("equal", "box")
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                ax_map.set_title(f"Row {row} (Map)", fontsize=8)
                plt.tight_layout()
                cbar_ax = fig.add_axes([0.5, 0.27, 0.2, 0.05])
                cb = plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap_used, norm=norm_used),
                    cax=cbar_ax,
                    orientation="horizontal",
                )
                if plot_k:
                    cb.set_ticks(tick_locs)
                    cb.set_ticklabels(tick_labels)
                    cb.set_label("K (ft/day)", fontsize=9)
                else:
                    cb.set_ticklabels(list(color_names), rotation=90, fontsize=9)
                cbar_ax.set_facecolor((0.5, 0.5, 0.5, 0.3))
                cbar_ax.patch.set_edgecolor("black")
                cbar_ax.patch.set_linewidth(1.5)
                pdf_rows.savefig(fig)
                plt.close(fig)
        print(f"Saved row cross-sections to: {outpdf_rows}")

        with PdfPages(outpdf_cols) as pdf_cols:
            for col in range(ncol_val):
                fig_height = 6
                fig, (ax_cs, ax_map) = plt.subplots(
                    1,
                    2,
                    figsize=(fig_width * 1.5, fig_height),
                    gridspec_kw={"width_ratios": [2.5, 1]},
                )
                xsect = flopy.plot.PlotCrossSection(
                    model=gwf,
                    ax=ax_cs,
                    line={"column": col},
                    extent=(0, nrow_val * mg.delc[0], bottom_elev, top_elev),
                )
                xsect.plot_array(arr, cmap=cmap_used, norm=norm_used)
                xsect.plot_grid(linewidth=0.5, color="black", alpha=0.2)
                ax_cs.set_title(f"Column {col} Cross-Section", fontsize=10)
                ax_cs.annotate(
                    text=f"Exag={exag}",
                    xy=(0.01, 0.01),
                    xycoords="axes fraction",
                    fontsize=7,
                )
                mbndry.plot(ax=ax_map, facecolor="none", edgecolor="k", lw=1)
                pmv = PlotMapView(model=gwf, ax=ax_map)
                pmv.plot_grid(alpha=0.1)
                segments = get_col_segments(gwf, col)
                lc = LineCollection(segments, color="red", linewidths=1.5)
                ax_map.add_collection(lc)
                # Overlay sand polygons on map panel:
                l5_sand.plot(
                    ax=ax_map, facecolor="none", edgecolor="magenta", linewidth=1.5
                )
                l7_sand.plot(
                    ax=ax_map, facecolor="none", edgecolor="cyan", linewidth=1.5
                )
                ax_map.set_aspect("equal", "box")
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                ax_map.set_title(f"Col {col} (Map)", fontsize=8)
                plt.tight_layout()
                cbar_ax = fig.add_axes([0.5, 0.27, 0.2, 0.05])
                cb = plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap_used, norm=norm_used),
                    cax=cbar_ax,
                    orientation="horizontal",
                )
                if plot_k:
                    cb.set_ticks(tick_locs)
                    cb.set_ticklabels(tick_labels)
                    cb.set_label("K (ft/day)", fontsize=9)
                else:
                    cb.set_ticklabels(list(color_names), rotation=90, fontsize=9)
                cbar_ax.set_facecolor((0.5, 0.5, 0.5, 0.3))
                cbar_ax.patch.set_edgecolor("black")
                cbar_ax.patch.set_linewidth(1.5)
                pdf_cols.savefig(fig)
                plt.close(fig)
        print(f"Saved column cross-sections to: {outpdf_cols}")


if __name__ == "__main__":
    draw_xsec = False  # If True, user picks cross-sections interactively
    make_rc_xsecs = True  # Generate row & column cross-sections
    plot_k = True  # Toggle to plot hydraulic conductivity values as discrete categories
    # with custom colors (and mask out inactive cells from idomain)
    # If not drawing interactively, use an empty GeoDataFrame or load from shapefile as needed
    xsecs = gpd.GeoDataFrame()
    plot_output(
        mod_dir=os.path.join("model_ws", "wahp_mf6"),
        ofpdir="history_plots",
        mod_name="wahp_mf6",
        sp=30,
        sp_info="stress_period_info.csv",
        sub=False,
        xsecs=xsecs,
        make_rc_xsecs=make_rc_xsecs,
        plot_k=plot_k,
    )
