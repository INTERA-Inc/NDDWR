#!/usr/bin/env python3
import os
import flopy
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Patch
from pyproj import CRS
from shapely.geometry import Polygon
from adjustText import adjust_text  # pip install adjustText
import numpy as np


# -------------------------------
# Data Processing: Condense Borehole Data and add Unique ID per xy_id
# -------------------------------
def condense_boring_data(df):
    """
    Condense contiguous borehole segments with the same 'Grouped' value into one row.
    Assumes:
      - Each boring is identified by 'xy_id'
      - Data are sorted top-to-bottom (i.e. descending start_z)
    For each contiguous segment, this function aggregates:
      - 'Grouped': lithology from the segment
      - 'start_z': the maximum start_z (top of the unit)
      - 'end_z': the minimum end_z (bottom of the unit)
      - 'start_x' and 'start_y': taken from the first row (assumed constant for the boring)
    After condensing, a new unique borehole id ("bh_id") is added for each unique xy_id.
    """
    # Sort by boring id and descending start_z
    df = df.sort_values(["xy_id", "start_z"], ascending=[True, False]).copy()

    # Mark contiguous segments where the lithology (Grouped) changes.
    df["group_block"] = df.groupby("xy_id")["Grouped"].transform(
        lambda x: (x != x.shift()).cumsum()
    )

    # Group by xy_id and the contiguous block indicator, and aggregate.
    df_condensed = (
        df.groupby(["xy_id", "group_block"])
        .agg(
            {
                "Grouped": "first",
                "Aquifer_Units": "first",  # if present
                "start_x": "first",
                "start_y": "first",
                "start_z": "max",  # top of the segment
                "end_z": "min",  # bottom of the segment
                "Top_Screen_Elev": "first",  # if needed
                "Bottom_Screen_Elev": "first",  # if needed
            }
        )
        .reset_index()
    )

    # Create a unique borehole id for each unique xy_id (all segments from the same xy_id get the same bh_id)
    df_condensed["bh_id"] = "bh_" + df_condensed["xy_id"].astype(
        "category"
    ).cat.codes.add(1).astype(str)

    return df_condensed


# -------------------------------
# Create a GeoDataFrame from the model grid
# -------------------------------
def grid_to_geodf(gwf):
    """
    Create a GeoDataFrame from a FloPy model grid.
    Each grid cell is represented as a polygon with 0-indexed row and column indices.
    """
    mg = gwf.modelgrid
    cells = []
    for row in range(mg.nrow):
        for col in range(mg.ncol):
            vertices = mg.get_cell_vertices(row, col)
            poly = Polygon(vertices)
            cells.append({"row_idx": row, "col_idx": col, "geometry": poly})
    gdf = gpd.GeoDataFrame(cells, crs="epsg:2265")
    return gdf


# -------------------------------
# Helper functions for grid segments (for mapping panels)
# -------------------------------
def get_row_segments(gwf, row):
    mg = gwf.modelgrid
    ncol = mg.ncol
    segments = []
    for col in range(ncol):
        verts = list(mg.get_cell_vertices(row, col))
        if verts[0] != verts[-1]:
            verts.append(verts[0])
        for i in range(len(verts) - 1):
            segments.append([verts[i], verts[i + 1]])
    return segments


def get_col_segments(gwf, col):
    mg = gwf.modelgrid
    nrow = mg.nrow
    segments = []
    for row in range(nrow):
        verts = list(mg.get_cell_vertices(row, col))
        if verts[0] != verts[-1]:
            verts.append(verts[0])
        for i in range(len(verts) - 1):
            segments.append([verts[i], verts[i + 1]])
    return segments


# -------------------------------
# Borehole plotting helper with single bh_id annotation per xy_id
# -------------------------------
def plot_boreholes(ax, borings, cell_size):
    """
    Plot borehole rectangles on axis ax.
    For each borehole segment in the GeoDataFrame "borings", a rectangle is drawn.
    The fill color is based on "Grouped" (blue for aquifer, brown for clay, grey for bedrock).
    For segments in the same grid cell:
      - If they have the same bh_id, they are drawn at the same horizontal location.
      - If they have different bh_ids, they are subdivided and drawn side-by-side.
    Leader arrows annotate each unique bh_id only once.
    Returns a list of annotation objects.
    """
    annotations = []  # collect annotation objects for adjust_text
    annotated_ids = set()  # track which bh_id has been annotated already

    # Determine grouping field for cell assignment.
    if "col_idx" in borings.columns and borings["col_idx"].notna().all():
        group_field = "col_idx"
    elif "row_idx" in borings.columns and borings["row_idx"].notna().all():
        group_field = "row_idx"
    else:
        group_field = None

    if group_field:
        groups = borings.groupby(group_field)
        for cell_idx, group in groups:
            # Group the segments in this cell by bh_id.
            bh_groups = group.groupby("bh_id")
            unique_ids = list(bh_groups.groups.keys())
            n = len(unique_ids)
            sub_width = cell_size / n  # horizontal slot width for each unique bh_id
            for j, bhid in enumerate(unique_ids):
                subgroup = bh_groups.get_group(bhid)
                x_offset = int(cell_idx) * cell_size + j * sub_width
                # Plot every segment from this borehole (they'll be drawn at the same x_offset)
                for _, bore in subgroup.iterrows():
                    # Determine vertical coordinates.
                    if pd.notna(bore["start_z"]) and pd.notna(bore["end_z"]):
                        z_top = bore["start_z"]
                        z_bot = bore["end_z"]
                    elif pd.notna(bore["Top_Screen_Elev"]) and pd.notna(
                        bore["Bottom_Screen_Elev"]
                    ):
                        z_top = bore["Top_Screen_Elev"]
                        z_bot = bore["Bottom_Screen_Elev"]
                    else:
                        print(
                            f"Skipping borehole at cell {bore[group_field]} due to missing elevation data."
                        )
                        continue
                    z_lower = min(z_top, z_bot)
                    z_height = abs(z_top - z_bot)

                    # Determine color based on "Grouped"
                    if pd.notna(bore["Grouped"]) and str(bore["Grouped"]).strip() != "":
                        grp_lower = str(bore["Grouped"]).strip().lower()
                        if grp_lower == "aquifer":
                            color = "blue"
                        elif grp_lower == "clay":
                            color = "brown"
                        elif grp_lower == "bedrock":
                            color = "grey"
                        elif grp_lower == "blank":
                            continue
                        else:
                            color = "blue"
                    else:
                        color = "blue"
                    facecolor = color
                    edgecolor = color

                    rect = Rectangle(
                        (x_offset, z_lower),
                        sub_width,
                        z_height,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linewidth=2,
                        alpha=1.0,
                        zorder=5,
                    )
                    ax.add_patch(rect)

                    # Annotate inside the rectangle if Aquifer_Units exists.
                    if (
                        pd.notna(bore.get("Aquifer_Units", ""))
                        and str(bore.get("Aquifer_Units", "")).strip() != ""
                    ):
                        ax.text(
                            x_offset + sub_width / 2,
                            z_lower + z_height / 2,
                            str(bore["Aquifer_Units"]),
                            color="black",
                            fontsize=8,
                            ha="center",
                            va="center",
                            zorder=6,
                        )
                # Now annotate the unique borehole id (if not already annotated).
                if (
                    pd.notna(bhid)
                    and str(bhid).strip() != ""
                    and bhid not in annotated_ids
                ):
                    # Place annotation above the top of the highest segment for this bh_id.
                    # We take the maximum (z_lower+z_height) among the segments.
                    top_y = subgroup.apply(
                        lambda r: min(r["start_z"], r["end_z"])
                        + abs(r["start_z"] - r["end_z"]),
                        axis=1,
                    ).max()
                    ann = ax.annotate(
                        str(bhid),
                        xy=(x_offset + sub_width / 2, top_y),
                        xytext=(x_offset + sub_width / 2, top_y + 2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                        fontsize=8,
                        ha="center",
                        va="bottom",
                        zorder=6,
                    )
                    annotations.append(ann)
                    annotated_ids.add(bhid)
    else:
        # Fallback if no grouping field exists.
        for idx, bore in borings.iterrows():
            if pd.notna(bore["start_z"]) and pd.notna(bore["end_z"]):
                z_top = bore["start_z"]
                z_bot = bore["end_z"]
            elif pd.notna(bore["Top_Screen_Elev"]) and pd.notna(
                bore["Bottom_Screen_Elev"]
            ):
                z_top = bore["Top_Screen_Elev"]
                z_bot = bore["Bottom_Screen_Elev"]
            else:
                print(
                    f"Skipping borehole at index {idx} due to missing elevation data."
                )
                continue
            z_lower = min(z_top, z_bot)
            z_height = abs(z_top - z_bot)
            if pd.notna(bore["Grouped"]) and str(bore["Grouped"]).strip() != "":
                grp_lower = str(bore["Grouped"]).strip().lower()
                if grp_lower == "aquifer":
                    color = "blue"
                elif grp_lower == "clay":
                    color = "brown"
                elif grp_lower == "bedrock":
                    color = "grey"
                elif grp_lower == "blank":
                    continue
                else:
                    color = "blue"
            else:
                color = "blue"
            facecolor = color
            edgecolor = color
            if "col_idx" in bore and pd.notna(bore["col_idx"]):
                x0 = int(bore["col_idx"]) * cell_size
                width = cell_size
            elif "row_idx" in bore and pd.notna(bore["row_idx"]):
                x0 = int(bore["row_idx"]) * cell_size
                width = cell_size
            else:
                print(f"Skipping borehole at index {idx} due to missing group index.")
                continue

            rect = Rectangle(
                (x0, z_lower),
                width,
                z_height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=2,
                alpha=1.0,
                zorder=5,
            )
            ax.add_patch(rect)
            if (
                pd.notna(bore.get("Aquifer_Units", ""))
                and str(bore.get("Aquifer_Units", "")).strip() != ""
            ):
                ax.text(
                    x0 + width / 2,
                    z_lower + z_height / 2,
                    str(bore["Aquifer_Units"]),
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=6,
                )
            bhid = bore.get("bh_id", "")
            if pd.notna(bhid) and str(bhid).strip() != "" and bhid not in annotated_ids:
                ann = ax.annotate(
                    str(bhid),
                    xy=(x0 + width / 2, z_lower + z_height),
                    xytext=(x0 + width / 2, z_lower + z_height + 2),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    zorder=6,
                )
                annotations.append(ann)
                annotated_ids.add(bhid)
    return annotations


# -------------------------------
# Main plotting function: Row & Column Cross‑Sections
# -------------------------------
def plot_row_and_col_xsecs(mod_dir, model_name):
    """
    Load the MODFLOW 6 model and produce PDFs of row and column cross-sections.
    Boreholes (from a CSV) are overlaid on the cross-section panels and annotated
    with their unique borehole id (bh_id). Annotation labels are adjusted to avoid overlap.
    """
    # Load simulation and model.
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=mod_dir, sim_name=f"{model_name}.nam", exe_name="mf6.exe"
    )
    gwf = sim.get_model(model_name)
    mg = gwf.modelgrid
    mg.set_coord_info(
        xoff=mg.xoffset, yoff=mg.yoffset, angrot=mg.angrot, crs="epsg:2265"
    )

    nrow_val = mg.nrow
    ncol_val = mg.ncol
    cell_size = mg.delr[0]
    top_elev = 1000.0
    bottom_elev = 300.0

    # Load base map shapefile.
    shp_path = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
    )
    mbndry = gpd.read_file(shp_path)
    mbndry = mbndry.to_crs(epsg=2265)

    # Create GeoDataFrame from model grid.
    grid_gdf = grid_to_geodf(gwf)

    # Read boreholes CSV and convert to GeoDataFrame.
    bore_csv = os.path.join("data", "analyzed", "wahp_wells_borings_combined_lith.csv")
    df_bore = pd.read_csv(bore_csv)

    # Condense borehole data and assign unique bh_id per xy_id.
    df_condensed = condense_boring_data(df_bore)
    out_csv = os.path.join(
        "data", "analyzed", "wahp_wells_borings_combined_lith_condensed.csv"
    )
    # df_condensed.to_csv(out_csv, index=False)
    # print(f"Condensed borehole data saved to: {out_csv}")

    # Create GeoDataFrame from condensed data.
    gdf_bore = gpd.GeoDataFrame(
        df_condensed,
        geometry=gpd.points_from_xy(df_condensed["start_x"], df_condensed["start_y"]),
        crs="epsg:2265",
    )
    # Spatial join: assign row and col indices from the grid.
    joined = gpd.sjoin(
        gdf_bore,
        grid_gdf[["geometry", "row_idx", "col_idx"]],
        how="left",
        predicate="intersects",
    )
    print("Unique Grouped values:", joined["Grouped"].unique())

    # Define legend handles.
    legend_handles = [
        Patch(facecolor="blue", edgecolor="blue", label="Aquifer"),
        Patch(facecolor="brown", edgecolor="brown", label="Clay"),
        Patch(facecolor="grey", edgecolor="grey", label="Bedrock"),
    ]

    fout = os.path.join(mod_dir, "prelim_figs")
    if not os.path.exists(fout):
        os.makedirs(fout)

    # -------------------------------
    # Row cross-sections PDF.
    # -------------------------------
    outpdf_rows = os.path.join(fout, "all_rows.pdf")
    with PdfPages(outpdf_rows) as pdf_rows:
        for row in range(nrow_val):
            fig, (ax_cs, ax_map) = plt.subplots(
                1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [2.5, 1]}
            )
            extent = (0, ncol_val * cell_size, bottom_elev, top_elev)
            xsect = flopy.plot.PlotCrossSection(
                model=gwf, ax=ax_cs, line={"row": row}, extent=extent
            )
            xsect.plot_grid(linewidth=0.5, color="black")
            ax_cs.set_title(f"Row {row} Cross-Section")

            bore_row = joined[joined["row_idx"] == row]
            annotations = plot_boreholes(ax_cs, bore_row, cell_size)
            # Filter out annotations with non-finite coordinates.
            valid_annotations = [
                ann for ann in annotations if np.all(np.isfinite(ann.xy))
            ]
            if valid_annotations:
                adjust_text(
                    valid_annotations,
                    ax=ax_cs,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )
            ax_cs.legend(handles=legend_handles, loc="upper right")

            mbndry.plot(ax=ax_map, facecolor="none", edgecolor="blue", lw=1)
            pmv = flopy.plot.PlotMapView(model=gwf, ax=ax_map)
            pmv.plot_grid(edgecolor="gray", lw=0.5)
            segments = get_row_segments(gwf, row)
            lc = LineCollection(segments, colors="red", linewidths=1.5)
            ax_map.add_collection(lc)
            ax_map.set_title(f"Row {row} Map")
            ax_map.set_aspect("equal")
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.scatter(
                joined.geometry.x, joined.geometry.y, color="purple", s=2, zorder=5
            )

            plt.tight_layout()
            pdf_rows.savefig(fig)
            plt.close(fig)
    print(f"Saved row cross-sections to: {outpdf_rows}")

    # -------------------------------
    # Column cross‑sections PDF.
    # -------------------------------
    outpdf_cols = os.path.join(fout, "all_cols.pdf")
    with PdfPages(outpdf_cols) as pdf_cols:
        for col in range(ncol_val):
            fig, (ax_cs, ax_map) = plt.subplots(
                1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [2.5, 1]}
            )
            extent = (0, nrow_val * cell_size, bottom_elev, top_elev)
            xsect = flopy.plot.PlotCrossSection(
                model=gwf, ax=ax_cs, line={"column": col}, extent=extent
            )
            xsect.plot_grid(linewidth=0.5, color="black")
            ax_cs.set_title(f"Column {col} Cross‑Section")

            bore_col = joined[joined["col_idx"] == col]
            # Specify grouping_field="row_idx" so that the boreholes are positioned
            # correctly along the horizontal axis (which represents row index in column cross‑sections)
            annotations = plot_boreholes(
                ax_cs, bore_col, cell_size, grouping_field="row_idx"
            )
            valid_annotations = [
                ann for ann in annotations if np.all(np.isfinite(ann.xy))
            ]
            if valid_annotations:
                adjust_text(
                    valid_annotations,
                    ax=ax_cs,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )
            ax_cs.legend(handles=legend_handles, loc="upper right")

            mbndry.plot(ax=ax_map, facecolor="none", edgecolor="blue", lw=1)
            pmv = flopy.plot.PlotMapView(model=gwf, ax=ax_map)
            pmv.plot_grid(edgecolor="gray", lw=0.5)
            segments = get_col_segments(gwf, col)
            lc = LineCollection(segments, colors="red", linewidths=1.5)
            ax_map.add_collection(lc)
            ax_map.set_title(f"Column {col} Map")
            ax_map.set_aspect("equal")
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.scatter(
                joined.geometry.x, joined.geometry.y, color="purple", s=2, zorder=5
            )

            plt.tight_layout()
            pdf_cols.savefig(fig)
            plt.close(fig)
    print(f"Saved column cross‑sections to: {outpdf_cols}")


if __name__ == "__main__":
    mod_dir = os.path.join("model_ws", "model_7layer")
    model_name = "model_7layer"
    plot_row_and_col_xsecs(mod_dir, model_name)
