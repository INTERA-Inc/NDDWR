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
# Create a GeoDataFrame from the model grid
# -------------------------------
def grid_to_geodf(gwf):
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
# Borehole plotting helper for Grouped geology (top cross‑section)
# -------------------------------
def plot_boreholes(ax, borings, cell_size, grouping_field=None):
    annotations = []  # collect annotation objects for adjust_text
    annotated_ids = set()  # track which bh_id has been annotated already

    # Determine grouping field.
    if grouping_field is not None:
        group_field = grouping_field
    elif "col_idx" in borings.columns and borings["col_idx"].notna().all():
        group_field = "col_idx"
    elif "row_idx" in borings.columns and borings["row_idx"].notna().all():
        group_field = "row_idx"
    else:
        group_field = None

    if group_field:
        groups = borings.groupby(group_field)
        for cell_idx, group in groups:
            bh_groups = group.groupby("bh_id")
            unique_ids = list(bh_groups.groups.keys())
            n = len(unique_ids)
            sub_width = cell_size / n  # horizontal slot width for each unique bh_id
            for j, bhid in enumerate(unique_ids):
                subgroup = bh_groups.get_group(bhid)
                x_offset = int(cell_idx) * cell_size + j * sub_width
                for _, bore in subgroup.iterrows():
                    # Determine vertical coordinates.
                    if pd.notna(bore["start_z"]) and pd.notna(bore["end_z"]):
                        z_top = bore["start_z"]
                        z_bot = bore["end_z"]
                    elif pd.notna(bore.get("Top_Screen_Elev")) and pd.notna(
                        bore.get("Bottom_Screen_Elev")
                    ):
                        z_top = bore["Top_Screen_Elev"]
                        z_bot = bore["Bottom_Screen_Elev"]
                    else:
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
                        else:
                            color = "blue"
                    else:
                        color = "blue"
                    rect = Rectangle(
                        (x_offset, z_lower),
                        sub_width,
                        z_height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2,
                        alpha=1.0,
                        zorder=5,
                    )
                    ax.add_patch(rect)
                    # Annotate unique bh_id only once.
                    if (
                        pd.notna(bhid)
                        and str(bhid).strip() != ""
                        and bhid not in annotated_ids
                    ):
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
        for idx, bore in borings.iterrows():
            if pd.notna(bore["start_z"]) and pd.notna(bore["end_z"]):
                z_top = bore["start_z"]
                z_bot = bore["end_z"]
            elif pd.notna(bore.get("Top_Screen_Elev")) and pd.notna(
                bore.get("Bottom_Screen_Elev")
            ):
                z_top = bore["Top_Screen_Elev"]
                z_bot = bore["Bottom_Screen_Elev"]
            else:
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
                else:
                    color = "blue"
            else:
                color = "blue"
            if "col_idx" in bore and pd.notna(bore["col_idx"]):
                x0 = int(bore["col_idx"]) * cell_size
                width = cell_size
            elif "row_idx" in bore and pd.notna(bore["row_idx"]):
                x0 = int(bore["row_idx"]) * cell_size
                width = cell_size
            else:
                continue
            rect = Rectangle(
                (x0, z_lower),
                width,
                z_height,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                alpha=1.0,
                zorder=5,
            )
            ax.add_patch(rect)
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
# New borehole plotting helper for Model Unit geology (bottom cross‑section)
# -------------------------------
def plot_boreholes_model_unit(ax, borings, cell_size, grouping_field=None):
    annotations = []
    annotated_ids = set()
    # Determine grouping field.
    if grouping_field is not None:
        group_field = grouping_field
    elif "col_idx" in borings.columns and borings["col_idx"].notna().all():
        group_field = "col_idx"
    elif "row_idx" in borings.columns and borings["row_idx"].notna().all():
        group_field = "row_idx"
    else:
        group_field = None

    # Color mapping for model_unit.
    def get_mu_color(mu):
        mu = str(mu).strip().lower()
        if mu == "wss":
            return "green"
        elif mu == "uc":
            return "tan"
        elif mu == "wsp":
            return "lightblue"
        elif mu == "dc":
            return "saddlebrown"
        elif mu == "wbv":
            return "navy"
        elif mu == "wr":
            return "purple"
        elif mu == "bot":
            return "black"
        else:
            return "blue"

    if group_field:
        groups = borings.groupby(group_field)
        for cell_idx, group in groups:
            bh_groups = group.groupby("bh_id")
            unique_ids = list(bh_groups.groups.keys())
            n = len(unique_ids)
            sub_width = cell_size / n
            for j, bhid in enumerate(unique_ids):
                subgroup = bh_groups.get_group(bhid)
                x_offset = int(cell_idx) * cell_size + j * sub_width
                for _, bore in subgroup.iterrows():
                    # Skip if the bore has both Top_Screen_Elev and Bottom_Screen_Elev defined,
                    # or if model_unit is "drop".
                    if (
                        pd.notna(bore.get("Top_Screen_Elev"))
                        and pd.notna(bore.get("Bottom_Screen_Elev"))
                    ) or (
                        pd.notna(bore["model_unit_4_lay"])
                        and str(bore["model_unit_4_lay"]).strip().lower() == "drop"
                    ):
                        continue

                    if pd.notna(bore["start_z"]) and pd.notna(bore["end_z"]):
                        z_top = bore["start_z"]
                        z_bot = bore["end_z"]
                    else:
                        continue
                    z_lower = min(z_top, z_bot)
                    z_height = abs(z_top - z_bot)
                    color = get_mu_color(bore["model_unit_4_lay"])
                    rect = Rectangle(
                        (x_offset, z_lower),
                        sub_width,
                        z_height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2,
                        alpha=1.0,
                        zorder=5,
                    )
                    ax.add_patch(rect)
                    if (
                        pd.notna(bhid)
                        and str(bhid).strip() != ""
                        and bhid not in annotated_ids
                    ):
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
        for idx, bore in borings.iterrows():
            # Skip if the bore has both Top_Screen_Elev and Bottom_Screen_Elev defined,
            # or if model_unit is "drop".
            if (
                pd.notna(bore.get("Top_Screen_Elev"))
                and pd.notna(bore.get("Bottom_Screen_Elev"))
            ) or (
                pd.notna(bore["model_unit_4_lay"])
                and str(bore["model_unit_4_lay"]).strip().lower() == "drop"
            ):
                continue

            if pd.notna(bore["start_z"]) and pd.notna(bore["end_z"]):
                z_top = bore["start_z"]
                z_bot = bore["end_z"]
            else:
                continue
            z_lower = min(z_top, z_bot)
            z_height = abs(z_top - z_bot)
            if "col_idx" in bore and pd.notna(bore["col_idx"]):
                x0 = int(bore["col_idx"]) * cell_size
                width = cell_size
            elif "row_idx" in bore and pd.notna(bore["row_idx"]):
                x0 = int(bore["row_idx"]) * cell_size
                width = cell_size
            else:
                continue
            color = get_mu_color(bore["model_unit_4_lay"])
            rect = Rectangle(
                (x0, z_lower),
                width,
                z_height,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                alpha=1.0,
                zorder=5,
            )
            ax.add_patch(rect)
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
# Helper function for plan view color mapping based on model_unit.
# -------------------------------
def get_planview_color(row):
    mu = str(row["model_unit_4_lay"]).strip().lower()
    if mu == "outside_grid":
        return "lightpink"
    elif mu == "drop":
        return "lightblue"
    else:
        return "purple"


# -------------------------------
# Main plotting function: Row & Column Cross‑Sections with both Grouped and Model Unit geology
# -------------------------------
def plot_row_and_col_xsecs_with_model_unit(mod_dir, model_name):
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
    top_elev = 1005.0
    bottom_elev = 300.0

    # Load base map shapefile.
    shp_path = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
    )
    mbndry = gpd.read_file(shp_path)
    mbndry = mbndry.to_crs(epsg=2265)

    # Create GeoDataFrame from model grid.
    grid_gdf = grid_to_geodf(gwf)

    # Read already condensed borehole data.
    bore_csv = os.path.join(
        "data", "analyzed", "wahp_wells_borings_combined_lith_condensed.csv"
    )
    df_bore = pd.read_csv(bore_csv)
    # Create GeoDataFrame from the borehole data.
    gdf_bore = gpd.GeoDataFrame(
        df_bore,
        geometry=gpd.points_from_xy(df_bore["start_x"], df_bore["start_y"]),
        crs="epsg:2265",
    )
    # Spatial join: assign row and col indices from the grid.
    joined = gpd.sjoin(
        gdf_bore,
        grid_gdf[["geometry", "row_idx", "col_idx"]],
        how="left",
        predicate="intersects",
    )

    # Define legend handles.
    legend_handles_grouped = [
        Patch(facecolor="blue", edgecolor="blue", label="Aquifer"),
        Patch(facecolor="brown", edgecolor="brown", label="Clay"),
        Patch(facecolor="grey", edgecolor="grey", label="Bedrock"),
    ]
    legend_handles_mu = [
        Patch(facecolor="green", edgecolor="green", label="WSS"),
        Patch(facecolor="tan", edgecolor="tan", label="UC"),
        Patch(facecolor="lightblue", edgecolor="lightblue", label="WSP"),
        Patch(facecolor="saddlebrown", edgecolor="saddlebrown", label="DC"),
        Patch(facecolor="navy", edgecolor="navy", label="WBV"),
        Patch(facecolor="purple", edgecolor="purple", label="WR"),
        Patch(facecolor="black", edgecolor="black", label="bot"),
    ]

    fout = os.path.join(mod_dir, "prelim_figs")
    if not os.path.exists(fout):
        os.makedirs(fout)

    # -------------------------------
    # Row cross‑sections PDF.
    # -------------------------------
    outpdf_rows = os.path.join(fout, "all_rows_with_model_structure.pdf")
    with PdfPages(outpdf_rows) as pdf_rows:
        for row in range(nrow_val):
            # Create a figure with GridSpec: left column (2 rows) and right column (1 col spanning both rows)
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[2.5, 1])
            ax_cs_grouped = fig.add_subplot(gs[0, 0])
            ax_cs_mu = fig.add_subplot(gs[1, 0])
            ax_map = fig.add_subplot(gs[:, 1])

            extent = (0, ncol_val * cell_size, bottom_elev, top_elev)

            # Top cross‑section: Grouped geology.
            xsect_grouped = flopy.plot.PlotCrossSection(
                model=gwf, ax=ax_cs_grouped, line={"row": row}, extent=extent
            )
            xsect_grouped.plot_grid(linewidth=0.5, color="black")
            ax_cs_grouped.set_title(f"Row {row} Cross‑Section (Grouped)")
            bore_row = joined[joined["row_idx"] == row]
            annotations_grouped = plot_boreholes(ax_cs_grouped, bore_row, cell_size)
            valid_annotations = [
                ann for ann in annotations_grouped if np.all(np.isfinite(ann.xy))
            ]
            if valid_annotations:
                adjust_text(
                    valid_annotations,
                    ax=ax_cs_grouped,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )
            ax_cs_grouped.legend(handles=legend_handles_grouped, loc="upper right")

            # Bottom cross‑section: Model Unit geology.
            xsect_mu = flopy.plot.PlotCrossSection(
                model=gwf, ax=ax_cs_mu, line={"row": row}, extent=extent
            )
            xsect_mu.plot_grid(linewidth=0.5, color="black")
            ax_cs_mu.set_title(f"Row {row} Cross‑Section (Model Unit)")
            annotations_mu = plot_boreholes_model_unit(ax_cs_mu, bore_row, cell_size)
            valid_annotations_mu = [
                ann for ann in annotations_mu if np.all(np.isfinite(ann.xy))
            ]
            if valid_annotations_mu:
                adjust_text(
                    valid_annotations_mu,
                    ax=ax_cs_mu,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )
            ax_cs_mu.legend(handles=legend_handles_mu, loc="upper right")

            # Plan view.
            joined_plan = joined[
                ~(
                    joined["Top_Screen_Elev"].notna()
                    & joined["Bottom_Screen_Elev"].notna()
                )
            ]
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
            # Use plan view color mapping.
            colors = joined_plan.apply(get_planview_color, axis=1)
            ax_map.scatter(
                joined_plan.geometry.x,
                joined_plan.geometry.y,
                color=colors,
                s=2,
                zorder=5,
            )

            plt.tight_layout()
            pdf_rows.savefig(fig)
            plt.close(fig)
    print(f"Saved row cross‑sections with model_unit to: {outpdf_rows}")

    # -------------------------------
    # Column cross‑sections PDF.
    # -------------------------------
    outpdf_cols = os.path.join(fout, "all_cols_with_model_structure.pdf")
    with PdfPages(outpdf_cols) as pdf_cols:
        for col in range(ncol_val):
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[2.5, 1])
            ax_cs_grouped = fig.add_subplot(gs[0, 0])
            ax_cs_mu = fig.add_subplot(gs[1, 0])
            ax_map = fig.add_subplot(gs[:, 1])

            extent = (0, nrow_val * cell_size, bottom_elev, top_elev)

            # For column cross‑sections, set grouping_field to "row_idx"
            # Top cross‑section: Grouped geology.
            xsect_grouped = flopy.plot.PlotCrossSection(
                model=gwf, ax=ax_cs_grouped, line={"column": col}, extent=extent
            )
            xsect_grouped.plot_grid(linewidth=0.5, color="black")
            ax_cs_grouped.set_title(f"Column {col} Cross‑Section (Grouped)")
            bore_col = joined[joined["col_idx"] == col]
            annotations_grouped = plot_boreholes(
                ax_cs_grouped, bore_col, cell_size, grouping_field="row_idx"
            )
            valid_annotations = [
                ann for ann in annotations_grouped if np.all(np.isfinite(ann.xy))
            ]
            if valid_annotations:
                adjust_text(
                    valid_annotations,
                    ax=ax_cs_grouped,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )
            ax_cs_grouped.legend(handles=legend_handles_grouped, loc="upper right")

            # Bottom cross‑section: Model Unit geology.
            xsect_mu = flopy.plot.PlotCrossSection(
                model=gwf, ax=ax_cs_mu, line={"column": col}, extent=extent
            )
            xsect_mu.plot_grid(linewidth=0.5, color="black")
            ax_cs_mu.set_title(f"Column {col} Cross‑Section (Model Unit)")
            annotations_mu = plot_boreholes_model_unit(
                ax_cs_mu, bore_col, cell_size, grouping_field="row_idx"
            )
            valid_annotations_mu = [
                ann for ann in annotations_mu if np.all(np.isfinite(ann.xy))
            ]
            if valid_annotations_mu:
                adjust_text(
                    valid_annotations_mu,
                    ax=ax_cs_mu,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )
            ax_cs_mu.legend(handles=legend_handles_mu, loc="upper right")

            # Plan view.
            # Filter out borings with both Top_Screen_Elev and Bottom_Screen_Elev
            joined_plan = joined[
                ~(
                    joined["Top_Screen_Elev"].notna()
                    & joined["Bottom_Screen_Elev"].notna()
                )
            ]
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
            colors = joined_plan.apply(get_planview_color, axis=1)
            ax_map.scatter(
                joined_plan.geometry.x,
                joined_plan.geometry.y,
                color=colors,
                s=2,
                zorder=5,
            )

            plt.tight_layout()
            pdf_cols.savefig(fig)
            plt.close(fig)
    print(f"Saved column cross‑sections with model_unit to: {outpdf_cols}")


if __name__ == "__main__":
    mod_dir = os.path.join("model_ws", "model_4_lay")
    model_name = "model_4_lay"
    plot_row_and_col_xsecs_with_model_unit(mod_dir, model_name)
