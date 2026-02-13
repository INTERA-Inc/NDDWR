#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


# -------------------------------
# Process boreholes to create a thickness summary per contiguous model_unit block,
# inserting 0-thickness rows for missing intermediate units per the expected stratigraphic order.
# Expected order: WSS, WBV, DC, WR.
# -------------------------------
def process_boreholes(input_csv):
    """
    Reads in the condensed borehole CSV and produces a DataFrame with one row per
    contiguous model_unit block, filling in 0 thickness for missing intermediate units,
    but only when a lower block exists.

    Expected stratigraphic order (in lower case): WSS, WBV, DC, WR.

    For each borehole (bh_id):
      - Contiguous segments with the same model_unit are grouped.
      - Thickness is computed as (max(start_z) - min(end_z)) for that block.
      - Blocks for units 'outside_grid' and 'drop' are skipped.
      - Blocks for unit 'bot' are not output, but if present they indicate that the
        underlying WR block is complete (partial=False).
      - For the effective (output) blocks, if the borehole ends (i.e. no lower block exists),
        the last block is marked partial=True – except that if the last effective block is WR
        and a bot block is present, then WR is marked as not partial.
      - Additionally, if a gap exists between consecutive effective blocks (in the expected order)
        then missing intermediate units are inserted with a thickness of 0.

    Returns a DataFrame with columns: bh_id, x, y, unit, thickness, partial.
    """
    import pandas as pd

    df = pd.read_csv(input_csv)

    required = {"bh_id", "start_x", "start_y", "start_z", "end_z", "model_unit_4_lay"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns: {required}")

    # Define the expected stratigraphic order in lower case.
    expected_order = ["wss", "wbv", "dc", "wr"]

    out_rows = []
    # Process each borehole individually.
    for bh, group in df.groupby("bh_id"):
        # Sort descending by start_z (top first)
        group_sorted = group.sort_values("start_z", ascending=False).reset_index(
            drop=True
        )
        # Group contiguous segments by model_unit.
        group_sorted["block"] = (
            group_sorted["model_unit_4_lay"] != group_sorted["model_unit_4_lay"].shift()
        ).cumsum()
        blocks = group_sorted.groupby("block")

        effective_blocks = []  # list of blocks that will be output
        bot_present = False  # flag if a 'bot' block is encountered in this borehole

        for _, block_df in blocks:
            unit = block_df.iloc[0]["model_unit_4_lay"].strip().lower()
            if unit in ["outside_grid", "drop"]:
                continue
            if unit == "bot":
                bot_present = True
                # Do not add 'bot' blocks to effective output.
                continue
            bh_id = block_df.iloc[0]["bh_id"]
            x = block_df.iloc[0]["start_x"]
            y = block_df.iloc[0]["start_y"]
            top_val = block_df["start_z"].max()
            bottom_val = block_df["end_z"].min()
            thickness = top_val - bottom_val
            try:
                order_index = expected_order.index(unit)
            except ValueError:
                order_index = None
            effective_blocks.append(
                {
                    "bh_id": bh_id,
                    "x": x,
                    "y": y,
                    "unit": unit,
                    "thickness": thickness,
                    "order_index": order_index,
                    # 'partial' will be assigned later
                }
            )

        # Assign 'partial' flag for effective blocks:
        if effective_blocks:
            # By default, mark the last effective block as partial.
            for i, block in enumerate(effective_blocks):
                block["partial"] = i == len(effective_blocks) - 1
            # If the last effective block is WR and a bot block was encountered,
            # then mark that WR as complete (partial=False).
            if effective_blocks[-1]["unit"] == "wr" and bot_present:
                effective_blocks[-1]["partial"] = False

        # Fill in gaps between effective blocks.
        new_blocks = []
        if effective_blocks:
            new_blocks.append(effective_blocks[0])
            for i in range(len(effective_blocks) - 1):
                current = effective_blocks[i]
                nxt = effective_blocks[i + 1]
                # Only fill gaps if both current and next have a valid order_index,
                # and if the next block is more than one step below the current.
                if (
                    current["order_index"] is not None
                    and nxt["order_index"] is not None
                    and current["unit"] != "wr"
                    and nxt["order_index"] > current["order_index"] + 1
                ):
                    for missing_idx in range(
                        current["order_index"] + 1, nxt["order_index"]
                    ):
                        missing_unit = expected_order[missing_idx]
                        new_blocks.append(
                            {
                                "bh_id": current["bh_id"],
                                "x": current["x"],
                                "y": current["y"],
                                "unit": missing_unit,
                                "thickness": 0.0,
                                "partial": False,
                                "order_index": missing_idx,
                            }
                        )
                new_blocks.append(nxt)
        else:
            new_blocks = []

        # Append blocks for this borehole to the overall output.
        for block in new_blocks:
            out_rows.append(
                {
                    "bh_id": block["bh_id"],
                    "x": block["x"],
                    "y": block["y"],
                    "unit": block["unit"].upper(),  # output unit in uppercase
                    "thickness": block["thickness"],
                    "partial": block["partial"],
                }
            )

    return pd.DataFrame(out_rows)


# -------------------------------
# Write separate CSVs for each model_unit
# -------------------------------
def write_unit_csvs(df, output_dir):
    """
    For each distinct model_unit in the processed DataFrame,
    writes a CSV (with columns: bh_id, x, y, unit, thickness, partial).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for unit, unit_df in df.groupby("unit"):
        filename = os.path.join(
            output_dir, f"{str(unit).strip().lower()}_thickness.csv"
        )
        unit_df.to_csv(filename, index=False)
        print(f"Wrote {filename}")


# -------------------------------
# Plot a map view for each unit CSV in a 2 x 2 grid.
# Points are colored by thickness (gradated using 'viridis').
# Two shapefiles are overlaid:
#    - wahp_outline_full.shp (blue edges)
#    - updated_wahp_model_extent.shp (red edges)
# -------------------------------
def plot_unit_maps(output_dir):
    """
    Reads all CSV files ending with '_thickness.csv' from output_dir and creates a
    2 x 2 grid of scatter plots (map views). Points with thickness > 0 are colored by
    the thickness value using the 'viridis' colormap, and points with thickness 0 are
    plotted in grey. Two shapefiles are overlaid:
      - wahp_outline_full.shp (blue edges)
      - updated_wahp_model_extent.shp (red edges)
    """
    import glob
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    # Load the shapefiles and reproject to epsg:2265.
    shp_outline = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
    )
    shp_extent = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "updated_wahp_model_extent.shp"
    )
    outline_gdf = gpd.read_file(shp_outline).to_crs(epsg=2265)
    extent_gdf = gpd.read_file(shp_extent).to_crs(epsg=2265)

    csv_files = glob.glob(os.path.join(output_dir, "*_thickness.csv"))
    csv_files.sort()  # sort alphabetically by unit name
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        unit_name = os.path.basename(csv_file).replace("_thickness.csv", "").upper()
        ax = axes.flat[i]

        # Separate points with nonzero thickness and zero thickness.
        df_nonzero = df[df["thickness"] != 0]
        df_zero = df[df["thickness"] == 0]

        if not df_nonzero.empty:
            sc = ax.scatter(
                df_nonzero["x"],
                df_nonzero["y"],
                c=df_nonzero["thickness"],
                cmap="viridis",
                s=30,
            )
            fig.colorbar(sc, ax=ax, label="Thickness")
        if not df_zero.empty:
            ax.scatter(df_zero["x"], df_zero["y"], color="grey", s=30)

        # Overlay the shapefiles.
        outline_gdf.plot(ax=ax, facecolor="none", edgecolor="blue", lw=1)
        extent_gdf.plot(ax=ax, facecolor="none", edgecolor="red", lw=1)

        ax.set_title(unit_name)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Hide any unused subplots.
    for j in range(i + 1, nrows * ncols):
        axes.flat[j].axis("off")

    plt.tight_layout()
    fig_out = os.path.join(output_dir, "model_unit_4_lay_maps.pdf")
    plt.savefig(fig_out)
    print(f"Saved model unit maps to: {fig_out}")


def plot_unit_pdf_with_partial(unit, output_dir):
    """
    Reads the CSV file for the given unit (e.g., "WBV" or "WR") from output_dir,
    creates a map view where points are colored by thickness (viridis for nonzero,
    grey for 0 thickness) and flags points with partial==True by over-plotting
    a red ring. The map is saved as a separate PDF.
    """
    csv_file = os.path.join(output_dir, f"{unit.lower()}_thickness.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file for unit {unit} not found in {output_dir}")
        return

    df = pd.read_csv(csv_file)

    # Load the shapefiles and reproject to EPSG:2265.
    shp_outline = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
    )
    shp_extent = os.path.join(
        "..", "..", "gis", "input_shps", "wahp", "updated_wahp_model_extent.shp"
    )
    outline_gdf = gpd.read_file(shp_outline).to_crs(epsg=2265)
    extent_gdf = gpd.read_file(shp_extent).to_crs(epsg=2265)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Separate points with nonzero thickness and those with 0 thickness.
    df_nonzero = df[df["thickness"] != 0]
    df_zero = df[df["thickness"] == 0]

    # Plot nonzero thickness points with a color scale.
    if not df_nonzero.empty:
        sc = ax.scatter(
            df_nonzero["x"],
            df_nonzero["y"],
            c=df_nonzero["thickness"],
            cmap="viridis",
            s=50,
        )
        fig.colorbar(sc, ax=ax, label="Thickness")
    # Plot points with 0 thickness in grey.
    if not df_zero.empty:
        ax.scatter(df_zero["x"], df_zero["y"], color="grey", s=50)

    # Flag points where partial == True with a red ring.
    df_partial = df[df["partial"] == True]
    if not df_partial.empty:
        # Overplot with larger markers with no fill and red edge.
        ax.scatter(
            df_partial["x"],
            df_partial["y"],
            s=70,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
        )

    # Overlay the shapefiles.
    outline_gdf.plot(ax=ax, facecolor="none", edgecolor="blue", lw=1)
    extent_gdf.plot(ax=ax, facecolor="none", edgecolor="red", lw=1)

    ax.set_title(unit.upper())
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    pdf_file = os.path.join(output_dir, f"{unit.lower()}_map.pdf")
    plt.savefig(pdf_file)
    plt.close(fig)
    print(f"Saved map for {unit} to: {pdf_file}")


if __name__ == "__main__":
    # Path to the condensed borehole CSV.
    input_csv = os.path.join(
        "data", "analyzed", "wahp_wells_borings_combined_lith_condensed.csv"
    )
    output_dir = os.path.join("data", "analyzed", "model_unit_thickness_4_lay")

    # Process the borehole data (including gap filling for 0 thickness where applicable).
    df_units = process_boreholes(input_csv)
    # Write separate CSVs for each model_unit.
    write_unit_csvs(df_units, output_dir)
    # Generate map views with the two shapefile overlays.
    plot_unit_maps(output_dir)

    for unit in ["WBV", "WR"]:
        plot_unit_pdf_with_partial(unit, output_dir)
