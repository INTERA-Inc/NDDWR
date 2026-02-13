import pandas as pd
import os

raw_data_folder = os.path.join("data", "raw")
analyzed_data_folder = os.path.join("data", "analyzed")

boring = pd.read_csv(os.path.join(raw_data_folder, "wahp_lith_grouped.csv"))
well = pd.read_excel(os.path.join(raw_data_folder, "wahp_waterlevels.xlsx"))

unique_wells = well.groupby("Site_Index").first().reset_index()

# Calculate the screen elevations:
# Assuming that Top_Screen and Bottom Screen are depths below Land_Surface_Elev_NVGD29.
unique_wells["Top_Screen_Elev"] = (
    unique_wells["Land_Surface_Elev_NVGD29"] - unique_wells["Top_Screen"]
)
unique_wells["Bottom_Screen_Elev"] = (
    unique_wells["Land_Surface_Elev_NVGD29"] - unique_wells["Bottom_Screen"]
)

# Create a unique identifier for wells based on their x,y coordinates (rounded to 3 decimals)
unique_wells["xy_id"] = (
    unique_wells["x"].round(3).astype(str)
    + "_"
    + unique_wells["y"].round(3).astype(str)
)

# --------------------------------------
# Step 2. Prepare the boring DataFrame
# --------------------------------------
# Create a unique identifier for boring records (based on start_x and start_y)
boring["xy_id"] = (
    boring["start_x"].round(3).astype(str)
    + "_"
    + boring["start_y"].round(3).astype(str)
)

# Since the boring DataFrame does not have well-specific fields,
# add columns for the well data, setting them to NaN.
boring_enriched = boring.copy()
boring_enriched["Top_Screen_Elev"] = pd.NA
boring_enriched["Bottom_Screen_Elev"] = pd.NA
boring_enriched["Site_Index"] = pd.NA
boring_enriched["Land_Surface_Elev_NVGD29"] = pd.NA  # well's surface elevation

# --------------------------------------
# Step 3. Prepare the well DataFrame to match boring format
# --------------------------------------
# For well records, we want the format to mimic the boring records.
# We map the well's x and y to the boring's start_x and start_y.
# The boring-specific fields (start_z, end_z, Grouped) are not available for wells.
wells_enriched = pd.DataFrame(
    {
        "xy_id": unique_wells["xy_id"],
        "start_z": pd.NA,  # No boring start depth
        "end_z": pd.NA,  # No boring end depth
        "start_x": unique_wells["x"],
        "start_y": unique_wells["y"],
        "Aquifer_Units": unique_wells["Aquifer"],  # Use well's aquifer info
        "Grouped": pd.NA,  # No lithologic grouping for wells
        "Top_Screen_Elev": unique_wells["Top_Screen_Elev"],
        "Bottom_Screen_Elev": unique_wells["Bottom_Screen_Elev"],
        "Site_Index": unique_wells["Site_Index"],
        "Land_Surface_Elev_NVGD29": unique_wells["Land_Surface_Elev_NVGD29"],
    }
)

# --------------------------------------
# Step 4. Combine the DataFrames
# --------------------------------------
# Vertically concatenate the boring records and the enriched unique well records.
combined_df = pd.concat([boring_enriched, wells_enriched], ignore_index=True)

combined_df.to_csv(
    os.path.join(analyzed_data_folder, "wahp_wells_borings_combined_lith.csv"),
    index=False,
)
