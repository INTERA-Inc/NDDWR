import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd


# -------------------------
# USER SETTINGS
# -------------------------
mou_folder = os.path.join("pest_mou_output")
mou_loc = os.path.join(mou_folder, "min_impact.dvpop.csv")
mou_count = os.path.join(mou_folder, "min_impact.count.csv")
LINKAGE_CSV = os.path.join(
    "data", "processed", "water_use", "deferred_investigation",
    "deferred_wel_linkage_summary.csv"
)

# <-- set this to your actual grid shapefile path
GRID_SHP = os.path.join("..", "..", "gis", "output_shps", "elk", "elk_cell_size_660ft_epsg2265_rot20.grid.shp")

OUT_DIR = os.path.join(mou_folder, "output_shapefiles", "mou_wells")
os.makedirs(OUT_DIR, exist_ok=True)

WRITE_SHP = True   # Shapefile (DBF field-name limits; keep fields short)
WRITE_GPKG = True  # GeoPackage (recommended)


# -------------------------
# Helpers
# -------------------------
IDX_RE = re.compile(r"_idx0:(-?\d+)_idx1:(-?\d+)_idx2:(-?\d+)\b")

CELL_RE = re.compile(r"k(\d+)_i(\d+)_j(\d+)", re.IGNORECASE)

def parse_cells_applied(cell_str: str):
    """
    Parse 'cells_applied' which may contain one or many tokens like:
      k01_i0015_j0018
    Returns list of (lay,row,col) ints (0-indexed).
    """
    if cell_str is None or (isinstance(cell_str, float) and np.isnan(cell_str)):
        return []

    s = str(cell_str).strip()
    if not s:
        return []

    # split on common delimiters, then parse each token
    parts = re.split(r"[,\;\|\s]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        m = CELL_RE.search(p)
        if not m:
            continue
        lay = int(m.group(1))
        row = int(m.group(2))
        col = int(m.group(3))
        out.append((lay, row, col))
    return out


def parse_pname_idxs(pname: str):
    """
    Extract (idx0, idx1, idx2) from strings like:
      pname:..._idx0:1_idx1:9_idx2:36
    Returns (lay,row,col) as ints, or (None,None,None) if not found.
    """
    m = IDX_RE.search(str(pname))
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _remove_shp(path_no_ext: str):
    for e in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = path_no_ext + e
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def to_cell_center_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert polygon geometries to point geometries at cell centers (centroids)."""
    out = gdf.copy()
    out["geometry"] = out.geometry.centroid
    return out

def write_outputs(gdf: gpd.GeoDataFrame, base_path_no_ext: str):
    """
    Write both .gpkg and .shp (optional) from a GeoDataFrame.
    For shapefile safety, keep field names <= 10 chars and avoid bool dtype.
    """
    # Clean up old shapefile set
    _remove_shp(base_path_no_ext)

    # Cast bools to int for DBF compatibility
    for c in gdf.columns:
        if c == "geometry":
            continue
        if pd.api.types.is_bool_dtype(gdf[c]):
            gdf[c] = gdf[c].astype(int)

    if WRITE_GPKG:
        gpkg_path = base_path_no_ext + ".gpkg"
        if os.path.exists(gpkg_path):
            try:
                os.remove(gpkg_path)
            except Exception:
                pass
        gdf.to_file(gpkg_path, driver="GPKG")
        print("Wrote:", gpkg_path)

    if WRITE_SHP:
        shp_path = base_path_no_ext + ".shp"
        gdf.to_file(shp_path)
        print("Wrote:", shp_path)


# -------------------------
# Read inputs
# -------------------------
mou_loc_df = pd.read_csv(mou_loc)
mou_count_df = pd.read_csv(mou_count)

# Count file: typically columns are ["Unnamed: 0", "0"]
# Make them sane:
if mou_count_df.shape[1] >= 2:
    c0 = mou_count_df.columns[0]
    c1 = mou_count_df.columns[1]
    mou_count_df = mou_count_df.rename(columns={c0: "pname", c1: "count"})
else:
    raise ValueError("Count CSV does not have 2 columns as expected.")

# Ensure numeric count
mou_count_df["count"] = pd.to_numeric(mou_count_df["count"], errors="coerce").fillna(0).astype(int)

# ------------------------------------------
# Build (lay,row,col,pod_id,permit) mapping WITHOUT aggregation
# ------------------------------------------
link = pd.read_csv(LINKAGE_CSV)

rows = []
for _, r in link.iterrows():
    pod = r.get("pod_id", None)
    perm = r.get("permit", None)
    cells = r.get("cells_applied", None)
    for (lay, row, col) in parse_cells_applied(cells):
        rows.append((lay, row, col, str(pod) if pd.notna(pod) else "", str(perm) if pd.notna(perm) else ""))

cell_pairs = pd.DataFrame(rows, columns=["lay", "row", "col", "pod_id", "permit"])

# total cells applied for each pod/permit combo (global, member-independent)
cells_total = (
    cell_pairs.drop_duplicates(subset=["lay", "row", "col", "pod_id", "permit"])
    .groupby(["pod_id", "permit"], as_index=False)
    .size()
    .rename(columns={"size": "cells_applied_total"})
)

print(f"Linkage combos: {len(cells_total)}")


# ------------------------------------------
# DVPOP long table (if you don't already have it)
# ------------------------------------------
id_cols = ["generation", "member"]
pname_cols = [c for c in mou_loc_df.columns if c not in id_cols]

dv_long = mou_loc_df.melt(
    id_vars=id_cols,
    value_vars=pname_cols,
    var_name="pname",
    value_name="dvpop",
)
dv_long["dvpop"] = pd.to_numeric(dv_long["dvpop"], errors="coerce").fillna(0).astype(int)

parsed = dv_long["pname"].apply(parse_pname_idxs)
dv_long["lay"] = [p[0] for p in parsed]
dv_long["row"] = [p[1] for p in parsed]
dv_long["col"] = [p[2] for p in parsed]
dv_long = dv_long.dropna(subset=["lay", "row", "col"])
dv_long[["lay", "row", "col"]] = dv_long[["lay", "row", "col"]].astype(int)

# Keep only dvpop==1 for the “wells appeared with a 1” requirement
dv1 = dv_long.loc[dv_long["dvpop"] == 1, ["generation", "member", "lay", "row", "col"]].copy()

# Join dvpop=1 cells to linkage pod/permit combos
dv1_linked = dv1.merge(
    cell_pairs.drop_duplicates(subset=["lay", "row", "col", "pod_id", "permit"]),
    on=["lay", "row", "col"],
    how="left",
)

# Some dvpop cells might not be in deferred linkage (pod/permit missing)
# Drop rows where pod_id/permit is missing for the summaries
dv1_linked = dv1_linked[(dv1_linked["pod_id"] != "") & (dv1_linked["permit"] != "")].copy()


# ------------------------------------------
# Make one CSV per member
# ------------------------------------------
OUT_MEMBER_DIR = os.path.join(OUT_DIR, "member_summaries")
os.makedirs(OUT_MEMBER_DIR, exist_ok=True)

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s[:120]

members = sorted(dv_long["member"].astype(str).unique())

for mem in members:
    # dvpop==1 unique cells by (pod,permit) for this member
    mem_df = dv1_linked.loc[dv1_linked["member"].astype(str) == str(mem)].copy()

    dv1_counts = (
        mem_df.drop_duplicates(subset=["lay", "row", "col", "pod_id", "permit"])
        .groupby(["pod_id", "permit"], as_index=False)
        .size()
        .rename(columns={"size": "dvpop1_cells"})
    )

    # Combine with total cells applied for that combo
    out = cells_total.merge(dv1_counts, on=["pod_id", "permit"], how="left")
    out["dvpop1_cells"] = out["dvpop1_cells"].fillna(0).astype(int)

    # OPTIONAL: keep only combos that had at least one dvpop==1 for this member
    # out = out.loc[out["dvpop1_cells"] > 0].copy()

    # Sort nicely
    out = out.sort_values(["pod_id", "permit"]).reset_index(drop=True)

    out_csv = os.path.join(OUT_MEMBER_DIR, f"member_{safe_name(mem)}_pod_permit_summary.csv")
    out.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

print("Done. Member CSVs in:", OUT_MEMBER_DIR)

# -------------------------
# Load grid shapefile
# -------------------------
grid = gpd.read_file(GRID_SHP)

# -------------------------
# Deferred linkage: add pod_id + permit by (lay,row,col)
# -------------------------
if os.path.exists(LINKAGE_CSV):
    link = pd.read_csv(LINKAGE_CSV)

    # Expect columns: pod_id, permit, cells_applied, ...
    need = {"pod_id", "permit", "cells_applied"}
    missing = need - set(map(str.lower, link.columns))
    # case-insensitive normalize if needed
    link.columns = [c.strip() for c in link.columns]

    # Build exploded table of one row per applied cell
    rows = []
    for _, r in link.iterrows():
        pod = r.get("pod_id", None)
        perm = r.get("permit", None)
        cells = r.get("cells_applied", None)
        for (lay, row, col) in parse_cells_applied(cells):
            rows.append((lay, row, col, pod, perm))

    if len(rows) == 0:
        print("[WARN] Linkage CSV found but no parsable cells_applied entries.")
        cell_map = pd.DataFrame(columns=["lay", "row", "col", "pod_id", "permit"])
    else:
        cell_map = pd.DataFrame(rows, columns=["lay", "row", "col", "pod_id", "permit"])

        # If multiple pods/permits map to same cell, aggregate as ';' separated strings
        def _agg_str(s):
            s = [str(x) for x in s if pd.notna(x) and str(x).strip() != ""]
            # preserve order but unique
            seen = set()
            out = []
            for x in s:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return ";".join(out)

        cell_map = (
            cell_map.groupby(["lay", "row", "col"], as_index=False)
            .agg({"pod_id": _agg_str, "permit": _agg_str})
        )

    # Merge onto both dv and count frames later using lay,row,col
    # We'll keep this as 'cell_map' and merge right before writing.
    print(f"Linkage map built for {len(cell_map)} unique cells.")
else:
    print(f"[WARN] Linkage CSV not found at: {LINKAGE_CSV}")
    cell_map = None


# Determine 0-based row/col from grid attributes
if "i" in grid.columns and "j" in grid.columns:
    gi = pd.to_numeric(grid["i"], errors="coerce").fillna(0).astype(int)      # row (0-based)
    gj = pd.to_numeric(grid["j"], errors="coerce").fillna(0).astype(int)      # col (0-based)
elif "row" in grid.columns and "col" in grid.columns:
    gi = pd.to_numeric(grid["row"], errors="coerce").fillna(1).astype(int) - 1  # row (1-based -> 0-based)
    gj = pd.to_numeric(grid["col"], errors="coerce").fillna(1).astype(int) - 1  # col (1-based -> 0-based)
else:
    raise ValueError("Grid shapefile must contain (i,j) or (row,col) fields.")

grid = grid.copy()
grid["__row__"] = gi
grid["__col__"] = gj
grid["__key__"] = list(zip(grid["__row__"].to_numpy(), grid["__col__"].to_numpy()))

# Fast lookup: (row,col) -> geometry (and any other grid attrs you want)
# We'll merge via __key__.


# -------------------------
# 1) DVPOP -> wells shapefile
# -------------------------
id_cols = ["generation", "member"]
pname_cols = [c for c in mou_loc_df.columns if c not in id_cols]

# Long format: one record per (generation, member, pname)
dv_long = mou_loc_df.melt(
    id_vars=id_cols,
    value_vars=pname_cols,
    var_name="pname",
    value_name="dvpop",
)

# Force dvpop to 0/1 int
dv_long["dvpop"] = pd.to_numeric(dv_long["dvpop"], errors="coerce").fillna(0).astype(int)

# Parse indices
parsed = dv_long["pname"].apply(parse_pname_idxs)
dv_long["lay"] = [p[0] for p in parsed]
dv_long["row"] = [p[1] for p in parsed]
dv_long["col"] = [p[2] for p in parsed]
dv_long = dv_long.dropna(subset=["row", "col"])  # keep only parseable ones
dv_long["row"] = dv_long["row"].astype(int)
dv_long["col"] = dv_long["col"].astype(int)

# Build join key to grid geometry
dv_long["__key__"] = list(zip(dv_long["row"].to_numpy(), dv_long["col"].to_numpy()))

# Merge geometry
dv_join = dv_long.merge(
    grid[["__key__", "geometry"]].copy(),
    on="__key__",
    how="left",
)

missing_geom = dv_join["geometry"].isna().sum()
if missing_geom > 0:
    print(f"[WARN] DVPOP: {missing_geom} records did not match any grid cell geometry (row/col out of range?)")

dv_gdf = gpd.GeoDataFrame(dv_join.drop(columns=["__key__"]), geometry="geometry", crs=grid.crs)

# Keep shapefile-safe short fields (<=10 chars); keep full 'pname' mainly for gpkg
# For SHP we’ll also write with short columns; GeoPackage keeps everything.
dv_out = dv_gdf.copy()
dv_out["gen"] = pd.to_numeric(dv_out["generation"], errors="coerce").fillna(-1).astype(int)
dv_out["mem"] = dv_out["member"].astype(str)

# For SHP: truncate member string (DBF can store longer, but keeping reasonable)
dv_out["mem"] = dv_out["mem"].str.slice(0, 50)

# Reorder / select columns
# - For SHP, long field names are auto-truncated by some stacks; we explicitly keep short ones.
dv_cols_keep = ["gen", "mem", "dvpop", "lay", "row", "col", "pname", "geometry"]
dv_out = dv_out[dv_cols_keep]

if cell_map is not None and not cell_map.empty:
    dv_out = dv_out.merge(cell_map, on=["lay", "row", "col"], how="left")
else:
    dv_out["pod_id"] = np.nan
    dv_out["permit"] = np.nan


write_outputs(dv_out, os.path.join(OUT_DIR, "min_impact_dvpop_wells"))


# -------------------------
# 2) COUNT -> wells shapefile
# -------------------------
ct = mou_count_df.copy()
parsed = ct["pname"].apply(parse_pname_idxs)
ct["lay"] = [p[0] for p in parsed]
ct["row"] = [p[1] for p in parsed]
ct["col"] = [p[2] for p in parsed]
ct = ct.dropna(subset=["row", "col"])
ct["row"] = ct["row"].astype(int)
ct["col"] = ct["col"].astype(int)

ct["__key__"] = list(zip(ct["row"].to_numpy(), ct["col"].to_numpy()))
ct_join = ct.merge(
    grid[["__key__", "geometry"]].copy(),
    on="__key__",
    how="left",
)

missing_geom = ct_join["geometry"].isna().sum()
if missing_geom > 0:
    print(f"[WARN] COUNT: {missing_geom} wells did not match any grid cell geometry (row/col out of range?)")

ct_gdf = gpd.GeoDataFrame(ct_join.drop(columns=["__key__"]), geometry="geometry", crs=grid.crs)

# Keep shapefile-safe short fields
ct_out = ct_gdf.copy()
ct_out["cnt"] = pd.to_numeric(ct_out["count"], errors="coerce").fillna(0).astype(int)

ct_cols_keep = ["cnt", "lay", "row", "col", "pname", "geometry"]
ct_out = ct_out[ct_cols_keep]

if cell_map is not None and not cell_map.empty:
    ct_out = ct_out.merge(cell_map, on=["lay", "row", "col"], how="left")
else:
    ct_out["pod_id"] = np.nan
    ct_out["permit"] = np.nan

# ✅ Convert grid-cell polygons to point features at cell centers
ct_out = to_cell_center_points(ct_out)

write_outputs(ct_out, os.path.join(OUT_DIR, "min_impact_count_wells_pts_baseline"))

print("Done. Outputs in:", OUT_DIR)
