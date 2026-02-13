#!/usr/bin/env python
# make_use_by_well.py
# -------------------------------------------------------------
import os, re, calendar
import numpy as np
import pandas as pd
import geopandas as gpd

# ---------- filenames (expected in same folder) ---------------
WHP_MONTHLY_WELLS = "P1822_P1898_Wells123_WaterUse_2003-2024_fixed_2024.xlsx"
WHP_ANNUAL_TXT = "WahpetonBVA_Water use annual Acft PerPOD.txt"
WHP_MONTHLY_TXT = "WahpetonBVA_Water use monthly gallons PerPermit.txt"
BRECK_EXCEL = "Water use_Breckenridge.xlsx"
BRECK_TAB = "City of Breckenridge"
WELLS_SHP = "WahpetonBV_Wells_with5721.shp"

# ---------------- constants -----------------------------------
ACFT_PER_GAL = 1 / 325_851
FT3_PER_ACFT = 43_560
MONTHS = [
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
]
MONTH_TO_INT = {m: i for i, m in enumerate(MONTHS, 1)}
KEYCOLS = ["Permit", "Year", "Month", "Well", "permit_holder_name", "use_acft"]


# --------------------------------------------------------------
def _clean_keys(df):
    """Standardise join-key dtypes."""
    df[["use_year", "permit_number"]] = (
        df[["use_year", "permit_number"]]
        .apply(pd.to_numeric, errors="coerce")
        .round()
        .astype("Int64")
    )
    return df


# =================  MAIN PIPELINE  ============================
def build_use_by_well(*, monthly=True, out_dir="."):

    # ── 1 ▸ read raw tables ───────────────────────────────────
    annual_df = pd.read_csv(
        WHP_ANNUAL_TXT,
        sep="\t",
        engine="python",
        comment="#",
        skip_blank_lines=True,
        quoting=3,
    )
    annual_clean = _clean_keys(annual_df.copy())

    permit_txt = pd.read_csv(
        WHP_MONTHLY_TXT,
        sep="\t",
        engine="python",
        comment="#",
        skip_blank_lines=True,
        quoting=3,
    )

    # ── 2 ▸ Breckenridge  -------------------------------------
    breck = pd.read_excel(BRECK_EXCEL, BRECK_TAB, engine="openpyxl").query(
        "permit_status.str.lower() != 'inactive'"
    )

    breck_map = {130573: "13304728DDAADB", 130574: "13304727CCCBBD"}
    use_cols = [c for c in breck.columns if re.fullmatch(r"use_\d{4}_mg", c)]

    breck_long = (
        breck[breck["well_number"].isin(breck_map)]
        .assign(Well=lambda d: d["well_number"].map(breck_map))
        .melt(
            id_vars=["Well"], value_vars=use_cols, var_name="ycol", value_name="use_mg"
        )
        .dropna(subset=["use_mg"])
    )
    breck_long["Year"] = breck_long["ycol"].str.extract(r"use_(\d{4})").astype(int)
    breck_long["use_acft_ann"] = breck_long["use_mg"] * 1_000_000 * ACFT_PER_GAL
    breck_long["Permit"] = breck_long["Well"].map({v: k for k, v in breck_map.items()})
    breck_long["permit_holder_name"] = "City of Breckenridge"

    breck_annual_block = breck_long.loc[breck_long["Year"] < 2000].assign(
        Month=pd.NA, use_acft=lambda d: d["use_acft_ann"]
    )[KEYCOLS]
    breck_monthly_block = (
        breck_long.loc[breck_long["Year"] >= 2000]
        .merge(pd.DataFrame({"Month": MONTHS}), how="cross")
        .assign(use_acft=lambda d: d["use_acft_ann"] / 12)[KEYCOLS]
    )

    # ── 3 ▸ Wahpeton by-well Excel (2003-24) ------------------
    excel = pd.ExcelFile(WHP_MONTHLY_WELLS, engine="openpyxl")
    rename_map = {
        "Permit_1822_Well2_gal": "13304720ABD",
        "Permit_1898_Well1_gal": "13304720BBA",
        "Permit_1898_Well3_gal": "13304720ADD",
    }
    rec = []
    for yr in [s for s in excel.sheet_names if s.isdigit()]:
        raw = pd.read_excel(excel, yr, header=None)
        start = raw.index[raw.iloc[:, 0].astype(str).str.upper() == "JANUARY"][0]
        df = raw.iloc[start : start + 12].dropna(axis=1, how="all").copy()

        header = [
            "Month",
            "Permit_1822_Well2_gal",
            "Permit_1898_Well1_gal",
            "Permit_1898_Well3_gal",
        ]
        extra = df.shape[1] - len(header)
        if extra == 2:
            header += ["Total_Pumped_gal", "Treated_Pumped_gal"]
        elif extra == 3:
            header += [
                "Permit_1898_Combined_gal",
                "Total_Pumped_gal",
                "Treated_Pumped_gal",
            ]
        else:
            raise ValueError(f"Unexpected cols ({df.shape[1]}) sheet {yr}")
        df.columns = header
        df.rename(columns=rename_map, inplace=True)
        df["Year"] = int(yr)
        rec.append(df)

    wells_m = (
        pd.concat(rec, ignore_index=True)
        .melt(
            id_vars=["Year", "Month"],
            value_vars=list(rename_map.values()),
            var_name="Well",
            value_name="gallons",
        )
        .dropna()
    )
    wells_m["Permit"] = wells_m["Well"].map(
        {"13304720ABD": 1822, "13304720BBA": 1898, "13304720ADD": 1898}
    )
    wells_m["use_acft"] = wells_m["gallons"] * ACFT_PER_GAL
    wells_m["permit_holder_name"] = "WAHPETON, CITY OF"
    wells_m = wells_m[KEYCOLS]

    # ── 4 ▸ permit_raw tidy / glitch fix ----------------------
    permit_raw = _clean_keys(permit_txt.copy())
    mpat = re.compile(r"^use_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
    mcols = [c for c in permit_raw.columns if mpat.match(c)]
    permit_raw[mcols] = (
        permit_raw[mcols]
        .replace(r"[,\s]", "", regex=True)
        .replace("-", np.nan)
        .apply(pd.to_numeric, errors="coerce")
    )
    permit_raw["total_use_gal"] = permit_raw[mcols].sum(axis=1, skipna=True)
    permit_raw["total_use_acft"] = permit_raw["total_use_gal"] * ACFT_PER_GAL
    glitch = permit_raw["permit_number"].isin([4121, 3898]) & permit_raw[
        "use_year"
    ].between(2021, 2022)
    permit_raw.loc[glitch, mcols + ["total_use_gal"]] *= 1_000_000
    permit_raw["total_use_acft"] = permit_raw["total_use_gal"] * ACFT_PER_GAL

    # ── 5 ▸ data blocks  --------------------------------------
    # 5-a  annual 73-79  (all permits, incl. Minn-Dak)
    annual73 = (
        annual_clean.query("1973<=use_year<=1979")
        .groupby(["permit_number", "use_year"], as_index=False)
        .agg(
            use_acft=("reported_acft", "sum"),
            permit_holder_name=("permit_holder_name", "first"),
        )
        .rename(columns={"permit_number": "Permit", "use_year": "Year"})
        .assign(Month=pd.NA, Well=pd.NA)[KEYCOLS]
    )

    # 5-b  annual totals 80-99
    totals = (
        permit_raw.query("1980<=use_year<=1999")
        .rename(columns={"use_year": "Year", "permit_number": "Permit"})
        .assign(Month=pd.NA, Well=pd.NA, use_acft=lambda d: d["total_use_acft"])[
            KEYCOLS
        ]
    )

    # 5-c  monthly permit rows 2000-24 (except WAH 1822/1898 after 2002)
    mmask = permit_raw["use_year"].between(2000, 2024) & ~(
        (permit_raw["permit_number"].isin([1822, 1898]))
        & (permit_raw["use_year"] >= 2003)
    )
    melted = (
        permit_raw.loc[mmask]
        .melt(
            id_vars=["use_year", "permit_number", "permit_holder_name"],
            value_vars=mcols,
            var_name="mcol",
            value_name="gallons",
        )
        .dropna()
    )
    MONTH_NAME = {f"use_{m.lower()}": m.upper() for m in MONTHS}
    melted["Month"] = melted["mcol"].map(MONTH_NAME)
    melted["Year"] = melted["use_year"].astype(int)
    melted["Permit"] = melted["permit_number"].astype(int)
    melted["use_acft"] = melted["gallons"] * ACFT_PER_GAL
    melted["Well"] = pd.NA
    permit_monthly = melted[KEYCOLS]

    # 5-d  glue everything
    combined = pd.concat(
        [
            annual73,
            totals,
            permit_monthly,
            wells_m,
            breck_annual_block,
            breck_monthly_block,
        ],
        ignore_index=True,
    )

    # ── 6 ▸ resolve to well-level ------------------------------
    frames = [combined[combined["Well"].notna()]]
    W1, W2, W3 = "13304720ABD", "13304720ADD", "13304720BBA"
    MD1, MD2 = "13304720ABDAC1", "13304720ABDB7"
    MALT = "13304718ABC"
    CARG = ["13304707CBB1", "13304707CDD1", "13304707CDD3"]
    W5721 = "permit_5721"

    def even(df_slice, wells, permit_val):
        if df_slice.empty:
            return pd.DataFrame(columns=KEYCOLS)
        d = df_slice.loc[df_slice.index.repeat(len(wells))].copy()
        d["Well"] = np.tile(wells, len(df_slice))
        d["Permit"] = [permit_val] * len(d)
        d["use_acft"] /= len(wells)
        return d

    na_well = combined["Well"].isna()

    frames.append(
        even(
            combined.query("Permit==1822 & 1973<=Year<=1979 & @na_well"),
            [W1, W2, W3],
            (1822, 1898),
        )
    )
    frames.append(
        even(combined.query("Permit==1822 & 1980<=Year<=1999 & @na_well"), [W1], 1822)
    )
    frames.append(
        even(
            combined.query("Permit==1898 & 1980<=Year<=1999 & @na_well"), [W2, W3], 1898
        )
    )
    frames.append(
        even(combined.query("Permit==1822 & 2000<=Year<=2002 & @na_well"), [W1], 1822)
    )
    frames.append(
        even(
            combined.query("Permit==1898 & 2000<=Year<=2002 & @na_well"), [W2, W3], 1898
        )
    )

    # Minn-Dak (4121 & 3898) **all years** → 50/50 split
    mdmask = combined["Permit"].isin([4121, 3898]) & na_well
    if mdmask.any():
        md = (
            combined.loc[mdmask]
            .groupby(
                ["Year", "Month", "permit_holder_name"], as_index=False, dropna=False
            )
            .agg(use_acft=("use_acft", "sum"))
        )
        md["Permit"] = [(4121, 3898)] * len(md)
        frames.append(even(md, [MD1, MD2], (4121, 3898)))

    frames.append(even(combined.query("Permit==2115  & @na_well"), [MALT], 2115))
    frames.append(even(combined.query("Permit==5721  & @na_well"), [W5721], 5721))
    frames.append(even(combined.query("Permit==4862  & @na_well"), CARG, 4862))

    use_by_well = pd.concat(frames, ignore_index=True)[KEYCOLS]

    # ── 7 ▸ attach coords -------------------------------------
    wells_gdf = gpd.read_file(WELLS_SHP)
    c2266 = wells_gdf[["site_locat", "geometry"]].rename(columns={"site_locat": "Well"})
    c2266["x_2266"], c2266["y_2266"] = c2266.geometry.x, c2266.geometry.y
    c2266.drop(columns="geometry", inplace=True)

    c2265 = wells_gdf.to_crs(epsg=2265)[["site_locat", "geometry"]].rename(
        columns={"site_locat": "Well"}
    )
    c2265["x_2265"], c2265["y_2265"] = c2265.geometry.x, c2265.geometry.y
    c2265.drop(columns="geometry", inplace=True)

    coords = c2266.merge(c2265, on="Well", how="outer").drop_duplicates("Well")
    use_by_well = use_by_well.merge(coords, on="Well", how="left")

    # ── 8 ▸ Month → int, CFD, sort -----------------------------
    use_by_well["Month"] = (
        use_by_well["Month"]
        .mask(
            use_by_well["Month"].notna(),
            use_by_well["Month"].str.upper().map(MONTH_TO_INT),
        )
        .astype("Int64")
    )

    def ndays(r):
        if pd.isna(r["Month"]):
            return 366 if calendar.isleap(int(r["Year"])) else 365
        return calendar.monthrange(int(r["Year"]), int(r["Month"]))[1]

    use_by_well["_days"] = use_by_well.apply(ndays, axis=1)
    use_by_well["cfd"] = use_by_well["use_acft"] * FT3_PER_ACFT / use_by_well["_days"]
    # use_by_well.drop(columns="_days", inplace=True)

    use_by_well = use_by_well.sort_values(["Well", "Year", "Month"], ignore_index=True)

    # ── 9 ▸ yearly aggregation (flag) --------------------------
    if not monthly:
        yearly = use_by_well.groupby(
            [
                "Permit",
                "Well",
                "Year",
                "permit_holder_name",
                "x_2266",
                "y_2266",
                "x_2265",
                "y_2265",
            ],
            as_index=False,
            dropna=False,
            sort=False,  # ← prevents int vs tuple comparisons
        ).agg(use_acft=("use_acft", "sum"), days=("_days", "sum"))
        yearly["cfd"] = yearly["use_acft"] * FT3_PER_ACFT / yearly["days"]
        use_by_well = yearly
        fname = "use_by_well_yearly.csv"
    else:
        fname = "use_by_well_monthly.csv"

    out_path = os.path.join(out_dir, fname)
    use_by_well.to_csv(out_path, index=False)
    print(f"✔  wrote {out_path}  ({len(use_by_well):,} rows)")
    return use_by_well


# -------- CLI entry -------------------------------------------
if __name__ == "__main__":
    well_use = build_use_by_well(monthly=False)  # set False for yearly output
