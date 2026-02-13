# -*- coding: utf-8 -*-
"""
Simplified SWB→MODFLOW recharge prep
Assumptions:
- NetCDF variable (net_infiltration) is mm/day.
- SWB CSV columns (gross_precipitation, net_infiltration) are in/day.
- Outputs (monthly and yearly products) are in/day.

Author: shjordan
"""

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression


NROW = 300
NCOL = 84
DELR = 660
DELC = 660

# ========== 1) Load PRISM precip ==========
def load_precip_for_rch() -> pd.DataFrame:
    """
    Returns a DataFrame with a single column 'precip':
      - Annual totals (AS-JAN) through 1999
      - Monthly totals (MS) 2000–2025
    Units follow the input file (typically inches).
    """
    precip = pd.read_csv(
        os.path.join('data', 'raw', 'PRISM_precip',
                     'PRISM_ppt_stable_4km_196501_202412_47.9507_-97.5454.csv'),
        skiprows=10, index_col=0, parse_dates=True
    )
    annual = precip.resample("AS-JAN").sum().loc[: "1999-01-01"]
    monthly = precip.resample("MS").sum().loc["2000-01-01":"2025-12-31"]
    precip_agg = pd.concat([annual, monthly]).sort_index()
    precip_agg.columns = ['precip']
    return precip_agg


# ========== 2) Regression: annual precip -> annual infiltration ==========
@dataclass
class InfilRegression:
    model: LinearRegression
    coef: float
    intercept: float
    r2: float
    n_years: int


def build_annual_regression_from_swb_csv(
    swb_csv: str,
    date_col: str = "date",
    col_precip: str = "gross_precipitation",   # in/day
    col_infil: str = "net_infiltration",       # in/day
    ) -> Tuple[InfilRegression, pd.DataFrame]:
    """
    Reads daily SWB totals (in/day), aggregates to annual (AS-JAN, in/year),
    fits I = a*P + b. Returns (regression, annual df used).

    Output annual df has cols: ['annual_precip','annual_infil'] in inches/year.
    """
    df = pd.read_csv(swb_csv, parse_dates=[date_col]).set_index(date_col).sort_index()
    
    # Sum daily in/day over each calendar year (AS-JAN) -> Units are in inches/year
    annual = df[[col_precip, col_infil]].resample("AS-JAN").sum().dropna()

    X = annual[[col_precip]].values  # (n,1) inches/year precip
    y = annual[col_infil].values     # inches/year infiltration
    model = LinearRegression(positive=True).fit(np.log10(X), np.log10(y))
    info = InfilRegression(
        model=model,
        coef=float(model.coef_[0]),
        intercept=float(model.intercept_),
        r2=float(model.score(X, y)),
        n_years=len(annual),
    )

    # Optional plot
    plot_infiltration_regression(
        annual, info,
        col_precip=col_precip, col_infil=col_infil,
        title="SWB (2000–2023): Infil vs Precip (annual)",
        one_to_one=False, out_path=None, show=False
    )

    return info, annual.rename(columns={col_precip: "annual_precip", col_infil: "annual_infil"})


def plot_infiltration_regression(
    annual: pd.DataFrame,
    info: InfilRegression,
    col_precip: str = "gross_precipitation",
    col_infil: str = "net_infiltration",
    title: Optional[str] = None,
    one_to_one: bool = False,
    out_path: Optional[str] = None,
    show: bool = False):

    x = np.log10(annual[col_precip].to_numpy())
    y = np.log10(annual[col_infil].to_numpy())
    z = annual.index.year.to_numpy()
    cmap = plt.cm.get_cmap("rainbow", len(z))

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(x, y, c= z, cmap=cmap, s=30,)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    yy = info.coef * xx + info.intercept
    ax.plot(xx, yy, linewidth=2, label="linear fit")
    if one_to_one:
        ax.plot(xx, xx, linestyle="--", linewidth=1, label="1:1")
    ax.set_xlabel("Annual precipitation (log(in))")
    ax.set_ylabel("Annual infiltration (log(in))")
    ax.set_title(title or "Annual Infiltration vs Precipitation Log Transformed")
    ax.legend()

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=year,
                          markerfacecolor=cmap(i), markersize=8)
               for i, year in enumerate(z)]
    ax.legend(handles=handles, title="Years", ncol=2)



    ax.grid(alpha=0.3)
    ax.text(0.02, 0.98,
            f"I = {info.coef:.3f}·P + {info.intercept:.3f}\n$R^2$ = {info.r2:.3f}\n n = {info.n_years}",
            transform=ax.transAxes, va="top")
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def predict_annual_infiltration(
        regression: InfilRegression,
        annual_precip: pd.Series,
        ) -> pd.Series:
    """
    Predict annual infiltration (in/year) from annual precip (in/year).
    """
    X = annual_precip.values.reshape(-1, 1)
    yhat = regression.model.predict(np.log10(X))
    yhat = 10**yhat
    return pd.Series(yhat, index=annual_precip.index, name="annual_infil_pred")


# ========== 3) Spatial pattern from NetCDF (2000–2023) ==========
def find_infil_var_name(ds: xr.Dataset, preferred: str = "net_infiltration") -> str:
    if preferred in ds.data_vars:
        return preferred
    for v in ds.data_vars:
        name = v.lower()
        if "inf" in name and "net" in name:
            return v
    return list(ds.data_vars)[0]


def compute_normalized_spatial_pattern(
        nc_path: str,
        var_name: Optional[str] = None,
        time_slice: Tuple[str, str] = ("2000-01-01", "2023-12-31"),
        normalize: str = "mean",  # "mean" => mean=1; "sum" => sum=1
        mask_nan: bool = True
        ) -> xr.DataArray:
    """
    Build a normalized, average-annual spatial pattern from SWB daily net infiltration (in/day).
    Steps:
      - select 2000–2023
      - sum daily in/day to annual in
      - average across years (in)
      - normalize across grid (dimensionless)
    """
    ds = xr.open_dataset(nc_path)
    vn = var_name or find_infil_var_name(ds)
    infil = ds[vn]

    if not np.issubdtype(infil['time'].dtype, np.datetime64):
        infil = xr.decode_cf(ds)[vn]

    infil = infil.sel(time=slice(*time_slice))

    annual_cell = infil.groupby("time.year").sum("time")   # in/year
    mean_annual = annual_cell.mean("year")              # in

    if mask_nan:
        mean_annual = mean_annual.where(np.isfinite(mean_annual))

    pattern = mean_annual / mean_annual.sum()

    pattern.name = "infiltration_pattern_norm"
    
    fig,ax = plt.subplots()
    im = ax.imshow(pattern)
    ax.set_title("Normalized pre-2000 RCH surface")
    plt.colorbar(im)
    #plt.show()
    
    return pattern


# ========== 4) Build per-year RCH arrays (in/day) ==========
def build_yearly_rch_arrays(
    pattern_norm: xr.DataArray,
    annual_infil_pred: pd.Series,  # in/year
    ) -> Dict[int, np.ndarray]:
    """
    Scales the normalized spatial pattern (mean=1) by annual depth (in/year),
    Returns {year: 2D array (ny, nx)} in in/day.
    """
    rch_arrays: Dict[int, np.ndarray] = {}
    for ts, annual_depth_in in annual_infil_pred.items():
        year = ts.year
        
        # Convert total inches of infiltration to a volume
        annual_depth_in = annual_depth_in * DELR * DELC  * NROW * NCOL  * (12**2) # x cell volume in feet, ft2 -> in2, x # of cells
        
        # Multiply normalized surface by the total infiltration (volume) for each year
        per_cell_depth_in = pattern_norm.values * float(annual_depth_in)  # in*3/year of infiltration
        
        # Convert back to a depth (in) by dividing by the cell area (in^2)
        per_cell_depth_in = per_cell_depth_in / (DELR * DELC * (12**2))
        
        # Convert to an in/day rate
        days = 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365
        per_cell_rate_in_day = per_cell_depth_in / days
        
        # Save recharge array
        rch_arrays[year] = per_cell_rate_in_day.astype(float)
        
    return rch_arrays


# ========== 5) Orchestrator ==========
def make_rch_from_swb(
    swb_csv_path: str,
    nc_path: str,
    precip_annual: pd.DataFrame,
    csv_date_col: str = "date",
    csv_precip_col: str = "gross_precipitation",  # in/day
    csv_infil_col: str = "net_infiltration",      # in/day
    nc_var_name: Optional[str] = None,
    pattern_normalize: str = "mean",
    years_to_build: Tuple[int, int] = (1965, 1999),
    ) -> Dict[int, np.ndarray]:
    """
    Full pipeline (outputs in/day):
      - regression from SWB CSV (daily → annual inches),
      - predict annual infiltration for [years_to_build],
      - spatial pattern from NetCDF (2000–2023),
      - build yearly RCH arrays (in/day).
    """
    reg, annual_df = build_annual_regression_from_swb_csv(
        swb_csv=swb_csv_path,
        date_col=csv_date_col,
        col_precip=csv_precip_col,
        col_infil=csv_infil_col,
    )
    print(f"Log Regression: I = {reg.coef:.4f} * P + {reg.intercept:.4f}  (R^2={reg.r2:.3f}, n={reg.n_years})")

    # Select target annual precip series (in/year)
    precip_annual_only = precip_annual.copy()
    mask = (precip_annual_only.index.year >= years_to_build[0]) & \
           (precip_annual_only.index.year <= years_to_build[1])
    precip_annual_only = precip_annual_only.loc[mask, 'precip']  # Series in/year

    infil_pred_in_year = predict_annual_infiltration(reg, precip_annual_only)
    # qui

    pattern_norm = compute_normalized_spatial_pattern(
        nc_path=nc_path,
        var_name=nc_var_name,
        time_slice=("2000-01-01", "2023-12-31"),
        normalize=pattern_normalize,
        mask_nan=True
        )

    rch_arrays_in_day = build_yearly_rch_arrays(
        pattern_norm=pattern_norm,
        annual_infil_pred=infil_pred_in_year
        )

    return rch_arrays_in_day


# ========== Function to plot SWB2 outputs ==========
def plot_monthly_infil_climatology(
    nc_path: str,
    var_name: Optional[str] = None,
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    units_label: str = "in/day",
    mask_negatives: bool = True,
    robust: bool = True,
    prc: Tuple[float, float] = (2.0, 98.0),
    bins: int = 60,
    out_pdf: Optional[str] = None,
    show: bool = False,
    ):
    """
    For each calendar month (Jan..Dec), compute the mean infiltration map across years
    and plot it as an imshow next to a histogram of cell values. Then append:
      • Yearly total (domain-mean depth, inches) bars for 2000–2023
      • Monthly total bars for each year (small multiples, 3×2 years per page)

    Notes:
      - Totals are computed as domain-mean depth (not area-integrated volume).
      - If mask_negatives=True, negative daily rates are omitted before all summaries.
    """
    # --- open and prep daily data
    ds = xr.open_dataset(nc_path)
    vn = var_name or find_infil_var_name(ds)
    da = ds[vn]

    dp = xr.open_dataset(nc_path.replace("net_infiltration", "gross_precipitation"))
    dp = dp['gross_precipitation']

    if not np.issubdtype(da["time"].dtype, np.datetime64):
        ds = xr.decode_cf(ds)
        da = ds[vn]

    # restrict time range
    da = da.sel(time=slice(start, end))
    dp = dp.sel(time=slice(start, end))

    # apply negative mask if requested (also affects totals below)
    if mask_negatives:
        da = da.where(da >= 0)

    # monthly mean *rate* (mean of daily rates within each month)
    monthly_rate = da.resample(time="MS").mean("time")

    # monthly climatology: average all Jan together, all Feb together, etc.
    clim = monthly_rate.groupby("time.month").mean("time")  # dims: month, y, x

    # choose shared vmin/vmax for imshow
    if robust:
        vmin = float(np.nanpercentile(clim.values, prc[0]))
        vmax = float(np.nanpercentile(clim.values, prc[1]))
    else:
        vmin = float(np.nanmin(clim.values))
        vmax = float(np.nanmax(clim.values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    # --- FIGURE 1: monthly maps + histograms (12 rows × 2 cols)
    fig1, axes = plt.subplots(nrows=12, ncols=2, figsize=(18, 60), constrained_layout=True)
    for m in range(1, 13):
        arr = clim.sel(month=m).values
        # left: imshow
        im = axes[m-1, 0].imshow(arr, vmin=vmin, vmax=vmax)  # viridis default
        axes[m-1, 0].set_title(f"{calendar.month_name[m]} – mean infiltration")
        axes[m-1, 0].set_xticks([]); axes[m-1, 0].set_yticks([])
        cbar = fig1.colorbar(im, ax=axes[m-1, 0], fraction=0.046, pad=0.04)
        cbar.set_label(units_label)

        # right: histogram (ignore NaNs and zeros)
        arr_hist = np.where(arr == 0, np.nan, arr)
        vals = arr_hist[np.isfinite(arr_hist)]
        axes[m-1, 1].hist(vals, bins=bins, range=(vmin, vmax))
        if vals.size:
            axes[m-1, 1].axvline(np.nanmean(vals), linestyle="--", linewidth=1, label="mean")
            axes[m-1, 1].axvline(np.nanmedian(vals), linestyle=":", linewidth=1, label="median")
        axes[m-1, 1].set_xlim(vmin, vmax)
        axes[m-1, 1].set_xlabel(units_label)
        axes[m-1, 1].set_ylabel("cell count")
        axes[m-1, 1].set_title(f"{calendar.month_name[m]} – histogram")
        axes[m-1, 1].grid(alpha=0.3)
        axes[m-1, 1].legend(loc="upper right", fontsize=8, frameon=False)

    # ======== Totals for bar charts (domain-mean depth, inches) ========
    # monthly total per cell: sum of daily rates over the month => depth (in)
    monthly_total_map = da.resample(time="MS").sum("time")  # (time, y, x), units: inches/month
    monthly_total_precip = dp.resample(time="MS").sum("time")  # (time, y, x), units: inches/month
    mask = monthly_total_precip[6] !=0
    monthly_total_precip = monthly_total_precip.where(mask, np.nan)  # mask negative precip
    monthly_total_map = monthly_total_map.where(mask, np.nan,)  # mask negative precip

    # domain-mean per month (single number)
    spatial_dims = [d for d in monthly_total_map.dims if d != "time"]
    monthly_total_mean = monthly_total_map.mean(spatial_dims, skipna=True)  # (time,)
    monthly_total_mean_precip = monthly_total_precip.mean(spatial_dims, skipna=True)  # (time,)

    # yearly totals (sum of monthly totals within year)
    yearly_total_mean = monthly_total_mean.resample(time="AS-JAN").sum("time")  # (yearly time,)
    yearly_total_mean_precip = monthly_total_mean_precip.resample(time="AS-JAN").sum("time")  # (yearly time,)

    # slice years 2000..2023 (in case start/end wider)
    ymask = (yearly_total_mean["time"].dt.year >= 2000) & (yearly_total_mean["time"].dt.year <= 2023)
    yearly_total_mean = yearly_total_mean.sel(time=ymask)
    yearly_total_mean_precip = yearly_total_mean_precip.sel(time=ymask)

    # palette for bars
    palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    # --- FIGURE 2: Yearly totals 2000–2023
    years = yearly_total_mean["time"].dt.year.values.astype(int)
    yvals = yearly_total_mean.values
    pvals = yearly_total_mean_precip.values
    fig2, ax2 = plt.subplots(figsize=(12, 4.2))
    ax2_ = ax2.twinx()
    colors_y = [palette[i % len(palette)] for i in range(len(years))]
    ax2.bar(years, yvals, color=colors_y, edgecolor="black", linewidth=0.6)
    ax2_.plot(years, pvals, color="blue", linewidth=2, marker='o', label="precip")

    ax2.set_title("Yearly total infiltration (domain-mean depth)")
    ax2.set_ylabel("Total infiltration (in)")
    ax2.set_xlabel("Year")
    ax2_.set_ylabel("Total precipitation (in)", color="blue")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xlim(years.min() - 0.6, years.max() + 0.6)

    # --- FIGURE 3..N: Monthly totals per year (small multiples, 3×2 years per page)
    # Prepare monthly totals series with month names on x-axis
    monthly_df = monthly_total_mean.to_series()
    precip_df = monthly_total_mean_precip.to_series()
    # keep 2000-2023 months
    monthly_df = monthly_df[(monthly_df.index.year >= 2000) & (monthly_df.index.year <= 2023)]
    precip_df = precip_df[(precip_df.index.year >= 2000) & (precip_df.index.year <= 2023)]
    # pivot to (year -> 12 monthly totals)
    month_names = [calendar.month_abbr[m] for m in range(1, 13)]
    # global y-limit for comparability
    global_ylim = float(np.nanmax(monthly_df.values)) * 1.25 if np.isfinite(np.nanmax(monthly_df.values)) else 1.0

    years_all = np.arange(2000, 2024)
    blocks = [years_all[i:i+6] for i in range(0, len(years_all), 6)]  # 6 years per page

    figs_small_multiples = []
    for block in blocks:
        figSM, axesSM = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), constrained_layout=True)
        axesSM = axesSM.ravel()
        for ax, yr in zip(axesSM, block):
            # extract that year's 12 months
            ysel = monthly_df[monthly_df.index.year == yr]
            ysel_prcip = precip_df[precip_df.index.year == yr]
            # ensure 12 slots
            vals = [ysel.get(pd.Timestamp(year=yr, month=m, day=1), np.nan) for m in range(1, 13)]
            pvals = [ysel_prcip.get(pd.Timestamp(year=yr, month=m, day=1), np.nan) for m in range(1, 13)]
            # bar colors cycling through the 5-color palette
            month_colors = [palette[(m-1) % len(palette)] for m in range(1, 13)]
            ax.bar(month_names, vals, color=month_colors, edgecolor="black", linewidth=0.4)
            ax_ = ax.twinx()
            ax_.plot(month_names,pvals,color="blue",linewidth=2,marker='o',label="precip")
            ax.set_title(f"{yr} – monthly totals")
            ax.set_ylim(0, global_ylim)
            ax_.set_ylabel("Total precipitation (in)", color="blue")
            ax_.set_ylim([0,4])
            ax.grid(axis="y", alpha=0.3)
            # tidy x labels
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(0)
                lbl.set_fontsize(8)
            ax.set_ylabel("in")
        # If last block has <6 years, blank the remaining axes
        for j in range(len(block), 6):
            axesSM[j].axis("off")
        figs_small_multiples.append(figSM)

    # --- Save / show
    if out_pdf:
        with PdfPages(out_pdf) as pdf:
            pdf.savefig(fig1, dpi=200, bbox_inches="tight")  # climatology page
            pdf.savefig(fig2, dpi=200, bbox_inches="tight")  # yearly totals
            for figSM in figs_small_multiples:               # monthly totals per-year pages
                pdf.savefig(figSM, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    # (optional) close figures to free memory when not showing
    plt.close(fig1); plt.close(fig2)
    for figSM in figs_small_multiples:
        plt.close(figSM)

    return clim, (fig1, axes)


# ========== 6) Monthly from NetCDF (in/day) ==========
def monthly_from_netcdf(
    nc_path: str,
    var_name: Optional[str] = None,
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    make_plots : bool = True,
    ) -> Dict[str, np.ndarray]:
    """
    Build monthly mean infiltration *rates* (in/day) per cell from SWB NetCDF.
    Average daily in/day within each month
    Returns dict keyed by 'YYYY-MM' with arrays (ny, nx) in/day.
    """
    ds = xr.open_dataset(nc_path)
    vn = var_name or find_infil_var_name(ds)
    da = ds[vn]

    if not np.issubdtype(da['time'].dtype, np.datetime64):
        da = xr.decode_cf(ds)[vn]
        
    # Grab data for specified monthly period (post-2000 monthly Stres Periods)
    da = da.sel(time=slice(start, end))
    monthly_rate = da.resample(time="MS").mean("time")  # in/day from nc arrays (avg over month)
    # potentially should sum and divide by days in month... but probably not necessary

    out = {}
    for t in pd.period_range(start=start[:7], end=end[:7], freq="M"):
        key = f"{t.year}-{t.month:02d}"
        arr_in_day = monthly_rate.sel(time=str(t)).values
        out[key] = arr_in_day
    
    if make_plots:
        out_pdf = os.path.join('figures','swb_rch.pdf')
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
        plot_monthly_infil_climatology(
            nc_path, var_name, start=start, end=end,
            units_label="in/day",
            out_pdf=out_pdf, show=False
        )
    return out


# ========== 7) Optional: quick audit ==========
def audit_monthlies(monthly_dict: Dict[str, np.ndarray], year: int):
    """
    Prints domain-average monthly rates (in/day) and annual total (in).
    Helps verify magnitudes (e.g., expect a few inches per year).
    """
    keys = sorted([k for k in monthly_dict if k.startswith(str(year))])
    if not keys:
        raise ValueError(f"No months found for {year}")
    rows = []
    for k in keys:
        arr = monthly_dict[k]
        # arr = np.where(arr==0,np.nan,arr)
        r_day = float(np.nanmean(arr))  # in/day
        dt = pd.Period(k, freq="M").to_timestamp()
        days = (dt + pd.offsets.MonthEnd(0)).day
        depth_in = r_day * days
        rows.append((k, r_day, depth_in))
    df = pd.DataFrame(rows, columns=["month", "rate_in_day", "depth_in"])
    print(df)
    print(f"Annual recharge (domain avg) for {year}: {df['depth_in'].sum():.2f} in")
    print(f"Max monthly-average daily rate: {df['rate_in_day'].max():.3f} in/day")


# Some basic plotting functions for the SWB daily output
def plot_swb(swb_csv_path):
    import seaborn as sns
    # Read in processed daily file
    swb_out = pd.read_csv(swb_csv_path,
                          index_col=0,
                          parse_dates=True)
    swb_out = swb_out[swb_out.index >= pd.Timestamp('01-01-2000')]
    swb_out_monthly = swb_out.resample('ME').sum()
    swb_out_monthly['month'] = swb_out_monthly.index.month
    season_map_long = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                       9: 'Fall', 10: 'Fall', 11: 'Fall'
                       }
    swb_out_monthly['season'] = swb_out_monthly.index.month.map(season_map_long)

    # Normalize and plot each variable eas a cumulative sum
    num = swb_out.select_dtypes(include=np.number).fillna(0)
    totals = num.clip(lower=0).sum(axis=0)
    totals = totals.replace(0, np.nan)
    norm = (num / totals).fillna(0)
    swb_cumsum = norm.cumsum()

    # ---- Plot all the variables in cumsum form
    fig, ax = plt.subplots(figsize=(14, 5))
    swb_cumsum.plot(ax=ax, lw=1.8)
    ax.plot([swb_cumsum.index.min(), swb_cumsum.index.max()],
            [0, 1],
            color='k',
            ls='--',
            lw=1)

    ax.grid()
    ax.set_title("Normalized Cumulative Sum of SWB Variables")
    ax.set_ylabel("Normalized Cumulative Sum")
    ax.set_xlabel("Year")

    # ---- Yearly Jointplot
    fig, ax = plt.subplots()
    sns.scatterplot(data=swb_out_monthly,
                    x='gross_precipitation',
                    y='net_infiltration',
                    color="#4CB391",
                    hue='season',
                    ax=ax
                    )
    ax.grid()
    ax.set_ylabel("Monthly Infiltration (in)")
    ax.set_xlabel("Monthly Precipitation (in)")


# ========== Main ==========
def main():
    precip_agg = load_precip_for_rch()  # inches
    swb_csv_path = os.path.join("data","swb", "water_balance_daily_in_day.csv")
    nc_path = os.path.join("data", "swb",
                           "swb2_net_infiltration__2000-01-01_to_2023-12-31__296_by_178.nc")

    assert os.path.exists(swb_csv_path), f"Missing SWB CSV: {swb_csv_path} - Need to run SWB!"
    assert os.path.exists(nc_path), f"Missing SWB NetCDF: {nc_path} - Need to run SWB!"

    # Yearly (1970–1999) arrays for MF RCH (in/day)
    rch_by_year = make_rch_from_swb(
        swb_csv_path=swb_csv_path,
        nc_path=nc_path,
        precip_annual=precip_agg,
        csv_date_col="date",
        csv_precip_col="gross_precipitation",
        csv_infil_col="net_infiltration",
        pattern_normalize="mean",
        years_to_build=(1965, 1999),
    )

    # Monthly (2000–2023) arrays for MF RCH (in/day)
    post_2000_monthly = monthly_from_netcdf(
        nc_path=nc_path,
        var_name="net_infiltration",
        start="2000-01-01",
        end="2023-12-31",
    )

    # Optional checks
    audit_monthlies(post_2000_monthly, 2000)

    # close all figs:
    plt.close('all')

    return rch_by_year, post_2000_monthly


if __name__ == "__main__":
    main()
