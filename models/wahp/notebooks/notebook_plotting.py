import os 
import sys
import shutil 
from datetime import datetime
sys.path.insert(0,os.path.abspath(os.path.join('..','..','..','dependencies')))
sys.path.insert(1,os.path.abspath(os.path.join('..','..','..','dependencies','flopy')))
sys.path.insert(2,os.path.abspath(os.path.join('..','..','..','dependencies','pyemu')))
import pyemu
import flopy
from typing import Tuple, Dict, Optional
from flopy.utils import CellBudgetFile, postprocessing as pp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Sequence, Union
from pyproj import CRS
import subprocess



def clean_file_fmts(mpws):    
    icell_files = [f for f in os.listdir(mpws) if f.startswith('npf_icelltype')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(mpws, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(mpws, file), icelltype_data, fmt='%i')

    icell_files = [f for f in os.listdir(mpws) if f.startswith('sto_iconvert')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(mpws, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(mpws, file), icelltype_data, fmt='%i')


def plot_hds(
    model_ws: Union[str, os.PathLike],
    sim_name: str = "wahp",
    kstpkper: tuple[int, int] = (0, 0),
    layers: Union[str, Sequence[int]] = "all",
    show_bc: bool = True,
    bc_colors: dict[str, str] | None = None,
    out_dir: str = "fig_heads",
    dpi: int = 300,
):
    """
    Colour-flood head maps with boundary conditions, compatible with
    older FloPy versions (uses PlotMapView.plot_bc).

    Parameters
    ----------
    model_ws
        Workspace containing the simulation.
    sim_name
        GWF model name (same as *.nam* prefix).
    kstpkper
        (kstp, kper) tuple to extract from *.hds*.
    layers
        'all' or iterable of 0-based layer indices.
    show_bc
        Toggle RIV, GHB, CHD overlays (DRN removed).
    bc_colors
        Dict of colours for each BC.
    out_dir
        Sub-folder for PNGs.
    dpi
        Output resolution.
    """
    bc_colors = bc_colors or dict(
        riv="dodgerblue",
        ghb="limegreen",
        chd="magenta",
    )

    model_ws = os.fspath(model_ws)

    # 1 - Load sim & heads
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, exe_name="mf6")
    gwf = sim.get_model(sim_name)
    grid = gwf.modelgrid
    nlay = gwf.dis.nlay.data

    # stats for the modified anisotropy mask area (same as before)
    riv_data = gwf.riv.stress_period_data.data
    riv_ss = pd.DataFrame(riv_data[0])
    riv_ss["k"] = riv_ss["cellid"].apply(lambda x: x[0])
    riv_ss["i"] = riv_ss["cellid"].apply(lambda x: x[1])
    riv_ss["j"] = riv_ss["cellid"].apply(lambda x: x[2])

    rows = np.arange(141, 152)
    cols = np.arange(29, 59)
    riv_cells = riv_ss[
        (riv_ss["i"].isin(rows)) & (riv_ss["j"].isin(cols))
    ].copy()
    stats_str = (
        "RIV cells in modified high-Kv area:\n"
        f"  min stage   : {riv_cells['stage'].min():.2f}\n"
        f"  max stage   : {riv_cells['stage'].max():.2f}\n"
        f"  mean stage  : {riv_cells['stage'].mean():.2f}\n"
        f"  median stage: {riv_cells['stage'].median():.2f}"
    )

    if layers == "all":
        layers = range(nlay)

    hfile = flopy.utils.HeadFile(os.path.join(model_ws, f"{sim_name}.hds"))
    heads = hfile.get_data(kstpkper=kstpkper)
    idomain = grid.idomain

    # 2 - Output folder
    figdir = os.path.join(model_ws, out_dir)
    os.makedirs(figdir, exist_ok=True)

    cmap = mpl.colormaps.get_cmap("viridis")

    # plot_bc key → FloPy ftype (DRN removed)
    tag2ftype = dict(riv="RIV", ghb="GHB", chd="CHD")

    for k in layers:
        fig, ax = plt.subplots(figsize=(8, 7))
        mview = flopy.plot.PlotMapView(model=gwf, layer=k, ax=ax)

        # mask inactive + fill values
        active = idomain[k] > 0
        h_msk = np.ma.masked_where(~active | (heads[k] >= 1e20), heads[k])
        vmin, vmax = np.nanmin(h_msk), np.nanmax(h_msk)

        mview.plot_array(h_msk, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(
            mpl.cm.ScalarMappable(
                cmap=cmap, norm=mpl.colors.Normalize(vmin, vmax)
            ),
            ax=ax,
            shrink=0.8,
            label="SS Head (ft)",
        )

        # 3 - Boundary conditions via plot_bc
        if show_bc:
            for tag, ftype in tag2ftype.items():
                if hasattr(gwf, tag):
                    mview.plot_bc(
                        name=ftype,
                        kper=0,
                        color=bc_colors[tag],
                        plotAll=False,
                    )
                    ax.scatter(
                        [],
                        [],
                        marker="o",
                        facecolors="none",
                        edgecolors=bc_colors[tag],
                        label=ftype,
                    )

        # cosmetics & save
        ax.set_title(f"{sim_name} – Layer {k + 1}", loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (ft)")
        ax.set_ylabel("Northing (ft)")

        # stats on the figure
        ax.text(
            0.02,
            0.98,
            stats_str,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        if show_bc:
            ax.legend(fontsize=8, frameon=True)

        fig.tight_layout()
        png = os.path.join(figdir, f"heads_layer_{k + 1:02d}.png")
        fig.savefig(png, dpi=dpi, facecolor="white")
        # plt.close(fig)
        print("saved →", png)
        

def plot_cross_section(
    model_ws: str,
    sim_name: str = "wahp",
    kstpkper: Tuple[int, int] = (0, 0),
    column: int = 44,                # 1-based model column for the section
    arrow_kw: Optional[Dict] = None,
):
    """
    Column cross-section with layer-4 shading, red specific-discharge
    arrows, and a blue head line through layer 4.

    Parameters
    ----------
    model_ws : str
        MF6 simulation workspace.
    sim_name : str
        GWF model name (.nam prefix).
    kstpkper : (int, int)
        (kstp, kper) tuple for the CBB / head file.
    column : int
        1-based model column number for the cross-section.
    arrow_kw : dict, optional
        Extra kwargs forwarded to PlotCrossSection.plot_vector.
    """
    # ── load simulation (only packages we need) ─────────────────────
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=model_ws, exe_name="mf6", load_only=["dis", "oc", "riv"]
    )
    gwf  = sim.get_model(sim_name)
    grid = gwf.modelgrid

    # ── specific discharge ─────────────────────────────────────────
    oc_file = gwf.oc.budget_filerecord.array[0][0]
    cbc     = CellBudgetFile(os.path.join(model_ws, oc_file), precision="double")
    spdis   = cbc.get_data(text="SPDIS", kstpkper=kstpkper)[0]
    qx, qy, qz = pp.get_specific_discharge(spdis, gwf)

    # ── heads for layer-4 line ─────────────────────────────────────
    heads = flopy.utils.HeadFile(
        os.path.join(model_ws, f"{sim_name}.hds")
    ).get_data(kstpkper=kstpkper)

    # ── set up figure & cross-section object ───────────────────────
    fig, ax = plt.subplots(figsize=(15, 8), dpi=500)
    xs = flopy.plot.PlotCrossSection(
        model=gwf,
        line={"column": column - 1},        # zero-based index
        geographic_coords=True,
        ax=ax,
    )

    xs.plot_grid(ax=ax, linewidth=0.2)
    xs.plot_inactive(color_noflow="grey")

    # shade layer-4 cells (index 3) in light gray
    layer_mask = np.full_like(grid.idomain, np.nan, dtype=float)
    layer_mask[3, :, :] = 0.5
    xs.plot_array(layer_mask, cmap="Greys", alpha=0.35, vmin=0, vmax=1)

    # river boundary
    xs.plot_bc(name="riv", color="cyan")

    # head line in layer 4 (blue)
    xs.plot_surface(
        heads[3],                     # layer 4
        color="blue",
        linewidth=1.2,
        linestyle="-",
    )

    # red flow arrows
    vkw = {"kstep": 2, "hstep": 2}
    if arrow_kw:
        vkw.update(arrow_kw)
    xs.plot_vector(
        qx, qy, qz, ax=ax,
        scale=2, color="red", width=0.002, **vkw
    )

    # ── cosmetics ──────────────────────────────────────────────────
    ax.set_title(f"Cross-section along Column {column}", loc="left")
    ax.set_xlabel("Horizontal coordinate (ft)")
    ax.set_ylabel("Elevation (ft)")
    plt.tight_layout()
    plt.show()

def plot_plan_column(
    model_ws: str,
    sim_name: str = "wahp",
    column: int = 44,                 # 1-based model column to highlight
):
    """
    Plan-view map showing

      • gray mask (rows 141-151, cols 29-58),
      • highlighted column in yellow,
      • river boundary cells (blue),
      • Wahp outline (red).

    Parameters
    ----------
    model_ws : str
        MF6 workspace (e.g. "model_ws/wahp_clean").
    sim_name : str
        GWF model name (.nam prefix).
    column : int
        1-based model column number to highlight.
    """
    # ── load model ─────────────────────────────────────────────────
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, exe_name="mf6", load_only=["dis", "riv"])
    gwf   = sim.get_model(sim_name)
    grid  = gwf.modelgrid
    nrows, ncols = grid.nrow, grid.ncol

    # ── build mask exactly as before ──────────────────────────────
    rows = np.arange(141, 152)
    cols = np.arange(29, 59)
    mask = np.zeros((nrows, ncols), dtype=int)
    mask[np.ix_(rows - 1, cols - 1)] = 1          # 0-based index

    # ── highlight the chosen column (yellow) ─────────────────────
    col_high = np.full_like(mask, np.nan, dtype=float)
    col_high[:, column - 1] = 1.0                 # NaN elsewhere, 1 in column

    # ── outline shapefile ─────────────────────────────────────────
    outline_fp = os.path.join(
        "..", "..", "..", "gis", "input_shps", "wahp", "wahp_outline_full.shp"
    )
    outline = gpd.read_file(outline_fp)


    # ── map view ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    mv = flopy.plot.PlotMapView(model=gwf, ax=ax)

    mv.plot_array(mask, cmap="gray", alpha=0.7, vmin=0, vmax=1)

    mv.plot_array(
        col_high,
        cmap="autumn",                     # oranges/yellows
        alpha=0.8,
        vmin=0,
        vmax=1
    )

    mv.plot_bc(package=gwf.riv, color="blue", lw=0.8, label="River package")

    outline.boundary.plot(ax=ax, color="red", linewidth=1.2, label="Wahp outline")

    ax.set_title(f"Plan view – highlighted Column {column}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_params_map_mel(
    model_ws: Union[str, os.PathLike],
    partype: str,
    sim_name: str = "wahp",
    layers: Union[str, Sequence[int]] = "all",
    kstpkper: tuple[int, int] = (0, 0),          # for recharge only
    logscale_hk: bool = True,
    cmap_overrides: dict[str, str] | None = None,
    out_dir: str = "fig_params",
    dpi: int = 300,
):
    """
    Colour-flood maps of HK, VK, SS, SY or RECH ― one PNG per layer ― with the
    Wahpeton project outline shown as a grey basemap.
    """

    # ── helper to unwrap LayeredArray/Util3D → ndarray
    def _to_array(obj):
        return obj.array if hasattr(obj, "array") else obj

    # ── colour maps
    cmaps = dict(hk="plasma", vk="plasma", ss="viridis",
                 sy="YlGn", rech="Blues")
    if cmap_overrides:
        cmaps.update({k.lower(): v for k, v in cmap_overrides.items()})

    partype = partype.lower()
    if partype not in {"hk", "vk", "ss", "sy", "rech"}:
        raise ValueError("partype must be hk, vk, ss, sy or rech")

    # ── load simulation & model
    model_ws = os.fspath(model_ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, exe_name="mf6")
    gwf = sim.get_model(sim_name)
    grid = gwf.modelgrid
    idomain = grid.idomain

    nlay = gwf.dis.nlay.data
    if layers == "all":
        layers = range(nlay)

    # ── read outline shapefile once and match CRS
    gis_dir = os.path.join("..","..", "..", "gis")
    outline_fp = os.path.join(gis_dir, "input_shps",
                              "wahp", "wahp_outline_full.shp")
    outline = gpd.read_file(outline_fp)


    # ── fetch requested parameter array
    if partype == "hk":
        arr3d = _to_array(gwf.npf.k)

    elif partype == "vk":
        raw = getattr(gwf.npf, "k33", None) or getattr(gwf.npf, "vk", None)
        if raw is None:
            raise ValueError("Model has neither K33 nor VK defined.")
        arr3d = _to_array(raw)

    elif partype == "ss":
        arr3d = _to_array(gwf.sto.ss)

    elif partype == "sy":
        arr3d = _to_array(gwf.sto.sy)

    elif partype == "rech":
        if not hasattr(gwf, "rch"):
            raise ValueError("Model has no RCH package.")

        full = _to_array(gwf.rch.recharge)          # 2-, 3- or 4-D
        if full.ndim == 4:                          # (nper, nlay, nrow, ncol)
            arr2d = full[kstpkper[1], 0, :, :]
        elif full.ndim == 3:                        # (nper, nrow, ncol)
            arr2d = full[kstpkper[1]]
        else:                                       # steady 2-D
            arr2d = full

        arr3d = arr2d[np.newaxis, ...]              # fake 3-D for loop
        layers = [0]

    # ── output folder
    figdir = os.path.join(model_ws, out_dir)
    os.makedirs(figdir, exist_ok=True)
    cmap = mpl.colormaps.get_cmap(cmaps[partype])

    # ── loop over layers
    for k in layers:
        data = arr3d[k]
        active = idomain[min(k, idomain.shape[0]-1)] > 0
        data_m = np.ma.masked_where(~active |
                                    (data <= -1e30) | (data >= 1e30), data)

        # colour norm
        if partype in {"hk", "vk"} and logscale_hk:
            data_m = np.ma.masked_where(data_m <= 0, data_m)
            norm = mpl.colors.LogNorm(vmin=data_m.min(), vmax=data_m.max())
        else:
            norm = mpl.colors.Normalize(vmin=data_m.min(), vmax=data_m.max())

        # ── plotting
        fig, ax = plt.subplots(figsize=(8, 7))

        # 2) parameter colour-flood
        mview = flopy.plot.PlotMapView(model=gwf, layer=k, ax=ax)
        mview.plot_array(data_m, cmap=cmap, norm=norm)
                # 1) grey outline basemap
        outline.plot(ax=ax, facecolor="none", edgecolor="0.4", linewidth=1.0)
        unq_d = np.unique(data_m)
        if len(unq_d)> 1:
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, shrink=0.8,
                        label=("Recharge (ft/d)" if partype == "rech"
                                else partype.upper()))

        title = (f"{sim_name} – RECH (kstpkper {kstpkper})"
                 if partype == "rech"
                 else f"{sim_name} – {partype.upper()} layer {k + 1}")
        ax.set_title(title, loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (ft)")
        ax.set_ylabel("Northing (ft)")
        fig.tight_layout()

        fname = (f"rech_kstpkper_{kstpkper[0]}_{kstpkper[1]}.png"
                 if partype == "rech"
                 else f"{partype}_layer_{k + 1:02d}.png")
        png = os.path.join(figdir, fname)
        fig.savefig(png, dpi=dpi)
        #plt.close(fig)
        print("saved →", png)
            
# ----------- modpath plotting --------------

def clean_file_fmts(mpws):    
    icell_files = [f for f in os.listdir(mpws) if f.startswith('npf_icelltype')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(mpws, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(mpws, file), icelltype_data, fmt='%i')

    icell_files = [f for f in os.listdir(mpws) if f.startswith('sto_iconvert')]
    for file in icell_files:
        # read in with numpy
        icelltype_data = np.loadtxt(os.path.join(mpws, file))
        # change everything to ints:
        icelltype_data = icelltype_data.astype(int)
        # write to a new file:
        np.savetxt(os.path.join(mpws, file), icelltype_data, fmt='%i')
    
    tdis_lines = open(os.path.join(mpws,'wahp.tdis'),'r').readlines()
    with open(os.path.join(mpws,'wahp.tdis'),'w') as f:
        for line in tdis_lines:
            if not 'START_DATE_TIME' in line.split():
                f.write(line)
        f.close()

            
def list_of_kij_to_add_particles(lys=np.arange(0,1),rows=np.arange(141, 152),cols=np.arange(29, 59)):
    #  meshgrid of all combinations
    ly_grid, row_grid, col_grid = np.meshgrid(
       lys, rows, cols, indexing='ij'
    )
    #row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    # stack:
    kij_tuples = np.column_stack((ly_grid.ravel(),row_grid.ravel(), col_grid.ravel()))
    kij = kij_tuples.tolist()
    
    return kij

def set_particle_groups_strt_loc(gwf,mpws,kij):
    nrow = gwf.modelgrid.nrow
    ncol = gwf.modelgrid.ncol
    nlays = gwf.modelgrid.nlay
    idom = gwf.dis.idomain.array.copy()
    slocs = []
    par_ids = []
    for k in range(nlays):
        for i in np.arange(0,nrow,2):
            for j in np.arange(0,ncol,2):
                if idom[k,i,j]>0:
                    if [k,i,j] in kij:
                        slocs.append((k,i,j))
                        par_ids.append(f'mp.k{k}.i{i}.j{j}')

    df = pd.DataFrame(data=par_ids,columns=['par_id_key'])
    df.to_csv(os.path.join(mpws,'parid_key.csv'),index_label='parid')

    p = flopy.modpath.ParticleData(slocs,structured=True,
                                    drape=0, timeoffset=0.0) 
    pg = flopy.modpath.ParticleGroup(particlegroupname='PG1', 
                                        particledata=p, filename='wahp.pg.sloc')
    particlegroups = [pg]
    return particlegroups

def setup_modpath_sim(mpws,gwf, particlegroups):
    """
    Set up a Modpath simulation with the given particle groups.
    """
    # make sure crs is set correctly, needed for modpath plotting
    crs = 2265  # State plane (feet)
    angrot = 40  # Rotation angle (degrees)
    cord_sys = CRS.from_epsg(crs)
    ll_corner = (2941428.2031374616, -260810.76576105086)
    gwf.modelgrid.set_coord_info(
        xoff=ll_corner[0], yoff=ll_corner[1], angrot=angrot, crs=cord_sys
    )
    
    mp = flopy.modpath.Modpath7(
        modelname=f"wahp_mp",
        flowmodel=gwf,
        exe_name="mp7",
        model_ws=mpws,
    )
    mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.15)
    mpsim = flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="forward",
        budgetoutputoption="summary",
        referencetime=[0, 0, 0.0],
        timepointdata=[1, [0]],
        zonedataoption="on",
        particlegroups=particlegroups,
    )
    # write modpath datasets
    mp.write_input()
    
    # run:
    # Build the absolute path to mp7.exe
    #exe_path = os.path.abspath(os.path.join(mpws, "mp7.exe"))
    #print("About to run (abs):", exe_path, "— exists?", os.path.isfile(exe_path))
    #subprocess.run([exe_path, os.path.join(mpws,"wahp_mp")], check=True)
    #print("Modpath simulation completed successfully.")

    #print("About to run:", exe_path, "— exists?", os.path.isfile(exe_path))
    print("Running Modpath simulation...")
    subprocess.run([os.path.join(mpws,"mp7"), "wahp_mp"],cwd=mpws)
    print("Modpath simulation completed successfully.")
    
    return mp, mpbas, mpsim, gwf

def plot_modpath(gwf, mpsim, mpws):
    from flopy.utils import PathlineFile
    from flopy.export.vtk import Vtk
    import math
    import pyvista as pv

    pf = PathlineFile(os.path.join(mpws, mpsim.pathlinefilename))
    pl = pf.get_alldata()

    hf = flopy.utils.HeadFile(os.path.join(mpws, f"wahp.hds"))
    hds = hf.get_data()

    for p in pl:
        for rec in p:
            rec['x'] += gwf.modelgrid.xoffset
            rec['y'] += gwf.modelgrid.yoffset

    # convert rotation angle to radians
    theta = math.radians(gwf.modelgrid.angrot)

    # rotate:
    x0 = gwf.modelgrid.xoffset
    y0 = gwf.modelgrid.yoffset

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def rotate_point(x, y, x0, y0):
        x_rel = x - x0
        y_rel = y - y0
        x_rot = x_rel * cos_t - y_rel * sin_t + x0
        y_rot = x_rel * sin_t + y_rel * cos_t + y0
        return x_rot, y_rot

    # rotate all pathline points
    for p in pl:
        for rec in p:
            rec['x'], rec['y'] = rotate_point(rec['x'], rec['y'], x0, y0)

    layer_index = 3  # Layer 3 is index 2
    filtered_pl = []
    # filter particles to only those that pass through WBV !!!!!!!!!!!!!!
    for particle in pl:
        if np.any(particle['k'] == layer_index):
            filtered_pl.append(particle)

    vtk = Vtk(model=gwf, binary=False, vertical_exageration=50, smooth=False)
    vtk.add_model(gwf)
    #vtk.add_pathline_points(pl)
    vtk.add_pathline_points(filtered_pl)

    grid, pathlines = vtk.to_pyvista()

    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)

    grid.rotate_z(-100, point=axes.origin, inplace=True)
    pathlines.rotate_z(-100, point=axes.origin, inplace=True)

    tracks = {}
    particle_ids = set()
    release_locs = []

    for i, t in enumerate(pathlines["time"]):
        pid = str(round(float(pathlines["particleid"][i])))
        loc = pathlines.points[i]

        if pid not in tracks:
            tracks[pid] = []
            particle_ids.add(pid)
            release_locs.append(loc)

        # store the particle location in the corresponding track
        tracks[pid].append((loc, t))

    release_locs = np.array(release_locs)

    tracks = {k: np.array(v, dtype=object) for k, v in tracks.items()}
    max_track_len = max([len(v) for v in tracks.values()])
    print("The maximum number of locations per particle track is", max_track_len)

    pv.set_plot_theme("document")
    #pv.set_jupyter_backend("static")
    pv.set_jupyter_backend('trame')  # or 'panel', 'pythreejs', etc.

    # create the plot and add the grid and pathline meshes
    layer_index = 3  # zero-based index for layer 3
    nlay, nrow, ncol = gwf.modelgrid.shape
    cell_k = np.repeat(np.arange(nlay), nrow * ncol)
    grid.cell_data['k'] = cell_k

    idomain = gwf.modelgrid.idomain  # shape: (nlay, nrow, ncol)
    idomain_flat = idomain.ravel(order="C")  # MODFLOW 6 uses C order
    grid.cell_data['idomain'] = idomain_flat

    active_grid = grid.threshold(value=0.5, scalars='idomain')  # keeps idomain > 0
    layer_index = 3  # layer 3
    layer_grid = active_grid.extract_cells(active_grid.cell_data['k'] == layer_index)

    p = pv.Plotter()
    p.add_mesh(grid, opacity=0.001)  # full grid
    p.add_mesh(layer_grid, color="yellow", opacity=0.05)  # highlighted layer
    #p.add_mesh(pathlines,scalars='time')
    p.add_mesh(pathlines,color="navy")

    # add a particle ID label to each 4th particle's starting point
    label_coords = []
    start_labels = []
    for pid, track in tracks.items():
        if int(pid) % 4 == 0:
            label_coords.append(track[0][0])
            start_labels.append(f"Particle {pid}")

    start_points = np.array([track[0][0] for track in tracks.values()])
    end_points   = np.array([track[-1][0] for track in tracks.values()])

    # plot riv boundaries:
    nlay, nrow, ncol = gwf.modelgrid.shape
    ncells = nlay * nrow * ncol

    cell_k = np.repeat(np.arange(nlay), nrow * ncol)
    cell_i = np.tile(np.repeat(np.arange(nrow), ncol), nlay)
    cell_j = np.tile(np.arange(ncol), nrow * nlay)

    grid.cell_data['k'] = cell_k
    grid.cell_data['i'] = cell_i
    grid.cell_data['j'] = cell_j

    spd0 = gwf.riv.stress_period_data.get_data()[0]  # list of [k,i,j,…]

    riv_mask = np.zeros(ncells, dtype=bool)
    for rec in spd0:
        kk, ii, jj = int(rec[0][0]), int(rec[0][1]), int(rec[0][2])
        riv_mask |= ((cell_k == kk) & (cell_i == ii) & (cell_j == jj))

    riv_grid = grid.extract_cells(riv_mask)
    
    p.add_mesh(riv_grid, color='cyan', opacity=0.3)
    
    start_pd = pv.PolyData(start_points)
    end_pd = pv.PolyData(end_points)
    p.add_mesh(start_pd, color="green", point_size=9, render_points_as_spheres=True)
    p.add_mesh(end_pd, color="red", point_size=9, render_points_as_spheres=True)

    # zoom in and show the plot
    #p.camera.zoom(1.8)
    p.camera_position = 'xy'
    p.show()


def plot_obs_v_sim(m_d):
    from matplotlib.colors import ListedColormap
    sim = pd.read_csv(os.path.join(m_d,'wahp.ss_head.obs.output.csv')).T
    sim = sim.reset_index()
    sim = sim.iloc[1:,:]
    sim.columns = ['obsnme','sim_head']
    sim['id'] = sim.obsnme.apply(lambda x: x.split(':')[1].split('_')[0])
    sim['id'] = sim['id'].astype(int)
    sim['k'] = sim.obsnme.apply(lambda x: int(x.split(':')[2].split('_')[0]))
    obs = pd.read_csv(os.path.join('..','data','processed','wahp_1970_ss_wls.csv'))

    obsvsim = obs.merge(sim, on='id', how='left')

    fig,ax = plt.subplots(figsize=(8, 6))
    # color based on k

    # plot one to one line:
    obs_mn = obsvsim['gwe_ft'].min()
    obs_mx = obsvsim['gwe_ft'].max()
    
    unique_k = np.sort(obsvsim['k'].unique())
    unique_k = unique_k[~np.isnan(unique_k)]
    n_k = len(unique_k)
    base_cmap = plt.get_cmap('tab10')
    if n_k > base_cmap.N:
        # fallback: if you need more than 10 colors, use 'tab20' or generate your own
        base_cmap = plt.get_cmap('tab20')
        if n_k > base_cmap.N:
            raise ValueError(f"Need at least {n_k} colors but 'tab20' only has {base_cmap.N}.")
    # build a ListedColormap of exactly n_k colors
    colors = base_cmap(np.arange(n_k))
    disc_cmap = ListedColormap(colors)
    k_to_idx = {kval: idx for idx, kval in enumerate(unique_k)}
    idx_array = obsvsim['k'].map(k_to_idx).to_numpy()

    sc = ax.scatter(
        obsvsim['gwe_ft'],
        obsvsim['sim_head'],
        s=25,
        c=idx_array,
        cmap=disc_cmap,
        norm=plt.Normalize(vmin=0, vmax=n_k-1),
        alpha=0.7,
        edgecolor='none',
    ) 
    ax.plot([obs_mn-5, obs_mx+5], [obs_mn-5, obs_mx+5], color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Observed (ft)')
    ax.set_ylabel('Simulated (ft)')
    ax.set_title('Observed vs Simulated Steady State Heads')
    ax.set_xlim(obs_mn-5, obs_mx+5)
    ax.set_ylim(obs_mn-5, obs_mx+5)
    
    # legend:
    for kval, idx in k_to_idx.items():
        ax.scatter([], [], color=disc_cmap(idx), label=f'k = {int(kval)}', s=50)
    ax.legend(title='Layer (k)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(m_d, 'obs_vs_sim_heads.png'), dpi=300)


if __name__ == "__main__":
    # load gw model:
    mpws = os.path.join('test_case_kv_1e-3')
    w_d = mpws
    clean_file_fmts(mpws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=mpws,version="mf6")
    gwf = sim.get_model()
    print("Model CRS:", gwf.modelgrid.crs)
    
    nrows = gwf.modelgrid.nrow
    ncols = gwf.modelgrid.ncol
    rows = np.arange(141, 152)
    cols = np.arange(29, 59)
    mask = np.zeros((nrows, ncols), dtype=int)
    mask[np.ix_(rows - 1, cols - 1)] = 1

    # if add second mask define and uncomment the merge line below:
    r2 = np.arange(80,108)
    c2 = np.arange(40, 55)
    mask2 = np.zeros((nrows, ncols), dtype=int)
    mask2[np.ix_(r2 - 1, c2 - 1)] = 1      # convert to 0-based Python index 
    #mask = (mask | mask2).astype(int) # merge
    
    kv_chnage_fig(w_d, gwf, mask)
    plot_params_map_mel(w_d,partype="vk", logscale_hk = True)
    plot_cross_section(model_ws=os.path.join(w_d))
    plot_plan_column(w_d, column=44)
    plot_hds(w_d)
    
    clean_file_fmts(w_d)
    # list of k,i,j to add particles:
    kij = list_of_kij_to_add_particles(lys=np.arange(3,4),rows=np.arange(141, 152),cols=np.arange(29, 59))
    # create particle groups with starting locations file:
    particle_groups = set_particle_groups_strt_loc(gwf,w_d,kij)
    # setup and run modpath simulation
    mp, mpbas, mpsim, gwf = setup_modpath_sim(w_d, gwf, particle_groups)

    plot_modpath(gwf, mpsim, w_d)