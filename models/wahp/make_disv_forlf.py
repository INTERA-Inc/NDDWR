#!/usr/bin/env python3
import os
import sys
import numpy as np
import flopy

# ---------------------------------------------------------------------
# 0. Define user inputs and create a transform function
# ---------------------------------------------------------------------
cwd = os.getcwd()

original_model_ws = os.path.join(cwd, "model_ws", "model_4_lay")  # original workspace
new_model_ws = os.path.join(cwd, "model_ws", "model_4_lay_disv")  # new workspace
nam_file = "mfsim.nam"
exe_name = "mf6"
model_name = "model_4_lay"

# For coordinate transformation:
xorigin = 2.94125e6  # shift in X
yorigin = -2.60511e5  # shift in Y
angrot = 40.0  # rotation in degrees

if not os.path.exists(new_model_ws):
    os.makedirs(new_model_ws)


def transform_xy(x, y, x0, y0, angle_deg):
    """
    Rotate (x, y) by angle_deg around (0,0),
    then shift by (x0, y0).

    If you want the rotation around (x0, y0), you must
    shift to origin, rotate, then shift back. We do that inside.
    """
    # Convert degrees to radians
    theta = np.radians(angle_deg)
    # Step 1: shift (x,y) so origin is (0,0)
    x_shift = x
    y_shift = y
    # Step 2: rotate about (0,0)
    x_rot = x_shift * np.cos(theta) - y_shift * np.sin(theta)
    y_rot = x_shift * np.sin(theta) + y_shift * np.cos(theta)
    # Step 3: shift back by (x0, y0)
    x_out = x0 + x_rot
    y_out = y0 + y_rot
    return x_out, y_out


# ---------------------------------------------------------------------
# 1. Load original simulation
# ---------------------------------------------------------------------
sim = flopy.mf6.MFSimulation.load(
    nam_file, sim_ws=original_model_ws, exe_name=exe_name, verbosity_level=1
)

sim.set_sim_path(new_model_ws)
gwf = sim.get_model(model_name)

# ---------------------------------------------------------------------
# 2. Extract arrays from DIS
# ---------------------------------------------------------------------
dis = gwf.dis
nlay = dis.nlay.array
nrow = dis.nrow.array
ncol = dis.ncol.array

idomain_3d = dis.idomain.array
top_2d = dis.top.array  # shape (nrow, ncol)
botm_3d = dis.botm.array  # shape (nlay, nrow, ncol)

delr = dis.delr.array  # length ncol
delc = dis.delc.array  # length nrow

# ---------------------------------------------------------------------
# 3. Identify and fix NaN values in top & botm
# ---------------------------------------------------------------------
nan_mask_top = np.isnan(top_2d)
nan_mask_botm = np.isnan(botm_3d)

# A) Replace NaNs in TOP
if np.any(nan_mask_top):
    valid_vals = top_2d[~nan_mask_top]
    if len(valid_vals) == 0:
        raise ValueError("All top values are NaN!")
    first_valid_top = valid_vals[0]
    top_2d[nan_mask_top] = first_valid_top

# B) Replace NaNs in BOTM, layer by layer
for k in range(nlay):
    layer_mask = nan_mask_botm[k]
    if np.any(layer_mask):
        valid_vals = botm_3d[k, ~layer_mask]
        if len(valid_vals) == 0:
            raise ValueError(f"Layer {k} has all NaN in botm!")
        first_valid_bot = valid_vals[0]
        botm_3d[k, layer_mask] = first_valid_bot

# C) Mark those cells inactive (idomain=0)
combined_nan_mask_3d = np.zeros((nlay, nrow, ncol), dtype=bool)
for k in range(nlay):
    # If top was NaN, it applies to all layers in that (row,col)
    combined_nan_mask_3d[k, nan_mask_top] = True
    # If botm layer k was NaN
    combined_nan_mask_3d[k, nan_mask_botm[k]] = True

idomain_3d[combined_nan_mask_3d] = 0

# Overwrite arrays in DIS so they're consistent
dis.top = top_2d
dis.botm = botm_3d
dis.idomain = idomain_3d

# ---------------------------------------------------------------------
# 4. Convert DIS -> DISV (structured approach), applying coordinate shift/rotation
# ---------------------------------------------------------------------
# Flatten arrays
ncpl = nrow * ncol

idomain_2d = np.zeros((nlay, ncpl), dtype=int)
for k in range(nlay):
    idomain_2d[k, :] = idomain_3d[k].ravel()

top_1d = top_2d.ravel()

botm_2d = np.zeros((nlay, ncpl))
for k in range(nlay):
    botm_2d[k, :] = botm_3d[k].ravel()

# Build original 0-based xgrid, ygrid with flipped y-axis
xgrid_0 = np.concatenate(([0.0], np.cumsum(delr)))  # length (ncol+1)
total_height = np.sum(delc)
# Reverse y so that row 0 is the northernmost row
ygrid_0 = total_height - np.concatenate(([0.0], np.cumsum(delc)))  # length (nrow+1)

# Create vertices with shift/rotation
vertices = []
iv = 0
for j in range(nrow + 1):
    for i in range(ncol + 1):
        # Original unrotated/unshifted coords using new ygrid_0
        x0 = xgrid_0[i]
        y0 = ygrid_0[j]
        # Transform
        xT, yT = transform_xy(x0, y0, xorigin, yorigin, angrot)
        vertices.append([iv, xT, yT])
        iv += 1

cell2d = []
icell = 0
for j in range(nrow):
    for i in range(ncol):
        # Cell center (in local 0-based coordinates using new ygrid_0)
        xc_local = 0.5 * (xgrid_0[i] + xgrid_0[i + 1])
        yc_local = 0.5 * (ygrid_0[j] + ygrid_0[j + 1])
        # Transform the center
        xc, yc = transform_xy(xc_local, yc_local, xorigin, yorigin, angrot)

        # Corner vertex indices (same as before)
        v1 = j * (ncol + 1) + i
        v2 = j * (ncol + 1) + i + 1
        v3 = (j + 1) * (ncol + 1) + (i + 1)
        v4 = (j + 1) * (ncol + 1) + i
        cell2d.append([icell, xc, yc, 4, v1, v2, v3, v4])
        icell += 1

# Remove old DIS and create DISV
gwf.remove_package("dis")

disv = flopy.mf6.ModflowGwfdisv(
    gwf,
    nlay=nlay,
    ncpl=ncpl,
    vertices=vertices,
    cell2d=cell2d,
    top=top_1d,
    botm=botm_2d,
    idomain=idomain_2d,
    length_units=(
        dis.length_units.array if dis.length_units.array is not None else "undefined"
    ),
    # You can still store xorigin, yorigin, angrot here, but Leapfrog won't use them:
    xorigin=xorigin,
    yorigin=yorigin,
    angrot=angrot,
)

# ---------------------------------------------------------------------
# 5. Write and (optional) run simulation
# ---------------------------------------------------------------------
sim.write_simulation()
# sim.run_simulation()  # Uncomment to run MF6 if needed
