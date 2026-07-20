import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


# Constants
CONST_CM  = 1.0
CONST_AMU = 1.660539040e-24

K_MYR_INV  = 1.0
T_COOL_MYR = 1.0 / K_MYR_INV

print("k [Myr^-1] =", K_MYR_INV)
print("t_cool [Myr] =", T_COOL_MYR)


# Configuration: output filenames
FILEOUT  = "fig__MetalDensity_plot"
FIG_NAME = "Metal Density v.s Time"

PREFIX     = '../'
LINE_WIDTH = 1
MARKERSIZE = 5.0
DPI        = 150


# Load data
density_all = []
time_all = []

file_list = sorted(glob.glob(os.path.join(PREFIX, "Data_*")))

for file_path in file_list:
    with h5py.File(file_path, "r") as f:
        UNIT_D = f["Info"]["InputPara"]["Unit_D"]

        density = (
            f["GridData"]["Metal"][0][0][0][0]
            * UNIT_D
            / (CONST_AMU / CONST_CM**3)
        )

        time = f["Info"]["KeyInfo"]["Time"][0]

        density_all.append(density)
        time_all.append(time)

density_all = np.array(density_all)
time_all = np.array(time_all)

density_norm = density_all / density_all[0]
time_cool = time_all / T_COOL_MYR


# Plot
fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(wspace=0.4)

ax.set_title(FIG_NAME)
ax.set_xlabel(r"$t/t_{\rm cool}$", fontsize="large")
ax.set_ylabel(
    r"$\rho_{\rm metal}/\rho_{\rm metal,0}$",
    fontsize="large"
)

ax.plot(
    time_cool,
    density_norm,
    'r-o',
    lw=LINE_WIDTH,
    mec='none',
    ms=MARKERSIZE,
    label="Numerical"
)

ax.set_xlim(0, time_cool[-1] * 1.05)
ax.set_yscale('linear')
ax.legend()


# Save outputs
plt.savefig(
    FILEOUT + ".png",
    bbox_inches='tight',
    pad_inches=0.05,
    dpi=DPI
)

# plt.show()