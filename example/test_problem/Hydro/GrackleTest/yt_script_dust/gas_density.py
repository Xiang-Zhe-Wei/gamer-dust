import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


# load the command-line parameters
parser = argparse.ArgumentParser(description="Plot dust temperature")
parser.add_argument("-s", type=int, required=True, help="Starting index")
parser.add_argument("-e", type=int, required=True, help="Ending index")
parser.add_argument("-d", type=int, required=True, help="Index step")
args = parser.parse_args()

# Constants
CONST_CM   = 1.0
CONST_AMU  = 1.660539040e-24
UNIT_L     = 3.08567758149e21
UNIT_M     = 1.9885e42
UNIT_D     = UNIT_M / UNIT_L**3
UNIT_T     = 3.15569252e13
K_MYR_INV  = 1.0
T_COOL_MYR = 1.0 / K_MYR_INV


# Configuration: output filenames
FILEOUT  = "fig__GasDensity_plot"
FIG_NAME = "Gas Density v.s Time"

PREFIX     = '../'
MARKERSIZE = 5.0
LINE_WIDTH = 1
DPI        = 150


gas_density_all = []
time_all = []

for idx in range(args.s, args.e+1, args.d):
    f = h5py.File(os.path.join(PREFIX, 'Data_%06d' % idx), "r")
    gas_density = f["GridData"]["Dens"][0][0][0][0] * UNIT_D / (CONST_AMU / CONST_CM**3)
    time = f["Info"]["KeyInfo"]["Time"][0]
    gas_density_all.append(gas_density)
    time_all.append(time)
    f.close()

gas_density_all = np.array(gas_density_all)
time_all = np.array(time_all)

gas_density_norm = gas_density_all / gas_density_all[0]
time_cool = time_all / T_COOL_MYR


#  plot
fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(wspace=0.4)

ax.set_xlabel(r"$t/t_{\rm cool}$", fontsize="large")
ax.set_title(FIG_NAME)
ax.plot(
    time_cool,
    gas_density_norm,
    'r-o',
    lw=LINE_WIDTH,
    mec='none',
    ms=MARKERSIZE,
    label="Numerical"
)

ax.set_xlim(0.0, time_cool[-1]*1.05)
ax.set_ylabel(
    r"$\rho_{\rm gas}/\rho_{\rm gas,0}$",
    fontsize="large"
)
ax.legend()

#  show/save figure
plt.savefig(FILEOUT+".png", bbox_inches='tight', pad_inches=0.05, dpi=DPI)
#  plt.show()