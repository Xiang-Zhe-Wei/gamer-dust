import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# Command-line parameters
# ============================================================

parser = argparse.ArgumentParser(
    description="Reference dust density evolution plotter"
)
parser.add_argument(
    "-option",
    type=str,
    required=True,
    choices=["edot_0", "edot_const"],
    help="Reference option",
)
args = parser.parse_args()


# ============================================================
# Constants
# ============================================================

UNIT_L = 3.08567758149e21
UNIT_T = 3.15569252e13
UNIT_V = UNIT_L / UNIT_T
UNIT_E = UNIT_V**2

GAMMA = 5.0 / 3.0
MU = 0.603877542851650
MU_TABLE = 0.6
M_H = 1.6735575e-24
K_B = 1.380649e-16
AMU_IN_G = 1.66054e-24
GYR_IN_S = 3.15569252e16
MYR_IN_S = GYR_IN_S / 1.0e3

GRAIN_RADIUS_UM = 0.1
OMEGA = 2.5


# ============================================================
# User settings
# ============================================================

DUST_TO_GAS = 0.1
GAS_RHO0_AMU_LIST = [0.01, 0.1, 1.0]
K_MYR_INV = 1.0
N_COOLING_TIME = 5.0
N_POINTS = 10000
DUST_TOL = 1.0e-3


for gas_rho0_amu in GAS_RHO0_AMU_LIST:
    gas_rho0_cgs = gas_rho0_amu * AMU_IN_G
    dust_rho0 = DUST_TO_GAS * gas_rho0_amu

    k = K_MYR_INV / MYR_IN_S
    t_cool_myr = 1.0 / K_MYR_INV
    t_end_myr = N_COOLING_TIME * t_cool_myr

    # ============================================================
    # Load initial temperature from Data_000000
    # ============================================================

    with h5py.File("../Data_%06d" % 0, "r") as data_file:
        t_grackle = float(
            np.asarray(data_file["GridData/GrackleTemp"]).ravel()[0]
        )
        mu_grackle = float(
            np.asarray(data_file["GridData/GrackleMu"]).ravel()[0]
        )

    t0 = t_grackle * MU_TABLE / mu_grackle

    # ============================================================
    # Unit conversion
    # ============================================================

    def temperature_to_code_energy(temperature):
        factor = (GAMMA - 1.0) * MU * M_H / K_B
        return temperature / (UNIT_E * factor)

    e_code = temperature_to_code_energy(t0)
    e_phys = e_code * UNIT_E

    # ============================================================
    # Reference functions
    # ============================================================

    def internal_energy(e_0, decay_rate, time):
        return e_0 * np.exp(-decay_rate * time)

    def sputtering_time(energy):
        coefficient_1 = (
            0.17
            * (GRAIN_RADIUS_UM / 0.1)
            * (1.0e-27 / gas_rho0_cgs)
            * GYR_IN_S
        )
        coefficient_2 = (
            (10.0**6.3 * K_B) / ((GAMMA - 1.0) * MU * M_H)
        ) ** OMEGA
        return coefficient_1 * (coefficient_2 / energy**OMEGA + 1.0)

    def dust_density_derivative(time, dust_density):
        energy = internal_energy(e_phys, k, time)
        return -3.0 / sputtering_time(energy) * dust_density

    # ============================================================
    # Time array
    # ============================================================

    time = np.linspace(0.0, t_end_myr, N_POINTS)
    time_cool = time / t_cool_myr

    # ============================================================
    # Reference solution
    # ============================================================

    if args.option == "edot_0":
        coefficient_1 = (
            0.17
            * (GRAIN_RADIUS_UM / 0.1)
            * (1.0e-27 / gas_rho0_cgs)
            * GYR_IN_S
        )
        coefficient_2 = (10.0**6.3 / t0) ** OMEGA
        tsp_myr = coefficient_1 * (coefficient_2 + 1.0) / MYR_IN_S

        rho_ref = dust_rho0 * np.exp((-3.0 / tsp_myr) * time)

    elif args.option == "edot_const":
        t_span = (0.0, t_end_myr * MYR_IN_S)
        t_eval = time * MYR_IN_S

        solution = solve_ivp(
            dust_density_derivative,
            t_span,
            [dust_rho0],
            t_eval=t_eval,
            rtol=1.0e-10,
            atol=1.0e-14,
        )

        rho_ref = solution.y[0]

    rho_ref_norm = rho_ref / rho_ref[0]

    # ============================================================
    # Find saturation point from reference
    # Same criterion as C++:
    #     (rho_now - rho_pred) / rho_0 < DUST_TOL
    #
    # where
    #     rho_pred = rho_now * exp(-t_cool/tau_dust)
    #     tau_dust = tsp/3
    # ============================================================

    time_seconds = time * MYR_IN_S
    energy_array = internal_energy(e_phys, k, time_seconds)
    tsp_array = sputtering_time(energy_array)
    tau_dust_myr = (tsp_array / 3.0) / MYR_IN_S

    rho_pred = rho_ref * np.exp(-t_cool_myr / tau_dust_myr)
    delta_dust_norm0 = (rho_ref - rho_pred) / rho_ref[0]

    # Avoid stopping immediately at t = 0
    after_min_time = time > 0.5 * t_cool_myr
    saturation_indices = np.where(
        after_min_time & (delta_dust_norm0 < DUST_TOL)
    )[0]

    if len(saturation_indices) > 0:
        saturation_index = saturation_indices[0]
        t_sat = time[saturation_index]
        t_sat_cool = time_cool[saturation_index]
        rho_now_norm = rho_ref[saturation_index] / rho_ref[0]

        print("====================================")
        print(f"k = {K_MYR_INV:.6g} Myr^-1")
        print(f"gas_density = {gas_rho0_amu:.6g} amu/cm^3")
        print(f"t/t_cool = {t_sat_cool:.6f}")
        print(f"t = {t_sat:.6e} Myr")
        print("====================================")

    else:
        t_sat = None
        t_sat_cool = None
        rho_now_norm = None

        print("====================================")
        print(f"k = {K_MYR_INV:.6g} Myr^-1")
        print(f"gas_density = {gas_rho0_amu:.6g} amu/cm^3")
        print("No saturation point found.")
        print("====================================")

    # ============================================================
    # Title and output filename
    # ============================================================

    k_string = f"{K_MYR_INV:.3g}"
    gas_string = f"{gas_rho0_amu:.3g}"

    figure_title = (
        rf"Reference Dust Density "
        rf"($k={k_string}\ \mathrm{{Myr}}^{{-1}}$, "
        rf"$\rho_{{\rm gas}}={gas_string}\ \mathrm{{amu\ cm}}^{{-3}}$)"
    )

    output_filename = (
        f"fig__DustDensity_reference_k_{k_string}_"
        f"gasdensity_{gas_string}_amu"
    )

    # ============================================================
    # Plot reference only
    # ============================================================

    fig, ax = plt.subplots(1, 1)

    ax.set_title(figure_title)
    ax.set_xlabel(
        r"Number of cooling times $(t/t_{\rm cool})$", fontsize="large"
    )
    ax.set_ylabel(r"$\rho_{\rm dust}/\rho_{\rm dust,0}$", fontsize="large")
    ax.plot(time_cool, rho_ref_norm, "b-", lw=1.8, label="Reference")

    # Mark saturation point
    if t_sat is not None:
        ax.axvline(
            t_sat_cool,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label=r"$(\rho(t)-\rho(t+t_{\rm cool}))/\rho_0 < 10^{-3}$",
        )

        ax.plot(t_sat_cool, rho_now_norm, "ko", ms=5.0)

        annotation = (
            rf"$k = {K_MYR_INV:.3g}\ \mathrm{{Myr}}^{{-1}}$" "\n"
            rf"$\rho_{{\rm gas}} = {gas_rho0_amu:.3g}\ "
            rf"\mathrm{{amu\ cm}}^{{-3}}$" "\n"
            rf"$t/t_{{\rm cool}} = {t_sat_cool:.3f}$" "\n"
            rf"$t = {t_sat:.3e}\ \mathrm{{Myr}}$"
        )

        ax.text(
            0.97,
            0.78,
            annotation,
            transform=ax.transAxes,
            fontsize=13,
            fontfamily="serif",
            math_fontfamily="stix",
            verticalalignment="top",
            horizontalalignment="right",
        )

    ax.set_yscale("linear")
    ax.set_xlim(0.0, N_COOLING_TIME)
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    plt.savefig(
        output_filename + ".png",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=150,
    )
    # plt.show()
