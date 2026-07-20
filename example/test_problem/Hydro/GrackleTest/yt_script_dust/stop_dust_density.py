import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# load the command-line parameters
parser = argparse.ArgumentParser(description="Reference dust density evolution plotter")
parser.add_argument(
    "-option",
    type=str,
    required=True,
    choices=["edot_0", "edot_const"],
    help="Reference option"
)
args = parser.parse_args()


# ============================================================
# Constants
# ============================================================

Const_cm  = 1.0
Const_amu = 1.660539040e-24
UNIT_L    = 3.08567758149e21
UNIT_M    = 1.9885e42
UNIT_D    = UNIT_M / UNIT_L**3
UNIT_T    = 3.15569252e13

a_um      = 0.1
omega     = 2.5

# Physical constants
gamma    = 5.0/3.0
mu       = 0.603877542851650
m_H      = 1.6735575e-24
k_B      = 1.380649e-16
amu_in_g = 1.66054e-24
Gyr_in_s = 3.15569252e16
Myr_in_s = Gyr_in_s / 1e3


# ============================================================
# User settings
# ============================================================

dust_to_gas = 0.1

gas_rho0_amu_list = [0.01, 0.1, 1.0]

for gas_rho0_amu in gas_rho0_amu_list:
    gas_rho0_cgs = gas_rho0_amu * amu_in_g
    dust_rho0    = dust_to_gas * gas_rho0_amu

    k_Myr_inv = 1.0
    k = k_Myr_inv / Myr_in_s

    t_cool_myr = 1.0 / k_Myr_inv

    n_cooling_time = 5.0
    t_end_myr  = n_cooling_time * t_cool_myr
    n_points   = 10000

    dust_tol = 1.0e-3


    # ============================================================
    # Load initial temperature from Data_000000
    # ============================================================

    with h5py.File("../Data_%06d" % 0, "r") as f:
        T_gamer   = float(np.asarray(f["GridData/Temp"]).ravel()[0])
        T_grackle = float(np.asarray(f["GridData/GrackleTemp"]).ravel()[0])
        mu_gra    = float(np.asarray(f["GridData/GrackleMu"]).ravel()[0])

    T0 = T_grackle * 0.6 / mu_gra


    # ============================================================
    # Unit conversion
    # ============================================================

    def T_to_ecode(T):
        UNIT_V = UNIT_L / UNIT_T
        UNIT_E = UNIT_V**2
        factor = (gamma - 1.0) * mu * m_H / k_B
        e_code = T / (UNIT_E * factor)
        return e_code


    UNIT_V = UNIT_L / UNIT_T
    UNIT_E = UNIT_V**2

    e_code = T_to_ecode(T0)
    e_phys = e_code * UNIT_E


    # ============================================================
    # Reference functions
    # ============================================================

    def internal_energy(e_0, k, t):
        return e_0 * np.exp(-k * t)


    def tsp_e(e_t):
        const_1 = 0.17 * (a_um / 0.1) * (1.0e-27 / gas_rho0_cgs) * Gyr_in_s
        const_2 = ((10.0**6.3 * k_B) / ((gamma - 1.0) * mu * m_H))**omega
        tsp = const_1 * (const_2 / e_t**omega + 1.0)
        return tsp


    def drho_dt(t, dust_rho):
        e_t = internal_energy(e_phys, k, t)
        tsp = tsp_e(e_t)
        return -3.0 / tsp * dust_rho


    # ============================================================
    # Time array
    # ============================================================

    time = np.linspace(0.0, t_end_myr, n_points)
    time_cool = time / t_cool_myr


    # ============================================================
    # Reference solution
    # ============================================================

    if args.option == "edot_0":

        const_1 = 0.17 * (a_um / 0.1) * (1.0e-27 / gas_rho0_cgs) * Gyr_in_s
        const_2 = (10.0**6.3 / T0)**omega
        tsp = const_1 * (const_2 + 1.0)
        tsp_myr = tsp / Myr_in_s

        rho_ref = dust_rho0 * np.exp((-3.0 / tsp_myr) * time)

    elif args.option == "edot_const":

        t_span = (0.0, t_end_myr * Myr_in_s)
        t_eval = time * Myr_in_s

        sol = solve_ivp(
            drho_dt,
            t_span,
            [dust_rho0],
            t_eval=t_eval,
            rtol=1.0e-10,
            atol=1.0e-14
        )

        rho_ref = sol.y[0]


    rho_ref_norm = rho_ref / rho_ref[0]


    # ============================================================
    # Find saturation point from reference
    # Same criterion as C++:
    #     (rho_now - rho_pred) / rho_0 < dust_tol
    #
    # where
    #     rho_pred = rho_now * exp(-t_cool/tau_dust)
    #     tau_dust = tsp/3
    # ============================================================

    t_sec_array = time * Myr_in_s

    e_t_array = internal_energy(e_phys, k, t_sec_array)
    tsp_array = tsp_e(e_t_array)

    tau_dust_myr = (tsp_array / 3.0) / Myr_in_s

    rho_pred = rho_ref * np.exp(-t_cool_myr / tau_dust_myr)

    delta_dust_norm0 = (rho_ref - rho_pred) / rho_ref[0]
    # delta_dust_norm0 = (rho_ref - rho_pred) / rho_ref

    # Avoid stopping immediately at t = 0
    after_min_time = time > 0.5 * t_cool_myr

    sat_indices = np.where((after_min_time) & (delta_dust_norm0 < dust_tol))[0]

    if len(sat_indices) > 0:

        sat_idx = sat_indices[0]

        t_sat = time[sat_idx]
        t_sat_cool = time_cool[sat_idx]

        rho_now = rho_ref[sat_idx]
        rho_now_norm = rho_now / rho_ref[0]

        delta_sat_norm0 = delta_dust_norm0[sat_idx]

        print("====================================")
        print(f"k = {k_Myr_inv:.6g} Myr^-1")
        print(f"gas_density = {gas_rho0_amu:.6g} amu/cm^3")
        print(f"t/t_cool = {t_sat_cool:.6f}")
        print(f"t = {t_sat:.6e} Myr")
        print("====================================")

    else:

        sat_idx = None
        t_sat = None
        t_sat_cool = None
        rho_now_norm = None
        delta_sat_norm0 = None

        print("====================================")
        print(f"k = {k_Myr_inv:.6g} Myr^-1")
        print(f"gas_density = {gas_rho0_amu:.6g} amu/cm^3")
        print("No saturation point found.")
        print("====================================")


    # ============================================================
    # Title and output filename
    # ============================================================

    k_str = f"{k_Myr_inv:.3g}"
    gas_str = f"{gas_rho0_amu:.3g}"

    k_tag = k_str
    gas_tag = gas_str

    fig_name = (
        rf"Reference Dust Density "
        rf"($k={k_str}\ \mathrm{{Myr}}^{{-1}}$, "
        rf"$\rho_{{\rm gas}}={gas_str}\ \mathrm{{amu\ cm}}^{{-3}}$)"
    )

    fileout = f"fig__DustDensity_reference_k_{k_tag}_gasdensity_{gas_tag}_amu"


    # ============================================================
    # Plot reference only
    # ============================================================

    fig, ax = plt.subplots(1, 1)

    ax.set_title(fig_name)
    ax.set_xlabel(r"Number of cooling times $(t/t_{\rm cool})$", fontsize="large")
    ax.set_ylabel(r"$\rho_{\rm dust}/\rho_{\rm dust,0}$", fontsize="large")

    ax.plot(time_cool, rho_ref_norm, "b-", lw=1.8, label="Reference")


    # Mark saturation point
    if t_sat is not None:

        ax.axvline(
            t_sat_cool,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label=r"$(\rho(t)-\rho(t+t_{\rm cool}))/\rho_0 < 10^{-3}$"
            # label=r"$(\rho(t)-\rho(t+t_{\rm cool}))/\rho(t) < 10^{-3}$"
        )

        ax.plot(
            t_sat_cool,
            rho_now_norm,
            "ko",
            ms=5.0
        )

        textstr = (
            rf"$k = {k_Myr_inv:.3g}\ \mathrm{{Myr}}^{{-1}}$" "\n"
            rf"$\rho_{{\rm gas}} = {gas_rho0_amu:.3g}\ \mathrm{{amu\ cm}}^{{-3}}$" "\n"
            rf"$t/t_{{\rm cool}} = {t_sat_cool:.3f}$" "\n"
            rf"$t = {t_sat:.3e}\ \mathrm{{Myr}}$"
        )

        ax.text(
            0.97, 0.78, textstr,
            transform=ax.transAxes,
            fontsize=13,
            fontfamily="serif",
            math_fontfamily="stix",
            verticalalignment="top",
            horizontalalignment="right"
        )

    ax.set_yscale("linear")
    ax.set_xlim(0.0, n_cooling_time)
    ax.set_ylim(0.0, 1.05)

    ax.legend()

    # plt.savefig(fileout + ".png", bbox_inches="tight", pad_inches=0.05, dpi=150)
    # plt.show()