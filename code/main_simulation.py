"""
Mercury's precession due to Jupiter: a simple model
Saúl Díaz Mansilla

Simulates the three-body problem (Sun–Mercury–Jupiter) using the
Euler–Cromer integrator, extracts orbital elements via ellipse fitting
at each perihelion passage, and produces publication-quality figures.
"""

import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import time
import numba as nb
from scipy.optimize import curve_fit
import utils
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

CURRENT_DIR = Path(__file__).parent
FIGURES_DIR = CURRENT_DIR / ".." / "figures"

# ── Physical constants and parameters ────────────────────────────────────────

# General parameters (astronomical units, years, solar masses)
G = 4 * π**2  # Gravitational constant in AU^3 / (yr^2 * M_sun)

# Mercury parameters
ECC_0 = 0.20564  # Mercury orbital eccentricity
A_0 = 0.3871  # Semimajor axis in AU
M_M = 1.659e-7  # Mercury mass divided by solar mass
T_MERCURY = A_0**1.5 * np.sqrt(1 / (1 + M_M))  # Orbital period of Mercury (years)
R_MIN = A_0 * (1 - ECC_0)  # Perihelion distance (AU)
V_MAX = np.sqrt(G * (1 + M_M) * (1 + ECC_0) / R_MIN)  # Speed at perihelion (AU/yr)

# Jupiter parameters
R_JUPITER = 5.2025  # Jupiter orbital radius (AU)
M_J_TRUE = 9.542e-4  # Jupiter mass divided by solar mass
T_JUPITER = R_JUPITER**1.5 * np.sqrt(1 / (1 + M_J_TRUE))  # Jupiter orbital period (years)
V_JUPITER = 2 * π * R_JUPITER / T_JUPITER  # Circular orbital speed for Jupiter (AU/yr)

# Simulation parameters
N_DIV = 5000
N_PERIODS = 500  # Number of Mercury periods to simulate
TF = N_PERIODS * T_MERCURY  # Total integration time (years)
DT = T_MERCURY / N_DIV  # Time step (years)


# ── Fitting functions ────────────────────────────────────────────────────────


def ellipse(theta, theta_0, a, ecc):
    """Ellipse equation (polar) for a Keplerian orbit around a focus.

    Used to fit the instantaneous orbit near a perihelion.
    """
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(theta - theta_0))
    return r


def empirical_fit(t, m, T, A, B, t_0, φ, n):
    """Empirical function to fit the perihelion angle evolution.

    Consists of a linear trend plus two sinusoidal oscillations.
    """
    return (
        -A * np.sin(2 * π / T * (t - t_0))
        + B * np.sin(π / T * (t - t_0) + φ)
        + m * t
        + n
    )


def empirical_2(t, T, A, B, t_0, φ, n):
    """Empirical function consisting of two sinusoidal oscillations."""
    return A * np.sin(2 * π / T * (t - t_0)) + B * np.sin(π / T * (t - t_0) + φ) + n


# ── N-body simulation functions ─────────────────────────────────────────────


@nb.njit
def three_body_problem(x, M_J):
    """N-body acceleration for Sun, Mercury, and Jupiter (Numba-accelerated).

    Array indexing convention:
        x[0], x[1] -> Mercury (xm, ym)
        x[2], x[3] -> Jupiter (xj, yj)
        x[4], x[5] -> Sun     (xs, ys)
    """
    xm, ym, xj, yj, xs, ys = x[0], x[1], x[2], x[3], x[4], x[5]
    dvxdt_M = (
        -G * (xm - xs) / ((xm - xs) ** 2 + (ym - ys) ** 2) ** 1.5
        - G * M_J * (xm - xj) / ((xm - xj) ** 2 + (ym - yj) ** 2) ** 1.5
    )
    dvydt_M = (
        -G * (ym - ys) / ((xm - xs) ** 2 + (ym - ys) ** 2) ** 1.5
        - G * M_J * (ym - yj) / ((xm - xj) ** 2 + (ym - yj) ** 2) ** 1.5
    )
    dvxdt_J = -G * (xj - xs) / ((xj - xs) ** 2 + (yj - ys) ** 2) ** 1.5
    dvydt_J = -G * (yj - ys) / ((xj - xs) ** 2 + (yj - ys) ** 2) ** 1.5
    dvxdt_S = -G * M_J * (xs - xj) / ((xs - xj) ** 2 + (ys - yj) ** 2) ** 1.5
    dvydt_S = -G * M_J * (ys - yj) / ((xs - xj) ** 2 + (ys - yj) ** 2) ** 1.5
    out = np.empty(6)
    out[0] = dvxdt_M
    out[1] = dvydt_M
    out[2] = dvxdt_J
    out[3] = dvydt_J
    out[4] = dvxdt_S
    out[5] = dvydt_S
    return out


@nb.njit
def euler_cromer(f, x0, v0, dt, tf, M_J):
    """Euler-Cromer (symplectic) integrator for the three-body problem.

    Parameters
    ----------
    f  : acceleration function
    x0 : initial positions (6-element array)
    v0 : initial velocities (6-element array)
    dt : time step
    tf : final time
    M_J: Jupiter mass parameter

    Returns
    -------
    x_out, v_out : position and velocity arrays of shape (nsteps+1, 6)
    """
    nsteps = int(tf / dt)
    x_out = np.empty((nsteps + 1, 6))
    v_out = np.empty((nsteps + 1, 6))
    for j in range(6):
        x_out[0, j] = x0[j]
        v_out[0, j] = v0[j]
    for i in range(nsteps):
        a = f(x_out[i, :], M_J)
        for j in range(6):
            v_out[i + 1, j] = v_out[i, j] + a[j] * dt
            x_out[i + 1, j] = x_out[i, j] + v_out[i + 1, j] * dt
    return x_out, v_out


# ── Main simulation and extraction of observables ───────────────────────────


def precession_main(M_J=M_J_TRUE):
    """Run the three-body simulation and extract orbital elements.

    Returns arrays for perihelion times, angles, eccentricities, and
    semi-major axes with their corresponding uncertainties.
    """
    time_simulation_start = time.time()

    # Initial conditions — set Sun velocity so the centre of mass is stationary
    vs = -(M_J * V_JUPITER + M_M * V_MAX) / (1 + M_J + M_M)
    x0 = np.array([R_MIN, 0, R_JUPITER, 0, 0, 0])
    v0 = np.array([0, V_MAX + vs, 0, V_JUPITER + vs, 0, vs])

    # Integrate equations of motion
    x, v = euler_cromer(three_body_problem, x0, v0, DT, TF, M_J)
    x = np.transpose(x)
    v = np.transpose(v)
    print(f"Integration time: {time.time() - time_simulation_start:.2f} seconds")

    # Distance from Mercury to the Sun and corresponding time array
    r = np.sqrt((x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2)
    t = np.linspace(0, TF, len(x[0]))

    # Estimate integration errors (local truncation error for Euler-Cromer)
    error_estimate_xm = np.abs(np.gradient(x[0], DT)) * (DT**2) / 2
    error_estimate_ym = np.abs(np.gradient(x[1], DT)) * (DT**2) / 2

    # Identify perihelion indices (local minima in r)
    perihelion_condition = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    i_perihelion = np.where(perihelion_condition)[0]

    # True polar angle of Mercury relative to the Sun
    theta = np.arctan2(x[1] - x[5], x[0] - x[4])
    theta_error = (x[0] * error_estimate_ym + x[1] * error_estimate_xm) / (r**2)

    # Fit ellipse near each perihelion to extract orbital elements
    theta_perihelion = []
    t_perihelion = []
    ecc_perihelion = []
    decc_perihelion = []
    a_perihelion = []
    da_perihelion = []

    for i in range(len(i_perihelion)):
        start = i_perihelion[i]
        end = i_perihelion[i + 1] if i < len(i_perihelion) - 1 else -1
        fitted, fitted_cov = curve_fit(
            ellipse,
            theta[start:end],
            r[start:end],
            p0=[theta[start], A_0, ECC_0],
            sigma=theta_error[start:end],
            absolute_sigma=True,
        )
        dfitted = np.sqrt(np.diag(fitted_cov))
        theta_perihelion.append(fitted[0])
        dtheta_perihelion = dfitted[0]
        ecc_perihelion.append(fitted[2])
        decc_perihelion.append(dfitted[2])
        a_perihelion.append(fitted[1])
        da_perihelion.append(dfitted[1])
        t_perihelion.append(t[start])

    theta_perihelion = np.unwrap(np.array(theta_perihelion))
    return (
        np.array(t_perihelion),
        theta_perihelion,
        np.array(dtheta_perihelion),
        np.array(ecc_perihelion),
        np.array(decc_perihelion),
        np.array(a_perihelion),
        np.array(da_perihelion),
    )


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_precession(t_perihelion, theta_perihelion, dtheta_perihelion, popt, perr):
    """Create and save the perihelion-angle precession figure."""
    fig, ax = plt.subplots()
    ax.errorbar(
        t_perihelion,
        theta_perihelion,
        yerr=dtheta_perihelion,
        fmt=".",
        ms=3,
        alpha=0.5,
        label="Perihelia",
    )
    ax.plot(
        t_perihelion,
        empirical_fit(t_perihelion, *popt),
        linewidth=2.5,
        label="Empirical fit",
    )

    # Text box with precession rate
    precession_value, precession_error = utils.format_value_error(
        popt[0] * 3600 * 100 * (180 / π), perr[0] * 3600 * 100 * (180 / π)
    )
    text_str = (
        f"$\\dot{{\\omega}} = {precession_value} \\pm {precession_error}$ arcsec/century"
    )
    props_textbox = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.95, 0.05,
        text_str,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props_textbox,
    )

    # Inset zoom
    ax_ins = inset_axes(
        ax,
        width="30%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(0.65, 0.18, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax_ins.errorbar(
        t_perihelion, theta_perihelion, yerr=dtheta_perihelion,
        fmt=".", ms=4, alpha=0.7, label="Perihelia",
    )
    ax_ins.plot(
        t_perihelion, empirical_fit(t_perihelion, *popt),
        linewidth=2, label="Empirical fit",
    )
    ax_ins.set_xlim(85.3, 99.9)
    ax_ins.set_ylim(0.005338, 0.005553)
    mark_inset(ax, ax_ins, loc1=2, loc2=1, fc="none", ec="0.5")

    ax.set_xlabel(r"$t$ (years)")
    ax.set_ylabel(r"$\theta$ (rad)")
    ax.legend(loc="upper left")

    plt.savefig(
        FIGURES_DIR / "precession.pdf", bbox_inches="tight",
        metadata={"Author": "Saúl Díaz Mansilla",
                   "Keywords": f"divisions per period: {N_DIV}, total periods: {N_PERIODS}"},
    )
    plt.savefig(FIGURES_DIR / "precession.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_eccentricity(t_perihelion, ecc_perihelion, decc_perihelion, popt_e):
    """Create and save the eccentricity evolution figure."""
    fig, ax = plt.subplots()
    ax.errorbar(
        t_perihelion, ecc_perihelion, yerr=decc_perihelion,
        fmt=".", capsize=5, label="Eccentricity from ellipse fits",
    )
    ax.plot(
        t_perihelion, empirical_2(t_perihelion, *popt_e),
        "-", color="C1", label="Empirical fit",
    )
    ax.axhline(
        y=ECC_0, color="r", linestyle="--",
        label=f"Initial eccentricity: {ECC_0:.6f}",
    )
    ax.set_xlabel(r"$t$ (years)")
    ax.set_ylabel(r"Eccentricity $\varepsilon$")
    ax.legend(loc="upper left")

    plt.savefig(
        FIGURES_DIR / "eccentricity.pdf", bbox_inches="tight",
        metadata={"Author": "Saúl Díaz Mansilla",
                   "Keywords": f"divisions per period: {N_DIV}, total periods: {N_PERIODS}"},
    )
    plt.savefig(FIGURES_DIR / "eccentricity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_semi_major_axis(t_perihelion, a_perihelion, da_perihelion, popt_a):
    """Create and save the semi-major axis evolution figure."""
    fig, ax = plt.subplots()
    ax.errorbar(
        t_perihelion, a_perihelion, yerr=da_perihelion,
        fmt=".", capsize=5, label="Semi-major axis from ellipse fits",
    )
    ax.plot(
        t_perihelion, empirical_2(t_perihelion, *popt_a),
        "-", color="C2", label="Empirical fit",
    )
    ax.axhline(
        y=A_0, color="r", linestyle="--",
        label=f"Initial semi-major axis: {A_0:.6f} AU",
    )
    ax.set_xlabel(r"$t$ (years)")
    ax.set_ylabel(r"Semi-major axis $a$ (AU)")
    ax.legend(loc="upper left")

    plt.savefig(
        FIGURES_DIR / "semi-major_axis.pdf", bbox_inches="tight",
        metadata={"Author": "Saúl Díaz Mansilla",
                   "Keywords": f"divisions per period: {N_DIV}, total periods: {N_PERIODS}"},
    )
    plt.savefig(FIGURES_DIR / "semi-major_axis.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    utils.use_latex_fonts()
    time_start = time.time()

    # ── Two-body calibration run (M_J = 0) to measure numerical bias ─────
    (
        t_perihelion_2b, theta_perihelion_2b, _,
        ecc_perihelion_2b, _, a_perihelion_2b, _,
    ) = precession_main(0.0)
    ecc_error = np.mean(ecc_perihelion_2b) - ECC_0
    a_error = np.mean(a_perihelion_2b) - A_0
    omegadot_error = np.mean(np.diff(theta_perihelion_2b) / np.diff(t_perihelion_2b))

    # ── Three-body simulation ────────────────────────────────────────────
    (
        t_perihelion, theta_perihelion, dtheta_perihelion,
        ecc_perihelion, decc_perihelion, a_perihelion, da_perihelion,
    ) = precession_main(M_J_TRUE)

    # Compensate for numerical bias
    theta_perihelion = theta_perihelion - omegadot_error * t_perihelion
    ecc_perihelion = ecc_perihelion - ecc_error
    a_perihelion = a_perihelion - a_error

    # ── Estimate synodic period from perihelion-angle maxima ──────────────
    c_max = (theta_perihelion[1:-1] > theta_perihelion[:-2]) & (
        theta_perihelion[1:-1] > theta_perihelion[2:]
    )
    t_max = t_perihelion[np.where(c_max)[0]]
    T_sinodico = np.mean(t_max[1:] - t_max[:-1])

    # ── Fit perihelion angle ─────────────────────────────────────────────
    p0_omegadot = [-2.4e-4, T_sinodico, 1e-3, 1e-4, 0, 0.1, theta_perihelion[0]]
    popt, pcov = curve_fit(
        empirical_fit, t_perihelion, theta_perihelion,
        p0=p0_omegadot, sigma=dtheta_perihelion, absolute_sigma=True,
    )
    perr = np.sqrt(np.diag(pcov))
    p_value_omegadot = utils.calculate_p_value_chi(
        t_perihelion, theta_perihelion, empirical_fit, popt,
        y_err=dtheta_perihelion, print_parameters=False,
    )[0]

    # ── Fit eccentricity ─────────────────────────────────────────────────
    p0_e = [T_sinodico, 1e-3, 1e-3, t_perihelion[0], 0, ECC_0]
    popt_e, pcov_e = curve_fit(
        empirical_2, t_perihelion, ecc_perihelion,
        p0=p0_e, sigma=decc_perihelion, absolute_sigma=True,
    )
    perr_e = np.sqrt(np.diag(pcov_e))

    # ── Fit semi-major axis ──────────────────────────────────────────────
    p0_a = [T_sinodico, 1e-4, 1e-4, t_perihelion[0], 0, A_0]
    popt_a, pcov_a = curve_fit(
        empirical_2, t_perihelion, a_perihelion,
        p0=p0_a, sigma=da_perihelion, absolute_sigma=True,
    )
    perr_a = np.sqrt(np.diag(pcov_a))

    # ── Print results ────────────────────────────────────────────────────
    print(f"\np-value for perihelion angle fit: {p_value_omegadot:.120f}")

    param_names_omega = [
        "m (rad/s)", "T (s)", "A (rad)", "B (rad)",
        "$t_0$ (s)", "$\\phi$ (rad)", "n (rad)",
    ]
    formatted_omega = [
        utils.latex_format(p, e) for p, e in zip(popt, perr)
    ]
    utils.latex_table_scientific(
        param_names_omega, formatted_omega, FIGURES_DIR / "precession.tex",
    )
    print("\nPerihelion fit parameters:")
    utils.print_scientific(popt, perr, param_names_omega)

    param_names_e = ["T (s)", "A", "B", "$t_0$ (s)", "$\\phi$ (rad)", "n"]
    formatted_e = [
        utils.latex_format(p, e) for p, e in zip(popt_e, perr_e)
    ]
    utils.latex_table_scientific(
        param_names_e, formatted_e, FIGURES_DIR / "eccentricity.tex",
    )
    print("\nEccentricity fit parameters:")
    utils.print_scientific(popt_e, perr_e, param_names_e)

    param_names_a = [
        "T (s)", "A (AU)", "B (AU)", "$t_0$ (s)", "$\\phi$ (rad)", "n (AU)",
    ]
    formatted_a = [
        utils.latex_format(p, e) for p, e in zip(popt_a, perr_a)
    ]
    utils.latex_table_scientific(
        param_names_a, formatted_a, FIGURES_DIR / "semi-major_axis.tex",
    )
    print("\nSemi-major axis fit parameters:")
    utils.print_scientific(popt_a, perr_a, param_names_a)

    print(f"\nTotal analysis time: {time.time() - time_start:.2f} seconds")

    # ── Generate and save figures ────────────────────────────────────────
    plot_precession(t_perihelion, theta_perihelion, dtheta_perihelion, popt, perr)
    plot_eccentricity(t_perihelion, ecc_perihelion, decc_perihelion, popt_e)
    plot_semi_major_axis(t_perihelion, a_perihelion, da_perihelion, popt_a)
    print("Figures saved to figures/")


if __name__ == "__main__":
    main()
