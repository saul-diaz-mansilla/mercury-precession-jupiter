"""
Mercury's precession due to Jupiter: a simple model
Saúl Díaz Mansilla
Updated: 16/12/2025
"""

import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
import time
import numba as nb
from scipy.optimize import curve_fit
import landaubeta as hasperdido
from pathlib import Path

"""--- Initialization ---"""

hasperdido.use_latex_fonts()
time_start = time.time()
current_dir = Path(__file__).parent 

"""--- Define constants and parameters ---"""

# General parameters (astronomical units, years, solar masses)
G = 4 * π**2            # Gravitational constant in AU^3 / (yr^2 * M_sun)

# Mercury parameters
ecc_0 = 0.20564           # Mercury orbital eccentricity
a_0 = 0.3871              # Semimajor axis in AU
M_M = 1.659e-7          # Mercury mass divided by solar mass
T = a_0**1.5 * np.sqrt(1 / (1 + M_M))  # Orbital period of Mercury (years)
rmin = a_0 * (1 - ecc_0)    # perihelion distance (AU)
vmax = np.sqrt(G * (1 + M_M) * (1 + ecc_0) / rmin)  # Approx. speed at perihelion (AU/yr)

# Jupiter parameters
R = 5.2025              # Jupiter orbital radius (AU)
M_J_true = 9.542e-4     # Jupiter mass divided by solar mass
T_J = R**1.5 * np.sqrt(1 / (1 + M_J_true))  # Jupiter orbital period (years)
v_J = 2 * π * R / T_J   # Circular orbital speed for Jupiter (AU/yr)

# Simulation parameters
n_div = 5000
n_periods = 500         # Number of Mercury periods to simulate
tf = n_periods * T            # Total integration time (years)
dt = T / n_div          # Time step (years)
N = int(tf / dt)        # Number of time steps


"""--- Define fitting functions ---"""

def ellipse(theta, theta_0, a, ecc):
    """Ellipse equation (polar) for a Keplerian orbit around a focus.

    Used to fit the instantaneous orbit near a perihelion.
    """
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(theta - theta_0))
    return r

def empirical_fit(t, m, T, A, B, t_0, phi, n):
    """Empirical function to fit the perihelion angle evolution.

    Consists of a linear trend plus two sinusoidal oscillations.
    """
    return -A * np.sin(2 * π / T * (t - t_0)) + B * np.sin(π / T * (t - t_0) + phi) + m * t + n

def empirical_2(t, T, A, B, t_0, phi, n):
    """Empirical function consisting of two sinusoidal oscillations.
    """
    return A * np.sin(2 * π / T * (t - t_0)) + B * np.sin(π / T * (t - t_0) + phi) + n


"""--- N-body simulation functions ---"""

@nb.njit
def three_body_problem(x, M_J):
    """N-body problem with Sun, Mercury, and Jupiter using Numba for speed."""
    """
    Array indexing convention for the output `x` and `v`:
        x[0], x[1] -> Mercury (xm, ym)
        x[2], x[3] -> Jupiter (xj, yj)
        x[4], x[5] -> Sun (xs, ys)
    """
    xm, ym, xj, yj, xs, ys = x[0], x[1], x[2], x[3], x[4], x[5]
    dvxdt_M = - G * (xm - xs) / ((xm - xs)**2 + (ym - ys)**2) ** 1.5 - G * M_J * (xm - xj) / ((xm - xj)**2 + (ym - yj)**2) ** 1.5
    dvydt_M = - G * (ym - ys) / ((xm - xs)**2 + (ym - ys)**2) ** 1.5 - G * M_J * (ym - yj) / ((xm - xj)**2 + (ym - yj)**2) ** 1.5
    dvxdt_J = - G * (xj - xs) / ((xj - xs)**2 + (yj - ys)**2) ** 1.5
    dvydt_J = - G * (yj - ys) / ((xj - xs)**2 + (yj - ys)**2) ** 1.5
    dvxdt_S = - G * M_J * (xs - xj) / ((xs - xj)**2 + (ys - yj)**2) ** 1.5
    dvydt_S = - G * M_J * (ys - yj) / ((xs - xj)**2 + (ys - yj)**2) ** 1.5
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
    """Euler-Cromer integrator for N-body problems using Numba for speed.
    Inputs:
        f: function that computes accelerations given positions
        x0: initial positions (numpy array)
        v0: initial velocities (numpy array)
        dt: time step
        tf: final time
    Outputs:
        x_out: array of positions at each time step
        v_out: array of velocities at each time step
    """
    nsteps = int(tf / dt)
    x_out = np.empty((nsteps + 1, 6))
    v_out = np.empty((nsteps + 1, 6))
    # initialize
    for j in range(6):
        x_out[0, j] = x0[j]
        v_out[0, j] = v0[j]
    for i in range(nsteps):
        a = f(x_out[i, :], M_J)
        for j in range(6):
            v_out[i+1, j] = v_out[i, j] + a[j] * dt
            x_out[i+1, j] = x_out[i, j] + v_out[i+1, j] * dt
    return x_out, v_out


"""--- Main simulation and extraction of observables ---"""

def precession_main(M_J=M_J_true):
    time_simulation_start = time.time()
    # Initial conditions
    # Set the Sun velocity so that the center of mass remains stationary
    vs = -(M_J * v_J + M_M * vmax) / (1 + M_J + M_M)
    x0 = np.array([rmin, 0, R, 0, 0, 0])
    v0 = np.array([0, vmax + vs, 0, v_J + vs, 0, vs])

    # Integrate equations of motion using Euler-Cromer
    x, v = euler_cromer(three_body_problem, x0, v0, dt, tf, M_J)
    x = np.transpose(x)
    v = np.transpose(v)
    print(f"Integration time: {time.time() - time_simulation_start:.2f} seconds")

    # Distance from Mercury to the Sun and corresponding time array
    r = np.sqrt((x[0] - x[4])**2 + (x[1] - x[5])**2)
    t = np.linspace(0, tf, len(x[0]))

    # Estimate integration errors for Mercury's x and y coordinates
    # using two-point central-difference error estimate (local truncation error)
    # For Euler-Cromer: error ~ O(dt^2) per step, accumulates to ~ O(dt) globally
    error_estimate_xm = np.abs(np.gradient(x[0], dt)) * (dt**2) / 2  # Local error estimate
    error_estimate_ym = np.abs(np.gradient(x[1], dt)) * (dt**2) / 2  # Local error estimate
    rms_error_xm = np.sqrt(np.mean(error_estimate_xm**2))
    rms_error_ym = np.sqrt(np.mean(error_estimate_ym**2))

    # Identify perihelion indices: a point is a perihelion if it's a local minimum
    condicion = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    i_perihelion = np.where(condicion)[0]
    r_perihelion = r[i_perihelion]


    # True polar angle of Mercury relative to the Sun position
    theta = np.arctan2(x[1] - x[5], x[0] - x[4])
    theta_error = (x[0] * error_estimate_ym + x[1] * error_estimate_xm) / (r**2)  # Propagated error in theta

    # Fit an ellipse near each perihelion to extract the perihelion angle, eccentricity, and semi-major axis
    theta_perihelion = []
    t_perihelion = []
    ecc_perihelion = []  # Store eccentricity from each fit
    decc_perihelion = []  # Store eccentricity uncertainties
    a_perihelion = []  # Store semi-major axis from each fit
    da_perihelion = []  # Store semi-major axis uncertainties
    for i in range(len(i_perihelion)):
        start = i_perihelion[i]
        # Use the next perihelion as the end of the fitting window (or end of series)
        end = i_perihelion[i+1] if i < len(i_perihelion) - 1 else -1
        fitted, fitted_cov = curve_fit(ellipse, theta[start:end], r[start:end], p0=[theta[start], a_0, ecc_0], sigma=theta_error[start:end], absolute_sigma=True)
        dfitted = np.sqrt(np.diag(fitted_cov))
        theta_perihelion.append(fitted[0])
        dtheta_perihelion = dfitted[0]
        ecc_fit = fitted[2]  # Extract the fitted eccentricity (third parameter)
        decc_fit = dfitted[2]  # Extract uncertainty in eccentricity
        a_fit = fitted[1]  # Extract the fitted semi-major axis (second parameter)
        da_fit = dfitted[1]  # Extract uncertainty in semi-major axis
        ecc_perihelion.append(ecc_fit)
        decc_perihelion.append(decc_fit)
        a_perihelion.append(a_fit)
        da_perihelion.append(da_fit)
        t_perihelion.append(t[start])
    theta_perihelion = np.unwrap(np.array(theta_perihelion))
    return np.array(t_perihelion), theta_perihelion, np.array(dtheta_perihelion), np.array(ecc_perihelion), np.array(decc_perihelion), np.array(a_perihelion), np.array(da_perihelion)

print(f"Setup time: {time.time() - time_start:.2f} seconds")
time_start = time.time()


"""--- Run simulations and analyze results ---"""

# Run the simulation for two-body problem to compensate for numerical inaccuracies
t_perihelion_2b, theta_perihelion_2b, _, ecc_perihelion_2b, _, a_perihelion_2b, _ = precession_main(0.0)
ecc_error = np.mean(ecc_perihelion_2b) - ecc_0
a_error = np.mean(a_perihelion_2b) - a_0
omegadot_error = np.mean(np.diff(theta_perihelion_2b) / np.diff(t_perihelion_2b))

# Run the main precession simulation
t_perihelion, theta_perihelion, dtheta_perihelion, ecc_perihelion, decc_perihelion, a_perihelion, da_perihelion = precession_main(M_J_true)
theta_perihelion = theta_perihelion - omegadot_error * t_perihelion  # Compensate for numerical precession error
ecc_perihelion = ecc_perihelion - ecc_error  # Compensate for numerical eccentricity error
a_perihelion = a_perihelion - a_error  # Compensate for numerical semi-major axis error

# Identify maxima in the perihelion angle sequence to estimate the synodic oscillation period
c_max = (theta_perihelion[1:-1] > theta_perihelion[:-2]) & (theta_perihelion[1:-1] > theta_perihelion[2:])
t_max = t_perihelion[np.where(c_max)[0]]
T_sinodico = np.mean(t_max[1:] - t_max[:-1])

# Empirical fit to the perihelion angle data
p0_omegadot =[-2.4e-4, T_sinodico, 1e-3, 1e-4, 0, 0.1, theta_perihelion[0]]
popt, pcov = curve_fit(empirical_fit, t_perihelion, theta_perihelion, p0=p0_omegadot, sigma=dtheta_perihelion, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
p_value_omegadot = hasperdido.calculate_p_value_chi(t_perihelion, theta_perihelion, empirical_fit, popt, y_err=dtheta_perihelion, print_parameters=False)[0]

# Fit eccentricity to "empirical_2"
p0_e = [T_sinodico, 1e-3, 1e-3, t_perihelion[0], 0, ecc_0]
popt_e, pcov_e = curve_fit(empirical_2, t_perihelion, ecc_perihelion, p0=p0_e, sigma=decc_perihelion, absolute_sigma=True)
perr_e = np.sqrt(np.diag(pcov_e))

res_e = ecc_perihelion - empirical_2(t_perihelion, *popt_e) if not np.all(np.isnan(popt_e)) else np.full_like(ecc_perihelion, np.nan)
chi2_e = np.nansum((res_e / decc_perihelion)**2) if np.all(np.isfinite(decc_perihelion)) else np.nan

# Fit semi-major axis to "empirical_2"
p0_a = [T_sinodico, 1e-4, 1e-4, t_perihelion[0], 0, a_0]
popt_a, pcov_a = curve_fit(empirical_2, t_perihelion, a_perihelion, p0=p0_a, sigma=da_perihelion, absolute_sigma=True)
perr_a = np.sqrt(np.diag(pcov_a))

res_a = a_perihelion - empirical_2(t_perihelion, *popt_a) if not np.all(np.isnan(popt_a)) else np.full_like(a_perihelion, np.nan)
chi2_a = np.nansum((res_a / da_perihelion)**2) if np.all(np.isfinite(da_perihelion)) else np.nan

# Print fit parameters
print(f"\np-value for perihelion angle fit: {p_value_omegadot:.120f}")
print(f"\nPerihelion fit parameters:")
parameter_list = ['m (rad/s)', 'T (s)', 'A (rad)', 'B (rad)', 't_0 (s)', 'phi (rad)', 'n (rad)']
for i in range(len(popt)):
    parameter_value, parameter_error = hasperdido.format_value_error(popt[i], perr[i])
    print(f"  {parameter_list[i]}: {parameter_value} ± {parameter_error}")

print(f"\nEccentricity fit parameters:")
parameter_list = ['T (s)', 'A', 'B', 't_0 (s)', 'phi (rad)', 'n']
for i in range(len(popt_e)):
    parameter_value, parameter_error = hasperdido.format_value_error(popt_e[i], perr_e[i])
    print(f"  {parameter_list[i]}: {parameter_value} ± {parameter_error}")

print(f"\nSemi-major axis fit parameters:")
parameter_list = ['T (s)', 'A (AU)', 'B (AU)', 't_0 (s)', 'phi (rad)', 'n (AU)']
for i in range(len(popt_a)):
    parameter_value, parameter_error = hasperdido.format_value_error(popt_a[i], perr_a[i])
    print(f" {parameter_list[i]}: {parameter_value} ± {parameter_error}")

print(f"\nTotal analysis time: {time.time() - time_start:.2f} seconds")
time_start = time.time()


"""--- Plot results ---"""

# Plot data and fits
plt.errorbar(t_perihelion, theta_perihelion, yerr=dtheta_perihelion, fmt='.', label="Perihelia")
plt.plot(t_perihelion, empirical_fit(t_perihelion, *popt), label="Empirical fit")

# Create text box with precession value in LaTeX notation
precession_value, precession_error = hasperdido.format_value_error(popt[0] * 3600 * 100 * (180/π), perr[0] * 3600 * 100 * (180/π))

text_str = f"$\dot{{\omega}} = {precession_value} \pm {precession_error}$ arcsec/century"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(
    0.95, 0.05,                    # x, y in axes coordinates (5% from left, 5% from bottom)
    text_str,
    transform=plt.gca().transAxes, # axes coords [0..1]
    fontsize=12,
    verticalalignment='bottom',    # anchor vertically at the bottom of the text
    horizontalalignment='right',   # anchor horizontally at the right of the text
    bbox=props
)

plt.xlabel(r'$t$ (years)')
plt.ylabel(r'$\theta$ (rad)')
plt.title("Precession of Mercury's perihelion with Jupiter Perturbation")
plt.legend(loc='upper left')
plt.savefig(current_dir / ".." / "figures" / "precession.pdf", bbox_inches='tight', metadata={'Author': 'Saúl Díaz Mansilla', 'Keywords': f"divisions per period: {n_div}, total periods: {n_periods}"})
# plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(t_perihelion, ecc_perihelion, yerr=decc_perihelion, fmt='.', capsize=5, label="Eccentricity from ellipse fits")
ax.plot(t_perihelion, empirical_2(t_perihelion, *popt_e), '-', color='C1', label='Empirical fit')
ax.axhline(y=ecc_0, color='r', linestyle='--', label=f"Initial eccentricity: {ecc_0:.6f}")
ax.set_xlabel(r'$t$ (years)')
ax.set_ylabel(r'Eccentricity $\varepsilon$')
ax.set_title(r'Mercury orbital eccentricity evolution due to Jupiter perturbation')
ax.legend(loc='upper left')
plt.legend(loc='upper left')
plt.savefig(current_dir / ".." / "figures" / "eccentricity.pdf", bbox_inches='tight', metadata={'Author': 'Saúl Díaz Mansilla', 'Keywords': f"divisions per period: {n_div}, total periods: {n_periods}"})
# plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(t_perihelion, a_perihelion, yerr=da_perihelion, fmt='.', capsize=5, label="Semi-major axis from ellipse fits")
ax.plot(t_perihelion, empirical_2(t_perihelion, *popt_a), '-', color='C2', label='Empirical fit')
ax.axhline(y=a_0, color='r', linestyle='--', label=f"Initial semi-major axis: {a_0:.6f} AU")
ax.set_xlabel(r'$t$ (years)')
ax.set_ylabel(r'Semi-major axis $a$ (AU)')
ax.set_title(r'Mercury orbital semi-major axis evolution due to Jupiter perturbation')
ax.legend(loc='upper left')
plt.legend(loc='upper left')
plt.savefig(current_dir / ".." / "figures" / "semi-major_axis.pdf", bbox_inches='tight', metadata={'Author': 'Saúl Díaz Mansilla', 'Keywords': f"divisions per period: {n_div}, total periods: {n_periods}"})
# plt.show()