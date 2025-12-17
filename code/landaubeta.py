"""
Utility functions for data analysis and formatting.
Saúl Díaz Mansilla
Updated: 17/12/2025
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def use_latex_fonts():
    """Configure matplotlib to use LaTeX font rendering"""
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

def use_IEEE_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],     # IEEE uses Times New Roman
        "font.size": 10,             # Standard IEEE body text size
        "axes.labelsize": 10,
        "legend.fontsize": 8,        # Slightly smaller legends
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (3.5, 3.5 / 1.618),
        "savefig.bbox": "tight",     # Minimizes white space around the plot
        "savefig.pad_inches": 0.05
})

def format_value_error(value, error):
    """Format value with precision matching error's significant figures"""
    
    # Calculate number of decimal places based on error
    if error == 0:
        # Handle zero error case
        return f"{value:.6g}", "0"
    
    # Determine the position of the first significant digit in error
    if error > 0:
        error_exp = math.floor(math.log10(error))
    else:
        error_exp = math.floor(math.log10(-error))
    
    # Number of decimal places for value (negative if digits before decimal)
    decimal_places = -error_exp
    
    # Format error with 1 significant figure
    formatted_error = f"{error:.1g}"
    
    # For the value, we need to match the decimal places
    if decimal_places <= 0:
        # Error has significant digits before decimal point
        # Show value with same number of decimal places as error's position
        int_places = -decimal_places + 1
        formatted_value = f"{value:.{int_places}f}"
    else:
        # Error has significant digits after decimal point
        # Show value with same number of decimal places
        formatted_value = f"{value:.{decimal_places}f}"
    
    return formatted_value, formatted_error

def calculate_p_value_chi(x, y, model_func, popt, y_err=None, print_parameters=False):
    """
    Calculates the chi-squared statistic and p-value for a given fit.
    """
    # Calculate residuals
    residuals = y - model_func(x, *popt)
    
    # Use provided errors, otherwise estimate from residuals
    if y_err is None:
        y_err = np.ones_like(y) * np.std(residuals)
    
    # Chi-squared statistic
    chi2_stat = np.sum((residuals / y_err)**2)
    
    # Degrees of freedom
    dof = len(y) - len(popt)
    
    # P-value
    p_value = 1.0 - chi2.cdf(chi2_stat, dof)
    
    if print_parameters:
        print(f"\nEmpirical fit statistics:")
        print(f"Chi-squared statistic: {chi2_stat:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_value:.120f}")
        print(f"Reduced chi-squared (χ²/dof): {chi2_stat / dof:.4f}")

    return p_value, chi2_stat, dof