import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import sem  # For standard error calculation

# Constants
gamma_N = 27.116e6  # Hz/T (15N gyromagnetic ratio)
gamma_H = 267.513e6  # Hz/T (1H gyromagnetic ratio)
bNH = 1.02e-10  # NH bond length in meters
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (N/A^2)
hbar = 1.0545718e-34  # Reduced Planck's constant (J.s)

# Frequency parameters (example: 600 MHz spectrometer)
omega_H = 2 * np.pi * 600.13e6  # Angular frequency of 1H in Hz
omega_N = omega_H / (gamma_H / gamma_N)  # Angular frequency of 15N in Hz

# Read input data
input_file = "ubq.r2r1.input"  # Replace with your file name
data = pd.read_csv(
    input_file,
    delim_whitespace=True,
    header=None,
    names=["Residue", "R2_R1", "dR2_R1"],
    dtype={"Residue": int, "R2_R1": float, "dR2_R1": float},  # Ensure correct data types
    skip_blank_lines=True,
    comment="#",  # Skip comments starting with #
)

# Function to calculate the model R2/R1 ratio
def R2_R1_model(D_iso, axial_ratio, omega_N):
    D_par = axial_ratio * D_iso
    D_perp = D_iso / 2
    tau_iso = 1 / (2 * (D_par + 2 * D_perp))
    tau_parallel = 1 / (4 * D_par)
    tau_perpendicular = 1 / (4 * D_perp)
    
    # Effective correlation times
    tau_eff = lambda tau: 1 / ((1 / tau) + (1 / tau_iso))
    J_0 = tau_eff(tau_perpendicular)
    J_w = tau_eff(tau_iso + tau_parallel / 2) / (1 + (omega_N**2) * (tau_eff(tau_parallel))**2)
    
    return (4 * J_0 + J_w) / (J_w + 3 * J_0)

# Objective function for global optimization
def global_objective(params, R2_R1_values, dR2_R1_values, omega_N):
    D_iso, axial_ratio = params
    R2_R1_pred = R2_R1_model(D_iso, axial_ratio, omega_N)
    residuals = (R2_R1_values - R2_R1_pred) / dR2_R1_values
    print("D_iso:", D_iso, "D_par/D_per:", axial_ratio)
    print("Predicted R2/R1:", R2_R1_pred)
    return np.sum(residuals**2)

# Extract data for global fitting
R2_R1_values = data["R2_R1"].values
dR2_R1_values = data["dR2_R1"].values

# Initial guesses and bounds
#initial_guess = [1e7, 1.5]  # Initial guess for D_iso and axial ratio
initial_guess = [1e8, 5]  # Example for a broader search
#bounds = [(1e6, 1e8), (1.0, 3.0)]  # Bounds for D_iso and axial ratio
bounds = [(1e6, 1e9), (0.1, 10)]  # Broader search range


# Perform global optimization
#result = minimize(global_objective, initial_guess, args=(R2_R1_values, dR2_R1_values, omega_N), bounds=bounds)
result = minimize(
    global_objective, 
    initial_guess, 
    args=(R2_R1_values, dR2_R1_values, omega_N),
    bounds=bounds, 
    method='trust-constr',  # Different optimizer
    options={"maxiter": 10000, "disp": True}
)


if result.success:
    # Extract optimized parameters
    D_iso, axial_ratio = result.x

    # Compute uncertainties using the Hessian (inverse of the second derivative matrix)
    hessian_inv = result.hess_inv  # Approximation of covariance matrix
    # Calculate parameter errors from the Hessian inverse
    if hasattr(hessian_inv, 'todense'):  # For sparse Hessians
        hessian_inv_array = hessian_inv.todense()  # Convert to dense format
    elif isinstance(hessian_inv, np.ndarray) and hessian_inv.ndim == 2:  # For dense 2D arrays
        hessian_inv_array = hessian_inv
    else:
        hessian_inv_array = None  # Handle invalid Hessian case
    
    # If hessian_inv is valid, compute errors; otherwise, set errors to NaN
    if hessian_inv_array is not None and hessian_inv_array.ndim == 2:
        param_errors = np.sqrt(np.diag(hessian_inv_array))
    else:
        print("Warning: Invalid Hessian inverse. Setting parameter errors to NaN.")
        param_errors = np.full(2, np.nan)

    # Print results
    print("****************** Axial Results *******************")
    print(f"Diso (1/s): {D_iso:.5E} +/- {param_errors[0]:.5E}")
    print(f"Dpar/Dper: {axial_ratio:.5E} +/- {param_errors[1]:.5E}")
else:
    print("Optimization failed.")
