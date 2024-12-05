"""
Relxation Rate Calculation from MD Simulations
"""

import MDAnalysis as mda
from MDAnalysis.analysis import align

import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

# missing elements warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.topology.PDBParser")

# Load the alanine dipeptide trajectory
u = mda.Universe("alanine-dipeptide.pdb", "alanine-dipeptide-0-250ns.xtc", 
                 in_memory=True, in_memory_step=10)

# Align trajectory to the first frame
ref = mda.Universe("alanine-dipeptide.pdb", "alanine-dipeptide.pdb")
aligner = align.AlignTraj(u, ref, select='name CA', in_memory=True).run()

# Step 1: Calculate NH bond vectors for each frame
def compute_nh_vectors(universe):
    """
    Calculate NH bond vectors for each frame in the trajectory.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis Universe object containing the trajectory.

    Returns
    -------
    nh_vectors: numpy.ndarray
        An array of NH bond vectors with shape (n_frames, n_pairs, 3).
        Each entry corresponds to a bond vector for a specific frame and pair.
    """
    # Select N and H atoms in the backbone
    nh_atoms = universe.select_atoms("backbone and (name N or name H)")
    residues = nh_atoms.residues

    # Determine number of frames and NH pairs
    n_frames = len(universe.trajectory)
    n_pairs = len(residues)

    # Pre-allocate an array for NH bond vectors
    nh_vectors = np.zeros((n_frames, n_pairs, 3), dtype=np.float32)
    
    # Loop over trajectory frames
    for i, _ in enumerate(universe.trajectory):
        for j, res in enumerate(residues):
            try:
                n = res.atoms.select_atoms("name N").positions[0]
                h = res.atoms.select_atoms("name H").positions[0]
                # Store the NH bond vector
                nh_vectors[i, j] = h - n  
            # Use NaN for missing residues
            except IndexError:
                nh_vectors[i, j] = np.nan

    return nh_vectors

nh_vectors = compute_nh_vectors(u)
print("NH vector shape:", nh_vectors.shape)

# # ALT: could use FT of the cos(theta) time series of normalized NH bond vectors to get J(w)
# from scipy.fft import fft, fftfreq

# def calculate_spectral_density_fft(nh_vectors, dt):
#     """
#     Calculate the spectral density J(omega) using FFT from NH bond vectors.

#     Parameters
#     ----------
#     nh_vectors : np.ndarray
#         Time series of NH bond vectors, shape (n_frames, n_bonds, 3).
#     dt : float
#         Time step between frames in the trajectory (ps or ns).

#     Returns
#     -------
#     frequencies : np.ndarray
#         Frequencies corresponding to the spectral density (rad/s).
#     spectral_density : np.ndarray
#         Spectral density J(omega) as a function of frequency.
#     """
#     # Normalize bond vectors to unit length
#     norm_vectors = nh_vectors / np.linalg.norm(nh_vectors, axis=2, keepdims=True)

#     # Compute cos(theta) time series
#     #cos_theta = np.einsum('ijk,jk->ij', norm_vectors, norm_vectors[0])
#     cos_theta = np.sum(norm_vectors * norm_vectors[0], axis=2)

#     # Average over all NH bonds
#     cos_theta_mean = np.mean(cos_theta, axis=1)

#     # Compute FFT
#     fft_result = fft(cos_theta_mean)
#     fft_magnitude = np.abs(fft_result)**2  # Power spectral density

#     # Compute frequencies
#     n_frames = len(cos_theta_mean)
#     frequencies = fftfreq(n_frames, dt) * 2 * np.pi  # Convert to rad/s

#     # Keep only positive frequencies
#     positive_freqs = frequencies[:n_frames // 2]
#     spectral_density = fft_magnitude[:n_frames // 2]

#     return positive_freqs, spectral_density

# positive_freqs, spectral_density = calculate_spectral_density_fft(nh_vectors, 1000)
# plt.plot(spectral_density)
# plt.show()

# Step 2: Compute ACF for the NH bond vectors => C_I(t)
def calculate_acf(vectors, max_lag):
    """
    Calculate the autocorrelation function (ACF) for NH bond vectors using the 
    second-Legendre polynomial.

    Parameters
    ----------
    vectors : numpy.ndarray
        A 3D array of shape (n_frames, n_bonds, 3), where each entry represents
        an NH bond vector at a specific time frame.
    max_lag : int
        Maximum lag time for which to calculate the autocorrelation function.

    Returns
    -------
    numpy.ndarray
        A 1D array of size `max_lag` containing the normalized autocorrelation
        function for each lag time.
    """
    # Normalize the NH bond vectors to unit vectors
    unit_vectors = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)

    # Initialize the array to store the ACF for each lag
    correlations = np.zeros(max_lag, dtype=np.float64)

    # Loop over lag times
    for lag in range(max_lag):
        # Compute dot products for all vectors separated by 'lag' frames
        dot_products = np.einsum(
            'ijk,ijk->ij', 
            unit_vectors[:-lag or None],    # Frames from 0 to len(vectors) - lag
            unit_vectors[lag:]              # Frames from lag to the end
        )
        
        # Apply the second-Legendre polynomial P2(x) = 0.5 * (3x^2 - 1)
        p2_values = 0.5 * (3 * dot_products**2 - 1)

        # Compute the mean over all bonds and all time points
        correlations[lag] = np.nanmean(p2_values)

    return correlations


# input max lag time
acf = calculate_acf(nh_vectors, 100)
print("ACF shape: ", acf.shape)
# # plot ACF
# plt.plot(acf)
# plt.xscale("log")
# plt.show()

# Step 3: Fit C_I(t) to a multi-exponential decay function

# Multi-exponential decay function
# TODO: add Ao offset
def multi_exp_decay(t, A, tau):
    """
    Multi-exponential decay function.
    
    Parameters
    ----------
    t : np.ndarray
        Time values for the ACF.
    A : np.ndarray
        Amplitudes of each exponential component (must sum to 1).
    tau : np.ndarray
        Correlation times of each exponential component.

    Returns
    -------
    np.ndarray
        The multi-exponential decay values.
    """
    return np.sum(A[:, None] * np.exp(-t / tau[:, None]), axis=0)

# Objective function to minimize (sum of squared residuals)
def objective(params, t, acf_values, n_exponentials):
    """
    Objective function for fitting.
    
    Parameters
    ----------
    params : np.ndarray
        Flattened array of amplitudes and correlation times.
    t : np.ndarray
        Time lags.
    acf_values : np.ndarray
        ACF values to fit.
    n_exponentials : int
        Number of exponential components.

    Returns
    -------
    float
        Sum of squared residuals between model and data.
    """
    # Split parameters into amplitudes and taus
    A = params[:n_exponentials]
    tau = params[n_exponentials:]
    # Calculate the multi-exponential decay
    fit = multi_exp_decay(t, A, tau)
    return np.sum((acf_values - fit)**2)

# Fit function with constraints
def fit_acf_minimize(acf_values, time_lags, n_exponentials=2):
    """
    Fit ACF data to a multi-exponential decay model using scipy.optimize.minimize.

    Parameters
    ----------
    acf_values : np.ndarray
        ACF values at different time lags.
    time_lags : np.ndarray
        Time lags corresponding to the ACF values.
    n_exponentials : int
        Number of exponential terms to fit.

    Returns
    -------
    dict
        Result dictionary containing optimized parameters, amplitudes, and correlation times.
    """
    # Initial guess for parameters: equal amplitudes and random time constants
    initial_amplitudes = np.ones(n_exponentials) / n_exponentials
    initial_taus = np.linspace(1, 10, n_exponentials)  # Initial guess for correlation times
    initial_guess = np.concatenate([initial_amplitudes, initial_taus])

    # Constraints: 
    # 1. Sum of amplitudes = 1
    # 2. Amplitudes and taus must be positive
    constraints = [
        {"type": "eq", "fun": lambda params: np.sum(params[:n_exponentials]) - 1},  # A1 + A2 + ... + An = 1
    ]
    bounds = [(0, None)] * (2 * n_exponentials)  # All parameters must be positive

    # Perform optimization
    result = minimize(
        objective,
        initial_guess,
        args=(time_lags, acf_values, n_exponentials),
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        # print convergence messages
        options={"disp": True}
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    # Extract optimized parameters
    optimized_params = result.x
    A = optimized_params[:n_exponentials]
    tau = optimized_params[n_exponentials:]

    return {"amplitudes": A, "correlation_times": tau, "result": result}

# Example ACF time lags: start, stop, num
time_lags = np.linspace(0, acf.shape[0], num=acf.shape[0])
acf_values = acf

# Fit the ACF to a multi-exponential decay
fit_result = fit_acf_minimize(acf_values, time_lags, n_exponentials=5)

# Extract fitted parameters
amplitudes = fit_result["amplitudes"]
timescales = fit_result["correlation_times"]

# # Plot the data and the fit
plt.plot(time_lags, acf_values, label="ACF Data")
plt.plot(time_lags, multi_exp_decay(time_lags, amplitudes, timescales), label="Multi-Exponential Fit", linestyle="--")
plt.xlabel("Time Lag")
plt.ylabel("ACF")
plt.legend()
plt.show()

# Print fitted amplitudes and timescales
print("Fitted amplitudes:", amplitudes, "SUM: ", np.sum(amplitudes))
print("Fitted correlation times:", timescales)

# Step 4: Spectral Density Function - Analytical FT of C(t), where C(t)=C_O(t)C_I(T)
def spectral_density(omega, amplitudes, correlation_times, tau_c):
    """
    Calculate the spectral density function J(omega) with an overall tumbling time tau_c.

    Parameters
    ----------
    omega : np.ndarray or float
        Angular frequency (rad/s) or an array of angular frequencies.
    amplitudes : np.ndarray
        Amplitudes of the exponential components (A_i).
    correlation_times : np.ndarray
        Correlation times of the exponential components (tau_i).
    tau_c : float
        Molecular tumbling time (global correlation time).

    Returns
    -------
    np.ndarray
        Spectral density values J(omega) at the specified angular frequencies.
    """
    # Compute effective correlation times
    tau_eff = (tau_c * correlation_times) / (tau_c + correlation_times)
    
    # Tumbling term (first term in the equation)
    # TODO: include the offset Ao term?
    J = (2 * tau_c) / (1 + (omega * tau_c)**2)
    
    # Add internal contributions
    J += np.sum(
        (2 * amplitudes * tau_eff)[:, None] / (1 + (omega * tau_eff[:, None])**2),
        axis=0
    )
    return J[0]

# # Example: Compute the spectral density with tumbling
# omega = 600.13 * 2 * np.pi * 1e6  # Proton frequency (rad/s)
# tau_c = 1e-9  # Overall tumbling time (seconds)

# # Compute the spectral density
# J = spectral_density(omega, amplitudes, timescales, tau_c)
# print("J:", J)

# Step 5: Compute R1, R2, hetNOE using standard expressions
# Constants
mu_0 = 4 * np.pi * 1e-7     # Permeability of free space (N·A^-2)
hbar = 1.0545718e-34        # Reduced Planck's constant (J·s)
gamma_H = 267.513e6         # Gyromagnetic ratio of 1H (rad·s^-1·T^-1)
gamma_N = -27.116e6         # Gyromagnetic ratio of 15N (rad·s^-1·T^-1)
r_NH = 1.02e-10             # N-H bond length (meters)
Delta_sigma = 0             # CSA value (ppm)

# Derived parameters
d_oo = (mu_0 / (4 * np.pi))**2 * gamma_H**2 * gamma_N**2 * hbar**2
d_oo *= r_NH**-6  # Scale by bond length to the power of -6
c_oo = 1 / 15  # Constant from the equation

# Compute R1, R2, and NOE
def compute_relaxation_parameters(omega_H, omega_N, tau_c, amplitudes, correlation_times):
    """
    Compute relaxation parameters R1, R2, and NOE.

    Parameters
    ----------
    omega_H : float
        Angular frequency of proton (rad/s).
    omega_N : float
        Angular frequency of nitrogen (rad/s).
    tau_c : float
        Overall tumbling correlation time.
    amplitudes : list
        Amplitudes of the correlation function.
    correlation_times : list
        Correlation times of the correlation function.

    Returns
    -------
    tuple
        R1, R2, and NOE values.
    """
    # Spectral density values
    J_omega_H = spectral_density(omega_H, amplitudes, correlation_times, tau_c)
    J_omega_N = spectral_density(omega_N, amplitudes, correlation_times, tau_c)
    J_omega_H_minus_N = spectral_density(omega_H - omega_N, amplitudes, correlation_times, tau_c)
    J_omega_H_plus_N = spectral_density(omega_H + omega_N, amplitudes, correlation_times, tau_c)
    J_0 = spectral_density(0, amplitudes, correlation_times, tau_c)

    # R1 calculation
    R1 = d_oo * (3 * J_omega_N + J_omega_H_minus_N + 6 * J_omega_H_plus_N) + c_oo * J_omega_N
    #print("BREAK: ", R1, J_omega_H)

    # R2 calculation
    R2 = (d_oo / 2) * (4 * J_0 + 3 * J_omega_N + J_omega_H_minus_N + 6 * J_omega_H_plus_N + 6 * J_omega_H) + \
         (c_oo / 6) * (4 * J_0 + 3 * J_omega_N)

    # NOE calculation
    NOE = 1 + (gamma_H / gamma_N) * d_oo * (6 * J_omega_H_plus_N - J_omega_H_minus_N)

    return R1, R2, NOE

# Example usage
tau_c = 1e-9  # Overall tumbling time (seconds)
omega_H = 600.13 * 2 * np.pi * 1e6  # Proton frequency (rad/s)
omega_N = omega_H / 10.0  # Nitrogen frequency (rad/s)

R1, R2, NOE = compute_relaxation_parameters(omega_H, omega_N, tau_c, amplitudes, timescales)

print(f"R1: {R1} s^-1")
print(f"R2: {R2} s^-1")
print(f"NOE: {NOE}")

# TODO: MF2 and MF3 analysis for S2 OPs and tau_internal?