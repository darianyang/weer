"""
Relxation Calculation from MD Simulations
- using 3 parameter model free analysis
"""

import MDAnalysis as mda
from MDAnalysis.analysis import align

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# missing elements warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.topology.PDBParser")

# Load the alanine dipeptide trajectory
u = mda.Universe("alanine-dipeptide.pdb", "alanine-dipeptide-0-250ns.xtc", 
                 in_memory=True, in_memory_step=1000)

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

    Parameters:
    ----------
    vectors : numpy.ndarray
        A 3D array of shape (n_frames, n_bonds, 3), where each entry represents
        an NH bond vector at a specific time frame.
    max_lag : int
        Maximum lag time for which to calculate the autocorrelation function.

    Returns:
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
def multi_exp_decay(t, *params):
    """
    Multi-exponential decay function.
    
    Parameters:
    -----------
    t : np.ndarray
        Time values for the ACF.
    *params : list
        Parameters for the exponential model, consisting of:
        - A1, A2, ..., An (amplitudes)
        - tau1, tau2, ..., taun (correlation times)
        
    Returns:
    --------
    np.ndarray
        The multi-exponential decay at each time t.
    """
    n = len(params) // 2  # Number of exponentials
    A = params[:n]  # Amplitudes
    tau = params[n:]  # Time constants

    # Sum the exponentials
    result = np.zeros_like(t)
    for i in range(n):
        result += A[i] * np.exp(-t / tau[i])
    return result

def fit_acf(acf_values, time_lags, n_exponentials=2):
    """
    Fit ACF data to a multi-exponential decay model with constraints.

    Parameters:
    ----------
    acf_values : np.ndarray
        ACF values at different time lags.
    time_lags : np.ndarray
        Time lags corresponding to the ACF values.
    n_exponentials : int
        Number of exponential terms to fit.

    Returns:
    --------
    popt : np.ndarray
        Optimized parameters (amplitudes A_i and timescales tau_i).
    """
    # Initial guess for parameters: amplitudes and correlation times
    initial_guess = []
    for i in range(n_exponentials):
        initial_guess.append(np.max(acf_values))  # Initial amplitude guess (max ACF)
        initial_guess.append(1.0)  # Initial guess for correlation time (tau)

    # Constraints: Sum of amplitudes must be 1, and both amplitudes and taus must be positive
    lower_bounds = [0] * n_exponentials + [0] * n_exponentials  # No negative values
    upper_bounds = [np.inf] * n_exponentials + [np.inf] * n_exponentials  # No upper bounds

    # Use `curve_fit` with bounds and constraints
    try:
        popt, _ = curve_fit(
            multi_exp_decay, time_lags, acf_values, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000
        )

        # Enforce the sum of the amplitudes constraint
        amplitudes = popt[:n_exponentials]
        amplitudes /= np.sum(amplitudes)  # Normalize the amplitudes to sum to 1
        popt[:n_exponentials] = amplitudes  # Update the amplitudes in the fit parameters

    except RuntimeError as e:
        print(f"Error during fitting: {e}")
        return None

    return popt

# Example ACF time lags
time_lags = np.linspace(0, 1, 100)  # Time lags in ps or ns
acf_values = acf

# Fit the ACF to a multi-exponential decay
popt = fit_acf(acf_values, time_lags, n_exponentials=3)

# Extract the fitted parameters (amplitudes and timescales)
amplitudes = popt[::2]  # Amplitudes (A1, A2, ...)
timescales = popt[1::2]  # Correlation times (tau1, tau2, ...)

# Plot the data and the fit
plt.plot(time_lags, acf_values, label="ACF Data")
plt.plot(time_lags, multi_exp_decay(time_lags, *popt), label="Multi-Exponential Fit", linestyle="--")
plt.xscale("log")
plt.xlabel("Time Lag (ps/ns)")
plt.ylabel("ACF")
plt.legend()
plt.show()

# Print fitted amplitudes and timescales
print("Fitted amplitudes:", amplitudes, "SUM: ", np.sum(amplitudes))
print("Fitted correlation times:", timescales)

import sys; sys.exit(0)

# Step 3: Fit the ACF to the model-free extended Lipari-Szabo model (3 parameter)
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)
def extended_lipari_szabo_model(t, S2, tau_e, tau_c):
    """
    Model-Free extended Lipari-Szabo model for the ACF fitting.
    """
    return S2 * (tau_e / tau_c) * (1 - np.exp(-t / tau_c)) + (1 - S2) * np.exp(-t / tau_e)

# Fit the ACF to extract S^2, tau_e, and tau_c
times = np.arange(len(acf)) * u.trajectory.dt / 1000  # Convert to ns
popt, _ = curve_fit(extended_lipari_szabo_model, times, acf, p0=(1.0, 1.0, 10.0))

S2, tau_e, tau_c = popt
print(f"S^2 = {S2}, tau_e = {tau_e}, tau_c = {tau_c}")

# Step 4: Calculate R1 and R2 using model-free analysis
def calculate_r1_r2(S2, tau_e, tau_c, omega):
    """
    Calculate R1 and R2 using model-free analysis.
    """
    R1 = (1 - S2) / tau_e + S2 / tau_c * (1 / (1 + (omega * tau_e)**2))
    R2 = S2 / tau_c * (1 / (1 + (omega * tau_e)**2))
    return R1, R2

# Larmor frequencies (in MHz)
omegas = np.linspace(100, 600, 10) * 2 * np.pi  # Convert MHz to radians per second

# Calculate R1 and R2 for each omega
r1_values = []
r2_values = []
for omega in omegas:
    R1, R2 = calculate_r1_r2(S2, tau_e, tau_c, omega)
    r1_values.append(R1)
    r2_values.append(R2)

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(omegas / (2 * np.pi), r1_values, label="R1", color="b")
plt.plot(omegas / (2 * np.pi), r2_values, label="R2", color="r")
plt.xlabel("Larmor Frequency (MHz)")
plt.ylabel("Relaxation Rate (R1, R2)")
plt.title("R1 and R2 Relaxation Rates for Alanine Dipeptide (Model-Free Analysis)")
plt.legend()
plt.show()

