
import MDAnalysis as mda
from MDAnalysis.analysis import align

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial

# missing elements warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.topology.PDBParser")

# TODO: eventually inherit from a Relaxation base class? then children for each spin system?
class NH_Relaxation:
    """
    Backbone Amide Relxation Rate Calculations from MD Simulations.
    """
    # Constants
    mu_0 = 4 * np.pi * 1e-7     # Permeability of free space (N·A^-2)
    hbar = 1.0545718e-34        # Reduced Planck's constant (J·s) (h/2pi)
    gamma_H = 267.513e6         # Gyromagnetic ratio of 1H (rad·s^-1·T^-1)
    gamma_N = -27.116e6         # Gyromagnetic ratio of 15N (rad·s^-1·T^-1)
    r_NH = 1.02e-10             # N-H bond length (meters)
    Delta_sigma = 0             # CSA value (ppm)

    # Derived parameters
    d_oo = (1 / 20) * (mu_0 / (4 * np.pi))**2 * hbar**2 * gamma_H**2 * gamma_N**2
    d_oo *= r_NH**-6  # Scale by bond length to the power of -6
    c_oo = (1 / 15) * Delta_sigma**2

    # Nuclei frequencies
    omega_H = 600.13 * 2 * np.pi * 1e6      # Proton frequency (rad/s)
    omega_N = omega_H / 10.0                # ~Nitrogen frequency (rad/s)

    def __init__(self, pdb, traj, traj_step=10, max_lag=100, n_exps=5, acf_plot=False, tau_c=1e-9):
        """
        Initialize the RelaxationCalculator with simulation and analysis parameters.

        Parameters
        ----------
        pdb : str
            Path to the PDB or topology file.
        traj : str
            Path to the trajectory file.
        traj_step : int, optional
            Step interval for loading the trajectory (default is 10).
        max_lag : int, optional
            Maximum lag time for ACF computation (default is 100).
        n_exps : int, optional
            Number of exponential functions for ACF fitting (default is 5).
        acf_plot : bool, optional
            Whether to plot the ACF and its fit (default is False).
        tau_c : float, optional
            Overall tumbling time in seconds (default is 1e-9).
        """
        self.pdb = pdb
        self.traj = traj
        self.traj_step = traj_step
        self.max_lag = max_lag
        self.n_exps = n_exps
        self.acf_plot = acf_plot
        self.tau_c = tau_c

        self.u = self.load_align_traj()

    def __repr__(self):
        """
        Printable representation. 
        """
        return (f"RelaxationCalculator(pdb={self.pdb!r}, traj={self.traj!r}, "
                f"traj_step={self.traj_step}, max_lag={self.max_lag}, "
                f"n_exps={self.n_exps}, acf_plot={self.acf_plot}, tau_c={self.tau_c})")

    def load_align_traj(self):
        """
        Load and align input trajectory.

        Returns
        -------
        u : MDAnalysis.Universe
            The MDAnalysis Universe object containing the trajectory.
        """
        # Load the alanine dipeptide trajectory
        u = mda.Universe(self.pdb, self.traj, in_memory=True, in_memory_step=self.traj_step)

        # Align trajectory to the first frame
        ref = mda.Universe(self.pdb, self.pdb)
        align.AlignTraj(u, ref, select='name CA', in_memory=True).run()

        return u

    def compute_nh_vectors(self):
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
        nh_atoms = self.u.select_atoms("backbone and (name N or name H)")
        residues = nh_atoms.residues

        # Determine number of frames and NH pairs
        n_frames = len(self.u.trajectory)
        n_pairs = len(residues)

        # Pre-allocate an array for NH bond vectors
        nh_vectors = np.zeros((n_frames, n_pairs, 3), dtype=np.float32)
        
        # Loop over trajectory frames
        for i, _ in enumerate(self.u.trajectory):
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

    # Compute ACF for the NH bond vectors => C_I(t)
    def calculate_acf(self, vectors):
        """
        Calculate the autocorrelation function (ACF) for NH bond vectors using the 
        second-Legendre polynomial.

        Parameters
        ----------
        vectors : numpy.ndarray
            A 3D array of shape (n_frames, n_bonds, 3), where each entry represents
            an NH bond vector at a specific time frame.

        Returns
        -------
        numpy.ndarray
            A 1D array of size `max_lag` containing the normalized autocorrelation
            function for each lag time.
        """
        # Normalize the NH bond vectors to unit vectors
        unit_vectors = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)

        # Initialize the array to store the ACF for each lag
        correlations = np.zeros(self.max_lag, dtype=np.float64)

        # Loop over lag times
        for lag in range(self.max_lag):
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

    def calculate_acf_fft(self, vectors):
        """
        Compute ACF using a fully vectorized FFT implementation.

        Parameters
        ----------
        vectors : np.ndarray
            A 3D array of shape (n_frames, n_bonds, 3).

        Returns
        -------
        np.ndarray
            The averaged ACF over all bonds.
        """
        # Normalize vectors
        unit_vectors = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)
        dot_products = np.einsum("ijk,ijk->ij", unit_vectors, unit_vectors)
        p2_values = 0.5 * (3 * dot_products**2 - 1)

        # Compute FFT for all bonds simultaneously
        n_frames, n_bonds = p2_values.shape
        # zero-padding to prevent aliasing effects in FFT calc
        fft_size = 2 * n_frames
        # FFT of the second-order Legendre polynomial values
        fft_data = np.fft.fft(p2_values, n=fft_size, axis=0)
        # compute power spectrum of each bond, multiply FFT output with complex conj
        ps_bonds = fft_data * np.conjugate(fft_data)
        # inverse FFT to transform power spectrum back to time domain
        # gives ACF for each bond as a function of time lag
        # only take real part of the IFFT output and truncate upto n_frames (traj length)
        acf_raw = np.fft.ifft(ps_bonds, axis=0).real[:n_frames]

        # Normalize each bond's ACF to start at 1
        acf_raw /= acf_raw[0, :]

        # Take the mean over all bonds and truncate to max_lag
        acf = np.nanmean(acf_raw[:self.max_lag, :], axis=1)

        return acf

    # Fit C_I(t) to a multi-exponential decay function
    # Multi-exponential decay function
    # TODO: add Ao offset
    def multi_exp_decay(self, t, A, tau):
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
    def objective(self, params, t, acf_values):
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

        Returns
        -------
        float
            Sum of squared residuals between model and data.
        """
        # Split parameters into amplitudes and taus
        A = params[:self.n_exps]
        tau = params[self.n_exps:]
        # Calculate the multi-exponential decay
        fit = self.multi_exp_decay(t, A, tau)
        return np.sum((acf_values - fit)**2)

    # Fit function with constraints
    def fit_acf_minimize(self, acf_values, time_lags=None):
        """
        Fit ACF data to a multi-exponential decay model using scipy.optimize.minimize.

        Parameters
        ----------
        acf_values : np.ndarray
            ACF values at different time lags.
        time_lags : np.ndarray
            Time lags corresponding to the ACF values. Default None.
            Will guess time_lags array when None provided based on acf_values shape.

        Returns
        -------
        #dict
        #    Result dictionary containing optimized parameters, amplitudes, and correlation times.
        A, tau, result
        """
        # guess time_lags array when None provided
        if time_lags is None:
            time_lags = np.linspace(0, acf_values.shape[0], num=acf_values.shape[0])

        # Initial guess for parameters: equal amplitudes and random time constants
        initial_amplitudes = np.ones(self.n_exps) / self.n_exps
        initial_taus = np.linspace(1, 10, self.n_exps)  # Initial guess for correlation times
        initial_guess = np.concatenate([initial_amplitudes, initial_taus])

        # Constraints: 
        # 1. Sum of amplitudes = 1
        # 2. Amplitudes and taus must be positive
        constraints = [
            {"type": "eq", "fun": lambda params: np.sum(params[:self.n_exps]) - 1},  # A1 + A2 + ... + An = 1
        ]
        bounds = [(0, None)] * (2 * self.n_exps)  # All parameters must be positive

        # Perform optimization
        result = minimize(
            partial(self.objective, t=time_lags, acf_values=acf_values),
            initial_guess,
            #args=(time_lags, acf_values, self.n_exps),
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
        A = optimized_params[:self.n_exps]
        tau = optimized_params[self.n_exps:]

        # Optionally plot the data and the fit (TODO: update to OOP plot)
        if self.acf_plot:
            plt.plot(time_lags, acf_values, label="ACF Data")
            plt.plot(time_lags, self.multi_exp_decay(time_lags, A, tau), label="Multi-Exponential Fit", linestyle="--")
            plt.xlabel("Time Lag")
            plt.ylabel("ACF")
            plt.legend()
            plt.show()

        # Print fitted amplitudes and timescales
        print("Fitted amplitudes:", A, "SUM: ", np.sum(A))
        print("Fitted correlation times:", tau)

        #return {"amplitudes": A, "correlation_times": tau, "result": result}
        return A, tau, result

    # Step 4: Spectral Density Function - Analytical FT of C(t), where C(t)=C_O(t)C_I(T)
    def spectral_density(self, omega, amplitudes, correlation_times):
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

        Returns
        -------
        np.ndarray
            Spectral density values J(omega) at the specified angular frequencies.
        """
        # Compute effective correlation times
        tau_eff = (self.tau_c * correlation_times) / (self.tau_c + correlation_times)
        
        # Tumbling term (first term in the equation)
        # TODO: include the offset Ao term?
        J = (2 * self.tau_c) / (1 + (omega * self.tau_c)**2)
        
        # Add internal contributions
        J += np.sum(
            (2 * amplitudes * tau_eff)[:, None] / (1 + (omega * tau_eff[:, None])**2),
            axis=0
        )
        return J[0]

    # Step 5: Compute R1, R2, hetNOE using standard expressions
    # Compute R1, R2, and NOE
    def compute_relaxation_parameters(self, amplitudes, correlation_times):
        """
        Compute relaxation parameters R1, R2, and NOE.

        Parameters
        ----------
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
        J_omega_H = self.spectral_density(self.omega_H, amplitudes, correlation_times)
        J_omega_N = self.spectral_density(self.omega_N, amplitudes, correlation_times)
        J_omega_H_minus_N = self.spectral_density(self.omega_H - self.omega_N, amplitudes, correlation_times)
        J_omega_H_plus_N = self.spectral_density(self.omega_H + self.omega_N, amplitudes, correlation_times)
        J_0 = self.spectral_density(0, amplitudes, correlation_times)

        # R1 calculation
        R1 = self.d_oo * (3 * J_omega_N + J_omega_H_minus_N + 6 * J_omega_H_plus_N) + \
                         (self.c_oo * self.omega_N**2 * J_omega_N)

        # R2 calculation
        R2 = (self.d_oo / 2) * (4 * J_0 + 3 * J_omega_N + J_omega_H_minus_N + 6 * J_omega_H + 6 * J_omega_H_plus_N) + \
             (self.c_oo * self.omega_N**2 / 6) * (4 * J_0 + 3 * J_omega_N)

        # NOE calculation
        NOE = 1 + (self.gamma_H / self.gamma_N) * self.d_oo * (1 / R1) * (6 * J_omega_H_plus_N - J_omega_H_minus_N)

        return R1, R2, NOE

    def run(self):
        """
        Main public method for calculating R1, R2, and NOE values from input MD simulation.
        """
        # calc NH bond vectors
        nh_vectors = self.compute_nh_vectors()
        # calc ACF of norm NH bond vectors
        acf_values = self.calculate_acf(nh_vectors)
        #acf_values = self.calculate_acf_fft(nh_vectors)
        # fit ACF with multiple exponentials
        A, tau, self.result = self.fit_acf_minimize(acf_values)
        # calc R1, R2, and NOE using J(w) from ACF fitting results
        r1, r2, noe = self.compute_relaxation_parameters(A, tau)
        return r1, r2, noe

    # TODO: MF2 analysis for S2 OPs and tau_internal?

if __name__ == "__main__":

    R1, R2, NOE = NH_Relaxation("alanine-dipeptide.pdb", "alanine-dipeptide-0-250ns.xtc", 100, acf_plot=True).run()

    print(f"R1: {R1:.4f} s^-1 | T1: {1/R1:.4f} s")
    print(f"R2: {R2:.4f} s^-1 | T2: {1/R2:.4f} s")
    print(f"NOE: {NOE:.4f}")