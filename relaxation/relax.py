import MDAnalysis as mda
from MDAnalysis.analysis import align

import numpy as np
from scipy.optimize import minimize, curve_fit
#from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
from functools import partial

import time

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
    Delta_sigma = -170 * 1e-6   # CSA value (ppm) -170 ppm --> dimensionless units
    #Delta_sigma = 0             # CSA value (ppm)

    # Derived parameters
    d_oo = (1 / 20) * (mu_0 / (4 * np.pi))**2 * hbar**2 * gamma_H**2 * gamma_N**2
    d_oo *= r_NH**-6  # Scale by bond length to the power of -6
    c_oo = (1 / 15) * Delta_sigma**2

    # Nuclei frequencies
    omega_H = 600.13 * 2 * np.pi * 1e6      # Proton frequency (rad/s)
    omega_N = omega_H / 10.0                # ~Nitrogen frequency (rad/s)

    def __init__(self, pdb, traj, traj_start=None, traj_stop=None, traj_step=10, 
                 max_lag=None, n_exps=5, acf_plot=False, tau_c=None):
        """
        Initialize the RelaxationCalculator with simulation and analysis parameters.

        Parameters
        ----------
        pdb : str
            Path to the PDB or topology file.
        traj : str
            Path to the trajectory file.
        traj_start : int, optional
            The starting frame index for the trajectory (default is None).
        traj_stop : int, optional
            The stopping frame index for the trajectory (default is None).
        traj_step : int, optional
            Step interval for loading the trajectory (default is 10).
        max_lag : int, optional
            Maximum lag time for ACF computation (default is None, uses entire traj).
        n_exps : int, optional
            Number of exponential functions for ACF fitting (default is 5).
        acf_plot : bool, optional
            Whether to plot the ACF and its fit (default is False).
        tau_c : float, optional
            Overall tumbling time in seconds (default is None).
            Can input a value or otherwise will calculate it from simulation.
        """
        self.pdb = pdb
        self.traj = traj
        self.traj_start = traj_start
        self.traj_stop = traj_stop
        self.traj_step = traj_step
        self.max_lag = max_lag
        self.n_exps = n_exps
        self.acf_plot = acf_plot
        self.tau_c = tau_c

        self.u = self.load_align_traj()

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

    def compute_nh_vectors(self, start=None, stop=None, step=None):
        """
        Calculate NH bond vectors for each frame in the trajectory.

        Parameters
        ----------
        start : int, optional
            The starting frame index.
        stop : int, optional
            The stopping frame index.
        step : int, optional
            The step size between frames.

        Returns
        -------
        nh_vectors: numpy.ndarray
            An array of NH bond vectors with shape (n_frames, n_pairs, 3).
            Each entry corresponds to a bond vector for a specific frame and pair.
        """
        # Select the atoms involved in NH bonds
        selection = self.u.select_atoms('name N or name H')

        # Determine the number of frames and NH pairs
        n_frames = len(self.u.trajectory[start:stop:step])
        n_pairs = len(selection) // 2

        # Pre-cast a numpy array to store NH bond vectors
        nh_vectors = np.zeros((n_frames, n_pairs, 3))

        # Iterate over the trajectory frames and calculate NH bond vectors
        for i, ts in enumerate(self.u.trajectory[start:stop:step]):
            nh_vectors[i] = selection.positions[1::2] - selection.positions[::2]

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
        #print("unit vector shape", unit_vectors.shape)

        # initialize max_lag if not provided
        if self.max_lag is None:
            # Use the entire trajectory length for the ACF (n_frames)
            # limit by default to first 30% of the trajectory? (TODO)
            self.max_lag = int(unit_vectors.shape[0] * 0.3)

        # Initialize the array to store the ACF for each lag
        correlations = np.zeros((self.max_lag, unit_vectors.shape[1]), dtype=np.float64)

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
            #print("P2 Shape: ", p2_values.shape)

            # Compute the mean over all time points for each NH bond vector
            correlations[lag, :] = np.nanmean(p2_values, axis=0)

        #print("Correlations Shape: ", correlations.shape)
        return correlations

    # TODO: currently not working, should provide some speedup if it works
    def calculate_acf_fft(self, vectors):
        """
        Compute ACF using a fully vectorized FFT implementation for each NH bond vector.

        Parameters
        ----------
        vectors : np.ndarray
            A 3D array of shape (n_frames, n_bonds, 3).

        Returns
        -------
        np.ndarray
            The ACF for each NH bond vector with shape (max_lag, n_bonds).
        """
        # Normalize vectors
        unit_vectors = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)

        n_frames = unit_vectors.shape[0]
        n_bonds = unit_vectors.shape[1]

        # Compute dot products for the entire trajectory
        dot_products = np.einsum("ijk,ijk->ij", unit_vectors, unit_vectors)

        # Apply the second-Legendre polynomial P2(x) = 0.5 * (3x^2 - 1)
        p2_values = 0.5 * (3 * dot_products**2 - 1)
        #print("P2 Shape: ", p2_values.shape)
        plt.plot(p2_values)
        plt.yscale('log')
        plt.show()

        ### testing attempt on one set of NH bond vector P2 values
        # TODO: confirm that the FFT works correctly and update this code
        # x = unit_vectors[:,0]

        # from scipy.fftpack import fft, ifft

        # xp = (x - np.average(x))/np.std(x)
        # n, = xp.shape
        # xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
        # f = fft(xp)
        # p = np.absolute(f)**2
        # pi = ifft(p)
        # acf = np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)
        # acf = acf[:self.max_lag]
        # print("acf shape: ", acf.shape)
        
        #plt.plot(acf)
        #plt.show()
        #import sys; sys.exit(0)

        # Initialize an array to store the ACF for each bond
        acf = np.zeros((self.max_lag, n_bonds))

        # compute FFT of the values for each bond vector (2x to prevent FT artifacts)
        fft_uv = np.fft.fft(unit_vectors, n=2*n_frames, axis=0)

        # compute FFT of the P2 values for each bond vector (2x to prevent FT artifacts)
        fft_p2 = np.fft.fft(p2_values, n=2*n_frames, axis=0)

        #np.testing.assert_allclose(fft_p2[:,0], fft_p2[:,1], rtol=1e-5)
        # plt.plot(fft_p2[:,0])
        # plt.plot(fft_p2[:,1])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        # compute power spectrum of each bond, multiply FFT output with complex conj
        # ps_bonds = fft_p2 * np.conjugate(fft_p2)
        # print("PS Shape: ", ps_bonds.shape)
        #np.testing.assert_allclose(ps_bonds[:,0], ps_bonds[:,1], rtol=1e-5)
        # plt.plot(ps_bonds[:,0])
        # plt.plot(ps_bonds[:,1])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        # TODO: up to the power spectrum, the NH bond values are unique
        #       then, the ACF values are nearly the same for each bond
        # inverse FFT to transform power spectrum back to time domain
        # gives ACF for each bond as a function of time lag
        # only take real part of the IFFT output and truncate upto n_frames (traj length)
        #acf_raw = np.fft.ifft(ps_bonds, axis=0).real[:n_frames]
        # np.testing.assert_allclose(acf_raw[:,0], acf_raw[:,1], rtol=1e-5)
        # plt.plot(acf_raw[:,0])
        # plt.plot(acf_raw[:,1])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        # Compute the inverse FFT of the product of the FFTs
        acf_full = np.fft.ifft(fft_p2 * np.conjugate(fft_p2), axis=0).real[:n_frames]
        #acf_full = np.fft.ifft(fft_uv * np.conjugate(fft_uv), axis=0).real[:n_frames]
        print("ACF Full Shape: ", acf_full.shape)
        plt.plot(acf_full[:,0])
        plt.plot(acf_full[:,1])
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        # Normalize the ACF
        # normalization_factors = np.arange(n_frames, 0, -1)
        # acf_full[:n_frames] /= normalization_factors[:, None]
        
        # # Truncate to max_lag and take the real part
        # acf = acf_full[:self.max_lag].real

        return acf
    
    # Method to estimate tau_c from the ACF
    # TODO: could update to give better initial guess, and check units
    #       there is prob also a more accurate or correct way to do this (maybe with MF)
    def estimate_tau_c(self, acf_values):
        """
        Estimate the rotational correlation time (tau_c) from the ACF by fitting it to a 
        single exponential decay function.

        Parameters
        ----------
        acf_values : np.ndarray
            The ACF values to fit, with shape (max_lag, n_bonds).
        
        Returns
        -------
        float
            Estimated rotational correlation time (tau_c).
        """
        # Define the exponential decay function
        # Global tumbling: C_O(t) = exp(-t/tau_c)
        def exp_decay(t, tau_c):
            return np.exp(-t / tau_c)
        
        # Generate time lags
        time_lags = np.arange(acf_values.shape[0])
        
        # Flatten the ACF values and repeat the time lags for global fitting
        flattened_acf_values = acf_values.flatten()
        repeated_time_lags = np.tile(time_lags, acf_values.shape[1])
        
        # Initial guess for the tau_c parameter
        initial_tau_c = self.tau_c
        
        # Perform the global fit using curve_fit
        popt, _ = curve_fit(exp_decay, repeated_time_lags, flattened_acf_values, p0=initial_tau_c)
        
        # Extract the optimized tau_c
        tau_c_estimate = popt[0]

        # Optionally plot the single exponential fit to the ACF
        if self.acf_plot:
            plt.figure()
            for i in range(acf_values.shape[1]):
                plt.plot(time_lags, acf_values[:, i], label=f'ACF {i}')
            plt.plot(time_lags, exp_decay(time_lags, tau_c_estimate), linestyle="--", label=f'Fit {i}')
            plt.title("tau_c Estimate from ACF")
            plt.legend()
            plt.show()

        return tau_c_estimate

    # Multi-exponential decay function
    def multi_exp_decay(self, t, A0, A, tau):
        """
        Multi-exponential decay function with an offset.
        
        Parameters
        ----------
        t : np.ndarray
            Time values for the ACF.
        A0 : float
            Offset constant (A_0 in the equation).
        A : np.ndarray
            Amplitudes of each exponential component (along with A0, must sum to 1).
        tau : np.ndarray
            Correlation times of each exponential component.

        Returns
        -------
        np.ndarray
            The multi-exponential decay values.
        """
        #print("A0: ", A0, "A: ", A, "tau: ", tau)
        return A0 + np.sum(A[:, None] * np.exp(-t / tau[:, None]), axis=0)


    # Fit C_I(t) to a multi-exponential decay function
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
        A = params[:self.n_exps + 1]
        tau = params[self.n_exps + 1:]
        #print("A: ", A, "tau: ", tau)
        # Calculate the multi-exponential decay
        fit = self.multi_exp_decay(t, A[0], A[1:], tau)
        residuals = acf_values - fit
        #print(f"Residuals: {residuals}")
        return np.sum(residuals**2)
        #return residuals

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

        # Initial guess for parameters: equal amplitudes and linear time constants
        initial_amplitudes = np.ones(self.n_exps + 1) / (self.n_exps + 1)
        #print("Initial Amplitudes: ", initial_amplitudes)
        
        # Initial guess for correlation times
        #initial_taus = np.linspace(0.1, 10, self.n_exps)
        # Create a range of initial guesses around tau_c to have similar timescales
        # this is important when calculating J(w) and tau_eff later
        initial_taus = np.linspace(0.1 * self.tau_c, 10 * self.tau_c, self.n_exps)

        initial_guess = np.concatenate([initial_amplitudes, initial_taus])
        #print("Initial Guess: ", initial_guess)

        # Constraints: 
        # 1. Sum of amplitudes = 1
        # 2. Amplitudes and taus must be positive
        constraints = [
            {"type": "eq", "fun": lambda params: np.sum(params[:self.n_exps + 1]) - 1},  # (A0) + A1 + A2 + ... + An = 1
        ]
        bounds = [(0, None)] * ((2 * self.n_exps) + 1)  # All parameters must be positive

        # Perform optimization
        result = minimize(
            partial(self.objective, t=time_lags, acf_values=acf_values),
            initial_guess,
            #args=(time_lags, acf_values, self.n_exps),
            constraints=constraints,
            bounds=bounds,
            method="SLSQP",
            # print convergence messages
            #options={"disp": True}
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        # Extract optimized parameters
        optimized_params = result.x
        A = optimized_params[:self.n_exps + 1]
        tau = optimized_params[self.n_exps + 1:]

        # Optionally plot the data and the fit (TODO: update to OOP plot)
        if self.acf_plot:
            plt.plot(time_lags, acf_values, label="ACF Data")
            plt.plot(time_lags, self.multi_exp_decay(time_lags, A[0], A[1:], tau), label="Multi-Exponential Fit", linestyle="--")
            plt.xlabel("Time Lag")
            plt.ylabel("ACF")
            plt.legend()
            plt.show()

        # Print fitted amplitudes and timescales
        # print("\nFitted amplitudes:", A, "SUM: ", np.sum(A))
        # print("Fitted correlation times:", tau)

        #return {"amplitudes": A, "correlation_times": tau, "result": result}
        return A, tau, result

    # another version which uses lmfit
    def residual(self, params, t, acf_values):
        """
        Residual function for multi-exponential decay fitting.

        Parameters
        ----------
        params : lmfit.Parameters
            Parameters object containing 'A' and 'tau' values.
        t : np.ndarray
            Time lags.
        acf_values : np.ndarray
            Autocorrelation function (ACF) values.

        Returns
        -------
        np.ndarray
            Residuals between the data and the multi-exponential model.
        """
        # Extract parameters
        A = np.array([params[f"A{i}"].value for i in range(self.n_exps + 1)])
        tau = np.array([params[f"tau{i}"].value for i in range(self.n_exps)])

        # Evaluate the multi-exponential decay function
        model = self.multi_exp_decay(t, A[0], A[1:], tau)

        # Residuals
        return acf_values - model

    def fit_acf_lmfit_minimize(self, acf_values, time_lags=None):
        """
        Fit ACF data to a multi-exponential decay model using lmfit.minimize.

        Parameters
        ----------
        acf_values : np.ndarray
            ACF values at different time lags.
        time_lags : np.ndarray, optional
            Time lags corresponding to the ACF values. Default None.
            If None, a time_lags array will be generated.

        Returns
        -------
        tuple
            - A (np.ndarray): Fitted amplitudes.
            - tau (np.ndarray): Fitted correlation times.
            - result (lmfit.MinimizerResult): Fitting result.
        """
        if time_lags is None:
            time_lags = np.linspace(0, acf_values.shape[0], num=acf_values.shape[0])

        # Initialize Parameters
        params = Parameters()
        for i in range(self.n_exps):
            params.add(f"A{i}", value=1 / (self.n_exps + 1), min=0, max=1)    # Amplitudes
            params.add(f"tau{i}", value=self.tau_c, min=0)                    # Correlation times
        # Offset amplitude
        params.add(f"A{self.n_exps + 1}", value=1 / (self.n_exps + 1), min=0, max=1)

        # Constraint: Sum of amplitudes = 1
        params.add('sum_A', expr='+'.join([f"A{i}" for i in range(self.n_exps + 1)]), value=1)

        # Perform the fit
        result = minimize(
            self.residual,
            params,
            args=(time_lags, acf_values),
            method="leastsq",
        )

        # Extract fitted amplitudes and correlation times
        A = np.array([result.params[f"A{i}"].value for i in range(self.n_exps)])
        tau = np.array([result.params[f"tau{i}"].value for i in range(self.n_exps)])

        # Optionally plot the ACF and the fit
        if self.acf_plot:
            fitted_acf = self.multi_exp_decay(time_lags, A, tau)
            plt.plot(time_lags, acf_values, label="ACF Data")
            plt.plot(time_lags, fitted_acf, label="Multi-Exponential Fit", linestyle="--")
            plt.xlabel("Time Lag")
            plt.ylabel("ACF")
            plt.legend()
            plt.show()

        # Return results
        return A, tau, result

    # Step 4: Spectral Density Function - Analytical FT of C(t), where C(t)=C_O(t)C_I(T)
    def spectral_density(self, omega, A, tau):
        """
        Calculate the spectral density function J(omega) with an overall tumbling time tau_c.

        Parameters
        ----------
        omega : np.ndarray or float
            Angular frequency (rad/s) or an array of angular frequencies.
        A : np.ndarray
            Amplitudes of the exponential components (A_i).
        tau : np.ndarray
            Correlation times of the exponential components (tau_i).

        Returns
        -------
        np.ndarray
            Spectral density values J(omega) at the specified angular frequencies.
        """
        # Compute effective correlation times
        tau_eff = (self.tau_c * tau) / (self.tau_c + tau)
        #print("tau_eff: ", tau_eff)

        # Tumbling term (first term in the equation)
        J = (A[0] * 2 * self.tau_c) / (1 + (omega * self.tau_c)**2)
        
        # Add internal contributions
        J += np.sum(
            (2 * A[1:] * tau_eff)[:, None] / (1 + (omega * tau_eff[:, None])**2),
            axis=0
        )
        
        # convert return from list to int value
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

        Returns
        -------
        tuple
            R1, R2, and NOE values. Where each value is a numpy array of relaxation rates.
            Each rate corresponds to a specific NH bond vector.
        """
        # calc NH bond vectors
        nh_vectors = self.compute_nh_vectors(self.traj_start, self.traj_stop)
        
        # calc ACF of norm NH bond vectors
        acf_values = self.calculate_acf(nh_vectors)
        #acf_values2 = self.calculate_acf_fft(nh_vectors)
        
        # plt.plot(acf_values, label="ACF std")
        # plt.plot(acf_values2, linestyle="--", label="ACF fft")
        # plt.legend()
        # plt.show()
        # import sys; sys.exit(0)

        #np.testing.assert_allclose(acf_values, acf_values2, rtol=1e-5)

        # get tau_c if not provided
        # TODO: update the estmate tau_c method, and check units
        if self.tau_c is None:
            self.tau_c = self.estimate_tau_c(acf_values)
            self.tau_c *= 10**-10 # temp conversion for larger contributions from ACF (TODO)
            #print("tau_c: ", self.tau_c)

        # Pre-allocate arrays to store R1, R2, and NOE values for each NH bond vector
        n_bonds = acf_values.shape[1]
        r1_values = np.zeros(n_bonds)
        r2_values = np.zeros(n_bonds)
        noe_values = np.zeros(n_bonds)

        # Loop over each NH bond vector's ACF
        for i in range(n_bonds):
            nh_acf = acf_values[:, i]
            #print(f"ACF for NH bond vector {i}:")
            #print(nh_acf)
            
            # Fit ACF with multiple exponentials
            A, tau, self.result = self.fit_acf_minimize(nh_acf)
            #A, tau, self.result = self.fit_acf_lmfit_minimize(nh_acf)
            
            # Calculate R1, R2, and NOE using J(w) from ACF fitting results
            r1, r2, noe = self.compute_relaxation_parameters(A, tau)
            
            # Store the results in the pre-allocated arrays
            r1_values[i] = r1
            r2_values[i] = r2
            noe_values[i] = noe

            # TODO: move plotting function here to show all NH bond vector fits?

        return r1_values, r2_values, noe_values

    # TODO: methods for MF2 analysis for S2 OPs and tau_internal?

if __name__ == "__main__":
    # Run the NH_Relaxation calculation
    # relaxation = NH_Relaxation("alanine_dipeptide/alanine-dipeptide.pdb", 
    #                            "alanine_dipeptide/alanine-dipeptide-0-250ns.xtc", 
    #                            traj_step=10, acf_plot=True, n_exps=5, tau_c=1e-9, max_lag=100)
    relaxation = NH_Relaxation("t4l/sim1_dry.pdb", 
                               "t4l/t4l-1ps/segment_001.xtc", 
                               traj_step=10, acf_plot=False, n_exps=5, tau_c=10e-9)
    R1, R2, NOE = relaxation.run()

    # Print the results
    n_vectors = None
    print(f"\ntau_c: {relaxation.tau_c} s\n")
    print(f"R1: {R1[:n_vectors]} s^-1 \nT1: {1/R1[:n_vectors]} s\n")
    print(f"R2: {R2[:n_vectors]} s^-1 \nT2: {1/R2[:n_vectors]} s\n")
    print(f"NOE: {NOE[:n_vectors]}\n")