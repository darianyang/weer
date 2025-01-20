"""
Testing relaxation calc and reweighting.

This version uses the actual experimental data for reweighting.
And uses updated relaxation calculation code, but still needs improvement.
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import absurder
import relax

# TODO: use actual NMR data and errors for RW
#       and run the relaxation calc again since the code is updated
class NH_Reweight:
    
    def __init__(self, field=600):
        """
        Initialize the NH_Reweight object with the reference trajectory,

        Parameters
        ----------
        field : int, optional
            The field strength for the relaxation data (default is 600 MHz (1H)).
        """
        self.field = field

    def extract_nmr_data(self, nmr_file="data-NH/500MHz-R1R2NOE.dat"):
        """
        Reference data and errors for reweighting.

        Returned data shape: 
            n_rates x n_vectors

        Also return error for nmr data with dimensions: 
            n_rates x n_vectors

        Parameters
        ----------
        nmr_file : str, optional
            Path to the NMR relaxation data file.

        Updates
        -------
        exp_residues : np.ndarray
            Residue list array.

        Returns
        -------
        nmr_rates : np.ndarray
            NMR relaxation rates.
        nmr_err : np.ndarray
            NMR relaxation rate errors
        """
        # load nmrfile: Res | R1 | R1_err | R2 | R2_err | NOE | NOE_err
        nmr_data = np.loadtxt(nmr_file)
        # residue list array
        self.exp_residues = nmr_data[:, 0]
        # shape: n_rates x n_vectors x 1 single traj
        self.nmr_rates = nmr_data[:, (1, 3, 5)].T
        # shape: n_rates x n_vectors
        self.nmr_err = nmr_data[:, (2, 4, 6)].T
        return self.nmr_rates, self.nmr_err

    def calc_md_data(self, md_data_save="t4l/trajs.npy"):
        """
        Load in each short trajectory, calculate relaxation data, build output array.

        Parameters
        ----------
        md_data_save : str, optional
            Path to save the MD relaxation data.
        """
        # shape: n_rates x n_vectors x n_trajs (blocks)
        # here using 107 blocks of 10ns each from MD simulation
        for i in tqdm(range(1, 107)):
            traj = f"t4l/t4l-10ps-imaged2/segment_{i:03d}.xtc"
            relaxation = relax.NH_Relaxation("t4l/sim1_dry.pdb", traj, max_lag=100,
                                            traj_step=10, acf_plot=False, n_exps=5, tau_c=10e-9)
            R1, R2, NOE = relaxation.run()
            # filter the relaxation data to only include the residues in the experimental data
            filtered_indices = [i for i, resid in enumerate(relaxation.residue_indices) 
                                if resid in self.exp_residues]

            # Filter the R1, R2, and NOE arrays
            R1 = R1[filtered_indices]
            R2 = R2[filtered_indices]
            NOE = NOE[filtered_indices]

            # make per block/traj array
            if i == 1:
                relax_data = np.array([R1, R2, NOE])
            else:
                relax_data = np.dstack((relax_data, [R1, R2, NOE]))

        # save and return data
        np.save(md_data_save, relax_data)
        return relax_data

    def run_reweight(self, theta=10, plot=False):
        """
        Run reweighting on the data.
        """
        if self.nmr_rates is None or self.nmr_err is None:
            self.extract_nmr_data()
        rex = self.nmr_rates
        eex = self.nmr_err
        rmd = 't4l/trajs.npy'

        rw = absurder.ABSURDer(rex, rmd, eex, thetas=np.array([theta]))

        #rw.plot_comparison(1)

        #np.savetxt("w0.txt", rw.w0)
        rw.reweight(1)
        np.savetxt(f"w_opt_{theta}.txt", rw.res[theta])

        # for i in range(3):
        #     rw.plot_phix2r(i)

        if plot:
            opt_theta = 100
            rw.plot_comparison(1, opt_theta, outfig='t4l/r2_compare2')

    def plot_weights(self, run_rw=False):
        """
        Plot the weights for each theta.
        """
        for theta in [100, 1000, 10000]:
            if run_rw:
                self.run_reweight(theta)
            w_opt = np.loadtxt(f"w_opt_{theta}.txt")
            plt.plot(w_opt, label=f'theta={theta}')
            # print the traj index with the highest weight
            print(f"theta={theta} \t segments of highest/lowest weight = {np.argmax(w_opt)} / {np.argmin(w_opt)}")
            plt.legend()
        plt.ylim(-0.1, 0.5)
        plt.xlabel("Trajectory Segment")
        plt.ylabel("Weight")
        plt.tight_layout()
        #plt.show()
        plt.savefig("weight_opt.pdf")

if __name__ == '__main__':
    nh = NH_Reweight()
    nh.extract_nmr_data()
    nh.calc_md_data()

    # plot weights
    #plot_weights(run_rw=True)
    #plot_weights(run_rw=False)

    # run rw to make separate plot of chi^2
    #run_reweight(plot=True)