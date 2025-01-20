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
    
    def __init__(self, field=500):
        """
        Initialize the NH_Reweight object with the reference trajectory,

        Parameters
        ----------
        field : int, optional
            The field strength for the relaxation data (default is 500 MHz (1H)).
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

        Returns
        -------
        nmr_rates : np.ndarray
            NMR relaxation rates.
        nmr_err : np.ndarray
            NMR relaxation rate errors
        """
        # load nmrfile: Res | R1 | R1_err | R2 | R2_err | NOE | NOE_err
        nmr_data = np.loadtxt(nmr_file)
        # shape: n_rates x n_vectors x 1 single traj
        self.nmr_rates = nmr_data[:, (1, 3, 5)].T
        # shape: n_rates x n_vectors
        self.nmr_err = nmr_data[:, (2, 4, 6)].T
        return self.nmr_rates, self.nmr_err

    def make_md_data(self):
        """
        Load in each short trajectory, calculate relaxation data, build output array.
        """
        # desired shape: n_rates x n_vectors x n_trajs (blocks)
        for i in tqdm(range(1, 107)):
            traj = f"t4l/t4l-10ps-imaged2/segment_{i:03d}.xtc"
            relaxation = relax.NH_Relaxation("t4l/sim1_dry.pdb", traj, 
                                            traj_step=10, acf_plot=False, n_exps=5, tau_c=10e-9)
            R1, R2, NOE = relaxation.run()
            if i == 1:
                relax_data = np.array([R1, R2, NOE])
            else:
                relax_data = np.dstack((relax_data, [R1, R2, NOE]))
        np.save('t4l/trajs.npy', relax_data)

    def load_and_check_array(self, array):
        """
        Load and check the shape of the array.
        """
        data = np.load(array)
        print(data.shape)
        return data

    def run_reweight(self, theta=10, plot=False):
        """
        Run reweighting on the data.
        """
        rex = 't4l/ref-traj.npy'
        rmd = 't4l/trajs.npy'
        eex = 't4l/exp_err.npy'

        #load_and_check_array(rex)

        rw = absurder.ABSURDer(rex, rmd, eex, thetas=np.array([theta]))

        #rw.plot_comparison(1)

        #np.savetxt("w0.txt", rw.w0)
        rw.reweight(1)
        np.savetxt(f"w_opt_{theta}.txt", rw.res[theta])

        # for i in range(3):
        #     rw.plot_phix2r(i)

        if plot:
            opt_theta = 10
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
    # make data
    nh.extract_nmr_data()

    #make_nmr_data()
    # make_md_data()
    #make_exp_err_data()

    # plot weights
    #plot_weights(run_rw=True)
    #plot_weights(run_rw=False)

    # run rw to make separate plot of chi^2
    #run_reweight(plot=True)