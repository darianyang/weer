"""
Testing relaxation calc and reweighting.
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import absurder
import relax

def make_nmr_data():
    """
    Load in 1 trajectory, calculate relaxation data, build output array.
    This will be the reference data for reweighting.
    """
    # desired shape: n_rates x n_vectors x 1 single traj
    traj = f"t4l/sim1-100ps-imaged2.xtc"
    relaxation = relax.NH_Relaxation("t4l/sim1_dry.pdb", traj, 
                                    traj_step=1, acf_plot=False, n_exps=5, tau_c=10e-9)
    R1, R2, NOE = relaxation.run()
    # TODO: manually manipulating R2 to be 90% of the original value ?
    relax_data = np.array([R1, R2*0.9, NOE])
    np.save('t4l/ref-traj.npy', relax_data)

def make_md_data():
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

def make_exp_err_data():
    """
    Error for nmr data with dimensions: n_rates x n_vectors
    Needed since at least 3 vectors are needed for error estimation
    using the toy model.
    """
    # generate some random error data
    exp_err = np.random.rand(3, 160)
    np.save('t4l/exp_err.npy', exp_err)

def load_and_check_array(array):
    """
    Load and check the shape of the array.
    """
    data = np.load(array)
    print(data.shape)
    return data

def run_reweight(theta=10, plot=False):
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

def plot_weights(run_rw=False):
    """
    Plot the weights for each theta.
    """
    for theta in [10, 100, 1000]:
        if run_rw:
            run_reweight(theta)
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

# make data
#make_nmr_data()
# make_md_data()
#make_exp_err_data()

# plot weights
#plot_weights(run_rw=True)
plot_weights(run_rw=False)

# run rw to make separate plot of chi^2
#run_reweight(plot=True)