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
    traj = f"t4l/sim1-100ps.xtc"
    relaxation = relax.NH_Relaxation("t4l/sim1_dry.pdb", traj, 
                                    traj_step=1, acf_plot=False, n_exps=5, tau_c=10e-9)
    R1, R2, NOE = relaxation.run()
    relax_data = np.array([R1, R2, NOE])
    np.save('t4l/ref-traj.npy', relax_data)

def make_md_data():
    """
    Load in each short trajectory, calculate relaxation data, build output array.
    """
    # desired shape: n_rates x n_vectors x n_trajs (blocks)
    for i in tqdm(range(1, 107)):
        traj = f"t4l/t4l-10ps/segment_{i:03d}.xtc"
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

# make data
#make_nmr_data()
#make_md_data()
#make_exp_err_data()

rex = 't4l/ref-traj.npy'
rmd = 't4l/trajs.npy'
eex = 't4l/exp_err.npy'

#load_and_check_array(rex)

rw = absurder.ABSURDer(rex, rmd, eex)

rw.plot_comparison(1)

rw.reweight(1)

# for i in range(3):
#     rw.plot_phix2r(i)

opt_theta = 100
rw.plot_comparison(1, opt_theta)