"""
Testing relaxation calc and reweighting.
"""

import numpy as np
import matplotlib.pyplot as plt

import absurder
import relax

# first try with example data
# then try with calc relaxation data from script
# rex = 'data/same_ff/nmr.npy'
# rmd = 'data/same_ff/md.npy'

def make_nmr_data():
    """
    Generate the npy binary files for testing the reweighting.
    """
    # load ired output data as experimental data
    # of format: [vector_number, R1, R2, NOE]
    # desired shape: n_rates x n_vectors
    # so skip the vector_number column
    # rotate to get n_rates x n_vectors (transpose)
    ired = np.loadtxt('alanine_dipeptide/ired.noe', usecols=(1,2,3)).T
    np.save('alanine_dipeptide/ired.npy', ired)

def make_exp_err_data():
    """
    Error for nmr data with dimensions: n_rates x n_vectors
    Needed since at least 3 vectors are needed for error estimation
    using the toy model.
    """
    # generate some random error data
    exp_err = np.random.rand(3, 2)
    np.save('alanine_dipeptide/exp_err.npy', exp_err)

def make_md_data():
    """
    Load in each of the 5 trajectories, calculate relaxation data, build output array.
    """
    # desired shape: n_rates x n_vectors x n_trajs (blocks)
    for i in range(0, 250, 50):
        traj = f"alanine_dipeptide/traj_{i}_{i+50}ns.xtc"
        relaxation = relax.NH_Relaxation("alanine_dipeptide/alanine-dipeptide.pdb", traj, 
                                        traj_step=10, acf_plot=False, n_exps=5, tau_c=None)
        R1, R2, NOE = relaxation.run()
        if i == 0:
            relax_data = np.array([R1, R2, NOE])
        else:
            relax_data = np.dstack((relax_data, [R1, R2, NOE]))
    np.save('alanine_dipeptide/trajs.npy', relax_data)

# # make data
# make_nmr_data()
# make_exp_err_data()
# make_md_data()

rex = 'alanine_dipeptide/ired.npy'
rmd = 'alanine_dipeptide/trajs.npy'
eex = 'alanine_dipeptide/exp_err.npy'

rw = absurder.ABSURDer(rex, rmd, eex)

rw.plot_comparison(0)

rw.reweight(0)

# for i in range(3):
#     rw.plot_phix2r(i)

opt_theta = 100
rw.plot_comparison(0, opt_theta)