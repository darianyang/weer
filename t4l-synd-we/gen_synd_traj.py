"""
Generate a synthetic trajectory for a given model.
"""
import numpy as np

from synd.models.discrete.markov import MarkovGenerator
from synd.core import load_model

import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter

synd_model = load_model("t4l.synd")

# each step = 1ps
trajectory = synd_model.generate_trajectory(
    initial_states=np.array([800]),
    n_steps=10000, # 10 ns
)

mapped_traj = synd_model.backmap(trajectory)
syn_u = mda.Universe("sim1_dry.pdb")
syn_u.load_new(mapped_traj[0], format="memory")

# Define the output file name
output_xtc = "synd_10ns.xtc"

# Create an XTCWriter object
with mda.coordinates.XTC.XTCWriter(output_xtc, n_atoms=syn_u.atoms.n_atoms) as writer:
    for ts in syn_u.trajectory:
        writer.write(syn_u.atoms)