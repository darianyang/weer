"""
Testing relaxation calc and reweighting.
"""

import absurder
import numpy as np
import matplotlib.pyplot as plt

# first try with example data
# then try with calc relaxation data from script
rex = 'data/same_ff/nmr.npy'
rmd = 'data/same_ff/md.npy'

#rw = absurder.ABSURDer(rex, rmd)

#rw.plot_comparison(0)

# rw.reweight(0)

# for i in range(3):
#     rw.plot_phix2r( i )

# opt_theta = 100
# rw.plot_comparison( 0, opt_theta )