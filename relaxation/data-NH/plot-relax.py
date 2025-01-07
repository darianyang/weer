"""
Plot the T4L NH NMR relaxation data at multiple field strengths.
"""

import numpy as np
import matplotlib.pyplot as plt

def data_extract_and_plot(filename, ax=None):
    """
    Extract the processed NH relaxation data from filename.
    Plot the data on the given axis.

    Parameters
    ----------
    filename : str
        The name of the file containing the relaxation data.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the data.
        If None, a new figure and axis will be created.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # load data
    data = np.loadtxt(filename, delimiter='\t')
    
    # plot R1 data
    ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt='o', label=filename[:6])

data_extract_and_plot("500MHz-R1R2NOE.dat")
plt.legend()
plt.show()

# TODO: make into plot class with input list of MHz values for a 3 panel plot