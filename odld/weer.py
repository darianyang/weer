
import numpy as np

from scipy.stats import entropy
from scipy.special import rel_entr, kl_div

class WEER:
    '''
    Weighted ensemble with ensemble reweighting.
    '''
    def __init__(self, pcoords, weights, true_dist):
        '''
        Parameters
        ----------
        pcoords : ndarray
            Last pcoord value of each segment for the current iteration.
        weights : ndarray
            Weight for each each segment of the current iteration.
        true_dist : ndarray
            Array of the true distibution values to compare pcoord values to.
            For multi-dimensional pcoords, the first dimension should be the 
            same metric as true_dist.
        TODO: add a way to only run reweighting every n iterations

        '''
        # pcoord dimension 0 should be the same metric as true_dist
        if pcoords.ndim == 1:
            self.pcoords = pcoords
        # for multiple pcoords, just use the first for comparison to true_dist
        elif pcoords.ndim >= 2:
            # TODO: check the shape to make sure this indexing works
            self.pcoords = pcoords[:,0]

        self.weights = weights
        self.true_dist = true_dist

    @staticmethod
    def kl_divergence(true_distribution, simulated_distribution):
        """
        Calculate KL divergence between two distributions.
        """
        return entropy(true_distribution, simulated_distribution)

def bin_data(data, bins):
    """
    Bin the data into discrete intervals.
    """
    # with density=True the first element of the return tuple will be 
    # the counts normalized to form a probability density, 
    # i.e., the area (or integral) under the histogram will sum to 1
    # this normalzation is needed for KL divergence calc
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return hist

def kl_divergence(true_distribution, simulated_distribution, bins=10, epsilon=1e-10):
    """
    Calculate KL divergence between two distributions using binning.
    """
    true_hist = bin_data(true_distribution, bins)
    simulated_hist = bin_data(simulated_distribution, bins)

    # Add a small constant epsilon to avoid division by zero
    true_hist += epsilon
    simulated_hist += epsilon

    return entropy(true_hist, simulated_hist)

if __name__ == "__main__":
    # test data (1D array of 1D ODLD endpoints)
    #pcoords = np.loadtxt('pcoords.txt')
    #weights = np.loadtxt('weights.txt')
    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    #pcoords = np.load('pcoords_full.npy')

    # test init data
    # pcoords = np.array([9.5] * 50).reshape(-1,1)
    # weights = np.array([0.02] * 50)

    # WEER test
    # TODO:

    # KL divergence test
    #p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    #q = np.array([90, 2.2, 28, 4.5, 5.5])

    # normalize so both sum to 1
    #p /= np.sum(p)
    #q /= np.sum(q)

    #print(np.sum(p), np.sum(q))

    # print(entropy(p, q))
    # print(rel_entr(p, q), sum(rel_entr(p, q)))
    # print(kl_div(p, q), sum(kl_div(p, q)))
    
    #print(entropy(p, q), sum(rel_entr(p, q)), sum(kl_div(p, q)))

    # can use bins to compare arrays of varying sizes
    # TODO: I should include a layer where I only compare hist from the pcoord 
    #       value range min / max from true_dist that I see on my simulated range
    true_distribution = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    simulated_distribution = np.array([1.2, 2.2, 2.8, 4.5, 5.5, 6.0, 5.0])

    kl_divergence_value = kl_divergence(true_distribution, simulated_distribution)
    print("KL Divergence:", kl_divergence_value)