
import numpy as np

from scipy.stats import entropy
from scipy.special import rel_entr, kl_div
from scipy.optimize import minimize

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
              and it would be useful to take data from n iteration back
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

    def bin_data(self, data, bins, weights=None):
        """
        Bin the data into discrete intervals.
        This is to account for different length arrays between true and simulated
        distributions. Could also 
        """
        # with density=True the first element of the return tuple will be 
        # the counts normalized to form a probability density, 
        # i.e., the area (or integral) under the histogram will sum to 1
        # this normalzation is needed for KL divergence calc
        # TODO: can use WE weights directly here, make sure it works as expected
        hist, bin_edges = np.histogram(data, bins=bins, density=True, weights=weights)
        return hist

    def kl_divergence(self, weights, true_distribution, simulated_distribution, 
                      bins=100, epsilon=1e-10):
        """
        Calculate KL divergence between two distributions using binning.

        Parameters
        ----------
        true_distribution : array
        simulated_distribution : array
        bins : int
        epsilon : float
            Small constant to avoid zero division error
        """
        # only need to weight simulated data from WE
        true_hist = self.bin_data(true_distribution, bins)
        simulated_hist = self.bin_data(simulated_distribution, bins, weights)

        # Add a small constant epsilon to avoid division by zero
        true_hist += epsilon
        simulated_hist += epsilon

        return entropy(true_hist, simulated_hist)

    def optimize_weights(self, true_distribution, initial_weights, simulated_distribution):
        """
        Optimize weights to minimize KL divergence.
        """
        objective_function = lambda weights: self.kl_divergence(weights, 
                                                                true_distribution, 
                                                                simulated_distribution)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # Bounds: Weights should be between 0 and 1
        bounds = [(0, 1) for _ in range(len(initial_weights))]

        # Optimization
        result = minimize(objective_function, initial_weights, method='SLSQP', 
                          constraints=constraints, bounds=bounds)

        if result.success:
            optimized_weights = result.x
            return optimized_weights
        else:
            raise ValueError("Optimization failed.")

    def odld_1d_potential(self, A=2, B=5, C=0.5, x0=1):
        x = np.arange(0.1, 10.1, 0.1) 
        twopi_by_A = 2 * np.pi / A
        half_B = B / 2

        xarg = twopi_by_A * (x - x0)

        eCx = np.exp(C * x)
        eCx_less_one = eCx - 1.0

        potential = -half_B / eCx_less_one * np.cos(xarg)

        # normalize the plot to have lowest value as baseline
        potential -= np.min(potential)

        plt.plot(x, potential, color='k', alpha=0.5, label='ODLD potential', linestyle="--")
        return potential

if __name__ == "__main__":
    # test data (1D array of 1D ODLD endpoints)
    #pcoords = np.loadtxt('pcoords.txt')
    #weights = np.loadtxt('weights.txt')
    #pcoords = np.loadtxt('3000i_pcoord.txt')
    # for this method I also will be better off using the entire pcoord data
    pcoords = np.loadtxt('3000i_pcoord_full.txt')
    weights = np.loadtxt('3000i_weight.txt')

    import matplotlib.pyplot as plt
    #plt.hist(pcoords, bins=50)
    print(pcoords.reshape(-1))
    hist, bin_edges = np.histogram(pcoords.reshape(-1), bins=100)
    # TODO: NEXT, make the pdist, use this to compare to true_dist

    #true_dist = np.loadtxt("true_1d_odld.txt")
    #plt.plot(true_dist[:,0], true_dist[:,1])

    plt.show()

    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    #pcoords = np.load('pcoords_full.npy')

    # test init data
    # pcoords = np.array([9.5] * 50).reshape(-1,1)
    # weights = np.array([0.02] * 50)

    # WEER test
    #reweight = WEER()

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
    # true_distribution = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # simulated_distribution = np.array([1.2, 2.2, 2.8, 4.5, 5.5, 6.0, 5.0])

    # kl_divergence_value = kl_divergence(true_distribution, simulated_distribution)
    # print("KL Divergence:", kl_divergence_value)