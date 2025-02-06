
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.special import rel_entr, kl_div
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

class WEER:
    '''
    Weighted ensemble with ensemble reweighting.
    '''
    def __init__(self, pcoords, weights, true_dist, bins=100):
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
        bins : int
        TODO: add a way to only run reweighting every n iterations
              and it would be useful to take data from n iteration back
        '''
        # TODO: update this to be comparible with entire pcoord array data for iter
        #       ndim should always be 2 then or more with more pcoords, error with 1D
        self.pcoords = pcoords
        # # pcoord dimension 0 should be the same metric as true_dist
        # if pcoords.ndim == 1:
        #     self.pcoords = pcoords
        # # for multiple pcoords, just use the first for comparison to true_dist
        # elif pcoords.ndim >= 2:
        #     # TODO: check the shape to make sure this indexing works
        #     self.pcoords = pcoords[:,0]

        self.weights = weights
        self.true_dist = true_dist
        # TODO: update this throughout the methods
        self.bins = bins
        # precalculate constant histrange
        # TODO: should this be for the pcoords or true dist?
        #self.histrange = (np.min(pcoords), np.max(pcoords))
        self.histrange = (np.min(true_dist[:,0]), np.max(true_dist[:,0]))

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

    def align_support(self, true_distribution, simulated_distribution):
        """
        Align the support of two distributions by adding zero probabilities where needed.
        """
        unique_values = np.union1d(true_distribution, simulated_distribution)
        true_distribution_aligned = np.zeros_like(unique_values, dtype=float)
        simulated_distribution_aligned = np.zeros_like(unique_values, dtype=float)

        for i, value in enumerate(unique_values):
            if value in true_distribution:
                true_distribution_aligned[i] = true_distribution[true_distribution == value].sum()
            if value in simulated_distribution:
                simulated_distribution_aligned[i] = simulated_distribution[simulated_distribution == value].sum()

        return true_distribution_aligned, simulated_distribution_aligned

    def kl_divergence(self, weights, true_distribution, simulated_distribution, 
                      bins=100, epsilon=1e-10):
        """
        Calculate KL divergence between two distributions using binning.
        # TODO: binning is best here? how to account for alternate array lengths?

        Parameters
        ----------
        true_distribution : array
        simulated_distribution : array
        bins : int
        epsilon : float
            Small constant to avoid zero division error
        """
        # only need to weight simulated data from WE
        # TODO: wait, I already weighted it during pdist generation
        # true_hist = self.bin_data(true_distribution, bins)
        # #simulated_hist = self.bin_data(simulated_distribution, bins, weights)
        # simulated_hist = self.bin_data(simulated_distribution, bins)

        ## Add a small constant epsilon to avoid division by zero
        #true_hist += epsilon
        #simulated_hist += epsilon

        # testing out the alignment instead
        # true_distribution, simulated_distribution = \
        #     self.align_support(true_distribution, simulated_distribution)

        # TODO: TEST of just using the pdist values from sim compared to true

        # make simulated dist pdist
        x, simulated_dist = self.make_pdist(simulated_distribution, weights)

        # TODO: temp fix here for shape mismatch, need better solution
        #       need same shape for KLD calc
        true_dist = true_distribution[:,1]
        simulated_dist = simulated_dist[:-4]

        # Ensure both distributions sum to 1 (normalized for KLD)
        true_dist = true_dist / np.sum(true_dist)
        simulated_dist = simulated_dist / np.sum(simulated_dist)

        return entropy(true_dist, simulated_dist)

    def optimize_weights(self, initial_weights, true_distribution, simulated_distribution):
        """
        Optimize weights to minimize KL divergence.
        """
        # # constant initial weights guess
        # initial_weights = np.ones(initial_weights.shape)
        # initial_weights /= np.sum(initial_weights)
        # #print(initial_weights)

        objective_function = lambda weights: self.kl_divergence(weights, 
                                                                true_distribution, 
                                                                simulated_distribution)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # Bounds: Weights should be between 0 and 1
        bounds = [(0, 1) for _ in range(len(initial_weights))]

        # Optimization
        # TODO: when dists are already close, using a maxiter of 30 seems to work better
        #       otherwise it over optimizes, which may be remedied with more data
        result = minimize(objective_function, initial_weights, method='SLSQP',
                          constraints=constraints, bounds=bounds, options={'maxiter':30})

        print(result)

        # if result.success:
        #     optimized_weights = result.x
        #     return optimized_weights
        # else:
        #     raise ValueError("Optimization failed.")

        optimized_weights = result.x
        return optimized_weights

    def smooth_distribution(self, original_distribution, epsilon=1e-3):
        """
        Smooths the given probability distribution by adding 
        small probabilities (epsilon) to missing symbols.

        Parameters
        ----------
        original_distribution : array 
            The original probability distribution.
        epsilon : float
            Small constant value.

        Returns
        -------
        smoothed_distribution : array 
            The smoothed probability distribution.
        """
        # Extract symbols from the original distribution
        symbols = np.arange(original_distribution.shape[0])

        # Create a set of all possible symbols
        all_symbols = np.arange(max(original_distribution.shape[0], self.true_dist.shape[0]))

        # Calculate missing symbols
        missing_symbols = np.setdiff1d(all_symbols, symbols)

        # Smooth the distribution by adding epsilon to missing symbols
        smoothed_distribution = np.copy(original_distribution)
        smoothed_distribution[missing_symbols] += epsilon
        # also add epsilon to all zero values
        smoothed_distribution[smoothed_distribution == 0] += epsilon

        # Normalize the distribution to ensure the probabilities sum to 1
        total_probability = np.sum(smoothed_distribution)
        smoothed_distribution /= total_probability

        return smoothed_distribution

    def make_pdist(self, pcoord, weights, bins=100):
        '''
        Make a 1D pdist in units of kT.

        Parameters
        ----------
        pcoord : array
        weights : array
        bins : int
        '''
        # make an 1D array to fit the hist values based off of bin count
        histogram = np.zeros((bins))

        # # precalculate constant histrange
        # histrange = (np.min(pcoord), np.max(pcoord))

        # loop each segment in the current iteration
        for seg in range(0, pcoord.shape[0]):
            counts, bins = np.histogram(pcoord[seg], bins=bins,
                                        range=self.histrange)

            # multiply counts vector by weight scalar from weight array
            counts = np.multiply(counts, weights[seg])

            # add all of the weighted walkers to total array for the 
            # resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins[:-1] + bins[1:]) / 2

        # set the 0 count bins of lowest prob to lowest non-zero count value
        # TODO: there def could be a better way to account for this
        histogram[histogram == 0] = np.min(histogram[histogram != 0])
        # trying data smoothing with epsilon from KLD PDF
        #epsilon = 10**-3
        #histogram[histogram == 0] = epsilon
        #print(histogram)
        #histogram = self.smooth_distribution(histogram)
        #print(histogram)
        
        # normalize hist to kT
        histogram = -np.log(histogram / np.max(histogram))

        # TODO: sometimes after norm to kT there is a zero val still
        histogram[histogram == 0] = np.min(histogram[histogram != 0])

        return midpoints_x, histogram

    def run_weer(self):
        '''
        Main public class method for WEER.
        '''

        # TODO: make sure new and old weights sum to 1
        #       also, ensure no zero values in new weights

        # calc KL divergence from true dist
        #kld = self.kl_divergence(self.weights, self.true_dist[:,1], simulated_dist)
        #print(simulated_dist)
        #plt.plot(x, simulated_dist)
        #plt.plot(self.plot_kde(self.pcoords))
        #plt.show()

        # test plot to compare to post weight opt
        # TODO: temp fix here for shape mismatch, need better solution
        #       need same shape for KLD calc

        # make simulated dist pdist
        # test_x, test_simulated_dist = self.make_pdist(self.pcoords, self.weights)
        # test_true_dist = self.true_dist[:,1]
        # test_simulated_dist = test_simulated_dist[:-4]
        # test_x = test_x[:-4]
        # plt.plot(test_x, test_true_dist)
        # plt.plot(test_x, test_simulated_dist)

        # # Ensure both distributions sum to 1 (normalized for KLD)
        # true_dist = true_dist / np.sum(true_dist)
        # simulated_dist = simulated_dist / np.sum(simulated_dist)

        # plt.plot(x, true_dist)
        # plt.plot(x, simulated_dist)

        #kld = entropy(self.true_dist[:,1], simulated_dist[4:])
        kld = self.kl_divergence(self.weights, self.true_dist, self.pcoords)
        print("Initial KL div: ", kld)

        # TODO: OPT test
        opt_weights = self.optimize_weights(self.weights, self.true_dist, self.pcoords)

        # new pdist with updated weights
        #x, new_sim_dist = self.make_pdist(self.pcoords, opt_weights)
        # plt.plot(x, new_sim_dist)
        # plt.show()

        # KDE test
        # sim_x, sim_y = self.plot_kde(simulated_dist)
        # true_x, true_y = self.plot_kde(self.true_dist[:,1])
        
        # print(np.sum(sim_y), np.sum(true_y))

        # # Ensure both distributions sum to 1 (normalized for KLD)
        # true_y = sim_y / np.sum(sim_y)
        # sim_y = sim_y / np.sum(sim_y)

        # print(np.sum(sim_y), np.sum(true_y))

        # plt.plot(true_x, true_y)
        # plt.plot(sim_x, sim_y)
        # plt.show()

        # ensure no zero values in new weights
        opt_weights[opt_weights == 0] = np.min(opt_weights[opt_weights != 0])
        # make sure new weights sum to 1
        opt_weights /= np.sum(opt_weights)

        return opt_weights

    def plot_kde(self, data):
        """
        Calc the KDE of the given data.
        TODO: replace the bins with KDE?
        """
        # set the data shape to be 1D
        data = data.reshape(-1)

        # You can adjust the bandwidth parameter
        kde = KernelDensity(bandwidth=0.6)
        kde.fit(data[:, np.newaxis])
        
        # TODO: update to bin midpoint x values?
        #       need to think about the theory for is this makes sense
        x_vals = np.linspace(self.histrange[0], self.histrange[1], self.bins)
        log_dens = kde.score_samples(x_vals[:, np.newaxis])
        dens = np.exp(log_dens)
        
        return x_vals, dens
    
    def plot_dist(self, pcoords, weights):
        # make simulated dist pdist
        test_x, test_simulated_dist = self.make_pdist(pcoords, weights)
        test_simulated_dist = test_simulated_dist[:-4]
        test_x = test_x[:-4]
        plt.plot(test_x, test_simulated_dist)

    def plot_true(self):
        plt.plot(self.true_dist[:,0], self.true_dist[:,1])


if __name__ == "__main__":
    # test data (1D array of 1D ODLD endpoints)
    #pcoords = np.loadtxt('pcoords.txt')
    #weights = np.loadtxt('weights.txt')
    #pcoords = np.loadtxt('3000i_pcoord.txt')
    # for this method I also will be better off using the entire pcoord data
    # TODO: even more data will prob be better, having e.g. n iters of pcoords
    #       guidelines could be enough to make a properly filled out pdist
    pcoords = np.loadtxt('3000i_pcoord_full.txt')
    weights = np.loadtxt('3000i_weight.txt')
    # pcoords = np.loadtxt('30i_pcoord_full.txt')
    # weights = np.loadtxt('30i_weight.txt')

    #plt.hist(pcoords, bins=50)
    #print(pcoords.reshape(-1))
    #hist, bin_edges = np.histogram(pcoords.reshape(-1), bins=100)
    # TODO: NEXT, make the pdist, use this to compare to true_dist
    # x, hist = make_pdist(pcoords, weights) # turn into static method?
    # plt.plot(x, hist)
    # plt.show()

    # TODO: note that I'm comparing the ODLD potential which
    #       is already in same units as kT from WE
    #       but for future, e.g. EPR distance, I would need
    #       to probably normalize dists to 1, but KL div already
    #       needs this done anyway... so should be fine
    true_dist = np.loadtxt("true_1d_odld.txt")
    #plt.plot(true_dist[:,0], true_dist[:,1])
    #plt.show()

    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    # TODO: be able to also handle multi-dim pcoord, but only compare dim 0
    #pcoords = np.load('pcoords_full.npy')

    # test init data
    # pcoords = np.array([9.5] * 50).reshape(-1,1)
    # weights = np.array([0.02] * 50)

    # WEER test
    reweight = WEER(pcoords, weights, true_dist)
    opt_weights = reweight.run_weer()
    reweight.plot_true()
    reweight.plot_dist(reweight.pcoords, reweight.weights)
    reweight.plot_dist(reweight.pcoords, opt_weights)

    # TODO: epsilon to smooth data and get same points (see KLD PDF)

    plt.show()

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