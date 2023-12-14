import numpy as np
from scipy.optimize import minimize

def kl_divergence(weights, true_distribution, simulated_distribution):
    """
    Calculate KL divergence between the true and simulated distributions.
    """
    weighted_simulated = np.sum(weights * simulated_distribution, axis=0)
    return np.sum(np.where(weighted_simulated != 0, true_distribution * np.log(true_distribution / weighted_simulated), 0))

def optimize_weights(true_distribution, initial_weights, simulated_distribution):
    """
    Optimize weights to minimize KL divergence.
    """
    objective_function = lambda weights: kl_divergence(weights, true_distribution, simulated_distribution)
    
    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: Weights should be between 0 and 1
    bounds = [(0, 1) for _ in range(len(initial_weights))]

    # Optimization
    result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)

    if result.success:
        optimized_weights = result.x
        return optimized_weights
    else:
        raise ValueError("Optimization failed.")

# Example usage:
true_distribution = np.array([0.2, 0.3, 0.5])  # Replace with your true distribution
initial_weights = np.array([0.1, 0.3, 0.6])  # Replace with your initial simulated weights
simulated_distribution = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])  # Replace with your simulated distribution

optimized_weights = optimize_weights(true_distribution, initial_weights, simulated_distribution)

print("Initial Weights:", initial_weights, np.sum(initial_weights))
print("Optimized Weights:", optimized_weights, np.sum(optimized_weights))
