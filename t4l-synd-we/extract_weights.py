"""
Input west.log file and extract weight arrays.
"""

import numpy as np
import matplotlib.pyplot as plt

def extract_weights(filename):
    we_weights = []
    absurder_weights = []

    current_we = None
    current_absurder = None
    collecting_we = False
    collecting_absurder = False
    we_buffer = []
    absurder_buffer = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Detect WE weights start
            if line.startswith("WE weights:"):
                collecting_we = True
                we_buffer = []
                continue

            # Detect ABSURDer weights start
            if line.startswith("ABSURDer weights:"):
                collecting_absurder = True
                absurder_buffer = []
                continue

            # Collect WE weights over multiple lines
            if collecting_we:
                we_buffer.append(line)
                if line.endswith("]"):
                    # Combine and parse once the full list is collected
                    weights_string = " ".join(we_buffer).replace("[", "").replace("]", "")
                    current_we = np.fromstring(weights_string, sep=" ")
                    we_weights.append(current_we)
                    collecting_we = False

            # Collect ABSURDer weights over multiple lines
            if collecting_absurder:
                absurder_buffer.append(line)
                if line.endswith("]"):
                    # Combine and parse once the full list is collected
                    weights_string = " ".join(absurder_buffer).replace("[", "").replace("]", "")
                    current_absurder = np.fromstring(weights_string, sep=" ")
                    absurder_weights.append(current_absurder)
                    collecting_absurder = False

    return np.array(we_weights), np.array(absurder_weights)

# Example usage
we_weights, absurder_weights = extract_weights("west.log")

# # Print the extracted weights for checking
# print("WE Weights:", we_weights.shape, we_weights)
# # for i, weights in enumerate(we_weights):
# #     print(f"Iteration {i+1}: {weights}")

# print("\nABSURDer Weights:", absurder_weights.shape, absurder_weights)
# # for i, weights in enumerate(absurder_weights):
# #     print(f"Iteration {i+1}: {weights}")

# Plot the weights
#fig, ax = plt.subplots(2, 1, figsize=(10, 6))
iterations = np.arange(1, we_weights.shape[0]+1)
segments = np.arange(0, we_weights.shape[1])

# # Plot WE weights
# for i, weights in enumerate(we_weights.T):
#     ax[0].scatter(iterations, weights, label=f"WE {i+1}")
# # Plot ABSURDer weights
# for i, weights in enumerate(absurder_weights.T):
#     ax[1].scatter(iterations, weights, label=f"ABSURDer {i+1}")

# for i, w in enumerate(we_weights):
#     ax[0].plot(segments, w, label=f"WE {i+1}")
# for i, w in enumerate(absurder_weights):
#     ax[1].plot(segments, w, label=f"ABSURDer {i+1}")

#ax[0].set_yscale("log")
#ax[1].set_yscale("log")

# TODO: plot the weight change from previous iteration instead of WE weights
#       will need to use west.h5 file: find the seg parent and take weight diff
# TODO: include N_eff in plots

fig, ax = plt.subplots(5, 2, figsize=(16, 10), sharex=True, sharey=True)

for i in range(10):
    row = i % 5
    col = i // 5
    ax[row, col].plot(segments, we_weights[i], label=f"WE")
    ax[row, col].plot(segments, absurder_weights[i], label=f"ABSURDer")
    
    # Scatter points for the top four ABSURDer weight segments
    top_indices = np.argsort(absurder_weights[i])[-8:]
    ax[row, col].scatter(top_indices, absurder_weights[i][top_indices], color='red', zorder=5)
    
    ax[row, col].set_yscale("log")
    ax[row, col].set_title(f"WE Iteration {i+1}")

ax[0,0].legend()
plt.tight_layout()
plt.show()