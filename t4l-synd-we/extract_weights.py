"""
Input west.log file and extract weight arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

def extract_weights(filename):
    we_weights = []
    absurder_weights = []

    current_we = None
    current_absurder = None
    collecting_we = False
    collecting_absurder = False
    we_buffer = []
    absurder_buffer = []
    chi2 = []
    phi_eff = []

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
            
            # Collect chi2 and phi_eff
            if line.startswith("# Overall chi square"):
                chi2.append(float(line.split(":")[1]))
            if line.startswith("ABSURDer phi_eff"):
                phi_eff.append(float(line.split(":")[1]))

    return np.array(we_weights), np.array(absurder_weights), np.array(chi2), np.array(phi_eff)

def extract_weights_from_h5(filename):
    """
    Extract weights and weight changes from previous iteration
    from the west.h5 file.

    Parameters
    ----------
    filename : str
        Path to the west.h5 file.
    
    Returns
    -------
    """
    # weight diffs
    weight_diffs = []

    # Open the west.h5 file
    with h5py.File(filename, 'r') as f:

        # Extract the weight segments
        init_we_weights = f[f"iterations/iter_{1:08d}/seg_index"]["weight"]
        for it in range(1, f.attrs["west_current_iteration"]):

            # Extract the weight segments
            we_weights = f[f"iterations/iter_{it:08d}/seg_index"]["weight"]
            # Extract the parent segments
            parent_segs = f[f"iterations/iter_{it:08d}/seg_index"]["parent_id"]

            # Extract the weight changes from the previous iteration
            weight_diff = we_weights - init_we_weights[parent_segs]
            # update the initial weights for the next iteration
            init_we_weights = we_weights
            # Append the weight changes
            weight_diffs.append(weight_diff)
    
    return np.array(weight_diffs)

def plot_weights(we_weights, absurder_weights):
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


if __name__ == "__main__":
    # Example usage
    #filename = "test-data/west.log"
    filename = "we_weight_input_False/west.log"
    filename = "we_weight_input_True/west.log"
    we_weights, absurder_weights, chi2, phi_eff = extract_weights(filename)
    print(f"\nCHI2:PHI_EFF {list(zip(chi2, phi_eff))}\n")
    plot_weights(we_weights, absurder_weights)

    h5_filename = "we_weight_input_True/west.h5"
    # w_diffs = extract_weights_from_h5(h5_filename)
    # print(w_diffs.shape)