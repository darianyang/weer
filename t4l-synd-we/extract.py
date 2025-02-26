"""
Input west.log file and extract weight arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

import numpy as np
import pickle
import re

plt.style.use("default.mplstyle")

def parse_parent_wtg_ids(s):
    """
    Parse a string like "[{5} {5} {0} {0} {1, 3} {2, 4}]".
    """
    # Remove surrounding square brackets
    s = s.strip()[1:-1]
    # Use regex to capture each {...} group
    groups = re.findall(r'\{([^}]*)\}', s)
    result = []
    for group in groups:
        # Split on commas (and possibly whitespace)
        numbers = [int(x.strip()) for x in group.split(',') if x.strip() != '']
        # append as a tuple
        #result.append(set(numbers)) # was using set for a single var but of course not subscriptable
        result.append(tuple(numbers))

    return result

# TODO: this is overall not the best, but it works for now
# eventually I can either save the data to h5 or parse everything from the h5 directly
def extract_data_from_log(filename, results='extracted_data.pkl'):
    """
    Extract data from the west.log file.

    Parameters
    ----------
    filename : str
        Path to the west.log file.
    results : str
        Path to save the extracted data as a pickle file.

    Returns
    -------
    dict
        Dictionary of extracted data:
        {"WE weights": np.array(we_weights),
         "New weights": np.array(new_we_weights),
         "ABSURDer weights": np.array(absurder_weights),
         "pcoords": np.array(pcoords),
         "parent_ids": np.array(parent_ids, dtype=int),
         "parent_wtg_ids": list of sets of tuples,
         "chi2": np.array(chi2),
         "phi_eff": np.array(phi_eff)
        }
    """
    # Define keywords and initialize storage dictionaries
    data_keys = {
        "WE weights:": "WE weights",
        "New weights:": "New weights",
        "ABSURDer weights:": "ABSURDer weights",
        "curr pcoords:": "pcoords",
        "curr parent ids:": "parent_ids",
        "curr parent wtg ids:": "parent_wtg_ids"
    }
    
    data_store = {key: [] for key in data_keys.values()}
    scalars = {"# Overall chi square": "chi2", "ABSURDer phi_eff": "phi_eff"}
    scalar_store = {key: [] for key in scalars.values()}
    
    collecting_key = None  # currently active data key
    buffer = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Check if the line signals the start of a data block
            if line in data_keys:
                # If we were collecting a previous block, process it.
                if collecting_key is not None:
                    block_data = " ".join(buffer)
                    if collecting_key == "parent_wtg_ids":
                        # Use the specialized parser for parent_wtg_ids
                        data_store[collecting_key].append(parse_parent_wtg_ids(block_data))
                    else:
                        # Remove any surrounding brackets and parse numeric data
                        cleaned = block_data.replace("[", "").replace("]", "")
                        data_store[collecting_key].append(np.fromstring(cleaned, sep=" "))
                # Set new collecting key and reset buffer
                collecting_key = data_keys[line]
                buffer = []
                continue

            # If we're in the middle of collecting multiline data, add to the buffer
            if collecting_key:
                buffer.append(line)
                if line.endswith("]"):
                    block_data = " ".join(buffer)
                    if collecting_key == "parent_wtg_ids":
                        data_store[collecting_key].append(parse_parent_wtg_ids(block_data))
                    else:
                        cleaned = block_data.replace("[", "").replace("]", "")
                        data_store[collecting_key].append(np.fromstring(cleaned, sep=" "))
                    collecting_key = None
                    buffer = []
                continue

            # Process scalar values in single lines
            for key, name in scalars.items():
                if line.startswith(key):
                    scalar_store[name].append(float(line.split(":")[1]))
                    break

    # Convert numerical lists to NumPy arrays
    for key in data_store:
        if key != "parent_wtg_ids":  # keep parent_wtg_ids as list of sets of tuples
            data_store[key] = np.array(data_store[key])
    for key in scalar_store:
        data_store[key] = np.array(scalar_store[key])

    # Save the extracted data as a pickle file
    with open(results, "wb") as f:
        pickle.dump(data_store, f)

    return data_store


def extract_weights_from_h5(filename, absurder_weights):
    """
    Extract weights and weight changes from previous iteration
    from the west.h5 file. Sort WE weights, WE weight diffs, and 
    ABSURDer weights together and consistently for comparison.

    Parameters
    ----------
    filename : str
        Path to the west.h5 file.
    absurder_weights : np.array
        Array of ABSURDer weights.
    
    Returns
    -------
    """
    # Open the west.h5 file
    with h5py.File(filename, 'r') as f:

        # Extract the weight segments
        #init_we_weights = f[f"iterations/iter_{1:08d}/seg_index"]["weight"]
        n_init_segs = f["summary"]["n_particles"][0]
        init_we_weights = np.ones(n_init_segs) / n_init_segs
        init_parent_weights = init_we_weights
        all_weights = np.zeros((4, f.attrs["west_current_iteration"]-1, n_init_segs))
        #print("Initial WE Weights:", init_we_weights.shape, init_we_weights)
        for it in range(1, f.attrs["west_current_iteration"]):

            # Extract the weight segments
            we_weights = f[f"iterations/iter_{it:08d}/seg_index"]["weight"]
            # Extract the parent segment ids and weights
            parent_segs = f[f"iterations/iter_{it:08d}/seg_index"]["parent_id"]
            if it == 1:
                parent_weights = init_parent_weights[parent_segs]
            else:
                parent_weights = f[f"iterations/iter_{it-1:08d}/seg_index"]["weight"][parent_segs]
            
            # Extract the weight changes from the previous iteration
            # TODO: make sure that this works for same weight walkers later when sorted
            weight_diff = we_weights - init_we_weights[parent_segs]

            # update the initial weights for the next iteration
            init_we_weights = we_weights
            init_parent_weights = parent_weights
            
            # Append the weight changes
            #all_weight_diffs.append(weight_diff)

            # Sort the weights and weight diffs together
            # smallest to largest weight indices (also how ABSURDer weights are sorted)
            sort_indices = np.argsort(we_weights)
            we_weights_sorted = we_weights[sort_indices]
            #parent_segs_sorted = parent_segs[sort_indices]
            weight_diff_sorted = weight_diff[sort_indices]
            # grab the corresponding ABSURDer weights
            absurder_weights_sorted = absurder_weights[it-1]
            # and the parent weights
            parent_weights_sorted = parent_weights[sort_indices]

            # Append the sorted values
            #all_weights.append((we_weights_sorted, weight_diff_sorted, absurder_weights_sorted, parent_weights_sorted))
            #all_weights.append((parent_weights_sorted, we_weights_sorted, weight_diff_sorted, absurder_weights_sorted))
            all_weights[0, it-1] = parent_weights_sorted
            all_weights[1, it-1] = we_weights_sorted
            all_weights[2, it-1] = weight_diff_sorted
            all_weights[3, it-1] = absurder_weights_sorted

    return all_weights

def plot_weights(weights):
    """
    Plot the extracted weights for comparison.

    Parameters
    ----------
    weights : np.array
        Array of weights: 
    """
    # # Print the extracted weights for checking
    # print("WE Weights:", we_weights.shape, we_weights)
    # # for i, weights in enumerate(we_weights):
    # #     print(f"Iteration {i+1}: {weights}")

    # print("\nABSURDer Weights:", absurder_weights.shape, absurder_weights)
    # # for i, weights in enumerate(absurder_weights):
    # #     print(f"Iteration {i+1}: {weights}")

    # extraction from h5 and log
    # we_weights = weights[:,0]
    # weight_diff = weights[:,1]
    # absurder_weights = weights[:,2]
    # parent_weights = weights[:,3]

    # extraction from log file
    we_weights = weights[0]
    new_we_weights = weights[1]
    weight_diff = weights[2]
    absurder_weights = weights[3]

    # Plot the weights
    #fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    n_iterations = we_weights.shape[0]
    iterations = np.arange(1, n_iterations + 1)
    n_segments = we_weights.shape[1]
    segments = np.arange(0, n_segments)

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

    for i in range(n_iterations):
        row = i % 5
        col = i // 5
        ax[row, col].plot(segments, we_weights[i], label=f"WE Original")
        #ax[row, col].plot(segments, weight_diff[i], label=f"WE Diff")
        ax[row, col].plot(segments, absurder_weights[i], label=f"ABSURDer")
        ax[row, col].plot(segments, new_we_weights[i], label=f"WE Updated", color='tab:blue', linestyle='--')
        
        # Scatter points for the top four ABSURDer weight segments
        top_indices = np.argsort(absurder_weights[i])[-(n_segments//3):]
        ax[row, col].scatter(top_indices, absurder_weights[i][top_indices], color='red', zorder=5)
        
        #ax[row, col].set_yscale("log")
        ax[row, col].set_title(f"WE Iteration {i+1}")
        #ax[row, col].hlines(0, 0, 24, color='gray', linestyle='--')
        ax[row, col].grid(True)
    # only put labels on the outer plots
    ax[4, 0].set_xlabel("Segment Index")
    ax[4, 1].set_xlabel("Segment Index")
    ax[0, 0].set_ylabel("Weight")
    ax[1, 0].set_ylabel("Weight")
    ax[2, 0].set_ylabel("Weight")
    ax[3, 0].set_ylabel("Weight")
    ax[4, 0].set_ylabel("Weight")

    ax[0,0].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    #filename = "test-data/west.log"
    # TODO: make comparison plot of the two conditions
    #       also include the Chi2 and Phi_eff values
    we_weight_input = False
    theta = 100
    filename = f"we_weight_input_{we_weight_input}_theta{theta}"
    #filename = "."

    #we_weights, new_we_weights, absurder_weights, chi2, phi_eff = extract_data_from_log(f"{filename}/west.log")
    data = extract_data_from_log(f"{filename}/west.log")
    we_weights = data["WE weights"]
    new_we_weights = data["New weights"]
    absurder_weights = data["ABSURDer weights"]
    pcoords = data["pcoords"]
    parent_ids = data["parent_ids"]
    parent_wtg_ids = data["parent_wtg_ids"]
    chi2 = data["chi2"]
    phi_eff = data["phi_eff"]
    print(data)
    #weights = np.array([we_weights, new_we_weights, new_we_weights-we_weights, absurder_weights])
    #print(f"WE Weights: {weights.shape}")
    print(f"\nCHI2:PHI_EFF {list(zip(chi2, phi_eff))}\n")
    # plot_weights(we_weights, absurder_weights)
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(chi2, linewidth=2)
    ax[0].set_ylabel("$\chi^2$")
    ax[1].plot(phi_eff, linewidth=2)
    ax[1].set_ylabel("$\phi_{eff}$")
    ax[1].set_xlabel("WE Iteration")

    # returns weights from h5
    weights = extract_weights_from_h5(f"{filename}/west.h5", absurder_weights)
    #print(f"WE Weights: {weights.shape}")
    # loop each iteration
    # TODO: make plotting function for weight diffs and absurder weights
    # for it in weights:
    #     we_weights, weight_diff, absurder_weights = it
    # plt.plot(weight_diff)
    # plt.plot(absurder_weights)
    #plot_weights(weights[:,3], weights[:,2])
    #plot_weights(weights)
        

    # TODO: is there a more intuitive way to sort the walkers per iteration?
    plt.tight_layout()
    plt.savefig("phi_chi2.pdf")
    plt.show()