"""
Input west.log file and extract weight arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

def extract_data_from_log(filename):
    """
    Extract data from the west.log file.

    Parameters
    ----------
    filename : str
        Path to the west.log file.

    Returns
    -------
    dict
        Dictionary of extracted data:
        {"WE weights": np.array(we_weights),
         "New weights": np.array(new_we_weights),
         "ABSURDer weights": np.array(absurder_weights),
         "pcoords": np.array(pcoords),
         "parent_ids": np.array(parent_ids),
         "chi2": np.array(chi2),
         "phi_eff": np.array(phi_eff)
        }
    """
    def collect_multiline_data(start_token, collecting_flag, buffer, data_list):
        if line.startswith(start_token):
            collecting_flag = True
            buffer = []
            return collecting_flag, buffer
        if collecting_flag:
            buffer.append(line)
            if line.endswith("]"):
                data_string = " ".join(buffer).replace("[", "").replace("]", "")
                data_list.append(np.fromstring(data_string, sep=" "))
                collecting_flag = False
        return collecting_flag, buffer

    we_weights, new_we_weights, absurder_weights = [], [], []
    pcoords, parent_ids = [], []
    chi2, phi_eff = [], []

    collecting_we = collecting_new_we = collecting_absurder = False
    collecting_pcoords = collecting_parent_ids = False
    we_buffer = new_we_buffer = absurder_buffer = []
    pcoords_buffer = parent_ids_buffer = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            collecting_we, we_buffer = collect_multiline_data("WE weights:", collecting_we, we_buffer, we_weights)
            collecting_new_we, new_we_buffer = collect_multiline_data("New weights:", collecting_new_we, new_we_buffer, new_we_weights)
            collecting_absurder, absurder_buffer = collect_multiline_data("ABSURDer weights:", collecting_absurder, absurder_buffer, absurder_weights)
            collecting_pcoords, pcoords_buffer = collect_multiline_data("curr pcoords:", collecting_pcoords, pcoords_buffer, pcoords)
            collecting_parent_ids, parent_ids_buffer = collect_multiline_data("curr parent ids:", collecting_parent_ids, parent_ids_buffer, parent_ids)

            if line.startswith("# Overall chi square"):
                chi2.append(float(line.split(":")[1]))
            if line.startswith("ABSURDer phi_eff"):
                phi_eff.append(float(line.split(":")[1]))

    #return np.array(we_weights), np.array(new_we_weights), np.array(absurder_weights), np.array(chi2), np.array(phi_eff)
    # return a dictionary of the extracted data
    return {
        "WE weights": np.array(we_weights),
        "New weights": np.array(new_we_weights),
        "ABSURDer weights": np.array(absurder_weights),
        "pcoords": np.array(pcoords),
        "parent_ids": np.array(parent_ids),
        "chi2": np.array(chi2),
        "phi_eff": np.array(phi_eff)
    }

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

    ax[0,0].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    #filename = "test-data/west.log"
    # TODO: make comparison plot of the two conditions
    #       also include the Chi2 and Phi_eff values
    we_weight_input = True
    theta = 1000
    #filename = f"we_weight_input_{we_weight_input}_theta{theta}"
    filename = "."

    #we_weights, new_we_weights, absurder_weights, chi2, phi_eff = extract_data_from_log(f"{filename}/west.log")
    data = extract_data_from_log(f"{filename}/west.log")
    we_weights = data["WE weights"]
    new_we_weights = data["New weights"]
    absurder_weights = data["ABSURDer weights"]
    pcoords = data["pcoords"]
    parent_ids = data["parent_ids"]
    chi2 = data["chi2"]
    phi_eff = data["phi_eff"]
    print(data)
    #weights = np.array([we_weights, new_we_weights, new_we_weights-we_weights, absurder_weights])
    #print(f"WE Weights: {weights.shape}")
    print(f"\nCHI2:PHI_EFF {list(zip(chi2, phi_eff))}\n")
    # plot_weights(we_weights, absurder_weights)

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
    plot_weights(weights)
        

    # TODO: is there a more intuitive way to sort the walkers per iteration?
    plt.show()