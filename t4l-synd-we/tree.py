"""
Make a simple resampling tree to visualize the resampling algorithm.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pickle
import os

import extract

plt.style.use("default.mplstyle")

# Function to generate grid positions
# TODO: rows is not needed I suppose
def generate_grid_positions(nodes, rows, cols):
    # Calculate the number of nodes in the grid
    positions = {}
    for idx, node in enumerate(nodes):
        row = idx // cols
        col = idx % cols
        positions[node] = (col, row)  # Positive row to go up from bottom
    return positions

# TODO: auto detection of row/cols and node_size calc
def plot_resampling_tree(node_names, weights, metrics, edges, rows=2, cols=3, 
                         node_size=500, node_labels=False, cbar_label='pcoord',
                         plot_title='WE Resampling Tree', savefig=None):
    """
    Plot a resampling tree with nodes, edges, weights, and metric values.
    """
    # Create a multi-directed graph (for multiple edges between nodes)
    # needed for splitting into multiple trajectories
    G = nx.MultiDiGraph()

    # Add nodes with weight and metric as attributes
    for i, node in enumerate(node_names):
        G.add_node(node, weight=weights[i], metric=metrics[i])

    # Add edges between nodes
    for edge in edges:
        #print("EDGEINIT", edge)
        # case with empty tuple (no edges), continue to next node
        if edge == [()]:
            continue
        # go through all edges in the edge list
        else:
            for sub_edge in edge:
                G.add_edge(sub_edge[0], sub_edge[1])

    # convert metrics to array if needed
    if isinstance(metrics, list):
        metrics = np.array(metrics)

    # Normalize metric values to range [0, 1]
    norm = mcolors.Normalize(vmin=metrics.min(), vmax=metrics.max())

    # Choose a colormap (e.g., 'viridis')
    cmap = plt.cm.viridis

    # Create a list of colors for the nodes
    # TODO: code like this has been slow in the past, might need to update
    node_colors = [cmap(norm(metrics[i])) for i in range(len(node_names))]

    # Generate grid positions (e.g., 2 rows, 3 columns)
    grid_positions = generate_grid_positions(node_names, rows, cols)
    #print(grid_positions)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the graph with node sizes proportional to the weights
    #print(G.nodes, G.edges)
    nx.draw(G, pos=grid_positions, ax=ax, with_labels=node_labels, 
            node_size=[G.nodes[node]['weight'] * node_size for node in node_names],
            node_color=node_colors, font_size=10, font_weight='regular', edge_color='gray')

    # Add a color bar using the ScalarMappable object
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # no data needed for color bar
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)

    # Show the plot
    plt.title(plot_title)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def w_tree(logfile="west.log", data="extracted_data.pkl", savefig=None):
    """
    Extract data from from west.log file, do data processing and formatting,
    then plot a resampling tree.
    """
    # TODO: eventually include a line plot sideways that shows phi_eff and chi2 for each iteration

    # TODO: add to args
    # extract data from log file (and save to pickle file)

    # # check if the datafile already exists
    # if not os.path.exists(data):
    #     # if it doesn't, extract the data from the log file
    #     data = extract.extract_data_from_log(logfile)
    # # otherwise, load the data from the pickle file
    # else:
    #     with open(data, "rb") as f:
    #         data = pickle.load(f)
    data = extract.extract_data_from_log(logfile)

    we_weights = data["WE weights"]
    new_we_weights = data["New weights"]
    absurder_weights = data["ABSURDer weights"]
    pcoords = data["pcoords"]
    parent_ids = data["parent_ids"]
    parent_wtg_ids = data["parent_wtg_ids"]
    chi2 = data["chi2"]
    phi_eff = data["phi_eff"]

    #print(parent_wtg_ids)
    #print("\nIDXED", parent_wtg_ids[1][5][0])

    # grab n_iters and n_segments per iter
    n_iterations, n_segments = we_weights.shape
    # TODO: last iteration specification?
    n_iterations //= 2

    # Create an array of node numbers and convert to strings using vectorized concatenation operator
    node_names = np.core.defchararray.add('n', 
                    np.arange(n_iterations * n_segments).astype(str)).reshape(n_iterations, n_segments)
    #print(node_names.shape)

    # create edge tuples for connections between nodes
    edges = []
    # loop through each iteration
    for iter_i in range(n_iterations):
        if iter_i == 0:
            # empty segment parent connections in inital iteration
            edges.append([[()] for _ in range(n_segments)])
        # otherwise, connect each segment to the parent segment
        else:
            # loop through each trajectory segment
            # make an edge connection tuple to connect from parent node to current node
            #print([(node_names[iter_i-1, parent_ids[iter_i, seg_i]]) for seg_i in range(n_segments)])
            #print(node_names[iter_i-1, parent_ids[iter_i, 0])
            # edges.append([(node_names[iter_i-1, parent], node_names[iter_i, seg_i])
            #               for seg_i in range(n_segments) 
            #               for parent in parent_wtg_ids[iter_i][seg_i]])
            seg_edges = []
            for seg_i in range(n_segments):
                multi_seg_edges = []
                for parent in parent_wtg_ids[iter_i][seg_i]:
                    multi_seg_edges.append((node_names[iter_i-1, parent], node_names[iter_i, seg_i]))
                seg_edges.append(multi_seg_edges)
            edges.append(seg_edges)

            # previous code without using parent_wtg_ids (single connections)
            # edges.append([(node_names[iter_i-1, int(parent_ids[iter_i, seg_i])], 
            #                node_names[iter_i, seg_i]) 
            #                for seg_i in range(n_segments)])

    # convert to numpy array
    #print(edges)
    edges = np.array(edges, dtype=object)
    #print(edges)
    # plot the resampling tree
    plot_resampling_tree(node_names.flatten(), we_weights.flatten(), absurder_weights.flatten(), edges.flatten(), 
                         rows=n_iterations, cols=n_segments, node_size=1000, cbar_label='ABSURDer Weight', savefig=savefig)

if __name__ == '__main__':
    # Example node attributes (these can be dynamically generated or read from a file)
    # TODO: eventually use pre-cast arrays here: str, float, float, obj
    # TODO: could use this as a test case eventually
    # node_names = ['T1', 'T2', 'T3', 'T4', 'T5']
    # weights = [2, 3, 5, 4, 1]
    # metrics = [0.1, 0.5, 0.9, 0.7, 0.3]
    # edges = [(('T1', 'T5'), ('T1', 'T4')), ('T2', 'T4'), (), ('T4', 'T5'), ('T5', 'T5')]
    #plot_resampling_tree(node_names, weights, metrics, edges)

    # plot a resampling tree from west.log data
    #w_tree(logfile="we_weight_input_True_theta1000/west.log", savefig="resampling_tree_we_input.pdf")
    #w_tree(logfile="we_weight_input_False_theta100/west.log")
    w_tree(logfile="we_weight_input_False_theta100/west.log", savefig="resampling_tree_theta100.pdf")