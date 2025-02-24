"""
Make a simple resampling tree to visualize the resampling algorithm.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import extract

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

def plot_resampling_tree(node_names, weights, metrics, edges, rows=2, cols=3):
    """
    Plot a resampling tree with nodes, edges, weights, and metric values.
    """
    # Create a multi-directed graph (for multiple edges between nodes)
    # needed for splitting into multiple trajectories
    G = nx.MultiDiGraph()

    # Add nodes with weight and metric as attributes
    for i, node in enumerate(node_names):
        G.add_node(node, weight=weights[i], metric=metrics[i])

    # Add edges based on history
    for edge in edges:
        if isinstance(edge[0], tuple):
            for sub_edge in edge:
                G.add_edge(sub_edge[0], sub_edge[1])
        else:
            G.add_edge(edge[0], edge[1])

    # Extract metric values for color mapping
    #metric_vals = np.array(metrics)
    metric_vals = np.array(metrics)

    # Normalize metric values to range [0, 1]
    norm = mcolors.Normalize(vmin=metric_vals.min(), vmax=metric_vals.max())

    # Choose a colormap (e.g., 'viridis')
    cmap = plt.cm.viridis

    # Create a list of colors for the nodes
    # TODO: code like this has been slow in the past, might need to update
    node_colors = [cmap(norm(metrics[i])) for i in range(len(node_names))]

    # Generate grid positions (e.g., 2 rows, 3 columns)
    grid_positions = generate_grid_positions(node_names, rows, cols)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the graph with node sizes proportional to the weights
    nx.draw(G, pos=grid_positions, ax=ax, with_labels=True, node_size=[G.nodes[node]['weight'] * 100 for node in node_names],
            node_color=node_colors, font_size=10, font_weight='bold', edge_color='gray')

    # Add a color bar using the ScalarMappable object
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # no data for color bar
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('ABSURDer Weight')

    # Show the plot
    plt.title("Trajectory History with Weighted Ensemble Resampling")
    plt.show()


if __name__ == '__main__':
    # Example node attributes (these can be dynamically generated or read from a file)
    # TODO: eventually use pre-cast arrays here: str, float, float, obj
    node_names = ['T1', 'T2', 'T3', 'T4', 'T5']
    weights = [2, 3, 5, 4, 1]
    metrics = [0.1, 0.5, 0.9, 0.7, 0.3]
    edges = [(('T1', 'T5'), ('T1', 'T4')), ('T2', 'T4'), ('T3', 'T5'), ('T4', 'T5'), ('T5', 'T5')]

    # TODO: okay, so I need data with node-name, WE weight, and absurder_weight, and edges/connections
    # TODO: eventually include a line plot sideways that shows phi_eff and chi2 for each iteration

    data = extract.extract_data_from_log(f"./west.log")
    we_weights = data["WE weights"]
    new_we_weights = data["New weights"]
    absurder_weights = data["ABSURDer weights"]
    pcoords = data["pcoords"]
    parent_ids = data["parent_ids"]
    chi2 = data["chi2"]
    phi_eff = data["phi_eff"]

    print(data)

    #plot_resampling_tree(node_names, weights, metrics, edges)