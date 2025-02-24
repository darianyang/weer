"""
Make a simple resampling tree to visualize the resampling algorithm.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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

def plot_resampling_tree(nodes, edges, weights, metric_values, rows=2, cols=3):
    """
    Plot a resampling tree with nodes, edges, weights, and metric values.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with weight and metric as attributes
    for node in nodes:
        G.add_node(node, weight=weights[node], metric=metric_values[node])

    # Add edges based on history
    G.add_edges_from(edges)

    # Extract metric values for color mapping
    metric_vals = np.array(list(metric_values.values()))

    # Normalize metric values to range [0, 1]
    norm = mcolors.Normalize(vmin=metric_vals.min(), vmax=metric_vals.max())

    # Choose a colormap (e.g., 'viridis')
    cmap = plt.cm.viridis

    # Create a list of colors for the nodes
    node_colors = [cmap(norm(metric_values[node])) for node in nodes]

    # Generate grid positions (e.g., 2 rows, 3 columns)
    grid_positions = generate_grid_positions(nodes, rows, cols)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the graph with node sizes proportional to the weights
    nx.draw(G, pos=grid_positions, ax=ax, with_labels=False, node_size=[G.nodes[node]['weight'] * 100 for node in nodes],
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
    # Example data: node weights and metrics
    nodes = ['T1', 'T2', 'T3', 'T4', 'T5']
    edges = [('T1', 'T5'), ('T2', 'T4'), ('T3', 'T5'), ('T4', 'T5')]

    # Precalculated metric for each trajectory (e.g., some property)
    metric_values = {'T1': 0.1, 'T2': 0.5, 'T3': 0.9, 'T4': 0.7, 'T5': 0.3}
    weights = {'T1': 2, 'T2': 3, 'T3': 5, 'T4': 4, 'T5': 1}

    # # Initialize an empty dictionary for nodes
    # nodes_data = {}

    # # Example node attributes (these can be dynamically generated or read from a file)
    # node_names = ['T1', 'T2', 'T3', 'T4', 'T5']
    # weights = [2, 3, 5, 4, 1]
    # metrics = [0.1, 0.5, 0.9, 0.7, 0.3]
    # edges = [['T5'], ['T4'], ['T5'], ['T5'], []]

    # # Populate the dictionary by iterating through the nodes data
    # for name, weight, metric, edge_list in zip(node_names, weights, metrics, edges):
    #     nodes_data[name] = {
    #         'weight': weight,
    #         'metric': metric,
    #         'edges': edge_list
    #     }

    # # Print the resulting dictionary
    # weights = {node: data['weight'] for node, data in nodes_data.items()}
    # print(weights)
    # print(nodes_data.keys())

    # TODO: okay, so I need to make a data structure that has node-name, edge/connection, WE weight, and absurder_weight

    # TODO: eventually include a line plot sideways that shows phi_eff and chi2 for each iteration

    plot_resampling_tree(nodes, edges, weights, metric_values)