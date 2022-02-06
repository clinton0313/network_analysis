#%%

import networkx as nx
import numpy as np
import pandas as pd
import os, itertools
import regex as re
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%
def read_adjacency(filename, override=False):
    '''Reads CSV file and returns adjacency matrix
    Args:
        filename: CSV file from Takes a csv file from https://sites.google.com/site/ucinetsoftware/datasets/covert-networks network data or equivalent format.
        override: Boolean. If true then do not enforce matrix symmetry (for debugging)
    Returns:
        Adjacency matrix in form of pandas data frame. 
    '''
    df = pd.read_csv(filename)
    if [str(x) for x in df.iloc[:,0].to_list()] == list(df.columns)[1:]:
        df = df.drop(columns=df.columns[0])
    elif len(df.index) == len(df.columns) - 1:
        print(f"First column and columns didn't match for {filename}, dropping first column anyways")
        df = df.drop(columns=df.columns[0])
    df = df.fillna(0).set_axis(df.columns)
    if not override:
        assert len(df.index) == len(df.columns), f"Adjacency matrix for {filename} is not symmetric"
    return df

def add_adjacencies(dfs, network="", merge_type = "sum"):
    '''
    Add together a list of pandas dataframes representing adjacency matrices. If dataframes do not have consistent columns then will merge instead.
    Args:
        dfs: List of pandas dataframes that are symmetric adjacency matrices
        network: String. The name of the network (for debugging)
        merge_type: "sum" (default) or "intersect". Indicates the way matrices will be merged. Sum for outputting weighted graph and intersect for unweighted.
    Returns:
        Final adjacency matrix as a pandas dataframe
    '''
    colnames = [list(df.columns) for df in dfs]
    if all(colname == colnames[0] for colname in colnames):
        result = dfs[0].copy()
        for df in dfs[1:]:
            result = result.add(df, fill_value=0)
    else:
        print(f"Adjacency matrices are not consistent for {network}, merging adjacencies instead using {merge_type}")
        result = merge_adjs(dfs, merge_type=merge_type)
    return result

def merge_adjs(adjs, merge_type='sum'):
    '''Pass in symmetric adjacency matrices that have differeing columns but some intersecting columns'''
    colnames = []
    for adj in adjs:
        assert list(adj.index) == list(adj.columns), "Index and columns don't match"
        colnames = colnames + [col for col in adj.columns]
    colnames = list(set(colnames))

    df = pd.DataFrame(data = 0, columns = colnames, index = colnames)
    for adj in adjs:
        for (i, j) in itertools.permutations(adj.columns, 2):
            if merge_type == "sum":
                df.loc[i, j] += adj.loc[i, j]
            else:
                df.loc[i,j] = adj.loc[i,j].copy()
    return df

def load_nx_graph(df, graph_name=""):
    '''
    Loads a pandas dataframe adjacency matrix to networkx graph object. Main function is to send colnames to an node attribute ("name") and use index numbers for the nodes
    Args:
        df: Pandas dataframe of adjacency matrix
        graph_name: Name of the graph
    Returns:
        Network X graph with graph["name"] attribute and nodes["name"] attributes. 
    '''
    node_names = {i:name for i, name in enumerate(df.columns)}
    df = df.rename(columns={name:i for i, name in enumerate(df.columns)})
    df = df.set_axis(df.columns)
    graph = nx.from_pandas_adjacency(df)
    nx.set_node_attributes(graph, node_names, name="name")
    graph.graph["name"] = graph_name
    return graph

def plot_nx_graph(graph, weighted = False, weight_multiplier = 2, remove_isolated = False, save=False, savepath = "", **plot_options):
    '''
    Plot network x graph helper function.
    Args:
        graph: Network X Graph Object
        weighted: Boolean. If true, plot edge widths based on edge weight.
        weight_multiplier: Adjust the width of the edges if using weighted plotting
        remove_isolated: Boolean. Removes isolated nodes from the graph before plotting
        save: Boolean. If true will save the graph using it's name.
        savepath: Dirpath to save the graph
        **plot_options: Extra kwargs to pass to network_x.draw function
    Returns:
        Matplotlib figure
    '''
    g = graph.copy()
    if remove_isolated:
        g.remove_nodes_from(list(nx.isolates(g)))
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title(g.graph["name"])
    if weighted:
        weights = np.array(list(nx.get_edge_attributes(g, "weight").values())) * weight_multiplier
        plot_options.update({"width":weights})
    nx.draw_kamada_kawai(g, ax=ax, **plot_options)
    if save:
        fig.savefig(os.path.join(savepath, f'{g.graph["name"]}.png'))
    return fig

def read_all_network_data(network_data_dir):
    '''
    Reads all network data from a directory and returns a list of the graphs. Assumes that within the directory, each individual network is contained within its own folder.
    Within each network folder contains only CSV files matching the ucinet format. Filters out files that match the regex .*attr.* (case insensitive).
    Args:
        network_data_dir: directory containing network data
    Returns:
        List of networkx graphs
    '''
    networks = os.listdir(network_data_dir)
    network_graphs = []
    for network in networks:
        files = list(filter(
        lambda x: not re.match(".*attr.*", x, re.IGNORECASE), 
        os.listdir(os.path.join(network_data_dir, network))
        ))
        adj_matrices = [read_adjacency(os.path.join(network_data_dir, network, file)) for file in files]
        adj = add_adjacencies(adj_matrices, network=network)
        graph = load_nx_graph(adj, graph_name=network)
        network_graphs.append(graph)
    return network_graphs



#%%

network_data_dir = "crime_network_data"
savepath = "figs"
os.makedirs("figs", exist_ok=True)
network_graphs = read_all_network_data(network_data_dir)
figs = [
    plot_nx_graph(graph, remove_isolated=True, save=True, savepath = savepath, node_size=50)
    for graph in network_graphs
]

# %%
