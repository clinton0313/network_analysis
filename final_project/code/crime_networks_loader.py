#%%
import networkx as nx
import numpy as np
import pandas as pd
import os, itertools
import regex as re
import matplotlib.pyplot as plt
import pickle

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
#%%
#FUNCTIONS

def read_adjacency(filename, override=False) -> pd.DataFrame:
    '''Reads CSV file and returns adjacency matrix. Not we arbitrarily eliminate self loops here as well as fill na's with 0s. 
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
    np.fill_diagonal(df.values, 0)  #Here we fill the diagonal with zeroes to eliminate self-loops. 
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
    '''Pass in symmetric adjacency matrices that have differing columns but some intersecting columns
    Must pass in matching index and column names'''
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

def weighted_edgelist_to_adj(edgelist: pd.DataFrame) -> pd.DataFrame:
    '''
    Pass a weighted edgelist in the form of a dataframe with columns source, target, weight in that order and 
    convert it to an undirected adjacency matrix with no self loops.
    '''
    node_list = set(list(edgelist.iloc[:,0].to_list() + edgelist.iloc[:,1].to_list()))
    adj = pd.DataFrame(0, index=node_list, columns=node_list)
    for entry in range(len(edgelist)):
        adj.loc[edgelist.iloc[entry, 0], edgelist.iloc[entry, 1]] += edgelist.iloc[entry, 2]
        adj.loc[edgelist.iloc[entry, 1], edgelist.iloc[entry, 0]] += edgelist.iloc[entry, 2]
    np.fill_diagonal(adj.values, 0)
    return adj

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

def plot_nx_graph(graph, weighted = False, weight_multiplier = 1, largest_component = False, remove_isolated = False, save=False, savepath = "", **plot_options):
    '''
    Plot network x graph helper function.
    Args:
        graph: Network X Graph Object
        weighted: Boolean. If true, plot edge widths based on edge weight.
        weight_multiplier: Adjust the width of the edges if using weighted plotting
        largest_component: Boolean. If true only plots the largest component
        remove_isolated: Boolean. Removes isolated nodes from the graph before plotting
        save: Boolean. If true will save the graph using it's name.
        savepath: Dirpath to save the graph
        **plot_options: Extra kwargs to pass to network_x.draw function
    Returns:
        Matplotlib figure
    '''
    g = graph.copy()
    if largest_component:
        g = g.subgraph(max(nx.connected_components(g)))
    elif remove_isolated:
        g.remove_nodes_from(list(nx.isolates(g)))
    fig, ax = plt.subplots(figsize=(20, 20))#Should maybe include some sort of standardization of edge widths like min-max scale
    ax.set_axis_off()
    ax.set_title(g.graph["name"],fontsize=30)
    if weighted: 
        weights = np.array(list(nx.get_edge_attributes(g, "weight").values()))
        weights = (weights - weights.min()+1)/(weights.max()+1) * weight_multiplier
        plot_options.update({"width":weights})
    pos = nx.drawing.layout.spring_layout(g, k=0.5 / np.sqrt(len(g.nodes)))
    nx.draw(g, ax=ax, pos=pos, **plot_options)
    if save:
        fig.savefig(os.path.join(savepath, f'{g.graph["name"]}.png'), dpi =300)
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

#Set filepaths
ucinet_dirpath = os.path.join("data", "ucinet_crime_network_data")
other_crime_dirpath = os.path.join("data", "other_crime_network_data")
savepath = os.path.join("figs", "network_graphs")
os.makedirs(savepath, exist_ok=True)

#%%

#Load and parse all Ucinet graphs
network_graphs = read_all_network_data(ucinet_dirpath)

#Load Ndrangheta mafia data. CSV is in form of co-located meetings so specific parsing
#Load and save node names
ndra = pd.read_csv(os.path.join(other_crime_dirpath, "ndrangheta_mafia", "NDRANGHETAMAFIA_2M.csv"))
ndra_names = ndra.iloc[:,0].to_list()
ndra = ndra.drop(columns=ndra.columns[0], axis=1)

#Iterate through columns taking index values of those that appear in the meeting and appending to adjacency twice
ndra_adj = np.zeros((len(ndra), len(ndra)))
for col in ndra.columns:
    colocated = list(ndra[ndra[col]==1].index)
    for i, j in itertools.combinations(colocated, 2):
        ndra_adj[i, j] += 1
        ndra_adj[j, i] += 1
ndra = pd.DataFrame(ndra_adj, columns = ndra_names)

#Eliminate self-loops and add graph to list
np.fill_diagonal(ndra.values, 0)
ndra_graph = load_nx_graph(ndra, "ndrangheta_mafia")
network_graphs.append(ndra_graph)

#Weighted edgelist format has own way of being loaded. 
#Load montagna data, convert to adjacency, merge, and add graph to list
montagna_meetings = pd.read_csv(os.path.join(other_crime_dirpath, "montagna_data", "Montagna_Meetings_Edgelist.csv"), delimiter=" ")
montagna_phone = pd.read_csv(os.path.join(other_crime_dirpath, "montagna_data", "Montagna_Phone_Calls_Edgelist.csv"), delimiter=",")

meetings_adj = weighted_edgelist_to_adj(montagna_meetings)
phone_adj = weighted_edgelist_to_adj(montagna_phone)

montagna_adj = merge_adjs([meetings_adj, phone_adj])
montagna_graph = load_nx_graph(montagna_adj, "montagna")
network_graphs.append(montagna_graph)

#%%
#Save file. Filename left blank to prevent accidental overwriting.
filename = ""
with open(os.path.join("data", "processed_data", filename), "wb") as outfile:
    pickle.dump(network_graphs, outfile)

#%%
#Plot and save all figs
figs = [
    plot_nx_graph(graph, weighted=True, weight_multiplier = 3, 
        largest_component=True, remove_isolated=True, 
        save=True, savepath = savepath, 
        node_size=100)
    for graph in network_graphs
]
# %%