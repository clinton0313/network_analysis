#%%
# %load_ext autoreload
# %autoreload 2
import numpy as np
from investigation import Investigation
import pickle, os, matplotlib, networkx
from matplotlib import pyplot as plt
from time import sleep

#%%

def simple_model(graph):
    q = 0.9
    p_dict = {}
    for node in graph.nodes:
        if graph.nodes[node].get("suspected"):
            p_dict[node] = (q * len(graph.edges(node)))
    return p_dict

def simple_strategy(current_investigation, random_catch):
    suspected = list(current_investigation.nodes(data="catch_proba"))
    if len(suspected) == 0:
        return "random", None
    suspected.sort(reverse=True, key = lambda x: x[1])
    return suspected[0][0], suspected[0][1]


with open(os.path.join("data", "processed_data", "raw_nwx_graphs.pkl"), "rb") as infile:
    network_graphs = pickle.load(infile)

inv = Investigation(crime_network = network_graphs[0], random_catch = 0.1)
inv.set_model(simple_model)
inv.set_strategy(simple_strategy, random_catch = inv.random_catch)
inv.plot()

inv.simulate(100, 20, update_plot=True)
sleep(5)
inv.reset()
inv.plot()


#%%
