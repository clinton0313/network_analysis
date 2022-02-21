#%%
import numpy as np
from investigation import Investigation
import pickle, os, matplotlib, networkx
from matplotlib import pyplot as plt
from time import sleep
from random import choice
import networkx as nx


#%%

#SAMPLE MODEL AND SIMULATION
def simple_model(graph, q):
    p_dict = {}
    for node in graph.nodes:
        if graph.nodes[node].get("suspected"):
            p_dict[node] = (q * len(graph.edges(node)))
    return p_dict

def simple_strategy(graph:nx.Graph, weighted:bool=True):
    '''Of highest probability suspects, take a random one.'''
    suspects = [suspect for suspect in graph.nodes if graph.nodes[suspect].get("suspected")]
    information = [degree for _, degree in graph.degree(suspects, weight=lambda _: "weight" if weighted else None)]
    if len(suspects) == 0:
        return "random"
    candidate_suspects = [suspect for suspect, info in zip(suspects, information) if info == max(information)]
    return choice(candidate_suspects)


with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    network_graphs = pickle.load(infile)

inv = Investigation(crime_network = network_graphs[2], random_catch = 0.1)
inv.set_model(simple_model, q = 0.2)
inv.set_strategy(simple_strategy, random_catch = inv.random_catch)


#%%
inv.simulate(100, 100, update_plot=True, sleep_time = 1, label = "Simple Model with a simple strategy")
sleep(30)
inv.fig.clear()
inv.fig.close()
print(inv.log)

#Not sure why it just doesn't quietly close to be honest. Somehow should fix that. Also doesn't plot final plot. 

#%%
