#%%

from investigation import Investigation
import pandas as pd
import os, pickle, matplotlib
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from time import sleep

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

#%%
with open(os.path.join("data", "processed_data", "raw_nwx_graphs.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

inv = Investigation(graphs[8], random_catch = 0.05)

#%%
#HELPER FUNCTIONS

def get_suspects(graph):
    return [suspect for suspect in graph.nodes if graph.nodes[suspect].get("suspected")]

def get_suspect_proba(graph, suspects):
    return [graph.nodes[suspect]["catch_proba"] for suspect in suspects]

def get_information(graph, suspects, weighted = True):
    information = {}
    for suspect in suspects:
        incident = [graph[i][j]["weight"] for i, j in graph.edges(suspect) if graph[i][j]["informed"]]
        if weighted:    
            information[suspect] = sum(incident)
        elif not weighted:
            information[suspect] = len(incident)
    return information

#MODELS

def exponential_model(graph, random_catch, lr, weighted=True):
    '''Each informative link adds exponentially more information'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect : max(1, random_catch * (1+lr) ** info - random_catch)
        for suspect, info in zip(suspects, information)}
    return suspect_proba

def constant_model(graph, c, weighted=True):
    '''Each informative link adds a constant amount of information'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect : max(1, c * info) for suspect, info in zip(suspects, information)}
    return suspect_proba

def decay_model(graph, random_catch, decay, weighted=True):
    pass

#STRATEGIES

def simple_greedy(graph):
    '''Of highest probability suspects, take a random one.'''
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)
    if len(suspects) == 0:
        return "random", None
    max_suspect = [index for index, proba in enumerate(suspect_proba) if proba == max(suspect_proba)]
    random_max = max_suspect[np.random.randint(len(max_suspect))]
    return suspects[random_max], suspect_proba[random_max]

def least_central(graph):
    '''Of lowest degree suspects, take a random one.'''
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)
    information = get_information(graph, suspects, weighted=True)
    min_central = [index for index, info in enumerate(information) if info == min(information)]
    random_min = min_central[np.random.randint(len(min_central))]
    return suspects[random_min], suspect_proba[random_min]

#%%

inv = Investigation(graphs[8])
inv.set_model(constant_model, c=0.05, weighted=True)
inv.set_strategy(least_central)

inv.simulate(10, 10, update_plot=True, label="Constant Model with Least Central Strategy")

inv.set_model(exponential_model, random_catch = inv.random_catch, lr=0.3)
inv.set_strategy(simple_greedy)
inv.reset(keep_fig=True)
inv.simulate(10, 10, update_plot=True, label="Exponential model with Simple Greedy Strategy")