#%%
import pickle, os
import networkx as nx
import pandas as pd
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
#%%

with open(os.path.join("data", "processed_data", "raw_nwx_graphs.pkl"), "rb") as infile:
    network_graphs = pickle.load(infile)

#%%

class Investigation(nx.Graph):
    def __init__(self, crime_network, p, id):
        super(Investigation, self).__init__()

        self.crime_network = crime_network
        self.current_investigation = None
        self.p = p
        self.id = id
        self.investigations = 0
        self.get_catch_prob = None
        self.strategy = None

    def set_model(self, func):
        '''
        Set underlying model of the chances of catching criminals. Return function that calculates probability of being caught given a node.
        func needs to accept a graph and return an array of probabilities for each node.
        '''
        self.get_catch_prob = func

    def set_strategy(self, strategy):
        '''
        Sets investigation strategy. strategy is a func that takes a graph that indicates suspects and caught criminals and returns the next suspect to investigate, with catching probability
        suspect should be node index
        '''
        self.strategy = strategy

    def investigate(self):
        suspect, p = self.strategy(self.current_investigation)
        if np.random.uniform() < p:
            self.crime_network.nodes[suspect]["caught"] = True
            for incident in self.crime_network.edges(suspect):
                self.crime_network.edges[incident]["informed"] = True
        self.investigations += 1

    def update_investigation(self):
        '''update graph view using node filter and edge filter'''
        pass