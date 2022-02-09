#%%
import pickle, os
import networkx as nx
import pandas as pd
import numpy as np
from functools import partial

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
#%%

with open(os.path.join("data", "processed_data", "raw_nwx_graphs.pkl"), "rb") as infile:
    network_graphs = pickle.load(infile)

#%%

class Investigation(nx.Graph):
    def __init__(self, crime_network:nx.Graph, first_criminal:int, random_catch:float, id = "name"):
        super(Investigation, self).__init__()

        self.crime_network = crime_network
        self.id = id
        nx.set_node_attributes(self.crime_network, False, "suspected")
        nx.set_node_attributes(self.crime_network, False, "caught")
        nx.set_edge_attributes(self.crime_network, False, "informed")

        self.investigations = 0
        self.current_investigation = None
        self._caught_suspect(first_criminal)
        self._update_investigation()

        self.random_catch = random_catch
        self.model_proba = None
        self.strategy = None

    def set_model(self, model):
        '''
        Set underlying model of the chances of catching criminals. Return function that calculates probability of being caught given a node.
        func needs to accept a graph and return an array of probabilities for each node.
        '''
        def node_filter(x):
            return self.current_investigation[x].get("suspected", False)
        def edge_filter(i, j):
            linked_to_suspect = self.current_investigation[i].get("suspected", False) or self.current_investigation[j].get("suspected", False)
            return linked_to_suspect
        investigation_suspects = nx.graphviews.subgraph_view(self.current_investigation, node_filter, edge_filter)

        model_proba = partial(model, investigation_suspects)

        self.model_proba = model_proba

    def set_strategy(self, strategy):
        '''
        Sets investigation strategy. strategy is a func that takes a graph that indicates suspects and caught criminals and returns the next suspect to investigate, with catching probability
        suspect should be node index, or return catch random?
        '''
        self.strategy =  partial(strategy, self.current_investigation, self.random_catch)

    def _catch_random(self):
        '''
        Catch a random criminal out there
        '''
        if np.random.uniform() < self.random_catch:
            unsuspected = [node for node, suspected in list(self.crime_network.nodes(data="suspected")) if not suspected]
            caught = unsuspected[np.random.randint(len(unsuspected))]
            self._caught_suspect(caught)
        pass

    def _caught_suspect(self, suspect):
        '''
        Update graph properties when suspect is caught
        '''
        self.crime_network.nodes[suspect]["caught"] = True
        self.crime_network.nodes[suspect]["suspected"] = False
        for i, j in list(self.crime_network.edges(suspect)):
            self.crime_network[i][j]["informed"] = True
            self.crime_network.nodes[j]["suspected"] = True
        self._update_investigation()

    def _set_probas(self, suspect_probas = {}):
        nx.set_node_attributes(self.crime_network, self.random_catch, name="catch_proba")
        caught_probas = {node : 0 for node, attr in self.crime_network.data if attr["caught"] == True}
        suspect_probas = {suspect : (proba + self.random_catch) for suspect, proba in suspect_probas.items()}
        nx.set_node_attributes(self.crime_network, suspect_probas, name="catch_proba")
        nx.set_node_attributes(self.crime_network, caught_probas, name="catch_proba")

    def investigate(self):
        '''
        Performs an investigation according to the set_strategy
        '''
        if self.strategy is None:
            print("No strategy set. Cannot investigate without a strategy.")
            return 
        if self.model_proba is None: 
            print("No underlying model is defined. Please define model.")
            return

        suspect, p = self.strategy(self.current_investigation)
        if suspect == "random":
            self._catch_random()
        elif np.random.uniform() < p:
            self._caught_suspect(suspect)

        self.investigations += 1

    def _update_investigation(self):
        '''update graph view using node filter and edge filter'''
        def filter_node(x):
            return self.crime_network.nodes[x].get("caught", False) or self.crime_network[x].get("suspected", False)
        def filter_edge(i, j):
            return self.crime_network[i][j].get("informed", False)
        self.current_investigation = nx.graphviews.subgraph_view(self.crime_network, filter_node, filter_edge)


#NEED TO TEST

#Need to handle catching probabilities. 

#%%

def simple_model(graph):
    q = 0.1
    p_dict = {}
    for node in graph.nodes:
        p_dict[node] = (q * len(graph.edges(q)))
    return p_dict

def simple_strategy(graph, random_catch):
    pass