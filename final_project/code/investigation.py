#%%
import pickle, os
import networkx as nx
import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

#%%
class Investigation(nx.Graph):
    def __init__(self, crime_network:nx.Graph, random_catch:float, id = "name", first_criminal = None):
        super(Investigation, self).__init__()

        self.crime_network = crime_network
        self.id = id
        nx.set_node_attributes(self.crime_network, False, "suspected")
        nx.set_node_attributes(self.crime_network, False, "caught")
        nx.set_edge_attributes(self.crime_network, False, "informed")

        self.caught_criminals = []
        self.suspected_criminals = []
        self.node_colors = ["blue" for _ in range(len(self.crime_network.nodes))]
        self.edge_colors = ["black" for _ in range(len(self.crime_network.edges))]
        self.layout = nx.layout.spring_layout(self.crime_network, k = 0.5 / np.sqrt(len(self.crime_network.nodes)))

        self.investigations = 0
        self.current_investigation = None
        if first_criminal == None:
            first_criminal = np.random.randint(len(self.crime_network.nodes))
        self._caught_suspect(first_criminal)

        self.random_catch = random_catch
        self.model_proba = None
        self.strategy = None
   
    def _set_probas(self, suspect_probas = {}):
        '''Set new capture provabilities'''
        nx.set_node_attributes(self.crime_network, self.random_catch, name="catch_proba")
        caught_probas = {node : 0 for node, attr in self.crime_network.nodes.data() if attr["caught"] == True}
        suspect_probas = {suspect : (proba + self.random_catch) for suspect, proba in suspect_probas.items()}
        nx.set_node_attributes(self.crime_network, suspect_probas, name="catch_proba")
        nx.set_node_attributes(self.crime_network, caught_probas, name="catch_proba")

    def _catch_random(self):
        '''Catch a random unsuspected criminal'''
        if np.random.uniform() < self.random_catch:
            unsuspected = [node for node, suspected in list(self.crime_network.nodes(data="suspected")) if not suspected]
            caught = unsuspected[np.random.randint(len(unsuspected))]
            self._caught_suspect(caught)

    def _caught_suspect(self, suspect):
        '''Update graph properties when suspect is caught'''
        self.crime_network.nodes[suspect]["caught"] = True
        self.caught_criminals.append(suspect)
        self.node_colors[suspect] = "black"
        self.crime_network.nodes[suspect]["suspected"] = False
        for i, j in list(self.crime_network.edges(suspect)):
            #Use provided order to choose source-target
            if j not in self.caught_criminals:
                self.crime_network.nodes[j]["suspected"] = True
                self.suspected_criminals.append(j)
                self.node_colors[j] = "red"

            #Reorder edges to index edges
            i, j = min(i, j), max(i, j)
            self.crime_network[i][j]["informed"] = True
            self.edge_colors[list(self.crime_network.edges).index((i, j))] = "orange"

        self._update_investigation()

    def _update_investigation(self):
        '''Update current investigation graphview with new suspected and informed'''
        def filter_node(x):
            return x in self.suspected_criminals or x in self.caught_criminals
        def filter_edge(i, j):
            return self.crime_network[i][j].get("informed", False) or self.crime_network[j][i].get("informed", False)
        self.current_investigation = nx.subgraph_view(self.crime_network, filter_node=filter_node, filter_edge=filter_edge)

    def set_model(self, model):
        '''
        Set underlying model of the chances of catching criminals. Return function that calculates probability of being caught given a node.
        func needs to accept a graph and return an array of probabilities for each node.
        '''
        self.model_proba = partial(model, self.current_investigation)

    def set_strategy(self, strategy, **kwargs):
        '''
        Sets investigation strategy. strategy is a func that takes a graph that indicates suspects and caught criminals and returns the next suspect to investigate, with catching probability
        suspect should be node index, or return catch random?
        '''
        self.strategy =  partial(strategy, **kwargs)

    def investigate(self, plot = False, **plot_kwargs):
        '''
        Makes an attempt to catch a criminal according to set strategy. Plotting optional. 
        '''
        if self.strategy is None:
            print("No strategy set. Cannot investigate without a strategy.")
            return 
        # if self.model_proba is None: 
        #     print("No underlying model is defined. Please define model.")
        #     return
        suspect_probas = self.model_proba()
        self._set_probas(suspect_probas)
        suspect, p = self.strategy(self.current_investigation)
        if suspect == "random":
            self._catch_random()
        elif np.random.uniform() < p:
            self._caught_suspect(suspect)

        self.investigations += 1

        if plot:
            fig = self.plot(**plot_kwargs)
            return fig

    def plot(self, ax = None, weighted = True, weight_multiplier = 3, **kwargs):
        return_fig = False
        if weighted:
            weights = np.array(list(nx.get_edge_attributes(self.crime_network, "weight").values()))
            weights = (weights - weights.min()+1)/(weights.max()+1) * weight_multiplier
            kwargs.update({"width":weights})
        if not ax:
            return_fig = True
            fig, ax = plt.subplots(figsize=(20, 20))
        nx.draw(self.crime_network, pos=self.layout, 
            ax=ax, node_color = self.node_colors,
            edge_color = self.edge_colors,
            **kwargs)
        ax.set_axis_off()
        ax.set_title(self.crime_network.graph["name"], fontsize=30)
        ax.text(x = 0.9, y = 0, 
            s = f"Investigations: {self.investigations}\nCaught Criminals: {self.caught_criminals}\nSuspects: {self.suspected_criminals}",
            transform=ax.transAxes)

        if return_fig:
            return fig, ax
        return ax


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

#%%
with open(os.path.join("data", "processed_data", "raw_nwx_graphs.pkl"), "rb") as infile:
    network_graphs = pickle.load(infile)

inv = Investigation(crime_network = network_graphs[0], random_catch = 0.1)
inv.set_model(simple_model)
inv.set_strategy(simple_strategy, random_catch = inv.random_catch)

#%%
from time import sleep
plt.ion()
fig, ax = inv.plot()
fig.show()
sleep(2)
for _ in range(5):
    fig.canvas.flush_events()
    inv.investigate()
    ax.clear()
    ax = inv.plot(ax)
    fig.canvas.draw()
    sleep(2)
    input()
input()

# %%
