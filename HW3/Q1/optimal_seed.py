#%%
import pickle, os, matplotlib
from turtle import update
import networkx as nx
import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from time import sleep
from typing import Callable, DefaultDict, Tuple
from random import choice
from tqdm import tqdm

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

#%%
class Investigation():
    def __init__(self, crime_network:nx.Graph, random_catch:float=0.05, lamb = 1, model:Callable = None, 
        strategy:Callable = None, first_criminal:int = None, compute_eigen = True, title:str = "", caught_color:str="black",
        suspect_color:str="red", criminal_color:str="blue", informed_color:str="orange", non_gc = False):
        '''
        Class to handle investigations simulations of criminal networks. Main method is to either call investigate or simulation to run investigate in a loop.
        the crime network will be initiated with node attributes: "suspected" and "caught, and edge attribute: "informed". 

        Args:
            crime_network: Undirected, weighted graph. Optional graph property "name" will be used for plotting.
            random_catch: Probability of catching a random criminal with no information.
            model: Underlying model that determines the probability of catching suspects. Function should take a graph of suspects and known criminals as an argument
                and return a dictionary of probabilities corresponding to {suspect: probability}
            strategy: Underlying algorithmn that investigate will use. Function should take at least a graph of suspects and known criminals as an argument and return a 
                suspect (int) and probability of capture (float) in that order. To use more arguments, use set_strategy method. Return string "random" in place of the suspect
                and an artbitrary value for p to catch a random unknown criminal instead.
            first_criminal: Optionally initialize the first criminal otherwise a random criminal will be used
            compute_eigen: Computes and stores the eigenvector centrality in handle "eigen" upon init. 
            title: Title used for plotting of graph.
            informed_color, caught_color, suspect_color, criminal_color: colors for plotting nodes and edges.
            non_gc: Default is False. If True, will not convert graph to giant component.

        Methods:
            set_model: Set the underlying model.
            set_strategy: Set the underlying strategy.
            investigate: Attempt to catch a criminal using the set strategy for the current investigation. 
            simulate: Run multiple investigations to a stopping criterion.
            reset: Resets the network but keeps the set model and strategy.
            plot: Plot the network.
            set_layout: Set plotting layout.
        
        Attributes:
            investigations, caught, suspects, fig, ax, title. 
        '''
        #Network intiializationa and attributes
        components = list(nx.connected_components(crime_network))
        if len(components) != 1:
            crime_network = crime_network.subgraph(max(components))
            print("Graph had multiple components, keeping only giant component")
        del components
        self.crime_network = crime_network
        nx.set_node_attributes(self.crime_network, False, "suspected")
        nx.set_node_attributes(self.crime_network, False, "caught")
        nx.set_edge_attributes(self.crime_network, False, "informed")
        nx.set_node_attributes(self.crime_network, criminal_color, "color")
        nx.set_edge_attributes(self.crime_network, informed_color, "color")

        #Intiialize thresholds
        self.lamb = lamb
        self.random_catch = random_catch
        if self.random_catch == "lambda":
            thresholds = np.random.normal(lamb, 0.5, len(self.crime_network.nodes))
            thresholds = {node:max(0, t) for node, t in zip(self.crime_network.nodes, thresholds)}
            nx.set_node_attributes(self.crime_network, thresholds , "catch_proba")
        self.eigen = False
        if compute_eigen:
            self._compute_eigen_centrality()

        self.investigations = 1
        self.caught = []
        self.suspects = []
        self.log = DefaultDict(list)

        self.current_investigation = None
        if first_criminal == None:
            first_criminal = choice(list(self.crime_network.nodes))
        if model == None:
            self.model_proba = None
        else:
            self.model_proba = partial(model, self.current_investigation)
        self.strategy = strategy
        
        #Plotting attributes
        if title == "":
            try:
                self.title = self.crime_network.graph["name"]
            except KeyError:
                print("Graph had no name property for titling plots")
        else:
            self.title = title
        self.criminal_color = criminal_color
        self.suspect_color = suspect_color
        self.caught_color = caught_color
        self.informed_color = informed_color

        self.fig = None
        self.ax = None
        self.layout = nx.layout.spring_layout(self.crime_network, k = 0.5 / np.sqrt(len(self.crime_network.nodes)))

        #Initialize
        if isinstance(first_criminal, list):
            for criminal in first_criminal:
                self._caught_suspect(criminal)
        else:
            self._caught_suspect(first_criminal)
        self._log_stats()

    def _compute_eigen_centrality(self, handle = "eigen"):
        ec = nx.eigenvector_centrality_numpy(self.crime_network)
        nx.set_node_attributes(self.crime_network, ec, name=handle)
        self.eigen = True

    def _model_check(self):
        if self.strategy is None:
            print("No strategy set. Please set a strategy using set_strategy method.")
            return False
        if self.model_proba is None: 
            print("No underlying model is defined. Please define model using set_model method.")
            return False

    def _set_probas(self, suspect_probas:dict = {}):
        '''Set new capture probabilities'''
        nx.set_node_attributes(self.crime_network, self.random_catch, name="catch_proba")
        caught_probas = {node : 0 for node, attr in self.crime_network.nodes.data() if attr["caught"] == True}
        suspect_probas = {suspect : (proba + self.random_catch) for suspect, proba in suspect_probas.items()}
        nx.set_node_attributes(self.crime_network, suspect_probas, name="catch_proba")
        nx.set_node_attributes(self.crime_network, caught_probas, name="catch_proba")

    def _catch_random(self):
        '''Catch a random unsuspected criminal'''
        if self.random_catch =="lambda":
            print("Using diffusion model, no need for random catching")
        elif np.random.uniform() < self.random_catch:
            unsuspected = [node for node, suspected in list(self.crime_network.nodes(data="suspected")) if not suspected]
            caught = choice(unsuspected)
            self._caught_suspect(caught, random = True)

    def _caught_suspect(self, suspect:int, random = False):
        '''Update graph properties when suspect is caught'''
        self.crime_network.nodes[suspect]["caught"] = True
        self.caught.append(suspect)
        self.crime_network.nodes[suspect]["color"] = self.caught_color
        self.crime_network.nodes[suspect]["suspected"] = False
        if self.investigations != 1 or random:
            self.suspects.remove(suspect)
        for i, j in list(self.crime_network.edges(suspect)):
            #Use provided order to choose source-target
            if j not in self.caught and j not in self.suspects: #COuld I just check if not an informed edge????
                self.crime_network.nodes[j]["suspected"] = True
                self.suspects.append(j)
                self.crime_network.nodes[j]["color"] = self.suspect_color
            #Reorder edges to index edges
            i, j = min(i, j), max(i, j)
            self.crime_network[i][j]["informed"] = True
            self.crime_network[i][j]["color"] = self.informed_color
        self._update_investigation()

    def _update_investigation(self):
        '''Update current investigation graphview with new suspected and informed'''
        def filter_node(x):
            return x in self.suspects or x in self.caught
        def filter_edge(i, j):
            return self.crime_network[i][j].get("informed", False) or self.crime_network[j][i].get("informed", False)
        self.current_investigation = nx.subgraph_view(self.crime_network, filter_node=filter_node, filter_edge=filter_edge)
    
    def _log_stats(self):
        self.log["caught"].append(len(self.caught))
        self.log["suspects"].append(len(self.suspects))
        self.log["informed"].append(len(self.current_investigation.edges))
        if self.eigen:
            self.log["captured_eigen"].append(np.sum([self.crime_network.nodes.data("eigen")[i] for i in self.caught]))
        self.log["investigation"].append(self.investigations)

    def set_model(self, model:Callable, **kwargs):
        '''Set underlying probability model for capture of suspects. Function should take a graph of suspects and known criminals as an argument
            and return a dictionary of probabilities corresponding to {suspect: probability}'''
        self.model_proba = partial(model, self.current_investigation, **kwargs)

    def set_strategy(self, strategy:Callable, **kwargs):
        '''Sets investigation strategy. Function should take at least a graph of suspects and known criminals as an argument and return a 
            suspect (int) and probability of capture (float) in that order. Return string "random" in place of the suspect and an artbitrary value for p
            to catch a random unknown criminal instead.'''
        self.strategy =  partial(strategy, **kwargs)
    
    def set_layout(self, pos):
        '''Set network x plotting layout.'''
        self.layout = pos

    def investigate(self, plot:bool = False, update_plot:bool = False, **plot_kwargs):
        '''
        Makes an attempt to catch a criminal according to strategy.

        Args:
            plot: If true calls plot method. 
            update_plot: If true attempts to update stored plot interactively. If no plot has been made, will call plot method normally.
        '''
        if self._model_check() == False:
            return
        #Set probabilities of the model
        if self.random_catch == "lambda":
            suspects = self.strategy(self.current_investigation)
            for suspect in suspects:
                if suspect not in self.suspects and suspect not in self.caught:
                    self._caught_suspect(suspect)
        else:
            suspect_probas = self.model_proba()
            self._set_probas(suspect_probas)
            suspect, p = self.strategy(self.current_investigation)
            if suspect == "random":
                self._catch_random()
            elif np.random.uniform() < p:
                self._caught_suspect(suspect)
        self.investigations += 1
        self._log_stats()


        if update_plot:
            if self.ax and self.fig:
                if not matplotlib.is_interactive():
                    plt.ion()
                self.fig.canvas.flush_events()
                self.ax.clear()
                self.plot(**plot_kwargs)
                self.fig.canvas.draw()
            else:
                self.plot(**plot_kwargs)
                self.fig.show()
        elif plot:
            self.plot(**plot_kwargs)

    def simulate(self, max_criminals:int = 0, max_investigations:int = 0, update_plot = False, sleep_time = 0.5, **kwargs):
        '''
        Investigates until either stopping criterion is met or entire network caught.

            Args:
                max_criminals: Number of criminals to catch before stopping investigation.
                max_investigations: Number of investigations to make before stopping investigation.
                condition: "and" or "or". How stopping criterion considers combines max_criminals and max_investigations conditions. 
                update_plot: Plots and updates as simulation runs. See investigate method.
                sleep_time: Sleep time between plot updates. Default 1. 
                **kwargs: Arguments such as plot and update_plot to be passed to investigate. 
        '''
        if self._model_check == False:
            return
        while len(self.caught) < max_criminals and self.investigations < max_investigations:
            if len(self.caught) == len(self.crime_network.nodes):
                break
            self.investigate(update_plot=update_plot, **kwargs)
            if update_plot:
                sleep(sleep_time)

    def plot(self, weighted:bool = True, weight_multiplier:float = 3, showfig:bool = True, label = "", **kwargs):
        '''
        Plots the network and saves to self.fig and self.ax. 

        Args:
            weighted: If true, plots weighted edge widths
            weight_multiplier: Adjusts the scale of weighted edge widths
            showfig: If true, calls fig.show(). 
            label: Text label under title.
            **kwargs: Extra arguments passed to nx.draw()
        '''
        if weighted:
            weights = np.array(list(nx.get_edge_attributes(self.crime_network, "weight").values()))
            weights = (weights - weights.min()+1)/(weights.max()+1) * weight_multiplier
            kwargs.update({"width":weights})
        
        if not self.ax:
            self.fig, self.ax = plt.subplots(figsize=(20, 20))

        statistics = f"Investigations: {self.investigations}\nCaught Criminals: {len(self.caught)}\nSuspects: {len(self.suspects)}"
        if self.eigen:
            captured_eigen = np.sum([self.crime_network.nodes.data("eigen")[i] for i in self.caught])
            statistics = statistics + f"\nCaptured EC:{round(captured_eigen, 2)}"
        
        nx.draw(self.crime_network, pos=self.layout, 
            ax=self.ax, node_color = dict(self.crime_network.nodes.data("color")).values(),
            edge_color = [color for _, _, color in self.crime_network.edges.data("color")],
            **kwargs)
        self.ax.set_axis_off()
        self.ax.set_title(self.title, fontsize=30)
        self.ax.text(x=0, y = 0, s=label, fontsize=15, transform=self.ax.transAxes)
        self.ax.text(x = 0.8, y = 0, 
            s = statistics,
            transform=self.ax.transAxes, fontsize=15)
        
        if showfig:
            self.fig.show()
    
    def reset(self, first_criminal:int = None, keep_fig:bool = False, verbose=False):
        '''
        Resets the network, and reinitializes. Does not reset the model and strategy.
        
        Args:
            first_criminal: Initialize with an optional first criminal index. Otherwise random.
            keep_fig: If true, does not reset the current figure and ax. Set to true and run subsequent simulations using update_plot
                to keep refreshing to a new figure
        '''

        nx.set_node_attributes(self.crime_network, False, "suspected")
        nx.set_node_attributes(self.crime_network, False, "caught")
        nx.set_edge_attributes(self.crime_network, False, "informed")
        nx.set_node_attributes(self.crime_network, self.criminal_color, "color")
        nx.set_edge_attributes(self.crime_network, self.informed_color, "color")

        #Intiialize thresholds
        if self.random_catch == "lambda":
            thresholds = np.random.normal(self.lamb, 0.5, len(self.crime_network.nodes))
            thresholds = {node:max(0, t) for node, t in zip(self.crime_network.nodes, thresholds)}
            nx.set_node_attributes(self.crime_network, thresholds , "catch_proba")
        self.eigen = False       
                
        self.investigations = 1
        self.caught = []
        self.suspects = []
        self.log = DefaultDict(list)

        self.current_investigation = None
        if first_criminal == None:
            first_criminal = choice(list(self.crime_network.nodes))

        if not keep_fig:
            self.fig = None
            
        if isinstance(first_criminal, list):
            for criminal in first_criminal:
                self._caught_suspect(criminal)
        else:
            self._caught_suspect(first_criminal)
        self._log_stats()
        if verbose:
            print("Crime network reset.")
    
    def refresh_fig(self):
        '''Refreshes the figure.'''
        if self.ax and self.fig:
            if not matplotlib.is_interactive():
                plt.ion()
            self.fig.canvas.flush_events()
            self.ax.clear()
            self.plot(showfig=False)
            self.fig.canvas.draw()

#HELPER FUNCTIONS

def get_suspects(graph:nx.Graph) -> list:
    '''Returns a list of suspects in the graph'''
    return [suspect for suspect in graph.nodes if graph.nodes[suspect].get("suspected")]

def get_suspect_proba(graph:nx.Graph, suspects:list) -> list:
    '''Returns a list of catching probabilities for each of the suspects'''
    return [graph.nodes[suspect]["catch_proba"] for suspect in suspects]

def get_information(graph:nx.Graph, suspects:list, weighted:bool = True) -> dict:
    '''Returns a dictionaryof information links for each suspect. If weighted, it will sum the edge weights, otherwise take the number
    of edges activated incident to each suspect.'''
    information = {}
    for suspect in suspects:
        incident = [graph[i][j]["weight"] for i, j in graph.edges(suspect) if graph[i][j]["informed"]]
        if weighted:    
            information[suspect] = sum(incident)
        elif not weighted:
            information[suspect] = len(incident)
    return information

def plot_simulations(investigation:Investigation, sims:int, max_criminals:int, max_investigations:int, 
    x:str="investigation", y:str="captured_eigen", title="", xlabel="Investigations", ylabel="", figsize:Tuple=(20,20), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    for _ in tqdm(range(sims), desc=f"Simulating {title}", position=1):
        investigation.reset()
        investigation.simulate(max_criminals, max_investigations)
        log = investigation.log
        ax.plot(log[x], log[y], **kwargs)
        ax.set_title(title, fontsize=20)
        ax.set_xlim(min(log[x]), max(log[x]))
    return fig, ax