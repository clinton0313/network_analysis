#%%
import pickle, os, matplotlib
from turtle import update
import networkx as nx
import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from time import sleep
from typing import Callable

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

#%%
class Investigation(nx.Graph):
    def __init__(self, crime_network:nx.Graph, random_catch:float, model:Callable = None, 
        strategy:Callable = None, first_criminal:int = None, title:str = "", caught_color:str="black",
        suspect_color:str="red", criminal_color:str="blue", informed_color:str="orange"):
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
            title: Title used for plotting of graph.

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
        super(Investigation, self).__init__()

        #Network intiializationa and attributes
        self.crime_network = crime_network
        nx.set_node_attributes(self.crime_network, False, "suspected")
        nx.set_node_attributes(self.crime_network, False, "caught")
        nx.set_edge_attributes(self.crime_network, False, "informed")
        self.random_catch = random_catch

        self.investigations = 1
        self.caught = []
        self.suspects = []

        self.current_investigation = None
        if first_criminal == None:
            first_criminal = np.random.randint(len(self.crime_network.nodes))
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
        self.node_colors = [self.criminal_color for _ in range(len(self.crime_network.nodes))]
        self.edge_colors = ["black" for _ in range(len(self.crime_network.edges))]
        self.fig = None
        self.ax = None
        self.layout = nx.layout.spring_layout(self.crime_network, k = 0.5 / np.sqrt(len(self.crime_network.nodes)))

        #Initialize
        self._caught_suspect(first_criminal)

    def _set_probas(self, suspect_probas:dict = {}):
        '''Set new capture probabilities'''
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

    def _caught_suspect(self, suspect:int):
        '''Update graph properties when suspect is caught'''
        self.crime_network.nodes[suspect]["caught"] = True
        self.caught.append(suspect)
        self.node_colors[suspect] = self.caught_color
        self.crime_network.nodes[suspect]["suspected"] = False
        for i, j in list(self.crime_network.edges(suspect)):
            #Use provided order to choose source-target
            if j not in self.caught:
                self.crime_network.nodes[j]["suspected"] = True
                self.suspects.append(j)
                self.node_colors[j] = self.suspect_color

            #Reorder edges to index edges
            i, j = min(i, j), max(i, j)
            self.crime_network[i][j]["informed"] = True
            self.edge_colors[list(self.crime_network.edges).index((i, j))] = self.informed_color

        self._update_investigation()

    def _update_investigation(self):
        '''Update current investigation graphview with new suspected and informed'''
        def filter_node(x):
            return x in self.suspects or x in self.caught
        def filter_edge(i, j):
            return self.crime_network[i][j].get("informed", False) or self.crime_network[j][i].get("informed", False)
        self.current_investigation = nx.subgraph_view(self.crime_network, filter_node=filter_node, filter_edge=filter_edge)

    def _model_check(self):
        if self.strategy is None:
            print("No strategy set. Please set a strategy using set_strategy method.")
            return False
        if self.model_proba is None: 
            print("No underlying model is defined. Please define model using set_model method.")
            return False

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
        suspect_probas = self.model_proba()
        self._set_probas(suspect_probas)
        suspect, p = self.strategy(self.current_investigation)
        
        if suspect == "random":
            self._catch_random()
        elif np.random.uniform() < p:
            self._caught_suspect(suspect)
        self.investigations += 1

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

    def plot(self, weighted:bool = True, weight_multiplier:float = 3, showfig:bool = True, **kwargs):
        '''
        Plots the network and saves to self.fig and self.ax. 

        Args:
            weighted: If true, plots weighted edge widths
            weight_multiplier: Adjusts the scale of weighted edge widths
            showfig: If true, calls fig.show(). 
            **kwargs: Extra arguments passed to nx.draw()
        '''
        if weighted:
            weights = np.array(list(nx.get_edge_attributes(self.crime_network, "weight").values()))
            weights = (weights - weights.min()+1)/(weights.max()+1) * weight_multiplier
            kwargs.update({"width":weights})
        
        if not self.ax:
            self.fig, self.ax = plt.subplots(figsize=(20, 20))

        nx.draw(self.crime_network, pos=self.layout, 
            ax=self.ax, node_color = self.node_colors,
            edge_color = self.edge_colors,
            **kwargs)
        self.ax.set_axis_off()
        self.ax.set_title(self.title, fontsize=30)
        self.ax.text(x = 0.8, y = 0, 
            s = f"Investigations: {self.investigations}\nCaught Criminals: {len(self.caught)}\nSuspects: {len(self.suspects)}",
            transform=self.ax.transAxes, fontsize=20)
        
        if showfig:
            self.fig.show()
    
    def simulate(self, max_criminals:int = 0, max_investigations:int = 0, update_plot = False, sleep_time = 1, **kwargs):
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
        max_criminals = min(max_criminals, len(self.crime_network.nodes))

        while len(self.caught) <= max_criminals and self.investigations <= max_investigations:
            self.investigate(update_plot=update_plot, **kwargs)
            if update_plot:
                sleep(sleep_time)
    
    def reset(self, first_criminal:int = None, keep_fig:bool = False):
        '''Resets the network and reinitializes using either first_criminal or random start if first_criminal not provided.'''
        nx.set_node_attributes(self.crime_network, False, "suspected")
        nx.set_node_attributes(self.crime_network, False, "caught")
        nx.set_edge_attributes(self.crime_network, False, "informed")
        self.investigations = 1
        self.caught = []
        self.suspects = []

        self.current_investigation = None
        if first_criminal == None:
            first_criminal = np.random.randint(len(self.crime_network.nodes))

        self.node_colors = [self.criminal_color for _ in range(len(self.crime_network.nodes))]
        self.edge_colors = ["black" for _ in range(len(self.crime_network.edges))]
        
        if not keep_fig:
            self.fig = None
            
        self._caught_suspect(first_criminal)
        # self.refresh_fig()
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


