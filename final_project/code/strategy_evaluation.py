#%%
from investigation import Investigation
from models_and_strategies import constant_model, max_diameter, balanced_diameter, greedy, naive_random
import pickle, os, matplotlib
from time import sleep
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, Union, Sequence
from tqdm import tqdm

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

#%%
def plot_simulations(investigation:Investigation, sims:int, max_criminals:int, max_investigations:int, 
    x:str="investigation", y:str="eigen_proportion", title="", xlabel="Investigations", ylabel="",
    ymax:float = 0, ymin:float = 0, xmax:float = 0, xmin: float = 0, figsize:Tuple=(16,16), **kwargs):
    '''
    Makes a multiline plot of all simulations run.

    Args:
        investigation: Investigation class object with strategy and model already set.
        sims: Number of simulations to run
        max_criminals: Maximum number of criminals to run simulations to. Condition is "and" with max_investigations.
        max_investigations: Maximum number of investigations to run simulations to. Condition is "and" with max_criminals.
        x: Name of node attribute to plot on x-axis.
        y: Name of node attribute to plot on y-axis.
        xlabel, ylabel: labels for x and y axis.
        ymax, ymin: for axis limits. If 0, will auto scale. 
        figsize: Size of figure. 
        **kwargs: Kwargs to be passed to ax.plot().
    
    Returns:
        Figure and axes of plot. 
    '''
    fig, ax = plt.subplots(figsize=figsize)
    for _ in tqdm(range(sims), desc=f"{investigation.crime_network.graph['name']}", position=0):
        investigation.reset()
        investigation.simulate(max_criminals, max_investigations, update_plot=False)
        log = investigation.log
        ax.plot(log[x], log[y], **kwargs)
        ax.set_title(title, fontsize=20)
        ax.set_xlim(min(log[x]), max(log[x]))
        if xmax != 0:
            ax.set_xlim(ymin, xmax)
        if ymax != 0:
            ax.set_ylim(ymin, ymax)
    return fig, ax


def evaluate_strategies(investigations:list, sims:int, max_criminals:int, max_investigations:int, savepath:str,
    models:dict, model_names:list, strategies:dict, strategy_names:list, overwrite:bool=False, **plot_kwargs):
    '''
    Evaluate many strategies and save plots by calling plot_simulation().

        Args:
            investiagions: A list of Investigation classes of networkx graphs to be evaluated
            sims: Number of simulations per graph.
            max_criminals, max_investigations: Stopping conditions for simulations. Both need to be met.
            models: Dictionary of model (functions) to model parameters in dictionary format.
            strategies: Dictionary of strategies (functions) to strategy parameters in dictionary format.
            model_names: List of model names to match models dictionary.
            strategy_names: List of strategy names to match strategies dictionary.
            overwrite: If true will overwrite existing files. 
            **plot_kwargs: plotting kwargs to be passed on to plot_simiulation(). 
        
        Returns:
            None
    '''

    #Set matplotlib backend so figures clear from memory properly
    matplotlib.use("TkAgg")
    #Create filepaths and directories for storing figures. 
    filepaths = {strat: strat.lower().replace(" ", "_") for strat in strategy_names}
    for filepath in filepaths.values():
        os.makedirs(os.path.join(savepath, filepath), exist_ok=True)

    for model_name, (model, model_params) in zip(model_names, models.items()):
        for strat_name, (strategy, strategy_params) in zip(strategy_names, strategies.items()):
            strat_dirpath = os.path.join(savepath, filepaths[strat_name])

            for inv in tqdm(investigations, desc="Investigating graph: ", position=1, colour="green"):
                filename = f"{inv.crime_network.graph['name']}_{filepaths[strat_name]}_{model_name.lower().replace(' ','')}.png"
                #Skip if graph already generated
                if os.path.exists(os.path.join(strat_dirpath, filename)) and not overwrite:
                    continue
                #Instnatiate investigation, set model and strategy.
                inv.set_model(model, **model_params)
                inv.set_strategy(strategy, **strategy_params)
                #Plot and save figure.
                fig, ax = plot_simulations(investigation=inv,
                    sims=sims, max_criminals=max_criminals, max_investigations=max_investigations, **plot_kwargs)
                ax.set_title(f"{inv.crime_network.graph['name']}\n{strat_name}; {model_name} Model\nCriminals Caught:{len(inv.caught)}/{len(inv.crime_network.nodes)}\n(final iteration)",
                    fontsize=20)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                fig.savefig(os.path.join(strat_dirpath, filename), facecolor="white", transparent=False)
                fig.clear()
                plt.close()


def inverse_eigen_probas(graph:nx.Graph, min_proba:float= 0.025, max_proba:float=0.075, weighted:bool=True) -> dict:
    '''
    Creates base probabilities based on eigenvector centralities. The lower the eigenvector centrality, the 
    higher the base probability. 
    
    Args:
        graph: NetworkX Graph.
        min_proba: Lower bound of base probabiltiies that eigenvector centralities are scaled to.
        max_proba: Upper bound of base probabilities that eigenvector cenralities are sclaed to.
        weighted: Consider graph as weighted.
    Returns:
        Dictionary of base probabilities {node: proba}    
    '''
    assert 0 <= max_proba <= 1, f"max_proba needs to be a valid probability, instead got {max_proba}"
    assert 0 <= min_proba <= 1, f"min_proba needs to be a valid probaiblity, instead got {min_proba}"
    #Get nodes and eigenvector centralities
    nodes = list(graph.nodes)
    try:
        eigens = nx.eigenvector_centrality_numpy(graph, weight= lambda _: "weight" if weighted else None)
    except TypeError:
        eigens = nx.eigenvector_centrality(graph, weight=lambda _: "weight" if weighted else None)
    
    #Scale eigenvector centralities
    scale = (max_proba - min_proba) / (max(eigens.values()) - min(eigens.values()))
    scaled_eigens = {
        s : max_proba - (eigen - min(eigens.values())) * scale
        for s, eigen in eigens.items()
        }
    base_proba = {node : scaled_eigens[node] for node in nodes}
    return base_proba

#%%
#Load graphs and set parameters. 
with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

graphs.pop(6) #omit paul_revere set. 

models = {constant_model:{"c":0.05, "weighted":True}}
strategies = {greedy:{"tiebreaker":"triangles"}}
model_names = ["Constant"]
strategy_names = ["Greedy triangles"]
savepath = os.path.join("figs", "simulations", "inverse_eigen")
#Use inverse_eigen_probas here
investigations = [Investigation(graph, random_catch = inverse_eigen_probas(graph)) for graph in graphs]

evaluate_strategies(investigations=investigations,
    sims=50,
    max_criminals=400,
    max_investigations=100,
    models=models,
    model_names=model_names,
    strategies=strategies,
    strategy_names=strategy_names,
    savepath=savepath,
    y="captured_eigen",
    ymin = 0, ymax = 0, xmax = 100,
    ylabel="EC Captured",
    color="blue",
    alpha=0.4)

# %%
