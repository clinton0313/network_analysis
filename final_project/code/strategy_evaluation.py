#%%
from investigation import Investigation
from models_and_strategies import least_central, constant_model, exponential_model, simple_greedy, least_central_criminal
import pickle, os, matplotlib
from time import sleep
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple
from tqdm import tqdm

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

#%%

#Are we sure the algorithm proceeds as we expect??? SANITY CHECK PLEASE

def plot_simulations(investigation:Investigation, sims:int, max_criminals:int, max_investigations:int, 
    x:str="investigation", y:str="captured_eigen", title="", xlabel="Investigations", ylabel="", figsize:Tuple=(16,16), **kwargs):
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
        figsize: Size of figure. 
        **kwargs: Kwargs to be passed to ax.plot().
    
    Returns:
        Figure and axes of plot. 
    '''
    fig, ax = plt.subplots(figsize=figsize)
    for _ in tqdm(range(sims), desc=f"Simulating {title}", position=0):
        investigation.reset()
        investigation.simulate(max_criminals, max_investigations, update_plot=False)
        log = investigation.log
        ax.plot(log[x], log[y], **kwargs)
        ax.set_title(title, fontsize=20)
        ax.set_xlim(min(log[x]), max(log[x]))
    return fig, ax


def evaluate_strategies(graphs:list, sims:int, max_criminals:int, max_investigations:int, 
    models:dict, model_names:list, strategies:dict, strategy_names:list, **plot_kwargs):
    '''
    Evaluate many strategies and save plots by calling plot_simulation().

        Args:
            graphs: A list of networkx graphs to be evaluated
            sims: Number of simulations per graph.
            max_criminals, max_investigations: Stopping conditions for simulations. Both need to be met.
            models: Dictionary of model (functions) to model parameters in dictionary format.
            strategies: Dictionary of strategies (functions) to strategy parameters in dictionary format.
            model_names: List of model names to match models dictionary.
            strategy_names: List of strategy names to match strategies dictionary.
            **plot_kwargs: plotting kwargs to be passed on to plot_simiulation(). 
        
        Returns:
            None
    '''

    #Set matplotlib backend so figures clear from memory properly
    matplotlib.use("TkAgg")
    #Create filepaths and directories for storing figures. 
    filepaths = {strat: strat.lower().replace(" ", "_") for strat in strategy_names}
    for filepath in filepaths.values():
        os.makedirs(os.path.join("figs", "simulations", filepath), exist_ok=True)

    for model_name, (model, model_params) in zip(model_names, models.items()):
        for strat_name, (strategy, strategy_params) in zip(strategy_names, strategies.items()):
            savepath = os.path.join("figs", "simulations", filepaths[strat_name])
            for graph in tqdm(graphs, desc="Investigating graph: ", position=1):
                #Instnatiate investigation, set model and strategy.
                inv = Investigation(graph)
                inv.set_model(model, **model_params)
                inv.set_strategy(strategy, **strategy_params)
                #Plot and save figure.
                fig, ax = plot_simulations(investigation=inv,
                    sims=sims, max_criminals=max_criminals, max_investigations=max_investigations,
                    title=f"{inv.crime_network.graph['name']}\n{strat_name}; {model_name} Model", **plot_kwargs)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                fig.savefig(os.path.join(savepath, 
                    f"{inv.crime_network.graph['name']}_{filepaths[strat_name]}_{model_name.lower().replace(' ','')}.png"),
                    facecolor="white", transparent=False)
                fig.clear()
                plt.close()

#%%
#Load graphs and set parameters. 
with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

models = {constant_model:{"c":0.05, "weighted":True}}
strategies = {least_central_criminal:{"use_eigen":False}}
model_names = ["Constant"]
strategy_names = ["Least Central Criminal"]

evaluate_strategies(graphs=graphs,
    sims=300,
    max_criminals=20,
    max_investigations=500,
    models=models,
    model_names=model_names,
    strategies=strategies,
    strategy_names=strategy_names,
    color="blue",
    alpha=0.2)


# %%
