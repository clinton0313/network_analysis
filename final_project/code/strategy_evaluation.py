#%%
from investigation import Investigation
from models_and_strategies import least_central, constant_model, exponential_model, simple_greedy
import pickle, os, matplotlib
from time import sleep
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple
from tqdm import tqdm
from functools import partial
matplotlib.use("TkAgg")

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
#%%
with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

#%%

#Are we sure the algorithm proceeds as we expect??? SANITY CHECK PLEASE ==> SEE GRAPH 11 ERROR

def plot_simulations(investigation:Investigation, sims:int, max_criminals:int, max_investigations:int, 
    x:str="investigation", y:str="captured_eigen", title="", xlabel="Investigations", ylabel="", figsize:Tuple=(20,20), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    for _ in tqdm(range(sims), desc=f"Simulating {title}", position=0):
        investigation.reset()
        investigation.simulate(max_criminals, max_investigations)
        log = investigation.log
        ax.plot(log[x], log[y], **kwargs)
        ax.set_title(title, fontsize=20)
        ax.set_xlim(min(log[x]), max(log[x]))
    return fig, ax


#%%

models = {constant_model:{"c":0.05, "weighted":True}}
strategies = {simple_greedy:{}, least_central:{}}
model_names = ["Constant"]
strategy_names = ["Simple Greedy", "Least Central"]

savepath = os.path.join("figs", "simulations")
for model_name, (model, model_params) in zip(model_names, models.items()):
    for strat_name, (strategy, strategy_params) in zip(strategy_names, strategies.items()):
        for graph in tqdm(graphs, desc="Investigating graph: ", position=1):
            inv = Investigation(graph)
            inv.set_model(model, **model_params)
            inv.set_strategy(strategy, **strategy_params)
            fig, ax = plot_simulations(investigation=inv,
                sims=100, max_criminals=10, max_investigations=200,
                title=f"{inv.crime_network.graph['name']}\n{strat_name}; {model_name} Model",color="blue", alpha=0.04)
            fig.savefig(os.path.join(savepath, 
                f"{inv.crime_network.graph['name']}_{strat_name.lower().replace(' ','')}_{model_name.lower().replace(' ','')}.png"),
                facecolor="white", transparent=False)
            fig.clear()
            plt.close()
# %%
