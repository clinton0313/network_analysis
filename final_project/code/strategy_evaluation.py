#%%
from investigation import Investigation
from models_and_strategies import least_central, constant_model, exponential_model, simple_greedy
import pickle, os, matplotlib
from time import sleep
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple
from tqdm import tqdm
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
    for _ in tqdm(range(sims)):
        investigation.reset()
        investigation.simulate(max_criminals, max_investigations)
        log = investigation.log
        ax.plot(log[x], log[y], **kwargs)
        ax.set_title(title, fontsize=20)
        ax.set_xlim(min(log[x]), max(log[x]))
    return fig, ax


#%%

# savepath = os.path.join("figs", "simulations")
# for graph in graphs:
#     inv = Investigation(graph)
#     inv.set_model(constant_model, c=0.05, weighted=True)
#     inv.set_strategy(simple_greedy)
#     fig, ax = plot_simulations(investigation=inv,
#         sims=100, max_criminals=100, max_investigations=200,
#         title=f"{inv.crime_network.graph['name']}\nSimple Greedy; Constant Model",color="blue", alpha=0.1)
#     fig.savefig(os.path.join(savepath, f"{inv.crime_network.graph['name']}_simplegreedy_constant.png"),
#         facecolor="white", transparent=False)
#     fig.clear()
#     plt.close()
# %%

inv = Investigation(graphs[10])
inv.set_model(constant_model, c=0.05, weighted=True)
inv.set_strategy(simple_greedy) #Also doesn't work with least_central
fig, ax = plot_simulations(investigation=inv,
    sims=100, max_criminals=100, max_investigations=200, ylabel="Captured EC",
    title=f"{inv.crime_network.graph['name']}\nSimple Greedy; Constant Model", color="blue", alpha=0.1)

#Something wrong with graphs[11] not adding or subtracting criminals/suspects properly

# %%
