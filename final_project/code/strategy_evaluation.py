#%%
from investigation import Investigation
from models_and_strategies import least_central, constant_model, exponential_model, simple_greedy
import pickle, os, matplotlib
from time import sleep
import matplotlib.pyplot as plt
import networkx as nx

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
#%%
with open(os.path.join("data", "processed_data", "raw_nwx_graphs.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

inv = Investigation(graphs[8], random_catch = 0.05)

#%%
inv = Investigation(graphs[8])
inv.set_model(constant_model, c=0.05, weighted=True)
inv.set_strategy(least_central)

fig, ax = plt.subplots()
for _ in range(20):
    inv.reset()
    inv.simulate(20, 100)
    log = inv.log
    ax.plot(log["investigation"], log["caught"], color="yellow", alpha = 0.03)

#%%
#Check that models and strategies are working as we want them to
#Check why simulate makes a figure still?
