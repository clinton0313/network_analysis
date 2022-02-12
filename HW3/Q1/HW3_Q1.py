#%%
from optimal_seed import Investigation, get_suspect_proba, get_information, get_suspects, plot_simulations
import os, pickle, itertools
from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from random import sample
from typing import DefaultDict
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%
with open("giant_component_crime_networks.pkl", "rb") as infile:
    networks = pickle.load(infile)

#%%
def seed_model(graph):
    return

def optimal_seeding(graph):
    suspects = get_suspects(graph)
    thresholds = [graph.nodes[s]["catch_proba"] for s, attr in list(graph.nodes.data("suspected")) if attr]
    lambdas = get_information(graph, suspects, weighted=False)
    diffused = [suspect for suspect, t in zip(suspects, thresholds) if lambdas[suspect] > t]
    return diffused

#%%

g = networks[1]
sims = 1

sim_records = []
for s1, s2 in tqdm(itertools.combinations(g.nodes, 2)):
    for lamb in [1, 2]:
        diffusion_sims = []
        for _ in range(sims):
            network = Investigation(networks[1], random_catch = "lambda", lamb=lamb, first_criminal=[s1, s2],
                compute_eigen=False, title="mali")
            network.set_model(seed_model)
            network.set_strategy(optimal_seeding)
            network.simulate(max_criminals = 1000, max_investigations = 4)
            diffusion_sims.append(len(network.caught) - 2)
        sim_records.append((lamb, s1, s2, np.mean(diffusion_sims), np.mean(diffusion_sims)/3))

optimal_seed_results = pd.DataFrame.from_records(data=sim_records, columns=["Lambda", "S1", "S2", "Diffusion", "Mean Diffusion"])
with open("optimal_seed_results.pkl", "wb") as outfile:
    pickle.dump(optimal_seed_results, outfile)

#%%

random_diffusion = DefaultDict(list)
for lamb in [1, 2]:
    for _ in tqdm(range(1000)):
        network = Investigation(networks[1], random_catch = "lambda", lamb=lamb, first_criminal=sample(g.nodes, 2),
                compute_eigen=False, title="mali")
        network.set_model(seed_model)
        network.set_strategy(optimal_seeding)
        network.simulate(max_criminals = 1000, max_investigations = 4)
        random_diffusion[lamb].append(len(network.caught))

with open("random_seed_results.pkl", "wb") as outfile:
    pickle.dump(random_diffusion, outfile)


for lamb in [1, 2]:
    optim = optimal_seed_results[(optimal_seed_results["Lambda"] == 1) & \
        (optimal_seed_results["Diffusion"] == optimal_seed_results["Diffusion"].max())]
    print(f"""For Lambda = {lamb}\nThe optimal seeds are {optim.loc[:,'S1']} and {optim.loc[:,'S2']}\n
        with diffusion of {optim.loc[:,'Diffusion']} over three periods of diffusion.\n\n
        Whereas using random seeding we have an expected diffusion of {np.mean(random_diffusion[lamb])}\n
        and variance of {np.var(random_diffusion[lamb])}""")
#%%