#%%
'''
README

The code here was adapted from code that had already been written to do a sort of cascades simulation on crime networks for 
the final project. That is why the variable names are all related to crime. The networks were downloaded and loaded from a different
script, but here I supply the networks already processed into networkx graph objects in pickle file format. 

The class allows you to define your own underlying probability model. Since the model here is fixed and does not need to be recomputed
the probabilities are hardcoded into the constructor of the class when calling random_catch ='lambda'. This simplifies the process
and so all that is needed is to pass a null model because we will not be dynamically computing the threshold. 

The strategy passed is just a simple comparison of number of activated incident edges versus the threshold as described in the question.
The built in method of simulate simply loops over a built in method called investigate that at each call follows the strategy to diffuse.
All these methods and attributes of the class can be viewed in the optimal_seed.py file. They are all imported here in the beginning to make
the script simple. Note: this is also why there are many methods within the investigation class that are coded there but not used. 
The methods that are relevant to this homework question. 
'''


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
#Load pre-parsed crime-networks. 

with open("giant_component_crime_networks.pkl", "rb") as infile:
    networks = pickle.load(infile)
#Choose the second crime network which happens to be Mali Terrorists
g = networks[1]
#%%

#Set a null model since we are going to be using the diffusion mode of the investigation class that was originally programmed
#For cascade type model. 
def seed_model(graph):
    return

#Optimal seeding strategy simply compares the thresholds with the lambdas which are the sum of all edges. Treat the graph as 
#unweighted for the homeowrk. Return the list of "suspects" to diffuse to. 
def optimal_seeding(graph):
    suspects = get_suspects(graph)
    thresholds = [graph.nodes[s]["catch_proba"] for s, attr in list(graph.nodes.data("suspected")) if attr]
    lambdas = get_information(graph, suspects, weighted=False)
    diffused = [suspect for suspect, t in zip(suspects, thresholds) if lambdas[suspect] > t]
    return diffused

#%%
#Simulate for optimal seeding:

sims = 60
sim_records = []
for s1, s2 in tqdm(itertools.combinations(g.nodes, 2)):
    for lamb in [1, 2]:
        diffusion_sims = []
        for _ in range(sims):
            #Key step here where we initiate the class and set random_catch to lambda. This is coded to generate threshold
            #values using the truncated normal as described. Otherwise we pass in the seeds, the lambda value and turn off the 
            #computing of eigenvalue centrality because it is unecessary. 
            network = Investigation(g,  random_catch = "lambda", lamb=lamb, first_criminal=[s1, s2],
                compute_eigen=False, title=g.graph['name'])
            #Here we set the model and strategy we initiated above. 
            network.set_model(seed_model)
            network.set_strategy(optimal_seeding)
            #Here we initiate giving max_criminals of 1000 so that we do not stop the simulation prematurely.
            #We go to 4 investigations because the class counts the initial seed as a step. 
            #For a fun visualization, you could set update_plot=True in the simulate method, with sims = 1 or outside of these loops. 
            network.simulate(max_criminals = 1000, max_investigations = 5)
            diffusion_sims.append(len(network.caught) - 2)
        #Append the results and compute the average diffusion rate as number of final nodes diffused to (less the two initial)/number of periods
        sim_records.append((lamb, s1, s2, np.mean(diffusion_sims), np.mean(diffusion_sims)/4))

#You can uncomment and run this part to see a visualization of the diffusion process.

# network = Investigation(networks[1], random_catch = "lambda", lamb=1, first_criminal=[0, 1],
#     compute_eigen=False, title="mali")
# network.set_model(seed_model)
# network.set_strategy(optimal_seeding)
# network.simulate(max_criminals = 1000, max_investigations = 5, update_plot=True)



optimal_seed_results = pd.DataFrame.from_records(data=sim_records, columns=["Lambda", "S1", "S2", "Diffusion", "Mean Diffusion"])
with open("optimal_seed_results.pkl", "wb") as outfile:
    pickle.dump(optimal_seed_results, outfile)

#%%
#Simulate for random diffusion: 

#Same as above, but instead we sample 2 random nodes as the starting seeds and save all results into a simple dictionary
random_diffusion = DefaultDict(list)
for lamb in [1, 2]:
    for _ in tqdm(range(100000)):
        network = Investigation(g, random_catch = "lambda", lamb=lamb, first_criminal=sample(g.nodes, 2),
                compute_eigen=False, title=g.graph["name"])
        network.set_model(seed_model)
        network.set_strategy(optimal_seeding)
        network.simulate(max_criminals = 1000, max_investigations = 4)
        random_diffusion[lamb].append(len(network.caught))

with open("random_seed_results.pkl", "wb") as outfile:
    pickle.dump(random_diffusion, outfile)

#%%

with open("optimal_seed_results.pkl", "rb") as infile:
    optimal_seed_results = pickle.load(infile)

with open("random_seed_results.pkl", "rb") as infile:
    random_diffusion = pickle.load(infile)


eigen = nx.eigenvector_centrality_numpy(g, weight=None)
network_name = g.graph["name"].split("_")
network_name = " ".join([s.capitalize() for s in network_name])

#Print statement
for lamb in [1, 2]:
    optim = optimal_seed_results[(optimal_seed_results["Lambda"] == 1) & \
        (optimal_seed_results["Diffusion"] == optimal_seed_results["Diffusion"].max())]
    res = optim.loc[optim.index[0],:].to_list()
    print(f"""For Lambda = {lamb}
    The optimal seeds are {res[1]} and {res[2]} 
    with eigenvector centralities of {round(eigen[res[1]], 2)} and {round(eigen[res[2]], 2)} respectively
    and diffusion of {round(res[3], 2)} over three periods of diffusion. \n
    Whereas using random seeding we have an expected diffusion of {np.mean(random_diffusion[lamb])}
    and variance of {round(np.var(random_diffusion[lamb]), 2)}""")


fig, ax = plt.subplots(figsize = (14, 14))
ax.scatter(eigen.keys(), eigen.values(), color = "black")
ax.scatter(x = [res[1], res[2]], 
    y = [eigen[res[1]], eigen[res[2]]],
    color="red", marker="o", label="Optimal Seeds")
ax.legend()
ax.set_xlabel("Seeds")
ax.set_ylabel("Eigenvector Centrality")
ax.set_title(f"Eigenvector Centrality of Nodes in {network_name} (Unweighted)", fontsize=20)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

os.makedirs("figs", exist_ok=True)
fig.savefig(os.path.join("figs", f"{g.graph['name']}_eigens.png"), facecolor="white", transparent=False, dpi=300)
#%%
node_colors = ["blue" for node in g.nodes]
node_colors[int(res[1])] = "red"
node_colors[int(res[2])] = "red"
fig2, ax2 = plt.subplots(figsize=(14, 14))
nx.draw_spring(g, ax=ax2, with_labels=True, node_color=node_colors)
ax2.set_axis_off()
ax2.set_title(network_name, fontsize=20)
fig2.savefig(os.path.join("figs", f"{g.graph['name']}_plot.png"), facecolor="white", transparent=False, dpi=300)
# %%
