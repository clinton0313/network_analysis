# %%

# create a graph with four nodes and two edges
from pathlib import Path

import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import os

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import random
from itertools import count
import copy
from IPython import display
import time
from itertools import combinations, groupby
basedir = os.path.dirname(os.path.realpath(__file__))
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]

# %% Q4

def printer(graph, pos, initial=True, **kwargs):
    """
    Plots a graph given a set of partitions.
    """
    # get color dict
    baseline_groups = dict(graph.nodes.data("group"))
    infected_group = max(baseline_groups.values())+1
    infected = [k for k,v in dict(graph.nodes.data("infected")).items() if v]
    mapping = {k:(v if k not in infected else infected_group) for k,v in baseline_groups.items()}
    groupmap = list(mapping.values())

    colorlist = ["#9E726F","#F3AB60", "#D6B2B1", "#7c8187"]
    if len(infected) == 0:
        colorlist = colorlist[:infected_group]
    else:
        colorlist = colorlist[:infected_group+1]
        colorlist[infected_group] = "#458B00"
    cmap = colors.ListedColormap(colorlist)

    # fix positions
    labels = nx.get_node_attributes(graph, 'group') 
    pos = {k: tuple([v[0], v[1]]) for k,v in pos.items()}

    # get title labels
    tally, iters, prob = kwargs.get('tally', None), kwargs.get('iters', None), kwargs.get('prob', None)
    if tally is not None:
        ifct, lucky, unafctd = tally["infected"], tally["lucky"], tally["unaffected"]

    if initial:
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        nx.draw_networkx(graph,pos, labels=labels, node_color=groupmap, cmap=cmap)
    elif tally and prob:
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        nx.draw_networkx(graph,pos, labels=labels, node_color=groupmap, cmap=cmap)
        ax.set_title(f"[INFECTED]: {ifct}, [LUCKY]: {lucky}, [UNAFFECTED]: {unafctd}. [PROB]: {prob}", fontsize = 14)
        fig.suptitle(f"Graph at t = {iters}", fontsize = 24)
    else:
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        nx.draw_networkx(graph,pos, labels=labels, node_color=groupmap, cmap=cmap)
        fig.suptitle(f"Graph at t = {iters}", fontsize = 24)
        
    # fig.savefig(basedir / "figures" / f'fig{iters}_4b.png', bbox_inches='tight', dpi=300)


def planted_graph(n:int, c:int, q:int, eps:int, **kwargs):
    """
    Generates a planted partition model.

    Parameters
    ----------
    n: a number of nodes
    c: a number of degrees per node
    q: a number of groups
    eps: relative difference between p_in and p_out
    probs (optional): a tuple for p_in & p_out. If not provided, these are computed using eps and c. 

    Output
    ----------
    graph: an updated networkx graph
    pin, pout: outcome probabilities 
    """
    edges = combinations(range(n), 2)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    z = {no:{"group": random.choice(range(q)), "infected": False, "evaluated": False} for no in graph.nodes}
    nx.set_node_attributes(graph, z)

    probs = kwargs.get('probs', None)
    if probs is None:
        pin = (2*c + eps)/(2*n)
        pout = (2*c - eps)/(2*n)
    else:
        pin, pout = probs[0], probs[1]

    for edge in edges:
        d0 = graph.nodes[edge[0]]["group"]
        d1 = graph.nodes[edge[1]]["group"]

        if d0 == d1:
            if random.random() < pin:
                graph.add_edge(*edge)
                nx.set_edge_attributes(graph, {edge:{"groups": tuple(sorted([d0,d1]))}})

        if d0 != d1:
            if random.random() < pout:
                graph.add_edge(*edge)
                nx.set_edge_attributes(graph, {edge:{"groups": tuple(sorted([d0,d1]))}})

    # Set all created nodes to non-evaluated
    edge_eval = {no:{"evaluated": False} for no in graph.edges}
    nx.set_edge_attributes(graph, edge_eval)

    return graph, (pin, pout)


def simulation(graph, pos, prob, print_figs=False, **kwargs):
    """
    Simulates a propagation model.

    Parameters
    ----------
    graph: a networkx graph, initialized and with established edges.
    pos: a coordinate position to print graph nodes
    prob: a probability that infection is spread
    print_figs: print and save iteration-specific graph figures

    Output
    ----------
    graph: an updated networkx graph
    tally: a dictionary with counts of infected, non-infected and those never exposed
    """

    timed = kwargs.get('timed', True)

    # initialize patient zero
    patient_zero = random.choice(range(len(graph.nodes)))
    spreader(graph, patient_zero, (True, True))
    cur_infected = 1
    it = 1

    if print_figs:
        printer(graph, pos, initial=False, iters=it, tally=None)

    while True:
        
        infected_nodes = [n for n,v in graph.nodes.data() if v['infected'] == True] 

        for infected in infected_nodes:
            contact_ls = list(graph.edges(infected))

            for infecter, contact in contact_ls:
                if graph.edges[(infecter, contact)]["evaluated"] is False:
                    if random.random() < prob:
                        spreader(graph, contact, (True, True))
                    else:
                        if graph.nodes[contact]["infected"] == True:
                            spreader(graph, contact, (True, True))
                        else:
                            spreader(graph, contact, (False, True))
                    nx.set_edge_attributes(graph, {(infecter, contact): {"evaluated": True}})
                    
                    t_tally = {
                        "unaffected": n - sum(dict(graph.nodes.data("evaluated")).values()),
                        "infected": sum(dict(graph.nodes.data("infected")).values()),
                        "lucky": sum(dict(graph.nodes.data("evaluated")).values()) - sum(dict(graph.nodes.data("infected")).values()),
                        "iteration": it
                    }

                    evaluated = sum([x[2] for x in graph.edges.data("evaluated")])

                    showcase_tally(t_tally["iteration"]+1, t_tally, timed=timed, evaluated=evaluated)
        if print_figs:
            printer(graph, pos, initial=False, iters=it+1, tally=t_tally, prob=prob)

        new_infected = sum(dict(graph.nodes.data("infected")).values())

        if cur_infected == new_infected:
            break
        else:
            it += 1
            cur_infected = new_infected

    general_tally = {
        "unaffected": n - sum(dict(graph.nodes.data("evaluated")).values()),
        "infected": sum(dict(graph.nodes.data("infected")).values()),
        "lucky": sum(dict(graph.nodes.data("evaluated")).values()) - sum(dict(graph.nodes.data("infected")).values()),
        "iteration": it
    }

    return general_tally


def spreader(graph, node, outcome):
    """
    Utility function to update node attributes within a simulation.
    """
    nx.set_node_attributes(graph, {node: {"infected": outcome[0], "evaluated": outcome[1]}})


def showcase_tally(iter, tally, evaluated, **kwargs):
    """
    Utility function to print iterations.
    """
    print(
        "[ITER]: %i, [INFECTED]: %i, [LUCKY]: %i, [UNAFFECTED NODES]: %i, [EVALUATED EDGES]: %i" % (iter, tally["infected"], tally["lucky"], tally["unaffected"], evaluated)
    )
    timed = kwargs.get('timed', None)
    if timed:
        time.sleep(0.05)
    display.clear_output(wait=True) 


################################################################################
################################################################################
# a)
n = 100         # number of vertices
c = 10          # number of degrees?! make it make sense
q = 2           # number of groups?! make it make sense
eps = [0,4,8]   # difference bettween in-degrees and out-degrees

g4, _ = planted_graph(n,c,q,eps[0])

pos = dict(nx.spring_layout(g4))

printer(g4, pos)
################################################################################
################################################################################
# %% b) preliminary

g4, _ = planted_graph(100,10,2,eps[2]*2)
pos = dict(nx.spring_layout(g4))

prob = 0.2

x = simulation(g4, pos, prob, print_figs=True, timed=True)


# %% b)

lsresults = []

n = 1000
c = 8
q = 2
eps = 0

for prob in tqdm(np.arange(0,1.01,0.01)):

    mini_batch = []
    for _ in range(3):
        g4b, _ = planted_graph(n,c,q,eps)

        results = simulation(g4b, "_", prob, print_figs=False, timed=False)

        mini_batch.append(results)

    avg_results = {j: (np.min([x[j] for x in mini_batch]) if j=="unaffected" else np.max([x[j] for x in mini_batch])) for j in mini_batch[0].keys()}
    lsresults.append(tuple([avg_results, prob]))

df = pd.merge(
    pd.DataFrame([x[0] for x in lsresults]),
    pd.DataFrame([x[1] for x in lsresults], columns=["prob"]),
    left_index=True, right_index=True
)

df.set_index("prob", inplace=True)

# %%
fig, ax = plt.subplots(
    2,2,
    figsize=(12,10),
    sharex=False, 
    sharey=False, 
    gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
)

label_dict = {
    "unaffected": "Number of unaffected individuals",
    "infected": "Number of infected individuals",
    "lucky": "Number of exposed but not infected individuals",
    "iteration": "Number of iterations t until disease stabilizes"
}

color_dict = {
    "unaffected": "#9E726F",
    "infected": "#F3AB60",
    "lucky": "#D6B2B1",
    "iteration": "#7c8187"
}

for i, val in enumerate(["unaffected", "infected", "lucky", "iteration"]):
    q, mod = divmod(i,2)

    tmp = df[[val]]
    sns.lineplot(
        data=tmp,
        x=tmp.index,
        y=val,
        ax=ax[mod,q],
        color=color_dict[val],
        linewidth=3
    )

    ax[mod,q].set_title(label_dict[val])
    ax[mod,q].set_xlabel("Probability of Infection")

    fig.savefig(os.path.join(basedir, "figures", 'fig_4b.png'), bbox_inches='tight', dpi=300)

# b) Comment: The infection does not propagate across the graph up to p=0.15. After than, some instability ensues
# where the disease may be more or less succesful depending on some aribtrary noise. This stabilizes at p=0.2, 
# by which point the disease always spreads. The fraction of susceptible loses approaches 1 quickly, and stabilizes
# around it by p=0.4. The duration of the simulation varies considerably: it increases early on and exhibits large variation
# around the noisy 0.15-0.2 area. After that, it stabilizes and slowly decreases as a more contagious disease
# ensures the spread is faster. Note that after 0.2 all individuals are at some point likely to catch the disease.

# Now, I'm unsure as to anything in the lecture slides that justifies the 0.15-0.2 range. Anyone?
# %%
# %% c)

lsresults4c = []

n = 1000
c = 8
q = 2
eps = [16]

for prob in tqdm(np.arange(0,1.05,0.05)):
    for epsilon in eps:
        mini_batch = []
        for _ in range(3):
            g4b, _ = planted_graph(n,c,q,epsilon)

            results = simulation(g4b, "_", prob, print_figs=False, timed=False)

            mini_batch.append(results)

        avg_results = {j: (np.min([x[j] for x in mini_batch]) if j=="unaffected" else np.max([x[j] for x in mini_batch])) for j in mini_batch[0].keys()}
        lsresults4c.append(tuple([avg_results, prob, epsilon]))
# %%
df4c = pd.merge(
    pd.DataFrame([x[0] for x in lsresults4c]),
    pd.DataFrame([x[1] for x in lsresults4c], columns=["prob"]),
    left_index=True, right_index=True
)
df4c = df4c.merge(
    pd.DataFrame([x[2] for x in lsresults4c], columns=["epsilon"]),
    left_index=True, right_index=True
)

df4c["prob"] = df4c["prob"].round(2)
df4c.set_index("prob", inplace=True)
# %%

plot_df = pd.pivot_table(df4c, values=["infected", "iteration"],index=["prob"], columns=["epsilon"])

plot_df["infected"] = plot_df["infected"]/1000
# %%

label_dict = {
    "infected": "Number of infected individuals",
    "iteration": "Number of iterations t"
}

color_dict = {
    "infected": "#F3AB60",
    "iteration": "#7c8187"
}


fig, ax = plt.subplots(
    1,2,
    figsize=(8,4),
    sharex=False, 
    sharey=True, 
    gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
)

for i, item in tqdm(enumerate(["infected", "iteration"])):

    tmp = plot_df.loc[:, item]
    sns.heatmap(
        tmp, 
        ax=ax[i],
        annot=False,  
        cmap="flare"
    )

    ax[i].set_title(label_dict[item])

fig.savefig(basedir / "figures" / 'fig_4c.png', bbox_inches='tight', dpi=300)
# %%
# Comment: Almost identical to the previous graph.
# Comment: Epsilon does not seem to impact the shape of the epidemic. As long as the probability of infecting
# neighboring nodes is sufficiently large (ie, >.15 and optimally >.2), the strength of the community structure
# seems to play no role in dictating the shape of the epidemic. Intuitively, although separated communities
# would a priori suggest higher resilience, these communities are consequently more interconnected than a
# homogeneous graph. This follows from the fact that p_in and p_out are perfectly correlated through epsilon.
# As such, even if the transmission between communities of a disease may take longer, once a member of a
# community is affected the whole community is likely to eventually catch the disease.   