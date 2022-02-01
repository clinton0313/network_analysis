#%%
import os, matplotlib
from igraph import Graph
import igraph as ig
import numpy as np
from itertools import combinations
from typing import Iterable
import matplotlib.pyplot as plt
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%
def get_group_degree(group, partition, edge_list):
    k = 0
    for i, j in edge_list:
        if partition[i] == group or partition[j] == group:
            k += 1
    return k

def log_likelihood(edge_list, partition:list, mode="DCSBM", directed=False):
    '''
    Computes the log likelihood
    Args:
        edge_list: list of edges
        partition: valid partition (iterable)
        mode: "SBM" or "DCSBM" (string)
    Returns:
        Log likelihood
    '''
    groups = set(partition)
    likelihoods = []
    if directed:
        m = len(edge_list)
    elif directed == False:
        m = 2 * len(edge_list)
    for u, v in combinations(groups, 2):
        N = partition.count(u) * partition.count(v)
        E = 0
        for i, j in edge_list:
            if (partition[i] == u and partition[j] == v) or (partition[i] == v and partition[j] == u):
                E +=1
        if mode == "SBM":
            l = E * np.log(E/N) + (N - E) * np.log((N-E)/N)
        elif mode == "DCSBM":
            ku = get_group_degree(u, partition, edge_list)
            kv = get_group_degree(v, partition, edge_list)
            if E == 0:
                l = 0
            else:
                l = (E/(m)) * np.log((E/(m))/((ku * kv)/(m**2)))
        likelihoods.append(l)
    return np.sum(likelihoods)
            
def makeAMove(graph:Graph, z:list, c:int, is_frozen:list, mode:str, directed=False):
    '''
    One step of a phase in the greedy heuristic.
    Args:
        graph: igraph Graph object
        z: Valid partition (iterable)
        c: Number of groups (int)
        is_frozen: Boolean list
        mode: "SBM" or "DCSBM" (string)
    Returns:
        Tuple of (node index to be frozen, z*, L*)
    '''
    #Store pairs of (partition, likelihood) for each node's best move
    likelihoods = []
    for i, node_group in enumerate(z):
        if is_frozen[i] == False:
            #Store (partition, likelihood) pair for each possible move for a node
            node_likelihoods = []
            for group in range(c):
                if node_group != group:
                    z_temp = z.copy()
                    z_temp[i] = group
                    node_likelihoods.append((z_temp, 
                        log_likelihood(graph.get_edgelist(), z_temp, mode, directed=directed)))
            likelihoods.append(max(node_likelihoods, key=lambda x: x[1]))
    
    z_news, likelihood_news = zip(*likelihoods)
    likelihoods = zip(range(len(z_news)), z_news, likelihood_news)
    return max(likelihoods, key=lambda x: x[2])

def runOnePhase(graph:Graph, z:list, c:int, mode:str, directed=False):
    '''
    Runs one phase of the greedy heuristic. 
    Args:
        graph: igraph Graph object
        z: Valid partition (iterable)
        c: Number of groups (int)
        mode: Either SBM or DCSBM (string)
    Returns:
        z*, L*, Likelihood list, h (binary halting criteria)
    '''
    n = len(z)
    is_frozen = [False for _ in range(n)]
    l0 = log_likelihood(g.get_edgelist(), z, mode) #Should I have this initial likelihood?
    likelihood_list = []
    h = 0

    for t in range(n):
        node_index, z_i, likelihood = makeAMove(graph, z, c, is_frozen, mode = mode, directed=directed)
        is_frozen[node_index] = True
        if likelihood_list == []:
            z_star = z_i
        elif likelihood > np.max(likelihood_list):
            z_star = z_i
        likelihood_list.append(likelihood)
    
    #Check if we use z_star or z:
    if np.max(likelihood_list) <= l0:
        h = 1
        z_star = z
    return z_star, np.max(likelihood_list), likelihood_list, h
    
def fitDCSBM(graph:Graph, c:int, T:int, mode = "DCSBM", directed=False):
    '''

    Args:
        graph: igraph Graph object
        c: Number of groups (int)
        T: Total number of allowable phases (int)
        mode: "SBM" or "DCSBM" (default) (string)
    Returns:
        z*, L*, number of phases taken, List of log-likelihoods
    '''

    z = list(np.random.randint(c, size=len(graph.vs)))
    z_star = z.copy()
    L_star = 0
    L_list = []
    for p in range(T):
        z, likelihood, likelihood_list, h = runOnePhase(graph, z, c, mode=mode, directed=directed)
        L_list.extend(likelihood_list)
        if likelihood > L_star:
            L_star = likelihood
            z_star = z.copy()
        if h:
            print("Halting criterion met! Terminating DCSBM.")
            return z_star, L_star, p+1, L_list
    return z_star, L_star, T, L_list 

def init_q2_graph(graph:Graph, c:int, colors:list):
    z_init = list(np.random.randint(3, size=len(g.vs)))
    l_orig = log_likelihood(g.get_edgelist(), z_init)
    fig, ax = plt.subplots()
    ig.plot(g, vertex_color = [colors[i] for i in g.vs["community"]], target=ax)
    ax.text(0.6, 1.5, s=f"Log-Likelihood: \n {round(l_orig, 3)}")
    ax.set_axis_off()
    return fig, z_init

def plot_dcsbm(g, z_star, l_star, colors):
    fig, ax = plt.subplots()
    ig.plot(g, 
    vertex_label=z_star, 
    vertex_color=[colors[i] for i in z_star],
    target=ax)
    ax.text(0.6, 1.5, s=f"Log-Likelihood: \n {round(l_star, 3)}")
    ax.set_axis_off()
    return fig

def plot_log_iterations(iterations, l_list):
    fig, ax = plt.subplots()
    ax.plot(range(iterations), l_list)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig            

#%%
# #Q2 graph:

edge_list = [(0, 1), (0, 7), (0, 8), (0, 6), (0, 5), (0, 2), (5, 2), (5, 3), (5, 4), (2, 3)]
z = [1, 1, 0, 0, 0, 0, 1, 1, 1]
g = Graph(edges=edge_list, directed=False)
g.vs["community"] = z
colors = ["blue", "red", "green"]

#%%
savepath = os.path.join("q3", "figs")
os.makedirs(savepath, exist_ok=True)
matplotlib.style.use("seaborn-bright")

#Part (a)

fig_a1, z_inita = init_q2_graph(g, c=3, colors=colors)

is_frozen_init = [False for _ in range(len(g.vs))]

_, z_star, l_star = makeAMove(g, z=z_inita, c=3, is_frozen=is_frozen_init, mode="DCSBM")

fig_a2 = plot_dcsbm(g, z_star, l_star, colors)

# %%

#part b

fig_b1, z_initb = init_q2_graph(g, c=3, colors=colors)

z_starb, l_starb, l_listb, _ = runOnePhase(g, z=z_initb, c=3, mode="DCSBM")

fig_b2 = plot_dcsbm(g, z_starb, l_starb, colors)

fig_b3 = plot_log_iterations(len(l_listb), l_listb)

# %%
#part (c)

z_starc, l_starc, p, l_listc = fitDCSBM(g, c=3, T=30)

fig_c1 = plot_dcsbm(g, z_starc, l_starc, colors)

max_likelihoods = []
multiplier = int(len(l_listc) / p)
for i in range(p-1):
    max_likelihoods.append(np.max(l_listc[ multiplier * i : multiplier * (i+1)]))
max_likelihoods.append(np.max(l_listc[multiplier * (p-1):]))

fig_c2 = plot_log_iterations(p, max_likelihoods)
# %%

fig_names = ["a1", "a2", "b1", "b2", "b3", "c1", "c2"]
figs = [fig_a1, fig_a2, fig_b1, fig_b2, fig_b3, fig_c1, fig_c2]
for name, fig in zip(fig_names, figs):
    fig.savefig(os.path.join(savepath, f"Q3_{name}.png"), facecolor="white", transparent=False)
#%%

#part d

#%%
def karate_search(max_phases, max_iterations):
    g = Graph()
    karate = g.Read_GraphML("zachary.graphml")
    karate_partition = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    karate_partition_inv = np.abs(1 - karate_partition)
    for i in tqdm(range(max_iterations)):
        z_d, l_d, _, _ = fitDCSBM(karate, c=2, T=max_phases)
        if z_d == list(karate_partition) or z_d == list(karate_partition_inv):
            print(f"Found the karate partition in {i+1} runs!")
            fig, _ = plot_dcsbm(karate, z_d, l_d, ["blue", "red"])
            fig.show()
            return z_d
    print(f"Did not find the correct partition even after {max_iterations} iterations!")
    return z_d
# %%
g = Graph()
karate = g.Read_GraphML("zachary.graphml")
karate_partition = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
ig.plot(karate, vertex_label=karate.vs["id"], vertex_color=[colors[i] for i in karate_partition])
# %%
z_d = karate_search(300, 300)
ig.plot(karate, vertex_label=karate.vs["id"], vertex_color=[colors[i] for i in z_d])
# %%