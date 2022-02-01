#%%

#Import all relevant libraries. 
#Installation instructions for graph_tool : https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions
import graph_tool as gt
from graph_tool.generation import random_graph, random_rewire
from graph_tool.topology import extract_largest_component
import os, pickle, time
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from itertools import combinations
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%

#Instantiate our variables from the problem
n_graphs = 1000
n = 100000
c = 2 * np.log(2)
m = int(c*n /2)


#From scratch implementation of a G(n,m) graph that was originally used but took too long since it still looped in python.

def gen_gnm_from_scratch(n, m):
    #Make sure you don't have too many edges
    if m >= n*(n-1)/2:
        print("Too many edges for number of nodes")
        return
    #Generate all possible edges which is n choose 2
    missing_edges = [edge for edge in combinations(range(n), 2)]

    #Randomly permute the edges
    random_edges = np.random.permutation(missing_edges)

    #Delete the original list to save memory
    del missing_edges

    #Build a graph adding n, verticies and the first m of the randomly permuted edges
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(random_edges[:m])
    return g

def gen_gnm(n:int, m:int):
    #Check that we don't call too many edges
    if m >= n*(n-1)/2:
        print("Too many edges for number of nodes")
        return
    #Instantiate graph
    g = gt.Graph(directed=False)

    #Add n vertices and an arbitrary m edges (these happen to all be self loops)
    g.add_vertex(n)
    g.add_edge_list(zip(range(m), range(m)))

    #Use graph_tools rewire function calling model ="erdos" to ensure it is rewired in a way consistent with an E-R model
    random_rewire(g, model="erdos")
    return g
#%%

#Generate a list to store componenet sizes
component_sizes = []

#A simple for loop that generates a graph and then extracts it's largest component's size for n_graph number of times
for _ in tqdm(range(n_graphs)):
    g = gen_gnm(n, m)
    component_sizes.append(extract_largest_component(g).num_vertices())


# %%

#Set the matplotlib style and a savepath for figures
savepath = "figs"
os.makedirs(savepath, exist_ok=True)
matplotlib.style.use("seaborn-bright")

#Plot the componentsize as a histogram
fig, ax = plt.subplots()
ax.hist(component_sizes, bins = 20)
ax.set_xlabel("Component Size")
ax.set_ylabel("Frequency")

#Add a line representing the analytical expected result
ax.axvline(n/2, color="red")
fig.savefig(os.path.join(savepath, "GNM_graph.png"), facecolor="white", transparent=False)


#Analytical Solution for n_G:

c = 2 * np.log(2)
#Generate x's finely from 0, 1 and then calculate our expeted value for S, the probability that a node is in the GC
xs = np.arange(0, 1, 0.00001)
ys = 1 - np.exp(-c * xs)

#Plot the ys as well as the identity line and find show their intersection at 0.5. 
fig, ax = plt.subplots()
ax.plot(xs, xs, linestyle="dashed", color="red")
ax.plot(xs, ys, color="black")
ax.plot([0.5], [0.5], 'o', color = "blue")
fig.savefig("figs/analytical.png", facecolor="white", transparent=False)

#Find their intersections which are 0 and 0.5
intersections = xs[xs==ys]

print(f"These lines intersect at {intersections[0]} and {intersections[1]}!")


# %%

#Part b

#From Scratch implementation that was also too slow 
def edge_list(n,p):
    #Generates an edge list by iterating through n(n-1) and drawing froma bernoulli to decide whether or not to include the edge
    edges = []
    for i in range(n):
        for j in range(n):
            if i !=j and np.random.rand() < p:
                edges.append([i,j])
    return edges

def gen_gnp_from_scratch(n,p):
    #Instantiate the graph and add the edge list generated using our binomial process
    g = gt.Graph()
    g.add_edge_list(edge_list(n,p))
    return g

#The faster way of doing it using graph_tool's internal process. We call a random poisson process with lambda = (n-1)p as the degree sampler and specify the "erdos" model
def gen_gnp(n,p):
    return random_graph(n, lambda: np.random.poisson((n-1)*p), directed=False, model="erdos")

#Sample for the G(n.p) model
c = 2 * np.log(2)

#We will try n nodes from 0, 10000 at 500 node intervals
ns = range(0,10000, 500)
ms = []
ems = []
for n in ns:
    p = c/(n-1)
    #Save the number of edges for 10 different graphs and take the mean
    m = [gen_gnp(n,p).num_edges() for _ in range(10)]
    ms.append(np.mean(m))

    #Save the expected value based on an analytical solution
    ems.append(n * (n-1) * p /2)

#Plot both the expected and the simulated plots. 
fig2, ax2 = plt.subplots()
ax2.plot(ns, ms)
ax2.plot(ns, ems, linestyle="dashed", color="red")
ax2.set_xlabel("Number of Nodes")
ax2.set_ylabel("Number of Edges")
fig2.savefig(os.path.join(savepath, "GNP_graph.png"), facecolor="white", transparent=False)

# %%
