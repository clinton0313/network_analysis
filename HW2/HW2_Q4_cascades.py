#%%
import igraph as ig
import numpy as np
import os, matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

os.chdir(os.path.dirname(os.path.realpath(__file__)))
matplotlib.style.use("seaborn-bright")
#%%


def gen_planted_partition(n, p, q, c):
    z = np.random.randint(c, size=n)
    g = ig.Graph(directed=False)
    g.add_vertices(n)
    g.vs["community"] = z
    for i, j in tqdm(combinations(range(len(z)), 2), f"Generating n={n} Planted Partition", position=1):
        if z[i] == z[j] and np.random.uniform() < p:
            g.add_edge(i, j)
        elif z[i] != z[j] and np.random.uniform() < q:
            g.add_edge(i, j)
    return g

def plot_planted_partition(graph, partition, colors, title):
    fig, ax = plt.subplots()
    ig.plot(graph, target=ax, vertex_color = [colors[i] for i in partition])
    ax.set_axis_off()
    ax.set_title(title)
    return fig

#%%
#part(a)
d=2
n=100
c=10
epsilon = [0,4,8]
colors = [f"tab:{color}" for color in ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]]
figs_a = []

for e in epsilon:
    p = (2*d + e)/(2*n)
    q = (2*d - e)/(2*n)
    planted_partition = gen_planted_partition(n=n, p=p, q=q, c=c)
    fig = plot_planted_partition(graph=planted_partition, 
        partition=planted_partition.vs["community"], 
        colors=colors, 
        title=f"Planted Partition with p={round(p, 2)}, q={round(q, 2)}, e={epsilon} and <k>=d={d}")
    figs_a.append(fig)

#%%
def get_pandemic_cluster_stats(graph):
    n = len(graph.vs)
    infected = int(np.random.randint(n, size=1))
    components = graph.clusters()
    c = components.membership[infected]
    cluster = components.subgraph(c)
    return len(cluster.vs)/len(graph.vs), cluster.diameter()
    
def get_pandemic_stats(cluster):
    return len(cluster.vs), cluster.diameter()

def pandemic_simulation(n, p_range, e, c, sims):
    pandemic_sizes = []
    pandemic_lengths = []
    for p in p_range:
        q = p - e/n
        average_size = []
        average_length = []
        for _ in tqdm(range(sims), f"Sims for p={p}", position=0):
            g = gen_planted_partition(n=n, p=p, q=q, c=c)
            size, diameter = get_pandemic_cluster_stats(g)
            average_size.append(size)
            average_length.append(diameter)
        pandemic_sizes.append(np.mean(average_size))
        pandemic_lengths.append(np.mean(average_length))
    return pandemic_sizes, pandemic_lengths
#%%
#part(b)
e = 0
p_range = np.arange(0, 1, 0.05)
p = 0.05
n=100
c=8
q = p - e/n
sims = 10

#Test here
# g = gen_planted_partition(n=n, p=p, q=q, c=c)
# fig = plot_planted_partition(graph=g,
#     partition=planted_partition.vs["community"],
#     colors=colors,
#     title="")

epidemic_sizes, epidemic_lengths = pandemic_simulation(n=n, p_range=p_range, e=e, c=c, sims=sims)


#%%
fig_b_size, ax_size = plt.subplots()
ax_size.plot(p_range, epidemic_sizes)
ax_size.set_xlabel("Probability of Transmission (p)")
ax_size.set_ylabel("Epidemic Size (s/n)")

fig_b_length, ax_length = plt.subplots()
ax_length.plot(p_range, epidemic_lengths)
ax_length.set_xlabel("Probability of Transmission (p)")
ax_length.set_ylabel("Epidemic Length (l)")

#%%

#part(c)
n=200
p_range = np.arange(0,1, 0.05)
sims=10
c=8
e_range= range(0,17,2)

sim_results = []
for e in e_range:
    e_sizes, e_lengths = pandemic_simulation(n=n, p_range = p_range, e=e, c=c, sims=sims)
    sim_results.append((p_range, e_sizes, e_lengths))

fig_c_size, ax_c_size = plt.subplots()
ax_c_size.set_xlabel("Probability of Transmission (p)")
ax_c_size.set_ylabel("Epidemic Size (s/n)")
ax_c_size.legend()

fig_c_length, ax_c_length = plt.subplots()
ax_c_length.set_xlabel("Probability of Transmission (p)")
ax_c_length.set_ylabel("Epidemic Length (l)")
ax_c_length.legend()

for i, p, size, length in enumerate(sim_results):
    ax_c_size.plot(p, size, color=colors[i], label=f"e={e_range[i]}")
    ax_c_length.plot(p, length, color=colors[i], label=f"e={e_range[i]}")

# %%
