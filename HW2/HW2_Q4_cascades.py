#%%
import igraph as ig
import numpy as np
import os, matplotlib, pickle, itertools
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
    for i, j in combinations(range(len(z)), 2):
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

def infect(graph, p):
    t = 0
    graph.es["used"] = np.zeros_like(graph.es) #0: unused, #1: used
    #List of infected
    infected = []
    #One time infecting stage
    infecting = list(np.random.randint(len(graph.vs), size=1))
    
    while len(infecting) > 0:
        #Take an infecting person
        infect = infecting.pop()

        #Get all incident edges that are unused
        links = [i for i in graph.incident(infect) if not graph.es["used"][i]]

        for link in links:
            #Set link to used
            graph.es["used"][link] = 1

            #If transmission successful
            if np.random.uniform() < p:
                i, j = graph.es[link].source, graph.es[link].target

                #Use source or target that is not the origin and not already infected and add to infecting
                if i != infect and i not in infected and i not in infecting:
                    infecting.append(i)
                elif j != infect and j not in infected and j not in infecting:
                    infecting.append(j)

        #Uptick period and add move infecting to infected
        t += 1
        infected.append(infect)
    return infected, t

def pandemic_simulation(n, d, e, c, prob_infection, sims):
    sizes = []
    lengths = []
    for _ in range(sims):
        p = (2*d + e)/(2*n)
        q = (2*d - e)/(2*n)
        g = gen_planted_partition(n=n, p=p, q=q, c=c)
        infected, t = infect(g, prob_infection)
        sizes.append(len(infected)/len(g.vs))
        lengths.append(t)
    return (np.mean(sizes), np.mean(lengths))


#%%
#part(a)
d=10
n=100
c=2
epsilon = [0,4,8]
colors = [f"tab:{color}" for color in ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]]
figs_a = []

for e in epsilon:
    p = (2*d + e)/(2*n)
    q = (2*d - e)/(2*n)
    planted_partition = gen_planted_partition(n=n, p=p, q=q, c=c)
    planted_partition.layout_fruchterman_reingold()
    fig = plot_planted_partition(graph=planted_partition, 
        partition=planted_partition.vs["community"], 
        colors=colors, 
        title=f"Planted Partition with p={round(p, 2)}, q={round(q, 2)}, e={e} and <k>=d={d}")
    figs_a.append(fig)

#%%
#part(b)
e = 0
p_range = np.arange(0, 1, 0.05) #Chance of infection along an edge
n=1000
c=2
d=8
sims = 100

stats = [pandemic_simulation(n, d, e, c, prob_infection, sims) for prob_infection in tqdm(p_range)]
epidemic_sizes, epidemic_lengths = zip(*stats)
        
    
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

# #part(c)
n=200
p_range = np.arange(0,1, 0.05)
sims = 100
c=2
d=8
e_range= range(0,17,2)

sim_results = [[pandemic_simulation(n, d, e, c, prob_infection=p, sims=sims) for p in p_range] for e in tqdm(e_range)]
with open ("sim_results.pkl", "wb") as outfile:
    pickle.dump(sim_results, outfile)

fig_c_size, ax_c_size = plt.subplots()
ax_c_size.set_xlabel("Probability of Transmission (p)")
ax_c_size.set_ylabel("Epidemic Size (s/n)")

fig_c_length, ax_c_length = plt.subplots()
ax_c_length.set_xlabel("Probability of Transmission (p)")
ax_c_length.set_ylabel("Epidemic Length (l)")

for i, e in enumerate(e_range):
    sizes, lengths = zip(*sim_results[i])
    ax_c_size.plot(p_range, sizes, color=colors[i], label=f"e={e}")
    ax_c_length.plot(p_range, lengths, color=colors[i], label=f"e={e}")

ax_c_size.legend()
ax_c_length.legend()

# %%
savepath = os.path.join("q4", "figs")
os.makedirs(savepath, exist_ok=True)

all_figs = figs_a + [fig_b_length, fig_b_size, fig_c_length, fig_c_size]
fig_names = [f"HW2_Q4_{name}.png" for name in ["a1", "a2", "a3", "b_length", "b_size", "c_length", "c_size"]]

for name, fig in zip(fig_names, all_figs):
    fig.savefig(os.path.join(savepath, name), facecolor="white", transparent=False)
# %%
