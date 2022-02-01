#%%
#Import all relevant libraries. 
#Installation instructions for graph_tool : https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib, os

import graph_tool as gt
from graph_tool.generation import random_graph
from graph_tool.topology import extract_largest_component
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%

#Define a higher order function that allows us to create a degree sample with any probability value for p1.
#Done in this way because the degree sample passed to graph_tool must call no arguments
def make_deg_sampler(p1):
    #we generate our specific three degree sampler simply by taking a uniform draw from 0 to 1 and checking if it is below p1 as specified by the question
    def three_deg_sampler():
        if np.random.rand(1) <= p1:
            return 1
        else:
            return 3
    return three_deg_sampler

#Function that generates a random configuration graph using our degree sampler and extracts the largest componenets fractional size
def gc_fractional_size(p1, n):
    #Make the degree sampler
    deg_sampler = make_deg_sampler(p1)
    #Generate the random graph using our degree sampler
    g = random_graph(n, deg_sampler, directed=False)

    #Extract the largest component, get it's size and divide by total number of nodes in the graph to get the fractional size
    fractional_size = extract_largest_component(g).num_vertices()/n
    return fractional_size

#This function simply calls gc_factional_size many times (sims) and gets and returns the mean to us.
def mean_fractional(sims, p1, n):
    fractional_sizes = [gc_fractional_size(p1, n) for _ in tqdm(range(sims), desc="Sims", position=1, color="blue")]
    return np.mean(fractional_sizes)

#%%
#Part a

#We do 100 sims and that is enough to get a good estimate
sims = 100
#Call our mean fractional function to get the mean fractional size for the model specified in the problem set
mean_frac_a = mean_fractional(sims, p1=0.6, n =10000)
print(f"The mean fractional size for the largest component is {round(mean_frac_a, 3)}")

#%%
#Part b
#We arrange all our probabilities we would like to sample. 100 sims produces a smooth enough curve
ps = np.arange(0, 1, 0.01)
sims = 100

#Get a list of the mean fractional sizes by calling our mean_fractional function
f_sizes = [mean_fractional(sims, p1 = p, n = 10000) for p in tqdm(ps, desc="Probability", position=0)]

#%%
#Plotting

#Configuration for plots and savepath
savepath = "figs"
os.makedirs("figs", exist_ok=True)
matplotlib.style.use("seaborn-bright")

#Plot the mean fractional size for varying levels of probability
fig, ax = plt.subplots()
ax.plot(ps, f_sizes)
ax.set_xlabel("Probaility of degree = 1 (p1)")
ax.set_ylabel("Fractional size of largest compoenent")
#Eyeball the phase transition and indicate it
ax.axvline(0.72, linestyle="dashed", color="red")
ax.text(x=0.75, y=0.8, s="p1=0.72", fontsize="medium")
fig.savefig(os.path.join(savepath, "frac_size.png"), facecolor="white", transparent=False)
fig.show()

# %%
