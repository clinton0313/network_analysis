# %%

# create a graph with four nodes and two edges
from pathlib import Path
import os
from tqdm import tqdm
from IPython import display

import numpy as np
import networkx as nx
import snap
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import csv
import random
import time

from itertools import combinations, groupby

basedir = Path("C:/Users/David/Dropbox/BSE/2_Networks/HW/HW1")

# # %%
# sizes = [30, 30, 30]
# probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
# g = nx.stochastic_block_model(sizes, probs, seed=0)
# len(g)
# # %%
# plt.figure(figsize=(8,5))
# nx.draw(g, node_color='lightblue', 
#         with_labels=True, 
#         node_size=500)
# # %%
# H = nx.quotient_graph(g, g.graph["partition"], relabel=True)
# # %%
# for v in H.nodes(data=True):
#     print(round(v[1]["density"], 3))
# # %%

# %% Q1
ethnicities = np.array([
    [0.258, 0.016, 0.035, 0.013],
    [0.012, 0.157, 0.058, 0.019],
    [0.013, 0.023, 0.306, 0.035],
    [0.005, 0.007, 0.024, 0.016]
])

sym_ethnicity = (ethnicities + ethnicities.T)*1000 # symmetric & rescale
np.fill_diagonal(sym_ethnicity, np.diag(sym_ethnicity)/2) # revert change in diag
sym_ethnicity = sym_ethnicity.astype(int) # to integer

# sum of nodes
m = np.sum(np.tril(sym_ethnicity))

# create "degree out" sums
d_out = sym_ethnicity.copy()
np.fill_diagonal(d_out, [0,0,0,0])
d_out = np.sum(d_out, axis=1)

Q = 0
for i in range(4):
    Q += sym_ethnicity[i,i]/m - ((sym_ethnicity[i,i]*2 + d_out[i]) / (2*m))**2

print("Q=",Q)

# %% Q2 testy
###################################################################################

# %%
Nu = 5
Nv = 4

Nuv = Nu*Nv
Nuu = math.factorial(Nu) // math.factorial(2) // math.factorial(Nu - 2)
Nvv = math.factorial(Nv) // math.factorial(2) // math.factorial(Nv - 2)

Euv = 2
Euu = 4
Evv = 4

hMuv = Euv / Nuv
hMuu = Euu / Nuu
hMvv = Evv / Nvv

# loglik = (
#     Euv * np.log(Euv/Nuv) + (Nuv - Euv) * np.log((Nuv-Euv)/Nuv)
#     # Euu * np.log(Euu) + (Nuu - Euu) * np.log(Nuu - Euu) - Nuu * np.log(Nuu) + 
#     # Euv * np.log(Evv) + (Nvv - Evv) * np.log(Nvv - Evv) - Nvv * np.log(Nvv) +
#     # 2 * (Euv * np.log(Euv) + (Nuv - Euv) * np.log(Nuv - Euv) - Nuv * np.log(Nuv))
# )

def sbm_lik(N, E):
    Nu, Nv = N[0], N[1]
    Euu, Euv, Evv = E[0], E[1], E[2]

    Nuv = Nu*Nv
    Nuu = math.factorial(Nu) // math.factorial(2) // math.factorial(Nu - 2)
    Nvv = math.factorial(Nv) // math.factorial(2) // math.factorial(Nv - 2)

    hMuv = Euv / Nuv
    hMuu = Euu / Nuu
    hMvv = Evv / Nvv
    
    hM = np.array([
        [hMuu, hMuv], [hMuv, hMvv]
    ])

    lik = (
        ((hMuv**Euv) * (1-hMuv)**(Nuv-Euv)) *
        ((hMuu**Euu) * (1-hMuu)**(Nuu-Euu)) *
        ((hMvv**Evv) * (1-hMvv)**(Nvv-Evv))
    )

    return hM, lik

N = [4,2]
E = [4,2,1]

M, lik = sbm_lik(N, E)

print("Lbad", lik)

N = [3,3]
E = [3,1,3]

M, lik = sbm_lik(N, E)

print("Lgood", lik)

# %% Q2 a)

N = [5,4]
E = [4,2,4]

M, lik = sbm_lik(N, E)

print("Q2 a) log-likelihood", np.log(lik))


# %% Q2 b)

k = [10,10]

def dcsbm_lik(N, E, k):
    Nu, Nv = N[0], N[1]
    Euu, Euv, Evv = E[0], E[1], E[2]
    ku, kv = k[0], k[1]
    m = ku + kv

    Nuv = Nu*Nv
    Nuu = math.factorial(Nu) // math.factorial(2) // math.factorial(Nu - 2)
    Nvv = math.factorial(Nv) // math.factorial(2) // math.factorial(Nv - 2)

    hMuv = Euv / Nuv
    hMuu = Euu / Nuu
    hMvv = Evv / Nvv
    
    hM = np.array([
        [hMuu, hMuv], [hMuv, hMvv]
    ])

    lik = (
        (Euv/(2*m)) * np.log( (Euv/(2*m)) / ( (ku/(2*m) * (kv/(2*m))) ) ) + 
        (Euu/(2*m)) * np.log( (Euu/(2*m)) / ( (ku/(2*m) * (ku/(2*m))) ) ) +
        (Evv/(2*m)) * np.log( (Evv/(2*m)) / ( (kv/(2*m) * (kv/(2*m))) ) ) 
    )

    return hM, lik

N = [5,4]
E = [4,2,4]
k = [10,10]

M, lik = dcsbm_lik(N, E, k)

print("Q2 b) log-likelihood", np.log(lik))

# Q2 c) The DC-SBM is more likely to  produce the observed netowrk. In networks with skewed degree distributions,
# the model tends to group vertices by degree, so that a higher probability cluster would be 1,3,6. This is because
# the model maximizes likelihood by having a small connected group have large degrees outside the cluster, so that
# they concentrate the mixing matrix non-diagonal weights in as few cells as possible.
# The DC-SBM modifies the generative model by adding propensity parameters to each vertex, which allows for an arbitrary
# dispersion of degrees within communities. Intuitively, DC-SBM seeks a partition that maximizes the information contained in the
# labels relative to a random graph with a given degree sequence. DSM seeks the same but for the ER model, and not a configuration model. 
# %%
