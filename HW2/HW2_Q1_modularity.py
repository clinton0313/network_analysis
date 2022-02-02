#%%

import numpy as np

adj = np.array([
    [0.258, 0.016, 0.035, 0.013],
    [0.012, 0.157, 0.058, 0.019],
    [0.013, 0.023, 0.306, 0.035],
    [0.005, 0.007, 0.024, 0.016]
])

males = np.array([[0.323, 0.247, 0.377, 0.053]])
females = np.array([[0.289, 0.204, 0.423, 0.084]])

random_prob = np.matmul(np.transpose(males), females)

X = adj - random_prob
modularity = (np.trace(X))
print(f"The modularity with respect to race is {modularity}")
#Since these probabilities sum to 1 there is an assumed 1 undirected edge amongst the entire matrix
# %%
#From 1/2n * \sum(A_ij - P_ij) * Ind(c_i != c_j)