#%%
from investigation import Investigation
import pandas as pd
import os, pickle, matplotlib
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from time import sleep
from random import choice

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))
#%%
#HELPER FUNCTIONS

def get_suspects(graph:nx.Graph) -> list:
    '''Returns a list of suspects in the graph'''
    return [suspect for suspect in graph.nodes if graph.nodes[suspect].get("suspected")]

def get_caught_criminals(graph:nx.Graph) -> list:
    '''Returns a list of caught criminals in the graph'''
    return [caught for caught in graph.nodes if graph.nodes[caught].get("caught")]

def get_suspect_proba(graph:nx.Graph, suspects:list) -> list:
    '''Returns a list of catching probabilities for each of the suspects'''
    return [graph.nodes[suspect]["catch_proba"] for suspect in suspects]

def get_information(graph:nx.Graph, suspects:list, weighted:bool = True) -> dict:
    '''Returns a dictionaryof information links for each suspect. If weighted, it will sum the edge weights, otherwise take the number
    of edges activated incident to each suspect.'''
    information = {}
    for suspect in suspects:
        incident = [graph[i][j]["weight"] for i, j in graph.edges(suspect) if graph[i][j]["informed"]]
        if weighted:    
            information[suspect] = sum(incident)
        elif not weighted:
            information[suspect] = len(incident)
    return information

def get_nearby_suspects(graph:nx.Graph, centroids: list):
    '''Gets all neighboring suspects of centroids'''
    candidates = []
    for center in centroids:
        candidate_suspects = [suspect for suspect in graph.neighbors(center) \
            if graph.nodes[suspect].get("suspected") and graph[center][suspect]["informed"]]
        candidates.extend(candidate_suspects)
    return candidates

def get_connected_centrality(graph, suspect, weighted, mode="eigen"):
    '''For the suspect return the sum of all connected criminals' eigenvector centralities'''
    if mode == "eigen":
        centrality_dict = nx.eigenvector_centrality_numpy(graph, weight=lambda _: "weight" if weighted else None)
    elif mode =="degree":
        centrality_dict = dict(graph.degree)
    connected_criminals = [linked for linked in graph.neighbors(suspect) \
        if graph.nodes[linked].get("caught") and graph[suspect][linked].get("informed")]
    centrality = [centrality_dict[connected] for connected in connected_criminals]
    return np.sum(centrality)

#MODELS

def exponential_model(graph, random_catch, lr, weighted=True):
    '''Each informative link adds exponentially more information'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect : min(1, random_catch * (1+lr) ** information[suspect] - random_catch)
        for suspect in suspects}
    return suspect_proba

def constant_model(graph, c, weighted=True):
    '''Each informative link adds a constant amount of information'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect : min(1, c * information[suspect]) for suspect in suspects}
    return suspect_proba

def decay_model(graph, random_catch, decay, weighted=True):
    '''Each informative link is exponentially less information (CDF of exponential function)'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect: min(1, (1- np.exp(-decay * information[suspect]))) for suspect in suspects}
    return suspect_proba

#STRATEGIES

def simple_greedy(graph:nx.Graph):
    '''Of highest probability suspects, take a random one.'''
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)
    if len(suspects) == 0:
        return "random", None
    max_suspect = [index for index, proba in enumerate(suspect_proba) if proba == max(suspect_proba)]
    random_max = max_suspect[np.random.randint(len(max_suspect))]
    return suspects[random_max], suspect_proba[random_max]

def least_central(graph:nx.Graph): #Incorrect strategy I was aiming for. 
    '''Of lowest degree suspects, take a random one.'''
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)
    information = get_information(graph, suspects, weighted=True)
    min_central = [index for index, info in information.items() if info == min(information.values())]
    random_min = min_central[np.random.randint(len(min_central))]
    return suspects[random_min], suspect_proba[random_min]

def least_central_criminal(graph:nx.Graph, suspects = None, use_eigen=True, weighted = True):
    '''
    Of lowest degree caught criminals. Take those suspects as candidates. 
        Args:
            graph: investigation graph.
            use_eigen: use minimum perceived eigenvector centrality to break ties between suspects
            weighted: If true, use weighted degrees.
    '''
    #Get suspect and suspect probabilities and set weight argument
    if suspects == None:
        suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)
    proba_dict = {s:p for s, p in zip(suspects, suspect_proba)}
    if weighted:
        weight = "weight"
    else:
        weight = None

    #Get degrees of current caught criminals and create of the ones withe the lowest degree
    caught = get_caught_criminals(graph)
    caught_degrees = get_information(graph, caught, weighted=weighted)
    degree_list = list(set(caught_degrees.values()))
    degree_list.sort()
    
    # Create list of candidate suspects. Check to make sure edge is informed. 
    candidates = []
    i = 0
    while candidates == []:
        least_degrees = [candidate for candidate, degree in caught_degrees.items() if degree == degree_list[i]]
        candidates = get_nearby_suspects(graph, least_degrees)
        i += 1
        assert i != len(degree_list) + 1, "Tried all degrees and did not find any suspect candidates..."
    
    #Break ties by random choice or eigenvector centrality minimum.
    if len(candidates) == 1:
        return candidates[0], proba_dict[candidates[0]]
    elif len(candidates) > 1:
        if use_eigen:
            eigens = nx.eigenvector_centrality_numpy(graph, weight=weight)
            filtered_eigens  = {s: eigen for s, eigen in eigens.items() if s in candidates}
            candidates = [cand for cand, eigen in filtered_eigens.items() if eigen == min(filtered_eigens.values())]
        suspect = choice(candidates)
        return suspect, proba_dict[suspect]
    
    #Should try to adjust this strategy to balance these choices and maximizing capture probability due to inforomation. 
    

def uncentral_greedy(graph:nx.Graph, mode:str="eigen", weighted:bool=True):
    '''Greedy search but break ties using least central adjacent criminal''' #Need to verify. 
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)

    if len(suspects) == 0:
        return "random", None
    max_suspects = {suspect:proba for suspect, proba in zip(suspects, suspect_proba) if proba == max(suspect_proba)}
    if len(max_suspects) == 1:
        return list(max_suspects.keys())[0], list(max_suspects.values())[0]
    else:
        #Get sum of all eigenvalue centralities of connected criminals for each suspect
        candidates = [(suspect, proba, get_connected_centrality(graph, suspect, weighted, mode=mode)) \
           for suspect, proba in max_suspects.items()]
        
        #Filter for the minimum eigenvalue
        minimum_eigen = np.min([x[2] for x in candidates])
        final_candidates = list(filter(lambda x: True if x[2]==minimum_eigen else False, candidates))

        #Return one of the random final candidates if there is a tie
        suspect = choice(final_candidates)
        return suspect[0], suspect[1]

def max_diameter(graph:nx.Graph): #Should make an adjustment for greediness sliding score between proba and diameter.
    '''Search by maximum diameter of caught criminals'''
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)

    #Get potential diameter if the suspect is included in the caught_criminals
    caught = get_caught_criminals(graph)

    suspected_diameter = get_potential_diam(graph, suspects, caught)
    
    #Filter for maximum diameter
    candidate_list = {suspect:proba 
        for suspect, proba, diameter in zip(suspects, suspect_proba, suspected_diameter) 
        if diameter == max(suspected_diameter)}
    #Filter for greediest choice
    candidate_list = [(suspect, proba) for suspect, proba in candidate_list.items() if proba == max(candidate_list.values())]
    #Random selection if still multiple
    candidate = choice(candidate_list)

    return candidate[0], candidate[1]

def get_potential_diam(graph, suspects, caught):
    suspected_diameter = []
    for suspect in suspects:
        potential_graph = nx.subgraph_view(graph, filter_node = lambda x: True if x in caught or x== suspect else False)
        suspected_diameter.append(nx.diameter(potential_graph))
    return suspected_diameter

def greedy_diameter(graph:nx.Graph): #NEED TO TEST
    '''Greedy search and break ties by maximum diameter of caught criminals'''
    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)

    #Filter for greediest choice
    candidate_list = [(suspect, proba) for suspect, proba in zip(suspects, suspect_proba) if proba == max(suspect_proba)]
    candidate_suspects = [suspect for suspect, _ in candidate_list] 
    #Get potential diameter if the suspect is included in the caught_criminals
    caught = get_caught_criminals(graph)
    suspected_diameter = get_potential_diam(graph, candidate_suspects, caught)

    #Filter for maximum diameter
    candidates, probas = zip(*candidate_list)
    candidate_list = [(suspect, proba) 
        for suspect, proba, diameter in zip(candidates, probas, suspected_diameter) 
        if diameter == max(suspected_diameter)]
    
    #Random selection if still multiple
    candidate = choice(candidate_list)

    return candidate[0], candidate[1]

def balanced_diameter(graph:nx.Graph, alpha: float = 0.5, weighted:bool = False): #NEED TO TEST
    '''Search by weighted score between potential diameter. Diameter is already divided by tenth to normalize a bit. Alpha should be between 0 and 1. 
    Larger alpha weights diameter more (depth first) and smaller weights greediness more (breadth first)'''

    assert 0 <= alpha <= 1, f"alpha should be between 0 and 1, instead got {alpha}"

    suspects = get_suspects(graph)
    suspect_proba = get_suspect_proba(graph, suspects)

    #Get potential diameter if the suspect is included in the caught_criminals and information of suspects
    caught = get_caught_criminals(graph)
    suspected_diameter = get_potential_diam(graph, suspects, caught)
    suspect_info = get_information(graph, suspects, weighted=weighted)

    #Compute scores as weighted average between one tenth of diameter and info. Get max score canddiates
    scores = [proba * (1 - alpha) + diam * 0.1 * alpha for diam, proba in zip(suspected_diameter, suspect_proba)]
    candidate_list = [(suspect, proba) for suspect, proba, score in zip(suspects, suspect_proba, scores) if score == max(scores)]
    
    #Random selection if still multiple
    candidate = choice(candidate_list)

    return candidate[0], candidate[1]




#%%

# #FOR TESTING

# with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
#     graphs = pickle.load(infile)
# graph_names = [(i, g.graph["name"]) for i, g in enumerate(graphs)]
# g = graphs[11]
# inv = Investigation(g)
# inv.set_model(constant_model, c = 0.05)
# inv.set_strategy(simple_greedy)
# inv.simulate(20, 200, update_plot=True, investigation_only=True, sleep_time= 0.5)
# plt.pause(10)
# %%
