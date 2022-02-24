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

def get_information(graph:nx.Graph, suspects:list, weighted:bool = True) -> dict:
    '''Returns a list of degree for each suspect. If weighted, it will sum the edge weights, otherwise take the number
    of edges activated incident to each suspect.'''
    return [degree for _, degree in graph.degree(suspects, weight=lambda _: "weight" if weighted else None)]

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

def constant_model(graph, c, weighted=True):
    '''Each informative link adds a constant amount of information'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect : min(1, c * info) for suspect, info in zip(suspects, information)}
    return suspect_proba

#STRATEGIES

def simple_greedy(graph:nx.Graph, weighted:bool=True):
    '''Of highest probability suspects, take a random one.'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted=weighted)
    if len(suspects) == 0:
        return "random"
    candidate_suspects = [suspect for suspect, info in zip(suspects, information) if info == max(information)]
    return choice(candidate_suspects)

def least_central_criminal(graph:nx.Graph, suspects = None, use_eigen=True, weighted = True): #SHOULD BE RETESTED IF USED
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

    #Get degrees of current caught criminals and create of the ones withe the lowest degree
    caught = get_caught_criminals(graph)
    caught_degrees = get_information(graph, caught, weighted=weighted)
    caught_degrees = {criminal: degree for criminal, degree in zip(caught, caught_degrees)}
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
        return candidates[0]
    elif len(candidates) > 1:
        if use_eigen:
            eigens = nx.eigenvector_centrality_numpy(graph, weight=lambda _: "weight" if weighted else None)
            filtered_eigens  = {s: eigen for s, eigen in eigens.items() if s in candidates}
            candidates = [cand for cand, eigen in filtered_eigens.items() if eigen == min(filtered_eigens.values())]
        suspect = choice(candidates)
        return suspect
    
    #Should try to adjust this strategy to balance these choices and maximizing capture probability due to inforomation. 
    

def uncentral_greedy(graph:nx.Graph, mode:str="eigen", weighted:bool=True): #SHOULD BE RETESTED IF USED
    '''Greedy search but break ties using least central adjacent criminal''' #Need to verify. 
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)

    if len(suspects) == 0:
        return "random", None
    max_suspects = [suspect for suspect, info in zip(suspects, information) if info == max(information)]
    if len(max_suspects) == 1:
        return max_suspects[0]
    else:
        #Get sum of all eigenvalue centralities of connected criminals for each suspect
        candidates = [(suspect, get_connected_centrality(graph, suspect, weighted, mode=mode)) \
           for suspect in max_suspects]
        
        #Filter for the minimum eigenvalue
        minimum_eigen = np.min([x[1] for x in candidates])
        final_candidates = list(filter(lambda x: True if x[1]==minimum_eigen else False, candidates))

        #Return one of the random final candidates if there is a tie
        suspect = choice(final_candidates)
        return suspect[0]

def max_diameter(graph:nx.Graph, weighted:bool=True): #Should make an adjustment for greediness sliding score between proba and diameter.
    '''Search by maximum diameter of caught criminals'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)

    #Get potential diameter if the suspect is included in the caught_criminals
    caught = get_caught_criminals(graph)
    suspected_diameter = get_potential_diam(graph, suspects, caught)
    
    #Filter for maximum diameter
    candidate_list = {suspect: info
        for suspect, info, diameter in zip(suspects, information, suspected_diameter.values()) 
        if diameter == max(suspected_diameter.values())}
    #Filter for greediest choice
    candidate_list = [suspect for suspect, info in candidate_list.items() if info == max(candidate_list.values())]
    #Random selection if still multiple
    return choice(candidate_list)

def get_potential_diam(graph, suspects, caught):
    suspected_diameter = {}
    for suspect in suspects:
        potential_graph = nx.subgraph_view(graph, filter_node = lambda x: True if x in caught or x== suspect else False)
        suspected_diameter[suspect] = nx.diameter(potential_graph)
    return suspected_diameter

def greedy_diameter(graph:nx.Graph, weighted:bool = True): #NEED TO TEST
    '''Greedy search and break ties by maximum diameter of caught criminals'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)

    #Filter for greediest choice
    candidate_suspects = [suspect for suspect, info in zip(suspects, information) if info == max(information)]

    #Get potential diameter if the suspect is included in the caught_criminals
    caught = get_caught_criminals(graph)
    suspected_diameter = get_potential_diam(graph, candidate_suspects, caught)

    #Filter for maximum diameter
    final_candidates = [suspect for suspect, diam in suspected_diameter.items() if diam == max(suspected_diameter.values())]
    
    #Random selection if still multiple
    return choice(final_candidates)

def balanced_diameter(graph:nx.Graph, alpha: float = 0.5, weighted:bool = False): #NEED TO TEST
    '''Search by weighted score between potential diameter. Diameter is already divided by tenth to normalize a bit. Alpha should be between 0 and 1. 
    Larger alpha weights diameter more (depth first) and smaller weights greediness more (breadth first)'''

    assert 0 <= alpha <= 1, f"alpha should be between 0 and 1, instead got {alpha}"

    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted
    )
    #Get potential diameter if the suspect is included in the caught_criminals and information of suspects
    caught = get_caught_criminals(graph)
    suspected_diameter = get_potential_diam(graph, suspects, caught)

    #Compute scores as weighted average between one tenth of diameter and info. Get max score canddiates
    scores = [info * (1 - alpha) + diam * alpha for diam, info in zip(suspected_diameter.values(), information)]
    candidate_list = [suspect for suspect, score in zip(suspects, scores) if score == max(scores)]
    
    #Random selection if still multiple
    return choice(candidate_list)


def greedy(graph:nx.Graph, tiebreaker = "random", weighted:bool = True): #NEEDS TO BE TESTED - SOMOETHING DOESNT WORK TRIANGLES AT LEAST
    '''Greedy search and break ties by maximum diameter of caught criminals'''
    assert tiebreaker in ["random", "eigenvector", "diameter", "triangles"], \
        f"Invalid tiebreaker strategy. Got {tiebreaker} and expected one of: random, eigenvector, triangles"

    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)

    #Filter for greediest choice
    candidate_suspects = [suspect for suspect, info in zip(suspects, information) if info == max(information)]

    #Get potential diameter if the suspect is included in the caught_criminals
    if tiebreaker == "random":
        return choice(candidate_suspects)
    elif tiebreaker=="diameter":
        caught = get_caught_criminals(graph)
        tiebreak_dict = get_potential_diam(graph, candidate_suspects, caught)
    elif tiebreaker == "eigenvector":
        tiebreak_dict = nx.eigenvector_centrality_numpy(graph, weight=lambda _: "weight" if weighted else None)
    elif tiebreaker == "triangles":
        tiebreak_dict = nx.triangles(graph)

    final_candidates = [suspect for suspect, score in tiebreak_dict.items() if score == max(tiebreak_dict.values())]
    #Random selection if still multiple
    return choice(final_candidates)

def naive_random(graph:nx.Graph):
    '''Completely random choice of suspects'''
    suspects = get_suspects(graph)
    return choice(suspects)


#%%

#FOR TESTING

# with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
#     graphs = pickle.load(infile)
# graph_names = [(i, g.graph["name"]) for i, g in enumerate(graphs)]
# g = graphs[11]
# random_catch = {node: float(np.random.normal(0.05, 0.01, 1)) for node in g.nodes}
# inv = Investigation(g, random_catch=random_catch)
# inv.set_model(constant_model, c = 0.05)
# inv.set_strategy(greedy, tiebreaker="triangles")
# inv.simulate(20, 200, update_plot=True, investigation_only=False, sleep_time= 0.5)
# plt.pause(10)
# %%
