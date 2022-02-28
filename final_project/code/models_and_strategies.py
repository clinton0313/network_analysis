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
import math


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

def get_investigations(graph:nx.Graph, suspects:list) -> dict:
    '''Returns a list with number of investigations for each suspect.'''
    investigations = nx.get_node_attributes(graph, "investigations")
    investigations_suspect = {suspect: investigations[suspect] for suspect in suspects}
    return list(investigations_suspect.values())

def get_probas(graph:nx.Graph, suspects:list) -> dict:
    '''Returns a list with starting catch probabilities for each suspect.'''
    probas = inverse_eigen_probas(graph, weighted=True)
    catch_probas = {suspect: probas[suspect] for suspect in suspects}
    return list(catch_probas.values())

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

def constant_model(graph:nx.Graph, c:float, weighted:bool=True):
    '''Each informative link adds a constant amount of information'''

    assert 0 <= c <= 1, f"c needs to be a valid probability, instead got {c}"

    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    suspect_proba = {suspect : min(1, c * info) for suspect, info in zip(suspects, information)}
    return suspect_proba

def plus_minus_model(graph:nx.Graph, plus:float, minus:float, weighted:bool=True):
    '''Probability of being caught chnages proportionally in number of degrees that are suspects (+)
    and number of unsuccesssful investigations made (-)'''
    suspects = get_suspects(graph)
    information = get_information(graph, suspects, weighted)
    investigations = get_investigations(graph, suspects)
    catch_probas = get_probas(graph, suspects)
    suspect_proba = {suspect : min(1, proba * ((1+plus)**info) * (minus**invest)) for suspect, proba, info, invest in zip(suspects, catch_probas, information, investigations)}
    return suspect_proba


#STRATEGIES

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


def greedy(graph:nx.Graph, tiebreaker = "random", weighted:bool = True):
    '''Greedy search and break ties by maximum diameter of caught criminals
    Args:
        graph: Graph of the current investigation.
        tiebreaker: Accepts "random", "eigenvector", "diameter", "triangles". 
            random: Breaks ties at random.
            eigenvector: Breaks tie wiht highest perceived eigenvector centrality.
            diameter: Breaks tie by the maximum diameter of the graph of criminals if the considered suspect is added.
            triangles: Breaks tie by the maximum number of triangles of the graph of criminals if the considered suspect is added.
        weighted: Consider the graph weighted or not.
    Returns:
        Suspect index.
    '''
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
        try:
            tiebreak_dict = nx.eigenvector_centrality_numpy(graph, weight=lambda _: "weight" if weighted else None)
        except TypeError:
            tiebreak_dict = nx.eigenvector_centrality(graph, weight=lambda _: "weight" if weighted else None)
    elif tiebreaker == "triangles":
        tiebreak_dict = nx.triangles(graph)

    tiebreak_dict = {candidate: tiebreak_dict[candidate] for candidate in candidate_suspects}
    final_candidates = [candidate for candidate, score in tiebreak_dict.items() 
        if np.abs(score - max(tiebreak_dict.values())) <= 1e-4]

    #Random selection if still multiple
    return choice(final_candidates)

def naive_random(graph:nx.Graph):
    '''Completely random choice of suspects'''
    suspects = get_suspects(graph)
    return choice(suspects)


#%%

#FOR TESTING
def inverse_eigen_probas(graph:nx.Graph, min_proba:float= 0.025, max_proba:float=0.075, weighted:bool=True) -> dict:
    '''
    Creates base probabilities based on eigenvector centralities. The lower the eigenvector centrality, the 
    higher the base probability. 
    
    Args:
        graph: NetworkX Graph.
        min_proba: Lower bound of base probabiltiies that eigenvector centralities are scaled to.
        max_proba: Upper bound of base probabilities that eigenvector cenralities are sclaed to.
        weighted: Consider graph as weighted.
    Returns:
        Dictionary of base probabilities {node: proba}    
    '''
    assert 0 <= max_proba <= 1, f"max_proba needs to be a valid probability, instead got {max_proba}"
    assert 0 <= min_proba <= 1, f"min_proba needs to be a valid probaiblity, instead got {min_proba}"
    #Get nodes and eigenvector centralities
    nodes = list(graph.nodes)
    try:
        eigens = nx.eigenvector_centrality_numpy(graph, weight= lambda _: "weight" if weighted else None)
    except TypeError:
        eigens = nx.eigenvector_centrality(graph, weight=lambda _: "weight" if weighted else None)
    
    #Scale eigenvector centralities
    scale = (max_proba - min_proba) / (max(eigens.values()) - min(eigens.values()))
    scaled_eigens = {
       s : max_proba - (eigen - min(eigens.values())) * scale
       for s, eigen in eigens.items()
       }
    # scaled_eigens = {s: -(2 * (eigen - min(eigens.values()))/(max(eigens.values()) - min(eigens.values())) - 1) for s, eigen in eigens.items()}
    # scaled_eigens = {s: 1/(1 + math.exp(-eigen)) for s, eigen in scaled_eigens.items()}
    base_proba = {node : scaled_eigens[node] for node in nodes}
    return base_proba


with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    graphs = pickle.load(infile)
graph_names = [(i, g.graph["name"]) for i, g in enumerate(graphs)]
g = graphs[4]
random_catch = {node: float(np.random.normal(0.05, 0.01, 1)) for node in g.nodes}
inv = Investigation(g, random_catch=inverse_eigen_probas(g, 0.05, 0.5))
inv.set_model(plus_minus_model, plus = 0.1, minus = 0.5)
inv.set_strategy(greedy, tiebreaker="eigenvector")
inv.simulate(100, 200, update_plot=True, investigation_only=False, sleep_time= 0.5)
plt.pause(10)
# %%
# %%
