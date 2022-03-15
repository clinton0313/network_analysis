#%%
import pickle, os
import numpy as np
from tqdm import tqdm
from IPython import display
import time

# from project.gitrepo.final_project.code.models_and_strategies import uncentral_greedy
# %%
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir(os.environ["NETWORKS"])

from investigation import Investigation
from strategy_evaluation import inverse_eigen_probas
from models_and_strategies import (
    constant_model, 
    greedy, 
    naive_random, 
    least_central_criminal, 
    uncentral_greedy, 
    max_diameter, 
    balanced_diameter
)

LOGPATH = "logs"
os.makedirs(LOGPATH, exist_ok=True)
#%%
#FUNCTIONS

def update_log(file, result:list, verbose=False):
    '''
    Update a log with new result.
    Args:
        file: pkl file filepath of the logfile
        result: a list of dictionaries to be appended to log file.
        verbose: If true, print a statement everytime and update is applied.
    '''
    with open(file, "rb") as infile:
        log = pickle.load(infile)
    assert isinstance(log, list), "Log file is not a list."
    log.extend(result)
    with open(file, "wb") as outfile:
        pickle.dump(log, outfile)
    if verbose:
        print(f"Log {file} updated.")

def create_log(dirpath:str, filename:str, result:list):
    '''
    Creates a new log file.
    Args:
        dirpath: Path to the directory to be saved.
        filename: Name of logfile.
        result: A list of dictionary results.
    '''
    os.makedirs(dirpath, exist_ok=True)
    file = os.path.join(dirpath, filename)
    with open(file, "wb") as outfile:
        pickle.dump(result, outfile)
    print(f"Log created for {filename}")

def log_simulations(sims:int, investigation:Investigation, max_criminals:int, max_investigations:int, 
    save_every:int, dirpath:str, filename:str, verbose=False, overwrite:bool=False, **kwargs): #NEEDS TESTING
    '''
    Collects results of simulations and saved them to file.
    Args:
        sims: Number of simulations to perform.
        investigation: An open investigation. This investigation will be reset.
        max_criminals, max_investigations: Stopping criterion for investigation simulation.
        save_every: Number of simulations to perform before saving results.
        dirpath: Directory path of logfile
        filename: Name of logfile.
        verbose: Print statement at each save. 
    '''
    results = []

    if overwrite:
        try:
            os.remove(os.path.join(dirpath, filename))
        except:
            pass
    for sim in range(sims):
        
        print("[STRATEGY]: %s, [GRAPH]: %s, [SIMUL]: %i" % (kwargs.get("strategy_name"), kwargs.get("graph_name"), sim))
        try:
            investigation.reset()
            investigation.simulate(max_criminals=max_criminals, max_investigations=max_investigations)
            log = investigation.log
            try:
                log["graph_name"] = kwargs.get("graph_name")
                log["strategy_name"] = kwargs.get("strategy_name")
                log["sim_run"] = sim
                log["num_nodes"] = kwargs.get("graph_nodes")
            except:
                pass
            results.append(log)
            if sim % save_every == 0:
                try:
                    update_log(os.path.join(dirpath, filename), result=results, verbose=False)
                except FileNotFoundError:
                    create_log(dirpath=dirpath, filename=filename, result=results)
                results = []
        except ZeroDivisionError as e:
            print("Skip due to zero-division:", e)
            pass
        except Exception as f:
            raise Exception(f)

        display.clear_output(wait=True)

    investigation.reset()


#%%

with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

graphs.pop(6) #omit paul_revere set. 

# %%
# Strategies to evaluate
strategy_candidates = {
    # "uncentral_greedy": {"object_name": uncentral_greedy, "args": {"mode": "eigen"}},
    "greedy_diameter": {"object_name": greedy, "args" : {"tiebreaker": "diameter"}},
    "greedy_triangles": {"object_name": greedy, "args" : {"tiebreaker": "triangles"}},
    # "max_diameter": {"object_name": max_diameter, "args": {}},
    # "balanced_diameter_1": {"object_name": balanced_diameter, "args": {"alpha": 0.1}},
    # "balanced_diameter_3": {"object_name": balanced_diameter, "args": {"alpha": 0.3}},
    # "balanced_diameter_5": {"object_name": balanced_diameter, "args": {"alpha": 0.5}},
    # "balanced_diameter_7": {"object_name": balanced_diameter, "args": {"alpha": 0.7}},
    # "balanced_diameter_9": {"object_name": balanced_diameter, "args": {"alpha": 0.9}}
}

for strat, params in strategy_candidates.items():
    for graph in graphs:
        
    
        # Set up investigation framework
        random_catch = {node: float(np.random.normal(0.05, 0.01, 1)) for node in graph.nodes}
        inv = Investigation(graph, random_catch=random_catch)
        inv.set_model(constant_model, c=0.05)
        inv.set_strategy(params["object_name"], **params["args"])

        # Run simulations, log results
        log_simulations(
            sims=500, 
            investigation=inv,
            max_criminals=300,
            max_investigations=1000,
            save_every=5,
            dirpath="logs",
            filename=f"{strat}_{graph.name}.pkl",
            overwrite=False,
            strategy_name=strat,
            graph_name=graph.name,
            graph_nodes=graph.number_of_nodes()
        )


# %%
