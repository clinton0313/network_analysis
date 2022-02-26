#%%
import pickle, os
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from investigation import Investigation
from strategy_evaluation import inverse_eigen_probas
from models_and_strategies import constant_model, greedy, naive_random

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
    save_every:int, dirpath:str, filename:str, verbose=False): #NEEDS TESTING
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
    for sim in tqdm(range(sims)):
        investigation.reset()
        investigation.simulate(max_criminals=max_criminals, max_investigations=max_investigations)
        results.append(investigation.log)
        if sim % save_every == 0:
            try:
                update_log(os.path.join(dirpath, filename), result=results, verbose=False)
            except FileNotFoundError:
                create_log(dirpath=dirpath, filename=filename, result=results)
            results = []

    investigation.reset()


#%%

with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    graphs = pickle.load(infile)

graphs.pop(6) #omit paul_revere set. 

