# %%
from tokenize import group
import pandas as pd
import pickle, os, re
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from joypy import joyplot
from tqdm import tqdm
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# %%
os.chdir(os.environ["NETWORKS"])

def qtiler(dataframe:pd.DataFrame, group_args:Union[list,str], target:str, thresh:float) -> dict:
    return (
        dataframe
        .groupby(group_args)
        .apply(lambda group: threshold(group, target, thresh))
        .set_index(["strategy_name", "graph_name", "sim_run"])["investigation"]
        .to_dict()
    )

def threshold(dataframe: pd.DataFrame, target:str, thresh:float):
    try:
        return dataframe.iloc[np.argmax(dataframe[target] >= thresh)]
    except:
        return dataframe.iloc[0] # can't think of a better solution, get investigation=1 and manually NA after

def data_compiler(directory:list) -> pd.DataFrame:
    outer_df = pd.DataFrame(
        columns=[
            "caught", "suspects", "informed", "captured_eigen", "eigen_proportion", 
            "investigation", "graph_name", "strategy_name", "sim_run"
        ]
    )
    for file in tqdm(directory):
        log = pd.read_pickle(file)
        inner_df = pd.DataFrame(
            columns=[
                "caught", "suspects", "informed", "captured_eigen", "eigen_proportion", 
                "investigation", "graph_name", "strategy_name", "sim_run"
            ]
        )
        # run through simulations
        for i in range(len(log)):
            run = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in log[i].items() ]))
            inner_df = pd.concat([inner_df, run], axis=0).reset_index(drop=True)

        inner_df[["graph_name", "strategy_name", "sim_run", "num_nodes"]] = (
            inner_df[["graph_name", "strategy_name", "sim_run", "num_nodes"]].fillna(method="ffill")
        )
        outer_df = pd.concat([outer_df, inner_df], axis=0).reset_index(drop=True)
    
    outer_df = (
        outer_df
        .assign(
            sim_run = lambda df: df.sim_run.astype(int),
            num_nodes = lambda df: df.num_nodes.astype(int),
            caught_proportion = lambda df: df.caught / df.num_nodes,
        )
        [[
            "strategy_name", "graph_name", "sim_run", "investigation",
            "caught", "suspects", "informed", "num_nodes", 
            "caught_proportion", "captured_eigen", "eigen_proportion"
        ]]
    )
    # Hacky fix, but some simulations have 1 suspect despite everyone being caught
    outer_df.loc[(outer_df.caught_proportion==1) & (outer_df.eigen_proportion!=1), "eigen_proportion"] = 1.
    return outer_df

def baseline_stats(dataframe:pd.DataFrame, vars:list, thresholds: list):
    lst = []
    data = []
    tuples = []
    for var in tqdm(vars):
        for th in thresholds:
            lst.append(
                qtiler(
                    dataframe=dataframe,
                    group_args=["strategy_name", "graph_name", "sim_run"],
                    target=var,
                    thresh=th
                )
            )
    for k in lst[0].keys():
        data.append([d[k] for d in lst])
        tuples.append(k)

    index = pd.MultiIndex.from_tuples(tuples, names=["strategy", "group", "sim"])
    columns = pd.MultiIndex.from_product([vars, thresholds])

    results = pd.DataFrame(data, index=index, columns=columns)
    # Account for artificial 1s from unfinished simulations
    results.iloc[
            :, 
            (results.columns.get_level_values(1)==0.75)|
            (results.columns.get_level_values(1)==1.0)
        ] = (
        results
        .iloc[
            :, 
            (results.columns.get_level_values(1)==0.75)|
            (results.columns.get_level_values(1)==1.0)
        ]
        .replace(1, np.nan) # inplace doesn't work?
    )
    results["unfinished"] = (
        results.apply(lambda row: np.where(any(row.isna()),1,0), axis=1)
    )
    return results


# %%

complete_data = []
for path, subdirs, files in os.walk("logs/combined"):
    for name in files:
        complete_data.append(os.path.join(path, name))

strats = [
    "naive_random",
    "greedy_random",
    "greedy_eigenvector",
    "least_central",
    # "balanced_diameter_1", 
    # "balanced_diameter_3", 
    # "balanced_diameter_5", 
    # "balanced_diameter_7", 
    # "balanced_diameter_9",
    # "greedy_diameter", 
    # "max_diameter"
]
for strat in strats:
    strat_files = [x for x in complete_data if strat in x]
    main_df = data_compiler(directory=strat_files)
    vars, thresholds = ["caught_proportion", "eigen_proportion"], [0.25, 0.5, 0.75, 1.]
    base_results = baseline_stats(main_df, vars=vars, thresholds=thresholds)
    base_results.to_pickle(f"data/simul_results/{strat}.pkl")
# %%
# Creating pickle and csv files for R
pickles = []
for path, subdirs, files in os.walk("data/simul_results"):
    for name in files:
        if ".pkl" in name:
            pickles.append(os.path.join(path, name))

base_results = []
for pickl in pickles:
    strat_results = pd.read_pickle(pickl)
    # strat_results.columns = strat_results.columns.to_flat_index()
    name = re.search('\\\(.*).pkl', pickl).group(1)
    # strat_results.to_csv(f"data/simul_results/{name}.csv")
    base_results.append(strat_results)
# %%
# Restore main results df
base_results = pd.concat(base_results)
# %% 31 march hack
# base_results["mean_simuls"] = base_results[[("caught_proportion", 1.0)]].groupby(level=["strategy", "group"]).transform("mean")
# base_results = base_results.loc[:, base_results.columns].div(base_results['mean_simuls'], axis=0)
# base_results.drop(columns=["mean_simuls"], inplace=True)
# base_results.to_csv("data/simul_results/normalized.csv")
# %%
# define basic stats
stats = (
    base_results
    .groupby(level=["strategy","group"])
    .describe()
    .loc[:, lambda df: df.columns.get_level_values(2).isin({"mean", "std"})]
)
stats.loc[:, lambda df: df.columns.get_level_values(2)=="mean"] = (
    stats.loc[:, lambda df: df.columns.get_level_values(2)=="mean"].round(2)
)
stats.loc[:, lambda df: df.columns.get_level_values(2)=="std"] = (
    stats.loc[:, lambda df: df.columns.get_level_values(2)=="std"].round(2)
)
stats["unfinished"] = base_results.groupby(level=["strategy", "group"])["unfinished"].sum()/500*100

# improve labelling for export
group_labels = {
    '17Nov_greece': '17N',
    '911_hijackers': '9/11 Hijackers', 
    'Islamic_state_group' : 'ISIL',
    'al_qaeda': 'Al Qaeda',
    'caviar': 'Caviar',
    'cocaine_acero': 'Operation Acero', 
    'cocaine_jake': 'Operation Jake', 
    'cocaine_juanes': 'Operation Juanes',
    'cocaine_mambo': 'Operation Mambo',
    'heroin_natarjan': "Heroin Natarjan",
    'italian_gangs': "Italian Gangs",
    'london_gang': "London Gangs",
    'mali_terrorists': "Mali Terrorists", 
    'montagna': "Montagna Operation", 
    'ndrangheta_mafia': "Ndrangheta", 
    'noordin_top': "Malasyian Extremists",
    'togo': "Project Togo"
}

strategy_labels = {
    "greedy_diameter": "Greedy Diameter",
    "max_diameter": "Maximum Diameter",
    "greedy_eigenvector": "Greedy Eigenvector",
    "greedy_random": "Greedy Random",
    "least_central": "Least Central",
    "naive_random": "Naive Random",
    "balanced_diameter_1": "Balanced Diameter (0.1)",
    "balanced_diameter_3": "Balanced Diameter (0.3)",
    "balanced_diameter_5": "Balanced Diameter (0.5)",
    "balanced_diameter_7": "Balanced Diameter (0.7)",
    "balanced_diameter_9": "Balanced Diameter (0.9)"
}

# rename and exprot
stats.reset_index(inplace=True)
stats["group"] = stats.group.replace(group_labels)
stats.set_index(["strategy", "group"], inplace=True)
for strat in stats.index.unique(level=0):
    stats.loc[(strat, slice(None), slice(None)),:].droplevel(0).to_latex(f"tables/{strat}.tex")
# %%
# build table that sorts models as rows, and reports average # sim for quantiles in cols
agg_results = base_results.groupby(level=["strategy"]).agg("mean").astype(int)
agg_results["unfinished"] = (base_results.groupby(level="strategy")["unfinished"].agg("mean")*100).round(2)
agg_results.loc[:,(["caught_proportion", "eigen_proportion", "unfinished"], slice(None))].to_latex("tables/model_results.tex")
# %%
base_results.reset_index(inplace=True)
# %%
for (strat, label) in strategy_labels.items():
    # Reshape dataset to feed to plot
    tmp_df = (
        base_results
        .loc[base_results.strategy==strat]
        [["group", "eigen_proportion"]]
        .assign(group=lambda df: df["group"].map(group_labels))
        .droplevel(level=0, axis=1)
    )
    tmp_df.rename(
        columns={
            "": "group",
            0.25: "First Quantile", 
            0.5: "Second Quantile",
            0.75: "Third Quantile",
            1.0: "Fourth Quantile"
        },
        inplace=True
    )

    # Create ridgeline plot
    plt.figure()
    # plt.tight_layout()
    fig, ax = joyplot(
        data=tmp_df, 
        by='group',
        column=["First Quantile", "Second Quantile", "Third Quantile", "Fourth Quantile"],
        color=["#682F2F", "#D6B2B1", "#B9C0C9", "#F3AB60"],
        legend=True,
        alpha=0.8,
        figsize=(8, 8),
        overlap=1,
        grid="both",
        title=f"Strategy: {label}"
    )
    plt.xlabel("Number of Investigations")
    plt.savefig(f"figs/simulations/long_simulations/{strat}.png", dpi=300)


# %%
# Auxiliaries
# from PIL import Image


# for img in [x for x in os.listdir("figs") if "batch" in x]:
#     im = Image.open("figs/"+img)
#     img_name = os.path.basename("figs/"+img)
#     width, height = im.size       
#     im = im.resize((int(width/4), int(height/4)), Image.ANTIALIAS)
#     # im.save(img_name)
#     #im.save('optimized_images/' + img_name, optimize=True, quality=95)
#     im.save(img_name, optimize=True, quality=60)
 

# %%

# %%
