# %%
import pandas as pd
import pickle, os
import numpy as np
from typing import Union
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
    for file in directory:
        log = pd.read_pickle(f"logs/{file}")
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
            caught_rel = lambda df: df.caught / (df.caught + df.suspects)
        )
        [[
            "strategy_name", "graph_name", "sim_run", "investigation",
            "caught", "suspects", "informed", "num_nodes", 
            "caught_proportion", "caught_rel", "captured_eigen", "eigen_proportion"
        ]]
    )
    # Hacky fix, but some simulations have 1 suspect despite everyone being caught
    outer_df.loc[(outer_df.caught_proportion==1) & (outer_df.caught_rel!=1), "caught_rel"] = 1.
    outer_df.loc[(outer_df.caught_proportion==1) & (outer_df.eigen_proportion!=1), "eigen_proportion"] = 1.
    return outer_df

def baseline_stats(dataframe:pd.DataFrame, vars:list, thresholds: list):
    lst = []
    data = []
    tuples = []
    for var in vars:
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
# build data
main_df = data_compiler(directory=os.listdir("logs"))
vars, thresholds = ["caught_proportion", "caught_rel", "eigen_proportion"], [0.25, 0.5, 0.75, 1.]
base_results = baseline_stats(main_df, vars=vars, thresholds=thresholds)

# %%
# define basic stats
stats = (
    base_results
    .groupby(level=["strategy","group"])
    .describe()
    .loc[:, lambda df: df.columns.get_level_values(2).isin({"mean", "std"})]
)

stats.loc[:, lambda df: df.columns.get_level_values(2)=="mean"] = (
    stats.loc[:, lambda df: df.columns.get_level_values(2)=="mean"].astype(int)
)

stats.loc[:, lambda df: df.columns.get_level_values(2)=="std"] = (
    stats.loc[:, lambda df: df.columns.get_level_values(2)=="std"].round(1)
)

stats.to_latex("tables/template.tex")
# %%
stats.loc[:, stats.columns.get_level_values(2).isin({"mean", "std"})]
# baser
# %%
one = main_df.loc[(main_df["graph_name"]=="al_qaeda")&(main_df["strategy_name"]=="least_central")]

# %%
# %%
# j.loc[(slice(None), "montagna", slice(None))]
# %%
# Ad-hoc addition 


# %%


# %%
