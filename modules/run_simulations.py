# %%
from os.path import join, abspath, dirname, isfile
from os import listdir

import sys
import json
import random
from pathlib import Path
root_path = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, root_path)

import pickle as pkl
import pandas as pd

from random import seed
from numpy import genfromtxt

from modules.helpers.utils import path_decompose
from modules.Simulator import *


# %%
DS_PATH = path_decompose("H:\\Research\\NIGraphs\\withPasive")
SEED_PATH = path_decompose("H:\\Research\\NIGraphs\\seeds")
path_to_json = join(root_path, "results")

# %%
simmulations_to_run = 10
threads_to_use = 22
CURRENT_SEED = 2024

# %%
seed(CURRENT_SEED)

# %%
all_graphs = [f for f in listdir(DS_PATH) if isfile(join(DS_PATH, f))]
all_seeds = [f for f in listdir(SEED_PATH) if isfile(join(SEED_PATH, f))]

# %%
gaph_data = {}
print("Building graph data for", len(all_graphs), "graphs and", len(all_seeds), "seeds.")

for graph_name in all_graphs:
    graph_config = graph_name.split("fromData")[0].split("]")
    num_nodes = int(graph_config[0].replace("[", ""))
    num_edges= int(graph_config[1].replace("[", ""))

    # Read Graph Data
    graph_path = join(DS_PATH, graph_name)

    gaph_data[f"{num_nodes}_{num_edges}"] = pkl.load(open(graph_path, 'rb'))
print("Loaded Graphs")
# %%
seed_data = {}

for seed_name in all_seeds:
        seed_tokens = seed_name.split("fromData")
        graph_config = seed_tokens[0].split("]")
        num_nodes = int(graph_config[0].replace("[", ""))
        num_edges= int(graph_config[1].replace("[", ""))
        percentile = seed_tokens[1].split("_")[1]

        # Read Graph Data
        seed_nodes = genfromtxt(join(SEED_PATH, seed_name), delimiter=",")

        seed_data[f"{num_nodes}_{num_edges}_{percentile}"] = seed_nodes
print("Loaded Seeds")
# %%
configurations = {}
blocked_nodes = {}
sim_config = pd.read_csv(join(path_to_json, 'simulationData.csv'))
sim_blocked = pd.read_csv(join(path_to_json, 'simulationNodes.csv'))

for index, row in sim_config.iterrows():
    num_nodes = row["num_nodes"]
    num_edges = row["num_edges"]
    percentile = row["percentile_name"]
    id = row["id"]

    composed_key = (f"{num_nodes}_{num_edges}_{percentile}", f"{num_nodes}_{num_edges}")

    if not composed_key in configurations:
        configurations[composed_key] = []

    configurations[composed_key].append(id)    

for index, row in sim_blocked.iterrows():
    blocked_nodes[row['id']] = [int(x) for x in row['nodes'][1:-2].split(",")]
print("Created Configuraions")

# %%
randomized_order = list(configurations.items())
random.shuffle(randomized_order)
for composed_key, ids in randomized_order:
    (seed_key, graph_key) = composed_key
    sim_name = join(path_to_json, "simulations", f"sim_{seed_key}.json")
    if isfile(sim_name):
        print("Skipping:", sim_name)
        continue
         
    Path(sim_name).touch()
    sim_results = []
    G = gaph_data[graph_key]
    graph_seed = seed_data[seed_key]

    sim = Simulator(G, graph_seed)
    print(f"Simulating {len(ids)} setups on {seed_key}")
    for id in ids:
        sim.add_blocked(id, blocked_nodes[id])
        results = sim.run(simmulations_to_run, threads_to_use)

        sim_results.append(results)    

    with open(sim_name, 'w') as f:
        json.dump(sim_results, f)



