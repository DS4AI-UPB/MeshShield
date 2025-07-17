import sys
from os import listdir
from os.path import join, isfile, abspath, dirname
sys.path.insert(0, abspath(join(dirname("."), '..')))

import argparse
from copy import deepcopy
from pathlib import Path

from modules.helpers.constants import PERCENTAGES, IMMUNIZATION_PERCENTAGES
from modules.helpers.runners import run_solver_against_configs
from modules.helpers.utils import path_decompose, get_git_root, get_seed_for_graph, get_graph_info_from_seed

DS_PATH = path_decompose("H:\\Research\\NIGraphs\\withPasive")
SEED_PATH = path_decompose("H:\\Research\\NIGraphs\\seeds")

full_config = ['Random', 'Degree', 'SparseShieldPlus',
               'SparseShieldSeedless', 'Dom', 'MeshShield']
folder_root = get_git_root(".")
folder_for_results = join(path_decompose(folder_root), "results")
standard_budgets = [1, 5, 10]

sys.path.append(folder_root)

Path(folder_for_results).mkdir(parents=True, exist_ok=True)


def run_to(graph, seed, startN, endN, stepN, experiment_config, NUM_THREADS):
    run_solver_against_configs(results_path=folder_for_results,
                               graph_file=graph,
                               seed_file=seed,
                               startNumber=startN,
                               endNumber=endN,
                               step=stepN,
                               algorithms_to_run=experiment_config,
                               just_solve=True,
                               num_threads=NUM_THREADS)


def run_experiment(experiment_config, REVERSE_INPUT, REVERSE_PERCENTAGES, REVERSE_I_PERCENTAGES):
    NUM_THREADS = len(experiment_config)
    all_graphs = [f for f in listdir(DS_PATH) if isfile(join(DS_PATH, f))]

    p_to_run = deepcopy(PERCENTAGES)
    i_p_to_run = deepcopy(IMMUNIZATION_PERCENTAGES)

    if REVERSE_INPUT:
        all_graphs.reverse()

    if REVERSE_PERCENTAGES:
        p_to_run.reverse()

    if REVERSE_I_PERCENTAGES:
        i_p_to_run.reverse()

    for graph in all_graphs:
        graph_path = join(DS_PATH, graph)
        graph_name = graph.replace(".pkl", "")

        for percentage in p_to_run:
            seed_name = get_seed_for_graph(graph_name, SEED_PATH, percentage)
            (num_nodes, _, percentage) = get_graph_info_from_seed(seed_name)

            for n in standard_budgets:
                run_to(graph_path, seed_name, n, n+1, 1,
                       experiment_config, NUM_THREADS)

            for p in i_p_to_run:
                budget = int(float(p)*float(num_nodes) + 1.0)
                run_to(graph_path, seed_name, budget, budget +
                       1, 1, experiment_config, NUM_THREADS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments with a set of configurations")
    parser.add_argument("solver_name", type=str)
    parser.add_argument("-ri", "--reverse_input",
                        type=int, default=0)
    parser.add_argument("-rp", "--reverse_percentages",
                        type=int, default=0)
    parser.add_argument("-rip", "--reverse_i_percentages",
                        type=int, default=0)
    args = parser.parse_args()

    run_experiment([args.solver_name],
                   args.reverse_input == 1,
                   args.reverse_percentages == 1,
                   args.reverse_i_percentages == 1)
