from os import system, linesep, sep
from os.path import join, abspath, exists

from joblib import Parallel, delayed
from modules import run_solver

RUNNER_PATH = abspath(run_solver.__file__)


def run_against_config(results_path, nodes_to_cut, algorithm_name,
                       graph_path, seed_path, just_solve):
    config_name = seed_path.split(sep)[-1].split(".")[0]
    out_name = f"result_{algorithm_name}_{nodes_to_cut}_{config_name}.json"
    output_name = join(results_path, out_name)

    # If the output already exists we can skip
    if exists(output_name):
        print("Skipping:", out_name)
        return None

    script_to_run = f"python {RUNNER_PATH} {graph_path} {seed_path}"
    arguments = f"{nodes_to_cut} {algorithm_name}"
    options = f"--outfile {output_name} --just_solve {1 if just_solve else 0}"
    full_command = f"{script_to_run} {arguments} {options}"

    with open(join(results_path, "commands.txt"), "a+") as myfile:
        myfile.write(full_command + linesep)

    print(full_command)
    system(full_command)


def run_solver_against_configs(results_path=None,
                               graph_file=None,
                               seed_file=None,
                               startNumber=10,
                               endNumber=1000,
                               step=10,
                               algorithms_to_run=[],
                               just_solve=True,
                               num_threads=1):
    if (results_path == None) or (graph_file == None) or (seed_file == None):
        print("Missing Input Files")
        return None

    if len(algorithms_to_run) == 0:
        print("Nothing to be done")
        return None
    
    configs = list([(nodes_to_cut, algorithm_name) for nodes_to_cut in range(
        startNumber, endNumber, step) for algorithm_name in algorithms_to_run])

    Parallel(n_jobs=num_threads)(
        delayed(run_against_config)(results_path,
                                    nodes_to_cut,
                                    algorithm_name,
                                    graph_file,
                                    seed_file,
                                    just_solve)
        for (nodes_to_cut, algorithm_name) in configs)
