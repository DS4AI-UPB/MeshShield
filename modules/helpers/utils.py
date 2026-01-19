from git import Repo
from os.path import join
from copy import deepcopy
from re import search


def get_graph_info_from_seed(seeds_name):
    pattern = "\[([0-9]+)\]\[([0-9]+)\].*_([0-9]+)percent_seeds.*"
    results = search(pattern, seeds_name)

    num_nodes = results.group(1)
    num_edges = results.group(2)
    percentage = results.group(3)

    return (num_nodes, num_edges, percentage)


def get_seed_for_graph(graph_config, seed_path, node_percentage):
    return join(seed_path, f"{graph_config}_{node_percentage}_seeds.csv")

def get_simple_seed_for_graph(seed_folder, graph_name):
    return join(seed_folder, f"{graph_name}.seeds.txt")


def get_subgraph(G, seed, other_seeds):
    node_adj = list([x for x in G.neighbors(seed)
                     if x not in other_seeds])
    multiplier = len(node_adj)

    if multiplier == 0:
        return (None, 0)

    sg = deepcopy(G.subgraph(node_adj))

    return (sg, multiplier)


def path_decompose(given_path):
    path_tokens = given_path.split('/')
    prev_path = path_tokens[0]
    if prev_path.endswith(":"):
        prev_path += "\\"
    for token in path_tokens[1:]:
        prev_path = join(prev_path, token)

    return prev_path


def get_git_root(path):
    git_repo = Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")

    return git_root
