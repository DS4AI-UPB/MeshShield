from git import Repo
from os.path import join
from copy import deepcopy

import scipy.sparse as sp


def get_seed_for_graph(graph_config, seed_path, node_percentage):
    return join(seed_path, f"{graph_config}_{node_percentage}_seeds.csv")

def get_simple_seed_for_graph(seed_folder, graph_name):
    return join(seed_folder, f"{graph_name}.seeds.txt")


def get_subgraph(G, seed, other_seeds):
    neighbors = None
    if sp.issparse(G):
        # Assuming G is a CSR matrix
        # Get neighbors of node 'i'
        start = G.indptr[seed]
        end = G.indptr[seed+1]
        neighbors = G.indices[start:end]
    else:
        neighbors = G.neighbors(seed)
    node_adj = list([x for x in neighbors
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
