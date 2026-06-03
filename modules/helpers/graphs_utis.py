import pandas as pd
import networkx as nx

import pickle
import random
import math

from pathlib import Path
from os.path import join

import statistics as stats

def load_seeds(seeds_file):
    """Load comma-separated seeds from text file."""
    with open(seeds_file, 'r') as f:
        seeds_str = f.read().strip()
    return [int(s.strip()) for s in seeds_str.split(',') if s.strip()]

def generate_experimental_seeds(graph_folder, pct=0.10, num_seeds=10, seed=None):
    """
    Generate seeds following the article's first experiment setup.
    
    For each graph: randomly select 10% of nodes (or fixed num_seeds).
    Save as comma-separated text file.
    
    Parameters
    ----------
    graph_folder : str/Path
        Folder with .pkl graphs.
    pct : float
        Fraction of nodes for seeds (default 0.10).
    num_seeds : int
        Fixed number if pct=0 (overrides).
    seed : int
        Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    
    graph_path = Path(graph_folder)
    for pkl_file in graph_path.glob("*.pkl"):
        with open(pkl_file, 'rb') as f:
            G = pickle.load(f)
        
        nodes = list(G.nodes())
        n = len(nodes)
        
        degrees = [d for _, d in G.degree()]  # list of all node degrees
        avg_deg = stats.mean(degrees)  # or sum(degrees) / len(degrees)
        med_deg = stats.median(degrees)  # Handles even/odd node counts
        
        if num_seeds > 0:
            k = min(num_seeds, n)  # Cap at graph size
        else:
            k = max(1, int(n * pct))

        passes = False
        while (not passes):  
            seeds = random.sample(nodes, k)     
            passes, _ = check_seeds_heuristic(G, seeds, avg_deg, med_deg)
            seeds.sort()  # Deterministic order 
        
        seeds_file = pkl_file.with_suffix('.seeds.txt')
        with open(seeds_file, 'w') as f:
            f.write(','.join(map(str, seeds)))
        
        print(f"{pkl_file.name}: {n} nodes → {k} seeds → {seeds_file.name}")


def graph_from_edge_csv(
    path,
    from_col="from",
    to_col="to",
    directed=False,
    edge_attr=None,
    has_header=True,
    separator=','
):
    """
    Create a NetworkX graph from a CSV edge list.

    Parameters
    ----------
    path : str or path-like
        Path to the CSV file.
    from_col : str
        Name of the column containing the source node.
    to_col : str
        Name of the column containing the target node.
    directed : bool
        If True, create a DiGraph, else an undirected Graph.
    edge_attr : str or list of str or None
        Column name(s) to use as edge attributes.
    has_header : bool
        If True, first row contains column names. If False, assign names=[from_col, to_col].
    separator : str
        CSV delimiter.

    Returns
    -------
    G : networkx.Graph or networkx.DiGraph
    """
    if has_header:
        df = pd.read_csv(path, sep=separator)
    else:
        df = pd.read_csv(path, header=None, names=[from_col, to_col], sep=separator)
    
    graph_type = nx.DiGraph() if directed else nx.Graph()

    G = nx.from_pandas_edgelist(
        df,
        source=from_col,
        target=to_col,
        edge_attr=edge_attr,
        create_using=graph_type,
    )
    return G

def build_graphs_from_folder(
    input_folder,
    output_folder,
    from_col="from",
    to_col="to",
    directed=False,
    edge_attr=None,
    has_header=True,
    separator=',',
    pattern="*.csv"
):
    """
    Read all CSV edge files from input_folder, build NetworkX graphs,
    and save them to output_folder using standard pickle.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing CSV edge files.
    output_folder : str or Path
        Folder to save .pkl files.
    from_col, to_col, directed, edge_attr, has_header, separator : see graph_from_edge_csv
    pattern : str
        Glob pattern for CSV files.

    Returns
    -------
    dict
        {csv_filename: graph} mapping
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    csv_files = list(input_path.glob(pattern))
    graphs = {}

    for csv_file in csv_files:
        basename = csv_file.stem
        print(f"Processing {csv_file.name}...")

        G = graph_from_edge_csv(
            str(csv_file),
            from_col=from_col,
            to_col=to_col,
            directed=directed,
            edge_attr=edge_attr,
            has_header=has_header,
            separator=separator,
        )

        output_file = output_path / f"{basename}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(G, f)
        
        graphs[str(csv_file)] = G
        print(f"Saved {output_file}")

    print(f"Processed {len(graphs)} graphs.")
    return graphs

def check_seeds_heuristic(G, seeds, avg, median):
    seed_degrees = [G.degree(seed) for seed in seeds if seed in G]
    if len(seed_degrees) < 3:  # Need min for median +2 above
        return False, "Too few seeds"
    
    avg_floor = math.floor(avg)
    
    has_above_avg = sum(1 for d in seed_degrees if d >= avg_floor) >= 1
    above_median = sum(1 for d in seed_degrees if d > median)
    has_two_above = above_median >= 2
    
    passes = has_above_avg and has_two_above
    return passes, {
        "avg_floor": avg_floor,
        "median": median,
        "has_exact_avg": has_above_avg,
        "nodes_above_median": above_median,
        "seed_count_used": len(seed_degrees)
    }

def get_highest_degree_seeds(G, num_seeds, seed=None):
    """
    Returns a list of 'num_seeds' with the highest degrees to start an infection.
    Ties in degree rank are broken at random.
    """
    if seed is not None:
        random.seed(seed)
        
    nodes = list(G.nodes())
    # 1. Shuffle the nodes first. This ensures that when we sort by degree, 
    # any nodes with the exact same degree will be in a random order.
    random.shuffle(nodes)
    
    # 2. Sort descending by degree. Python's sort is stable, preserving the 
    # random order for ties.
    sorted_nodes = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
    
    # 3. Sample the requested number of highest degree nodes
    seed_nodes = sorted_nodes[:num_seeds]
    
    return seed_nodes

# High probability for a dense graph
def get_er_graph(n_nodes = 200, num_infection_seeds = None, p_edge = 0.8 , seed=42, debug=False):
    # ==========================================
    # Erdős-Rényi Graph (Dense)
    # ==========================================  
    er_graph = nx.gnp_random_graph(n_nodes, p_edge, seed=42)

    # Get seeds
    if num_infection_seeds != None:
        er_seeds = get_highest_degree_seeds(er_graph, num_infection_seeds, seed=seed)
    else:
        er_seeds = None

    if debug:
        print("--- Dense Erdős-Rényi Graph ---")
        print(f"Graph generated with {er_graph.number_of_nodes()} nodes and {er_graph.number_of_edges()} edges.")
        print(f"Selected {num_infection_seeds} seed nodes for infection:")
        for node in er_seeds:
            print(f"  - Node ID: {node}, Degree: {er_graph.degree(node)}")

    return er_graph, er_seeds

# Weak modularity: p_out is very close to p_in
def get_sbm_graph(n_nodes = 200, num_infection_seeds = None, p_in = 0.30, p_out = 0.25 , seed=42, debug=False):
    # ==========================================
    # Stochastic Block Model (Weakly Modular)
    # ==========================================
    sizes = [int(n_nodes/4), int(n_nodes/4), int(n_nodes/4), int(n_nodes/4)]        

    # Probability matrix
    probs = [
        [p_in, p_out, p_out, p_out],
        [p_out, p_in, p_out, p_out],
        [p_out, p_out, p_in, p_out],
        [p_out, p_out, p_out, p_in]
    ]

    sbm_graph = nx.stochastic_block_model(sizes, probs, seed=seed)

    # Get seeds
    if num_infection_seeds != None:
        # Get seeds
        sbm_seeds = get_highest_degree_seeds(sbm_graph, num_infection_seeds, seed=seed)
    else:
        sbm_seeds = None

    if debug:
        print("\n--- Weakly Modular SBM Graph ---")
        print(f"Graph generated with {sbm_graph.number_of_nodes()} nodes and {sbm_graph.number_of_edges()} edges.")
        print(f"Selected {num_infection_seeds} seed nodes for infection:")
        for node in sbm_seeds:
            print(f"  - Node ID: {node}, Degree: {sbm_graph.degree(node)}")

    return sbm_graph, sbm_seeds

def save(output_path, graph_name, graph, seeds, debug=False):        
    output_file = join(output_path, f"{graph_name}.pkl")
    with open(join(output_path, output_file), 'wb') as f:
        pickle.dump(graph, f)
    
        if debug:
            print(f"Saved {output_file} in {output_path}")

    for seed in seeds:        
        seeds_file = f"{graph_name}.{len(seed)}.seeds.txt"
        with open(join(output_path, seeds_file), 'w') as f:
            f.write(','.join(map(str, seed)))
        
            if debug:
                print(f"Saved {seeds_file} in {output_path}")
        
    print("Saved graph and seeds for", graph_name)