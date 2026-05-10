# %%
import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
from os.path import join

random.seed(2026)

# %%
# --- CONFIGURATION ---
DS_PATH = "/mnt/43b8df78-b48a-4d67-b3aa-8c9e00a3ecbd/Research/Datasets"          # Replace with your path
graph_name = "[208702][1392677]fromData.pkl" # Replace with your filename
TARGET_EDGES = 10_300_400
TOP_K_NODES = 5         # How many top nodes per replica to connect to the hub
TOP_N_SAVE = 1000       # Number of top nodes from the FINAL graph to save

# %%

def load_and_scale_graph(ds_path, graph_name):
    # 1. LOAD GRAPH
    print(f"Loading {graph_name}...")
    graph_path = join(ds_path, graph_name)
    
    with open(graph_path, 'rb') as f:
        G = pkl.load(f)
        
    original_edge_count = G.number_of_edges()
    original_node_count = G.number_of_nodes()
    
    print(f"Original Graph: {original_node_count:,} nodes, {original_edge_count:,} edges")

    # 2. PREPROCESSING (Map to Integers 0..N-1)
    print("Mapping nodes to contiguous integers...")
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    
    # Extract Edges into a Numpy Array
    edges_raw = np.array(list(G.edges()), dtype=object) 
    
    mapped_edges = np.empty(edges_raw.shape, dtype=np.int32)
    
    for i, (u, v) in enumerate(edges_raw):
        mapped_edges[i, 0] = node_map[u]
        mapped_edges[i, 1] = node_map[v]
        
    del edges_raw 
    
    # Identify Top Nodes (by degree) for HUB connection
    print("Identifying Top Nodes for Hub connection...")
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_node_ids = np.array([node_map[n] for n, d in sorted_nodes[:TOP_K_NODES]], dtype=np.int32)
    
    del G, node_map, node_list, sorted_nodes
    
    # 3. CALCULATE SCALING
    multiplier = int(TARGET_EDGES / original_edge_count)
    node_offset_step = original_node_count 
    super_node_id = (node_offset_step * multiplier) + 1
    
    print(f"Scaling Factor: {multiplier}x")
    print(f"Target Structure: {multiplier} replicas connected to SuperNode {super_node_id}")

    # 4. VECTORIZED REPLICATION
    print("Replicating Graph Structure...")
    offsets = np.arange(multiplier, dtype=np.int32) * node_offset_step
    
    # A. Replicate Edges
    all_edges = np.tile(mapped_edges, (multiplier, 1))
    edge_offsets = np.repeat(offsets, original_edge_count)
    
    all_edges[:, 0] += edge_offsets
    all_edges[:, 1] += edge_offsets
    
    del edge_offsets, mapped_edges 

    # 5. CONNECT TO SUPER NODE
    print("Connecting Top Nodes to Hub...")
    top_nodes_tiled = np.tile(top_node_ids, multiplier)
    top_node_offsets = np.repeat(offsets, TOP_K_NODES)
    
    hub_sources = top_nodes_tiled + top_node_offsets
    hub_targets = np.full_like(hub_sources, super_node_id)
    
    hub_edges = np.column_stack((hub_sources, hub_targets))
    
    # 6. FINALIZE MATRIX CONSTRUCTION
    print("Stacking final edge list...")
    final_edges = np.vstack((all_edges, hub_edges))
    
    del all_edges, hub_edges, hub_sources, hub_targets, offsets
    
    print("Building CSR Matrix...")
    data = np.ones(len(final_edges), dtype=np.float32) 
    matrix_dim = super_node_id + 1
    
    big_graph_matrix = sp.csr_matrix(
        (data, (final_edges[:, 0], final_edges[:, 1])),
        shape=(matrix_dim, matrix_dim)
    )
    
    print(f"Final Matrix Shape: {big_graph_matrix.shape}")
    
    # 7. EXTRACT TOP 1000 NODES (FROM FINAL GRAPH)
    print("Calculating degrees of the final massive graph...")
    
    # Calculate Total Degree = In-Degree + Out-Degree
    # axis=1 sums rows (out-degree), axis=0 sums columns (in-degree)
    # We flatten them to 1D arrays
    out_degrees = np.array(big_graph_matrix.sum(axis=1)).flatten()
    in_degrees = np.array(big_graph_matrix.sum(axis=0)).flatten()
    total_degrees = out_degrees + in_degrees
    
    print(f"Finding top {TOP_N_SAVE} nodes...")
    
    # explicit sort is safer than argpartition for getting exactly sorted list
    # but for memory efficiency on huge arrays, we use argpartition first
    if len(total_degrees) > TOP_N_SAVE:
        # Get indices of the top N (unsorted)
        top_indices_unsorted = np.argpartition(total_degrees, -TOP_N_SAVE)[-TOP_N_SAVE:]
        # Get their values
        top_values_unsorted = total_degrees[top_indices_unsorted]
        # Sort these small arrays (descending)
        sorted_local_indices = np.argsort(top_values_unsorted)[::-1]
        
        final_top_indices = top_indices_unsorted[sorted_local_indices]
        final_top_degrees = top_values_unsorted[sorted_local_indices]
    else:
        # Fallback if graph is tiny
        final_top_indices = np.argsort(total_degrees)[::-1][:TOP_N_SAVE]
        final_top_degrees = total_degrees[final_top_indices]

    # Pack into list of dicts
    top_nodes_final = []
    for idx, deg in zip(final_top_indices, final_top_degrees):
        top_nodes_final.append({
            "node_id": int(idx),
            "degree": int(deg)
        })

    print(f"Top node ID: {top_nodes_final[0]['node_id']} with degree {top_nodes_final[0]['degree']}")
    
    return big_graph_matrix, top_nodes_final

# %%
# --- EXECUTION ---
if __name__ == "__main__":
    scaled_matrix, top_nodes = load_and_scale_graph(DS_PATH, graph_name)

    # 2. Save Matrix
    print("Saving matrix to disk...")
    sp.save_npz(join(DS_PATH, f'scaled_matrix_{TARGET_EDGES}.npz'), scaled_matrix)
    
    # 3. Save Top Nodes
    print(f"Saving top {len(top_nodes)} final nodes to pickle...")
    with open(join(DS_PATH, f'top_{TOP_N_SAVE}_final_nodes_{TARGET_EDGES}.pkl'), 'wb') as f:
        pkl.dump(top_nodes, f)        
        
    just_nodes = [node['node_id'] for node in top_nodes]
    for num_seeds in [1, 2, 3, 4, 6, 12, 18, 24]:
        random_sample = random.sample(just_nodes, num_seeds)

        with open(join(DS_PATH, f'graph_{TARGET_EDGES}_{num_seeds}.seeds.txt'), 'w') as f:
            f.write(','.join(map(str, random_sample)))
        
    print("All Saved.")