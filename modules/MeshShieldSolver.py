import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from copy import deepcopy

from modules.SparseShieldSolver import SparseShieldSolver
from modules.Solver import *
from modules.helpers.utils import get_subgraph


from joblib import Parallel, delayed
SKIP_RESULT = (0, [])

class MeshShieldSolver(Solver):
    def run_ss(self, multiplier, subG, seeds, k, params):
        try:
            ss_subgraphs = SparseShieldSolver(subG, seeds, k, **params)        
            blocked = ss_subgraphs.sparse_shield()
            return (multiplier, blocked)
        except:
            return SKIP_RESULT

    def combine_blocked_nodes(self, list_of_blocks):
        node_stats = {}

        for (multiplier, blocked) in list_of_blocks:
            for node in blocked:
                if node in node_stats:
                    node_stats[node] += multiplier
                else:
                    node_stats[node] = multiplier

        node_h = list(node_stats.items())
        node_h.sort(key=lambda tup: tup[1], reverse=True)

        return [x[0] for x in node_h[:self.k]]
    
    def work_on_graphs(self, seed):
        (sub_graph, mul) = get_subgraph(self.G, seed, self.seeds)

        if sub_graph == None:
            return SKIP_RESULT
        
        new_k = min(2 * self.k, len(sub_graph) - 1)
        return self.run_ss(mul, sub_graph, [seed], new_k, self.params,)
    
    def run(self):
        t1 = time.time()
        print(f"Running against {len(self.seeds)} seeds.")

        results = Parallel(n_jobs=-1)(delayed(self.work_on_graphs)(seed)for seed in self.seeds)

        blocked = self.combine_blocked_nodes(results)

        t2 = time.time()
        self.log['Total time'] = t2-t1

        self.log['Blocked nodes'] = [int(node) for node in blocked]