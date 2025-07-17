import networkx as nx
import time

from scipy.sparse.linalg import eigsh
from heapq import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.Solver import *
from modules.PriorityQueue import PriorityQueue

class SparseShieldSolver(Solver):
    def sparse_shield(self):
        G = self.G.to_undirected()
        nodelist = list(G.nodes())
        M = len(G)
        indexes = list(range(M))
        inverse_index = {}
        for i in indexes:
            inverse_index[nodelist[i]] = i

        t1 = time.time()
        A = nx.to_scipy_sparse_array(
            G, nodelist=nodelist, weight=None, dtype='f')
        W, V = eigsh(A, k=1, which='LM')
        max_eig = W[0]
        max_eigvec = V[:, 0].reshape((V.shape[0],))

        self.log["Eigenvalue"] = max_eig

        scores = 2*max_eig*(max_eigvec**2)
        pk = PriorityQueue(zip(scores.tolist(), indexes))

        S = set()
        for _ in range(self.k):
            next_best = pk.pop_task()
            S.add(next_best)
            for n in G.neighbors(nodelist[next_best]):
                j = inverse_index[n]
                if j not in S:
                    pk.update_task_add(
                        j, -2 * max_eigvec[next_best] * max_eigvec[j])

        t2 = time.time()
        self.log['Total time'] = t2-t1

        return [nodelist[i] for i in S]

    def run(self):
        blocked = self.sparse_shield()
        self.log['Blocked nodes'] = [int(node) for node in blocked]
