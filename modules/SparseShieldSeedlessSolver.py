import networkx as nx
import time
from heapq import *

from scipy.sparse.linalg import eigsh
from modules.Solver import *
from modules.PriorityQueue import PriorityQueue

class SparseShieldSeedlessSolver(Solver):
    def __init__(self, G, seeds, k, **params):
        Solver.__init__(self, G, seeds, k, **params)
        self.to_add = len(seeds)

    def sparse_shield_seedless(self):
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
        for _ in range(self.k + self.to_add):
            next_best = pk.pop_task()
            S.add(next_best)
            for n in G.neighbors(nodelist[next_best]):
                j = inverse_index[n]
                if j not in S:
                    pk.update_task_add(
                        j, -2 * max_eigvec[next_best] * max_eigvec[j])

        t2 = time.time()
        self.log['Total time'] = t2-t1

        return list([nodelist[i] for i in S if (nodelist[i] not in self.seeds)])[:self.k]

    def run(self):
        blocked = self.sparse_shield_seedless()
        self.log['Blocked nodes'] = [int(node) for node in blocked]
