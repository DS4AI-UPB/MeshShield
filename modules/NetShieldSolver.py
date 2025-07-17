import networkx as nx
import time

from scipy.linalg import eigh
from heapq import *

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.Solver import *
from modules.PriorityQueue import PriorityQueue

class NetShieldSolver(Solver):

    def net_shield(self):
        G = self.G.to_undirected()
        nodelist = [n for n in G.nodes()]
        inverse_index = {}
        for i in range(len(nodelist)):
            inverse_index[nodelist[i]] = i

        t1 = time.time()
        A = nx.to_numpy_matrix(G, nodelist=nodelist, weight=None)
        M = len(G)
        W, V = eigh(A, eigvals=(M-1, M-1), type=1, overwrite_a=True)
        max_eig = W[0]
        max_eigvec = V[:,0].reshape((V.shape[0],))

        self.log["Eigenvalue"] = max_eig

        scores = 2*max_eig*(max_eigvec**2)
        pk = PriorityQueue(zip(scores.tolist(), list(range(len(G)))))

        S = set()
        for it in range(self.k):
            next_best = pk.pop_task()
            S.add(next_best)
            for n in G.neighbors(nodelist[next_best]):
                j = inverse_index[n]
                if j not in S:
                    pk.update_task_add(j, -2 * max_eigvec[next_best] * max_eigvec[j])

        t2 = time.time()
        self.log['Total time'] = t2-t1

        return [nodelist[i] for i in S]

    def run(self):
        blocked = self.net_shield()
        self.log['Blocked nodes'] = [int(node) for node in blocked]
