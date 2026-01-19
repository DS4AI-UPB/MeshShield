import networkx as nx
import networkx.algorithms.community as nx_comm
import time
import numpy as np
from Solver import *


def Contain(G,seeds, budget, r, dr, T=100):
    nbs=[set(G.neighbors(s))|{s} for s in seeds]
    # print(nbs)
    subs=[G.subgraph(n) for n in nbs]
    agg=nx.compose_all(subs)
    seed_group=next(nx.connected_components(agg))
    # print(seed_group)
    idx = 0
    ration_change = True 
    old_ratios = budget
    while True:
        comms = nx_comm.louvain_communities(G, resolution=r, seed=42)
        intersects = [x for c in comms if (x := (seed_group & c))]
        ratios = [(x, len(x)/len(seed_group)) for x in intersects]
        # print(len(ratios))
        if len(ratios) >= budget or ration_change == False:
            sorted_ratios = sorted(ratios, key=lambda item:item[1], reverse=True)[0:budget]
            return sorted_ratios, r
        # else:
        #     return [], -1
        r += dr
        # code for stopping in case the ratios do not change after a number of iterations
        if old_ratios == len(ratios):
            idx +=1
            if idx == T:
                ration_change = False
        else:
            idx = 0
            old_ratios = len(ratios)

class ContainSolver(Solver):

    def run(self):
        t1 = time.time()
        isBigGraph = self.G.size() > 20000
        r_value = 1 if isBigGraph else 0.01
        dr_value = 0.25 if isBigGraph else 0.05
        ranks, _ = Contain(self.G, seeds=self.seeds, budget=self.k, r=r_value, dr=dr_value)
        order_on_equality = {}
        node_importance = {}
        for (solution, importance) in ranks:
            order = 0
            for node in solution:
                if node not in node_importance:
                    node_importance[node] = 0
                
                node_importance[node] += importance

                if node not in order_on_equality:
                    order_on_equality[node] = order
                
                order += 1

        nodes = list(node_importance.keys())
        sorted_nodes = sorted(
            nodes,
            key=lambda n: (-node_importance[n], order_on_equality[n])
        )        
        immunised_nodes = sorted_nodes[:self.k]
        t2 = time.time()

        self.log['Total time'] = (t2-t1)
        self.log['Blocked nodes'] = [int(node) for node in immunised_nodes]