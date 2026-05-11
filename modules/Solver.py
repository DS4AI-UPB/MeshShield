import time
import scipy.sparse as sp
import networkx as nx

class Solver:
    def __init__(self, G, seeds, k, **params):
        self.G = nx.from_scipy_sparse_array(G) if sp.issparse(G) else G.copy()
        self.graph_size = self.G.number_of_nodes()
        if self.graph_size == 0:
            raise Exception("Graph can not be empty")
        if len(seeds) == 0:
            raise Exception("Seeds can not be empty")
        if k > self.graph_size - len(seeds):
            raise Exception("Seeds can not be blocked: too large k")
        if k == 0:
            raise Exception("k should be greater than 0")
        
        self.G = nx.from_scipy_sparse_array(G) if sp.issparse(G) else G.copy()
        self.seeds = [int(node) for node in seeds]
        self.k = int(k)
        self.log = {}
        self.log['created'] = time.time()
        self.params = params
        self.clear()

    def clear(self):
        pass

    def get_name(self):
        return self.__class__.__name__
    
    def get_graph_size(self):
        return self.graph_size
