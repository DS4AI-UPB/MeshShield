import time
import argparse
import numpy as np
import json
import pickle as pkl

import os
from os.path import isfile
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.Simulator import *
from modules.SparseShieldSolver import *
from modules.SparseShieldSeedlessSolver import *
from modules.SparseShieldPlusSolver import *
from modules.NetShieldSolver import *
from modules.NetShapeSolver import *
from modules.RandomSolver import *
from modules.DomSolver import *
from modules.DegreeSolver import *
from modules.MeshShieldSolver import *

if __name__ == "__main__":
    t1 = time.time()
    parser = argparse.ArgumentParser(
        description="Run solver on a single graph with seeds")
    parser.add_argument("graph", type=str)
    parser.add_argument("seeds", type=str)
    parser.add_argument("nodes_to_block", type=int)
    parser.add_argument("algorithm", type=str)
    parser.add_argument("-j", "--simulation_iterations",
                        type=int, default="100")
    parser.add_argument("-o", "--outfile", type=str, default="a.out")
    parser.add_argument("-p", "--other_params", type=str, nargs="*")
    parser.add_argument("-s", "--just_solve",
                        type=int, default=1)
    args = parser.parse_args()

    if isfile(args.outfile):
        print("Skipping:", args.outfile)
    else:
        print("Locking:", args.outfile)
        Path(args.outfile).touch()

        G, seeds = pkl.load(open(args.graph, 'rb')), np.atleast_1d(
            np.loadtxt(args.seeds, delimiter=","))
        k = args.nodes_to_block
        problem_params = {}
        if args.other_params:
            for i in range(int(len(args.other_params)/2)):
                if args.problem_params[2*i+1].isdigit():
                    problem_params[args.other_params[2*i]
                                ] = int(args.other_params[2*i+1])
                else:
                    problem_params[args.other_params[2*i]
                                ] = args.other_params[2*i+1]
        z = dict(problem_params)

        Solver = eval(args.algorithm + "Solver")
        solver = Solver(G, seeds, k, **z)
        solver.run()

        try:
            # in case the are numpy values
            solver.log["Eigenvalue"] = solver.log["Eigenvalue"].item()
        except:
            None

        print("%s blocked %d nodes in a graph of size %d." %
            (solver.get_name(), k, len(G)))

        if args.just_solve == 0:
            print("Running simulations...")
            simulator = Simulator(G, seeds)
            simulator.add_blocked(0, solver.log['Blocked nodes'])
            results = simulator.run(args.simulation_iterations)
            solver.log.update({"simulation": results['solvers'][0]})
            
            print("Solver Time: %1.5fs; Objective (saved): %1.1f; Total time: %1.5s" % (
                solver.log["Total time"], results['solvers'][0]["saved nodes"]["mean"], (time.time() - t1)))
        else:        
            print("Solver Time: %1.5fs; Total time: %1.5s" % (
                solver.log["Total time"], (time.time() - t1)))

        json.dump(solver.log, open(args.outfile, "w"))
        print("Logs saved to {}.".format(args.outfile))
