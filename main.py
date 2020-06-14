import argparse
from experiments.benchmark_model.benchmark import BenchmarkStren
from algorithms.mcmc_algorithm import EvolutionaryMC
from utils.func_support import *

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--steps', type=int, default=1000000, metavar='int',
                    help='evaluation steps')
parser.add_argument('--seed', type=int, default =1, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.1, metavar='float',
                    help='bitflip probability')
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=40, metavar='int',
                    help = 'number of evaluations')

args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)
    #initialize the prior probability, leak probaility and association probaility
    simulation = EvolutionaryMC(BenchmarkStren(args.pflip, args.pcross))

    results = initialze_storage(simulation.model.settings)

    for k in range(args.eval):
        print('Currently at {}/{}'.format(k, args.eval))
        #initialize goal parameters and the corresponing data
        simulation.model.generate_parameters()
        simulation.model.generate_data()
        simulation.compute_lh()

        #loop over possible proposal methods
        for method in simulation.model.settings:
            bestSolution, fitHistory = simulation.run_mc(method, args.steps)
            results[method].append(fitHistory)

            text_output(method,k,bestSolution,simulation)

    create_plot(results, 'results/benchmark/')


