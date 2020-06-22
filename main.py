import argparse
from experiments.benchmark_model.benchmark import QMR_DT
from algorithms.mcmc_algorithm import EvolutionaryMC
from utils.func_support import *
import multiprocessing as mp
import pickle as pkl

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--parallel', default=True, action='store_false',
                    help='Flag to run the simulation in parallel processing')
parser.add_argument('--steps', type=int, default=1000000, metavar='int',
                    help='evaluation steps')
parser.add_argument('--seed', type=int, default =41, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.1, metavar='float',
                    help='bitflip probability')
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=1, metavar='int',
                    help = 'number of evaluations')

args = parser.parse_args()
# np.random.seed(args.seed)
results = {'mut': [], 'mut+crx': [], 'mut+xor': []}
res_error = {}
dist_plot ={}


def run(seed):
    print(seed)
    np.random.seed(seed)
    simulation = EvolutionaryMC(QMR_DT(args.pflip, args.pcross))
    result = {}


    simulation.model.generate_parameters() #create b truth
    simulation.model.generate_data() #sample findings for the generated instance
    simulation.compute_target()

    #loop over possible proposal methods
    for method in simulation.model.settings:
        bestSolution, fitHistory = simulation.run_mc(method, args.steps)
        result[method] = fitHistory

        text_output(method,seed,bestSolution,simulation)


    return result

def collect_result(result):
    # for result in result_list:
    global results
    for key in result:
        results[key].append(result[key])



def parallel():
    pool = mp.Pool(processes=5)

    for k in range(args.eval):
        pool.apply_async(run, (k,), callback=collect_result)

    pool.close()
    pool.join()

def sequential():
    for k in range(args.eval):
        np.random.seed(k)

        simulation = EvolutionaryMC(QMR_DT(args.pflip, args.pcross))
        # results = initialze_storage(simulation.model.settings)

        print('Currently at {}/{}'.format(k, args.eval-1))
        #initialize goal parameters and the corresponing data
        simulation.model.generate_parameters()
        simulation.model.generate_data()
        simulation.compute_target()

        #loop over possible proposal methods
        for method in simulation.model.settings:
            bestSolution, fitHistory, fitDist, error = simulation.run_mc(method, args.steps)

            global results
            results[method].append(fitHistory)

            global res_error
            res_error[method] = error

            global dist_plot
            dist_plot[method] = fitDist

            text_output(method,k,bestSolution,simulation)



if __name__ == '__main__':

    if args.parallel:
        parallel()
    else:
        sequential()

    pkl.dump(res_error, open('results/benchmark/pop_error.pkl', 'wb'))
    pkl.dump(results, open('results/benchmark/best_error.pkl', 'wb'))

    create_plot(results, 'results/benchmark/')
    plot_pop(res_error, 'error')
    plot_pop(dist_plot, 'target_dist')





