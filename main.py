import argparse
from experiments.QMR_DT.benchmark import QMR_DT
from methods.mcmc import EvolutionaryMC
from utils.func_support import *
import multiprocessing as mp
import pickle as pkl

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--parallel', default=True, action='store_false',
                    help='Flag to run the simulation in parallel processing')
parser.add_argument('--steps', type=int, default=40000, metavar='int',
                    help='evaluation steps') #200000
parser.add_argument('--seed', type=int, default=1, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.1, metavar='float',
                    help='bitflip probability')
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=10, metavar='int',
                    help = 'number of evaluations')

args = parser.parse_args()


#define settings to use
Strens = {'mut': 1., 'mut+xor': 0.5, 'mut+crx': 0.66}
Proposal_prob = {'mut': 1., 'mut+xor': 0.5, 'mut+crx': 0.66, 'braak':1.} #first try on sequeential setting
results = {'mut': [],  'mut+xor': [],'mut+crx': []} #, 'braak': []}
dist_plot = {'mut': [], 'mut+crx': [], 'mut+xor': []}
pop_error = {'mut': [], 'mut+crx': [], 'mut+xor': []}
# res_error = {}
# dist_plot ={}



def run(run_seed, simulation):
    print(run_seed)

    result = {}
    dist = {}

    #for every run compute a different b_truth, and data
    np.random.seed(run_seed)


    simulation.initialize_chains()  # initialzie the chains as fixed for all runs, compute their fitness
    simulation.compute_fitness()


    #loop over possible proposal methods
    for method in simulation.settings:
        bestSolution, fitHistory, fitDist, error = simulation.run_mc(method, args.steps)
        result[method] = fitHistory
        pop_error[method] = error
        dist[method] = fitDist

        text_output(method,run_seed,bestSolution,simulation)

    # plot_pop(dist, 'posterior' + str(run_seed))


    return (result, dist, pop_error)


def parallel():

    np.random.seed(args.seed+4) #to keep the initialization settings the same across runs
    simulation = EvolutionaryMC(QMR_DT(),args.pflip, args.pcross, settings=Strens)

    #old spot
    simulation.model.generate_parameters() #create b truth
    simulation.model.generate_data() #sample findings for the generated instance


    pool = mp.Pool(processes=5)

    for k in range(args.eval):
        pool.apply_async(run, (k,simulation), callback=collect_result)

    pool.close()
    pool.join()


def collect_result(outcome):
    # for result in result_list:
    result, dist, pop = outcome
    global results
    for key in result:
        results[key].append(result[key])

    global dist_plot
    for key in dist:
        dist_plot[key].append(dist[key])

    global pop_error
    for key in pop:
        pop_error[key].append(pop[key])



def sequential():
    print('Sequential run')
    # for k in range(1):
    k=0

    np.random.seed(args.seed)
    global simulation
    simulation = EvolutionaryMC(QMR_DT(),args.pflip, args.pcross, settings=Strens)

    #initialize goal parameters and the corresponing data
    simulation.model.generate_parameters()
    simulation.model.generate_data()

    np.random.seed(k)
    simulation.initialize_chains()
    simulation.compute_fitness()

    #loop over possible proposal methods
    for method in simulation.settings:
        print('Method: {}'.format(method))
        print(simulation.chains[0])
        bestSolution, fitHistory, fitDist, error = simulation.run_mc(method, args.steps)

        global results
        results[method].append(fitHistory)

        global res_error
        res_error[method] = error

        global dist_plot
        dist_plot[method] = fitDist

        text_output(method,k,bestSolution,simulation)

    return simulation.model.posterior(simulation.model.b_truth)




if __name__ == '__main__':

    if args.parallel:
        parallel()
        create_plot(dist_plot, 'results/benchmark/proposal_dist', 'posterior')
        pkl.dump(dist_plot, open('results/benchmark/posterior.pkl', 'wb'))
        create_plot(pop_error,'results/benchmark/pop_error', 'error')

    else:
        true_posterior = sequential()
        plot_pop(res_error, 'error')
        plot_pop(dist_plot, 'target_dist', true_posterior)


    # pkl.dump(res_error, open('results/benchmark/pop_error.pkl', 'wb'))
    pkl.dump(results, open('results/benchmark/error.pkl', 'wb'))
    create_plot(results, 'results/benchmark/benchmark_Stren', 'error')






