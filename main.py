import argparse
from experiments.QMR_DT.benchmark import QMR_DT
from methods.mcmc import EvolutionaryMC
from utils.func_support import *
import multiprocessing as mp
import pickle as pkl
import sys
import time

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--sequential', default=False, action='store_true',
                    help='Flag to run the simulation in parallel processing')
parser.add_argument('--steps', type=int, default=20000, metavar='int',
                    help='evaluation steps') #500000
parser.add_argument('--seed', type=int, default=0, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.1, metavar='float',
                    help='bitflip probability')
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=2, metavar='int',
                    help = 'number of evaluations')
parser.add_argument('--exp', type=str, default='stren', metavar='str',
                    help='proposal selection')


args = parser.parse_args()

SEED_MODEL=1
globalvar = sys.modules[__name__]
globalvar.xlim=None


def run(run_seed, simulation):
    print(run_seed)
    start_time = time.time()

    result = {}
    dist = {}


    '''
    For every run initialize the chains with different initial  distribution
    '''
    np.random.seed(run_seed)
    simulation.initialize_chains()
    simulation.compute_fitness()


    #loop over possible proposal methods
    for method in simulation.settings:

        bestSolution, fitHistory, fitDist, error, x = simulation.run_mc(method, args.steps)
        print('x {}'.format(x))
        result[method] = fitHistory
        pop_error[method] = error
        dist[method] = fitDist
        globalvar.xlim=x
        print('global {}'.format(globalvar.xlim))

        global store
        text_output(method,run_seed,bestSolution,simulation, store)

    print('for run {} time ---- {} minutes ---'.format(run_seed, (time.time() - start_time) / 60))
    return (result, dist, pop_error)


def parallel(settings):
    print('running python in parallel mode with seed {}'.format(args.seed))


    '''
    keep the underlying model same across all experiments with Seed_model
    '''
    np.random.seed(SEED_MODEL)
    simulation = EvolutionaryMC(QMR_DT(),args.pflip, args.pcross, settings=settings, info=args.exp)


    '''
    Sample different underlying parameter settings for each experiment with args.seed
    '''

    np.random.seed(args.seed)
    simulation.model.generate_parameters() #create b truth
    simulation.model.generate_data() #sample findings for the generated instance


    pool = mp.Pool(processes=2)

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

    global post_dist
    for key in dist:
        post_dist[key].append(dist[key])

    global pop_error
    for key in pop:
        pop_error[key].append(pop[key])



def sequential(settings):
    print('running python in sequential mode')
    k=0

    np.random.seed(SEED_MODEL)
    global simulation
    simulation = EvolutionaryMC(QMR_DT(),args.pflip, args.pcross, settings=settings, info=args.exp)


    #initialize goal parameters and the corresponing data
    np.random.seed(args.seed)
    simulation.model.generate_parameters()
    simulation.model.generate_data()

    np.random.seed(k)
    simulation.initialize_chains()
    simulation.compute_fitness()

    #loop over possible proposal methods
    for method in simulation.settings:
        print('Proposal: {}'.format(method))
        print(simulation.chains[0])
        bestSolution, fitHistory, fitDist, error = simulation.run_mc(method, args.steps)

        global results
        results[method].append(fitHistory)

        global pop_error
        pop_error[method] = error

        global post_dist
        post_dist[method] = fitDist

        global store
        text_output(method,k,bestSolution,simulation,store)

    return simulation.model.posterior(simulation.model.b_truth)




if __name__ == '__main__':

    Strens = {'mut': 1., 'mut+xor': 0.5, 'mut+crx': 0.66}
    Braak = ['de-mc', 'de-mc1', 'de-mc2']

    set_proposals = Strens if args.exp == 'stren' else Braak
    store='results/benchmark/' if args.exp == 'stren' else 'results/de-mc/'

    results = {prop:[] for prop in set_proposals}
    post_dist = {}
    pop_error = {}


    if args.sequential:
        true_posterior = sequential(set_proposals)
        plot_pop(pop_error, 'error')
        plot_pop(post_dist, 'target_dist', true_posterior)
    else:

        for prop in set_proposals:
            post_dist[prop] = []
            pop_error[prop] = []

        parallel(set_proposals)
        create_plot(post_dist, globalvar.xlim, store + 'proposal_dist', 'posterior', transform=True)
        pkl.dump(post_dist, open(store+'posterior.pkl', 'wb'))
        create_plot(pop_error, globalvar.xlim, store+'pop_error', 'error')

    pkl.dump(results, open(store+'error.pkl', 'wb'))
    create_plot(results, globalvar.xlim, store+args.exp, 'error')








