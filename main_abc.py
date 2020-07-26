import argparse
from experiments.QMR_DT.benchmark import QMR_DT
from methods.abc import ABC_Discrete
from utils.func_support import *
import multiprocessing as mp
import pickle as pkl
import os
import time

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--sequential', default=False, action='store_true',
                    help='Flag to run the simulation in parallel processing')
parser.add_argument('--steps', type=int, default=800000, metavar='int',
                    help='evaluation steps') #600000
parser.add_argument('--seed', type=int, default=0, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.005, metavar='float',
                    help='bitflip probability') #0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=15, metavar='int',
                    help = 'number of evaluations')
parser.add_argument('--exp', type=str, default='stren', metavar='str',
                    help='proposal selection')


args = parser.parse_args()

SEED_MODEL=1






def run(run_seed, simulation):
    print(run_seed)
    start_time = time.time()

    result = {}
    dist = {}
    pop={}
    x={}


    '''
    For every run initialize the chains with different initial  distribution
    '''
    np.random.seed(run_seed)
    simulation.initialize_chains()
    simulation.compute_fitness()


    #loop over possible proposal methods
    for method in simulation.settings:

        bestSolution, fitHistory, fitDist, error, x_pos = simulation.run_mc(method, args.steps)
        result[method] = fitHistory
        pop[method] = error
        dist[method] = fitDist
        x[method] = x_pos

        global store
        text_output(method,run_seed,bestSolution,simulation, store)

    print('for run {} time ---- {} minutes ---'.format(run_seed, (time.time() - start_time) / 60))
    return (result, dist, pop, x)


def parallel(settings):
    print('settings {} & running python in parallel mode with seed {}'.format(args.exp,args.seed))


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


    pool = mp.Pool(processes=args.eval)

    for k in range(args.eval):
        pool.apply_async(run, (k,simulation), callback=collect_result)

    pool.close()
    pool.join()


def collect_result(outcome):
    # for result in result_list:
    result, dist, pop, x = outcome
    global results
    for key in result:
        results[key].append(result[key])

    global post_dist
    for key in dist:
        post_dist[key].append(dist[key])

    global pop_error
    for key in pop:
        pop_error[key].append(pop[key])

    global xlim
    for key in x:
        xlim[key].append(x[key])




def sequential(settings):
    print('running python in sequential mode')
    k=0

    np.random.seed(SEED_MODEL)
    global simulation
    simulation = ABC_Discrete(QMR_DT(), args.pflip, args.pcross, settings=settings, info=args.exp)


    #initialize goal parameters and the corresponing data
    np.random.seed(args.seed)
    simulation.model.generate_parameters() #create the true underlying parameter settings
    print('true model parameters {}'.format(simulation.model.b_truth))
    simulation.model.generate_data(n=10) #generate 10 true data points
    print('data points \n')
    print(simulation.model.data)

    np.random.seed(k)
    simulation.initialize_chains()

    #loop over possible proposal methods
    for method in simulation.settings:
        print('Proposal: {}'.format(method))

        error, x = simulation.run_abc(method, args.steps)

        global pop_error
        pop_error[method] = error

        global xlim
        xlim[method] = x
        #
        # global store
        # text_output(method,k,bestSolution,simulation,store)



if __name__ == '__main__':

    set_proposals = {'de-mc':None, 'mut+xor':0.5}
    store = 'results/abc/'

    # store += str(args.seed)
    # if not os.path.exists(store):
    #     os.makedirs(store)

    results = {prop:[] for prop in set_proposals}
    post_dist = {}
    pop_error = {}
    xlim = {}

    if args.sequential:
        true_posterior = sequential(set_proposals)
        plot_pop(pop_error, 'error')
        # plot_pop(post_dist, 'target_dist', true_posterior)


    else:

        for prop in set_proposals:
            post_dist[prop] = []
            pop_error[prop] = []
            xlim[prop]=[]

        parallel(set_proposals)
        create_plot(post_dist, xlim, store + '/proposal_dist', 'posterior', True)
        pkl.dump(post_dist, open(store+'/posterior.pkl', 'wb'))
        create_plot(pop_error, xlim, store+'/pop_error', 'error')

    pkl.dump(results, open(store+'/error.pkl', 'wb'))
    create_plot(results, xlim, store+'/'+args.exp, 'error')








