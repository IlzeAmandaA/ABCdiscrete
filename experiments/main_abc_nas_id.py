import argparse
import multiprocessing as mp
import pickle as pkl
import sys
import os

PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/experiments'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.nas import NAS
from algorithms.abc import ABC_Discrete
from utils.func_support import *


parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--steps', type=int, default=120000, metavar='int',
                    help='evaluation steps')  # 600000
parser.add_argument('--seed', type=int, default=0, metavar='int',
                    help='seed')
parser.add_argument('--N', type=int, default=24, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.01, metavar='float',
                    help='bitflip probability')  # 0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=5, metavar='int',
                    help='number of evaluations')
parser.add_argument('--epsilon', type=float, default=0.04, metavar='float',
                    help='distance threshold')
parser.add_argument('--ens', type=int, default=1, metavar='int',
                    help='number of last iterations to store')


args = parser.parse_args()

SEED_MODEL = 1
MAX_PROCESS=1


def execute(method, simulation, runid):
    np.random.seed(runid)
    simulation.initialize_population()
    error, x_pos, ac_ratio, population = simulation.run(method, args.steps, runid)

    return (method, runid, error, x_pos, ac_ratio, population)

def parallel(simulation):
    pool = mp.Pool(processes=MAX_PROCESS)

    for k in range(args.eval):
        for prop in simulation.settings:
            pool.apply_async(execute, (prop, simulation, k), callback=log_result)

    pool.close()
    pool.join()


def log_result(result):
    method, runid, error, x_pos, ac_ratio, population = result

    global pop_error
    pop_error[method].append(error)

    global xlim
    xlim[method].append(x_pos)

    global acceptance_r
    acceptance_r[method].append(ac_ratio)

    global pop_store
    pop_store[str(runid)][method] = population



if __name__ == '__main__':

    set_proposals = {'id-samp': 1}

    store = 'results/abc/nas'
    if not os.path.exists(store):
        os.makedirs(store)

    pop_error = {}
    xlim = {}
    acceptance_r = {}
    pop_store = {}

    for prop in set_proposals:
        pop_error[prop] = []
        xlim[prop] = []
        acceptance_r[prop] = []

    for id in range(args.eval):
        pop_store[str(id)] = {}

    '''
    Initialze the algorithm and select the use case
    keep the underlying model same across all experiments with Seed_model
    '''

    np.random.seed(SEED_MODEL)


    alg = ABC_Discrete(NAS(), settings=set_proposals, epsilon=args.epsilon, store=args.ens)

    np.random.seed(args.seed)

    '''

    Run the algortihm in parallel mode
    '''
    parallel(alg)

    '''
    Report the results 

    '''

    print('finished parallel computing')
    pkl.dump(xlim, open(store + '/xlim' + str(args.epsilon) + '.pkl', 'wb'))
    pkl.dump(pop_error, open(store + '/pop_error' + str(args.epsilon) + '.pkl', 'wb'))
    pkl.dump(pop_store, open(store + '/pop_store' + str(args.epsilon) + '.pkl', 'wb'))
    create_plot(pop_error, xlim, store + '/pop_error' + str(args.epsilon), 'error')

    report(compute_avg(acceptance_r), args.epsilon, store + '/acceptance_ratio')
    print('Finished')









