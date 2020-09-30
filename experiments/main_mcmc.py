import argparse
import multiprocessing as mp
import pickle as pkl
import os
import sys

# PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/experiments'
PYTHONPATH = '/home/iaa510/ABCdiscrete/experiments'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.qmr_dt import QMR_DT
from algorithms.mcmc import PB_MCMC
from utils.func_support_temp import *


parser = argparse.ArgumentParser(description='Likelihood-based inference')
parser.add_argument('--steps', type=int, default=10000, metavar='int',
                    help='evaluation steps')
parser.add_argument('--seed', type=int, default=4, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.01, metavar='float',
                    help='bitflip probability') #0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=40, metavar='int',
                    help = 'number of evaluations')
parser.add_argument('--N', type=int, default=24, metavar='int',
                    help = 'population size')
parser.add_argument('--fB', default=True, action='store_false',
                    help='flag to use either a fixed or alternating underling b')


args = parser.parse_args()
SEED_MODEL=1
MAX_PROCESS=15


def execute(method, simulation, runid):


    if not args.fB:
        np.random.seed(runid)
        simulation.simulator.generate_parameters()  # create b truth
        simulation.simulator.generate_data()  # sample findings for the generated instance


    simulation.initialize_population()
    simulation.compute_fitness()

    bestSolution, fitHistory, fitDist, error, x_pos = simulation.run(method, args.steps, runid)

    global store
    text_output(method,runid,bestSolution,simulation, store) #CHECK THIS

    return (method, fitHistory, fitDist, error, x_pos)



def parallel(simulation):


    pool = mp.Pool(processes=MAX_PROCESS)

    for k in range(args.eval):
        for proposal in simulation.settings:
            pool.apply_async(execute, (proposal,simulation, k), callback=log_result)

    pool.close()
    pool.join()


def log_result(result):

    method, error, dist, pop, x = result
    global results
    best_error[method].append(error)

    global post_dist
    post_dist[method].append(dist)


    global pop_error
    pop_error[method].append(pop)

    global xlim
    xlim[method].append(x)



if __name__ == '__main__':

    # proposals = {'mut': 1., 'mut+xor': 0.5, 'mut+crx': 0.66,
    #             'dde-mc':1, 'dde-mc1':1, 'dde-mc2':1}

    proposals = {'mut': 1., 'mut+xor': 0.5, 'mut+crx': 0.66, 'dde-mc':1, 'id-samp':1}


    store = 'results/mcmc/qmr-dt/seed' + str(args.seed)
    if not os.path.exists(store):
        os.makedirs(store)

    best_error = {}
    post_dist = {}
    pop_error = {}
    xlim = {}

    for p in proposals:
        best_error[p]=[]
        post_dist[p] = []
        pop_error[p] = []
        xlim[p]=[]

    '''
    Initialze the underling model with a fixed seed 
    
    '''
    np.random.seed(SEED_MODEL)
    alg = PB_MCMC(QMR_DT(m=20, f=80), settings=proposals, pflip=args.pflip, pcross=args.pcross, N=args.N)

    '''

    Run the algorithm in parallel mode
    
    '''
    if args.fB:
        np.random.seed(args.seed)
        alg.simulator.generate_parameters()  # create b truth
        alg.simulator.generate_data()  # sample findings for the generated instance

    parallel(alg)

    '''
    Report the results 

    '''
    pkl.dump(post_dist, open(store + '/posterior.pkl', 'wb'))
    create_plot(post_dist, xlim, store + '/proposal_dist', 'posterior', True)
    create_plot(pop_error, xlim, store+'/pop_error', 'error')

    pkl.dump(best_error, open(store+'/error.pkl', 'wb'))
    create_plot(best_error, xlim, store+'/error_bestparams', 'error')








