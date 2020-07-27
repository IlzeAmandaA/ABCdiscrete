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
parser.add_argument('--steps', type=int, default=100000, metavar='int',
                    help='evaluation steps') #600000
parser.add_argument('--seed', type=int, default=10, metavar='int',
                    help='seed')
parser.add_argument('--N', type=int, default=24, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.01, metavar='float',
                    help='bitflip probability') #0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=40, metavar='int',
                    help = 'number of evaluations')
parser.add_argument('--exp', type=str, default='abc', metavar='str',
                    help='proposal selection')
parser.add_argument('--epsilon', type=float, default=1., metavar='float',
                    help='distance threshold')


args = parser.parse_args()

SEED_MODEL=1



def run(run_seed, simulation):
    print(run_seed)
    start_time = time.time()

    pop={}
    x={}
    ratio={}


    '''
    For every run initialize the chains with different initial  distribution
    '''
    np.random.seed(run_seed)
    # np.random.seed(args.seed)
    simulation.model.generate_parameters() #create b truth
    simulation.model.generate_data(n=50) #sample findings for the generated instance

    simulation.initialize_chains()

    #loop over possible proposal methods
    for method in simulation.settings:
        error, x_pos, ac_ratio = simulation.run_abc(method, args.steps)
        pop[method] = error
        x[method] = x_pos
        ratio[method] = ac_ratio

    print('for run {} time ---- {} minutes ---'.format(run_seed, (time.time() - start_time) / 60))
    return (pop, x, ratio)


def parallel(settings):
    print('settings {} & running python in parallel mode with seed {}'.format(args.exp,args.seed))


    '''
    keep the underlying model same across all experiments with Seed_model
    '''
    np.random.seed(SEED_MODEL)

    simulation = ABC_Discrete(QMR_DT(),args.pflip, args.pcross, settings=settings, info=args.exp, epsilon=args.epsilon, nchains=args.N)

    '''
    Sample different underlying parameter settings for each experiment with args.seed
    '''

    #old location for parameters

    pool = mp.Pool(processes=15)

    for k in range(args.eval):
        pool.apply_async(run, (k,simulation), callback=collect_result)

    pool.close()
    pool.join()


def collect_result(outcome):
    # for result in result_list:
    pop, x, r = outcome

    global pop_error
    for key, value in pop.items():
        pop_error[key].append(value)

    global xlim
    for key, value in x.items():
        xlim[key].append(value)

    global acceptance_r
    for key,value in r.items():
        acceptance_r[key].append(value)




def sequential(settings):
    print('running python in sequential mode')
    k=0

    np.random.seed(SEED_MODEL)
    global simulation
    simulation = ABC_Discrete(QMR_DT(), args.pflip, args.pcross, settings=settings, info=args.exp, epsilon=args.epsilon, nchains=args.N)

    #initialize goal parameters and the corresponing data
    np.random.seed(args.seed)
    simulation.model.generate_parameters() #create the true underlying parameter settings
    print('true model parameters {}'.format(simulation.model.b_truth))
    simulation.model.generate_data(n=50) #generate 10 true data points
    print('data points')
    print(simulation.model.data)

    np.random.seed(k)
    simulation.initialize_chains()

    #loop over possible proposal methods
    for method in simulation.settings:
        print('Proposal: {}'.format(method))

        error, x, ratio = simulation.run_abc(method, args.steps)
        print('Acceptance ratio : {}'.format(ratio))

        global pop_error
        pop_error[method] = error

        global xlim
        xlim[method] = x



if __name__ == '__main__':

    set_proposals = {'de-mc':None, 'mut+xor':0.5}
    store = 'results/abc'

    pop_error = {}
    xlim = {}
    acceptance_r ={}

    if args.sequential:
        sequential(set_proposals)
        plot_single(pop_error, xlim, 'error', store + '/pop_error')


    else:

        for prop in set_proposals:
            pop_error[prop] = []
            xlim[prop]=[]
            acceptance_r[prop] = []

        parallel(set_proposals)
        pkl.dump(xlim, open(store + '/xlim'+ str(args.epsilon)+'.pkl', 'wb'))
        pkl.dump(pop_error, open(store+'/pop_error'+ str(args.epsilon)+ '.pkl', 'wb'))
        create_plot(pop_error, xlim, store +'/pop_error'+ str(args.epsilon), 'error')

        report(compute_avg(acceptance_r), args.epsilon, store+'/acceptance_ratio')









