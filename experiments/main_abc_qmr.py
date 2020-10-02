import argparse
import multiprocessing as mp
import pickle as pkl
import os
import sys

# PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/experiments'
PYTHONPATH = '/home/iaa510/ABCdiscrete/experiments'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.qmr_dt import QMR_DT
from algorithms.abc import ABC_Discrete
from utils.func_support import *


parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--steps', type=int, default=80000, metavar='int',
                    help='evaluation steps')#
parser.add_argument('--seed', type=int, default=4, metavar='int',
                    help='seed')
parser.add_argument('--N', type=int, default=24, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.01, metavar='float',
                    help='bitflip probability') #0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=80, metavar='int',
                    help = 'number of evaluations')
parser.add_argument('--epsilon', type=float, default=2, metavar='float',
                    help='distance threshold')

args = parser.parse_args()

SEED_MODEL=1
MAX_PROCESS=15


def execute(method, simulation, runid):

    '''
    For every run initialize the chains with different initial  distribution
    '''
    np.random.seed(runid)
    simulation.simulator.generate_parameters() #create underlying true parameters
    simulation.simulator.generate_data(n=10) #sample K data for the given parameter settings
    run_var = compute_variability(simulation.simulator.data)

    simulation.initialize_population()


    error_pop, error, x_pos, ac_ratio, population = simulation.run(method, args.steps, runid)
    # error, x_pos, ac_ratio, chains = simulation.run(method, args.steps, runid)
    post = report_posterior(simulation, runid, method, population, store+'/posterior' +str(args.epsilon))
    return (method, runid, error_pop, error, x_pos, ac_ratio, run_var, post)



def parallel(simulation):

    pool = mp.Pool(processes=MAX_PROCESS)

    for k in range(args.eval):
        for proposal in simulation.settings:
            pool.apply_async(execute, (proposal,simulation, k), callback=log_result)

    pool.close()
    pool.join()

def log_result(result):
    method, runid, error_pop, error, x_pos, ac_ratio, run_var, post = result

    global pop_error
    pop_error[method].append(error_pop)

    global min_error
    min_error[method].append(error)

    global xlim
    xlim[method].append(x_pos)

    global acceptance_r
    acceptance_r[method].append(ac_ratio)

    global variability
    variability[str(runid)][method] = run_var

    global output_post
    output_post[runid + 1][method] = post[0]

    global output_true
    output_true[runid + 1][method] = post[1]

    global post_val
    post_val[method].append(post[2])



def compute_variability(matrix):
    results = []
    for col_id in range(matrix.shape[1]):
        null=0
        ones=0
        for row_id in range(matrix.shape[0]):
            if matrix[row_id,col_id] == 0 or matrix[row_id,col_id] == -1:
                null+=1
            else:
                ones+=1
        per = max(null/matrix.shape[0], ones/matrix.shape[0])
        results.append(per)
    return sum(results)/matrix.shape[1]



if __name__ == '__main__':

    set_proposals = {'dde-mc':1, 'mut+xor':0.5, 'ind-samp':1}
    store = 'results/abc/qmr-dt/'
    if not os.path.exists(store):
        os.makedirs(store)

    pop_error = {}
    min_error = {}
    xlim = {}
    acceptance_r ={}
    variability = {}
    output_post = {}
    output_true = {}
    post_val = {}

    '''
    keep the underlying model same across all experiments with Seed_model
    '''
    np.random.seed(SEED_MODEL)
    alg = ABC_Discrete(QMR_DT(), settings=set_proposals, epsilon=args.epsilon, e_fixed=True)

    #moved here
    # alg.simulator.generate_parameters() #create underlying true parameters
    # alg.simulator.generate_data(n=10) #sample K data for the given parameter settings

    for run in range(args.eval):
        variability[str(run)]={}
        output_true[run+1]={}
        output_post[run+1]={}

    for prop in set_proposals:
        pop_error[prop] = []
        min_error[prop] = []
        xlim[prop]=[]
        acceptance_r[prop] = []
        post_val[prop] = []

    '''

    Run the algortihm in parallel mode
    '''

    parallel(alg)


    '''
    Report the results 

    '''
    print('finished parallel computing')
    pkl.dump(xlim, open(store + '/xlim'+ str(args.epsilon)+'.pkl', 'wb'))
    pkl.dump(pop_error, open(store+'/pop_error'+ str(args.epsilon)+ '.pkl', 'wb'))
    pkl.dump(min_error, open(store + '/min_error' + str(args.epsilon) + '.pkl', 'wb'))
    create_plot(pop_error, xlim, store +'/pop_error'+ str(args.epsilon), 'avg error')
    create_plot(min_error, xlim, store + '/min_error' + str(args.epsilon), 'error')

    report(compute_avg(acceptance_r), args.epsilon, store+'/acceptance_ratio')
    report_variablitity(variability, store+'/acceptance_ratio')
    plot_dist(output_post, output_true, store +'/dist'+ str(args.epsilon))
    pkl.dump(output_post, open(store+'/dist_post'+ str(args.epsilon)+ '.pkl', 'wb'))
    pkl.dump(output_true, open(store + '/dist_true' + str(args.epsilon) + '.pkl', 'wb'))
    pkl.dump(post_val, open(store + '/dist_all' + str(args.epsilon) + '.pkl', 'wb'))
    print('Finished')











