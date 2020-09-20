import argparse
from experiments.mnist_numpy import MNIST
from experiments.qmr_dt import QMR_DT
from experiments.boltzmann_sim import Bolztmann_Net
from methods.abc import ABC_Discrete
from methods.mcmc import DDE_MC
from utils.func_support import *
import multiprocessing as mp
import pickle as pkl
import os
import time

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--seq', default=False, action='store_true',
                    help='Flag to run the simulation in parallel processing')
parser.add_argument('--steps', type=int, default=5000, metavar='int',
                    help='evaluation steps') #600000
parser.add_argument('--seed', type=int, default=0, metavar='int',
                    help='seed')
parser.add_argument('--N', type=int, default=24, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.01, metavar='float',
                    help='bitflip probability') #0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=5, metavar='int',
                    help = 'number of evaluations')
parser.add_argument('--exp', type=str, default='dde-mc', metavar='str',
                    help='proposal selection')

parser.add_argument('--epsilon', type=float, default=0.04, metavar='float',
                    help='distance threshold')

parser.add_argument('--alg', type=str, default = 'abc', metavar='str',
                    help = 'algorithm specification, options mcmc or abc')


args = parser.parse_args()

SEED_MODEL=1



def process(method, simulation, runid):

    error, x_pos, ac_ratio, population = simulation.run(method, args.steps, runid)

    return (method, runid, error, x_pos, ac_ratio, population)

    # return (pop, x, ratio, run_var, run_seed, chains)



def parallel(simulation):
    print('settings {} & running python in parallel mode with seed {}'.format(args.exp,args.seed))

    pool = mp.Pool(processes=15)


    for k in range(args.eval):
        '''
        For every run initialize the chains with different initial  distribution
        '''
        np.random.seed(k)
        simulation.initialize_population()
        start_time = time.time()
        for proposal in simulation.settings:
            pool.apply_async(process, (proposal,simulation, k), callback=log_result)
        print('for run {} time ---- {} minutes ---'.format(k, (time.time() - start_time) / 60))

    pool.close()
    pool.join()

def log_result(result):
    method, runid, error, x_pos, ac_ratio, population = result

    global pop_error
    pop_error[method].append(error)

    global xlim
    xlim[method].append(x_pos)

    global acceptance_r
    acceptance_r[method] = ac_ratio

    global pop_store
    pop_store[runid][method] = population



def collect_result(outcome):
    # for result in result_list:
    #pop, x, r, var, run_id, post = outcome
    pop, x, r, var, run_id, pop_s = outcome

    global pop_error
    for key, value in pop.items():
        pop_error[key].append(value)

    global xlim
    for key, value in x.items():
        xlim[key].append(value)

    global acceptance_r
    for key,value in r.items():
        acceptance_r[key].append(value)

    # global pop_c
    # for key,value in post.items():
    #     pop_c[key].append(value)

    global pop_store
    pop_store[str(run_id)] = pop_s



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



def sequential(simulation):
    print('running python in sequential mode')


    #initialize goal parameters and the corresponing data

    np.random.seed(args.seed)
    # simulation.model.generate_parameters() #create the true underlying parameter settings
    # simulation.model.generate_data(n=10) #generate 10 true data points
    simulation.initialize_chains()

    #loop over possible proposal methods
    for method in simulation.settings:
        print('Proposal: {}'.format(method))

        error, x_pos, ac_ratio, population = simulation.run_abc(method, args.steps)
        print('Acceptance ratio : {}'.format(ac_ratio))

        global pop_error
        pop_error[method] = error

        global xlim
        xlim[method] = x_pos



if __name__ == '__main__':

    set_proposals = {'de-mc':None, 'mut+xor':0.5}

    store = 'results/' + args.alg + '/' + 'bnn_mnist'
    if not os.path.exists(store):
        os.makedirs(store)


    pop_error = {}
    xlim = {}
    acceptance_r ={}
    pop_c ={}
    pop_store={}

    for prop in set_proposals:
        pop_error[prop] = []
        xlim[prop]=[]
        acceptance_r[prop] = []
        pop_c[prop] =[]

    for id in range(args.eval):
        pop_store[str(id)]={}


    '''
    Initialze the algorithm and select the use case
    keep the underlying model same across all experiments with Seed_model
    '''

    np.random.seed(SEED_MODEL)

    image_size = (14, 14)
    hidden_units = 20

    labels = [0,1]
    use_case = MNIST(l1=labels[0], l2=labels[1], image_size=image_size, H=hidden_units)


    alg = ABC_Discrete(use_case, args.pflip, args.pcross, settings=set_proposals,
                       epsilon=args.epsilon, nchains=args.N)

    np.random.seed(args.seed)

    '''
        
    Run the algortihm in parallel mode
    '''
    parallel(alg)

    '''
    Report the results 
    
    '''

    print('finihsed parallel computing')
    pkl.dump(xlim, open(store + '/xlim'+ str(args.epsilon)+'.pkl', 'wb'))
    pkl.dump(pop_error, open(store+'/pop_error'+ str(args.epsilon)+ '.pkl', 'wb'))
    #pkl.dump(pop_c,open(store+'/pop_c'+ str(args.epsilon)+ '.pkl', 'wb'))
    pkl.dump(pop_store,open(store+'/pop_store'+ str(args.epsilon)+ '.pkl', 'wb'))
    create_plot(pop_error, xlim, store +'/pop_error'+ str(args.epsilon), 'error')


    report(compute_avg(acceptance_r), args.epsilon, store+'/acceptance_ratio')
    print('Finished')









