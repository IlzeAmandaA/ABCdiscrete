import argparse
from experiments.kitchen_sinks import RandomKitchenSinks
from methods.abc import ABC_Discrete
from utils.func_support import *
import multiprocessing as mp
import pickle as pkl
import os
import time

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--seq', default=False, action='store_true',
                    help='Flag to run the simulation in parallel processing')
parser.add_argument('--steps', type=int, default=20000, metavar='int',
                    help='evaluation steps') #600000
parser.add_argument('--seed', type=int, default=0, metavar='int',
                    help='seed')
parser.add_argument('--N', type=int, default=24, metavar='int',
                    help='seed')
parser.add_argument('--pflip', type=float, default=0.01, metavar='float',
                    help='bitflip probability') #0.1
parser.add_argument('--pcross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--eval', type=int, default=15, metavar='int',
                    help = 'number of evaluations')

parser.add_argument('--exp', type=str, default='dde-mc', metavar='str',
                    help='proposal selection')

parser.add_argument('--epsilon', type=float, default=0.2, metavar='float',
                    help='distance threshold')

parser.add_argument('--alg', type=str, default = 'abc', metavar='str',
                    help = 'algorithm specification, options mcmc or abc')


parser.add_argument('--dataN', type=int, default=20000, metavar='int',
                    help = 'number of data points')

parser.add_argument('--lr', type=float, default=0.01, metavar='float',
                    help='learning rate') #0.1


args = parser.parse_args()

SEED_MODEL=1



def process(run_seed, algorithm):
    print(run_seed)
    start_time = time.time()

    pop={}
    x={}
    ratio={}
    chains = {}


    '''
    For every run initialize the chains with different initial  distribution
    '''
    np.random.seed(run_seed)
    algorithm.initialize_population()


    #loop over possible proposal methods
    for method in algorithm.settings:
        error, x_pos, ac_ratio, population = algorithm.run(method, args.steps, run_seed)

        pop[method] = error
        x[method] = x_pos
        ratio[method] = ac_ratio
        chains[method] = population

    print('for run {} time ---- {} minutes ---'.format(run_seed, (time.time() - start_time) / 60))

    return (pop, x, ratio, chains)



def parallel(algorithm):
    print('settings {} & running python in parallel mode with seed {}'.format(args.exp,args.seed))

    pool = mp.Pool(processes=15)

    for k in range(args.eval):
        pool.apply_async(process, (k,algorithm), callback=collect_result)

    pool.close()
    pool.join()



def collect_result(outcome):
    # for result in result_list:
    pop, x, r, chains = outcome

    global pop_error
    for key, value in pop.items():
        pop_error[key].append(value)

    global xlim
    for key, value in x.items():
        xlim[key].append(value)

    global acceptance_r
    for key,value in r.items():
        acceptance_r[key].append(value)

    global pop_c
    for key,value in chains.items():
        pop_c[key].append(value)



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

    set_proposals = {'de-mc':1., 'mut+xor':0.5}

    #create dict where to store the results
    store = 'results/' + args.alg + '/' + 'bcnn_mnist'
    if not os.path.exists(store):
        os.makedirs(store)


    #objects to store the results in
    pop_error = {}
    xlim = {}
    acceptance_r ={}
    pop_c ={}

    for prop in set_proposals:
        pop_error[prop] = []
        xlim[prop]=[]
        acceptance_r[prop] = []
        pop_c[prop] =[]



    '''
    Initialze the algorithm and select the use case
    keep the underlying model same across all experiments with Seed_model
    '''
    np.random.seed(SEED_MODEL)

    image_size = (14, 14)
    hidden_units = 20

    labels = [0,1]
    use_case = RandomKitchenSinks(N_data=args.dataN, lr=args.lr)


    alg = ABC_Discrete(use_case, args.pflip, args.pcross, settings=set_proposals,
                       epsilon=args.epsilon, nchains=args.N)

    np.random.seed(args.seed)


    '''
    Run the algortihm in parallel mode
    '''
    # alg.initialize_population()

    # loop over possible proposal methods
    # for method in alg.settings:
    #     error, x_pos, ac_ratio, population = alg.run(method, args.steps, 0)
    parallel(alg)


    '''
    Report the results 
    '''

    print('finihsed parallel computing')
    pkl.dump(xlim, open(store + '/xlim'+ str(args.epsilon)+'.pkl', 'wb'))
    pkl.dump(pop_error, open(store+'/pop_error'+ str(args.epsilon)+ '.pkl', 'wb'))
    pkl.dump(pop_c,open(store+'/pop_c'+ str(args.epsilon)+ '.pkl', 'wb'))
    create_plot(pop_error, xlim, store +'/pop_error'+ str(args.epsilon), 'error')


    report(compute_avg(acceptance_r), args.epsilon, store+'/acceptance_ratio')
    print('Finished')








