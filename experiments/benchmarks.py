import numpy as np
import pickle as pkl
import os
import sys
import argparse

"""
Main file to run to obtain the results of the benchmark experiments 
For code to run please define the correct PYTHONPATH to where the repository is located 
"""

PYTHONPATH = '/home/iaa510/ABCdiscrete/experiments'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.benchmark_testbed import BenchmarkStren
from algorithms.mcmc_algorithm import Metropolis
from utils.visualization import create_plot


parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--p_mut', type=float, default=0.1, metavar='float',
                    help='mutation probability')
parser.add_argument('--p_cross', type=float, default=0.5, metavar='float',
                    help='crossover probability')
parser.add_argument('--temp', action='store_true', default=False,
                    help='initialize temperature')

args = parser.parse_args()


if __name__ == '__main__':

    #possible transformation kernels
    transformation_kernels = ['xor','mutation','cross']

    #model settings
    diseases = 20
    findings = 80

    #experiment settings
    number_iter = 80000
    num_repetitions = 20 #40
    population = 1000

    results_all = {}
    results_dir = 'results/benchmark/'

    if args.temp:
        filename = results_dir + '_p_mut_' + str(args.p_mut) + '_temp' + '_p_cross_' + str(args.p_cross) +'_'
    else:
        filename = results_dir + '_p_mut_' + str(args.p_mut) + '_p_cross_' + str(args.p_cross)+'_'


    np.random.seed(0)
    model = BenchmarkStren(diseases, findings)
    alg = Metropolis(model=model, p_flip=args.p_mut, p_cross=args.p_cross, num_iterations=number_iter, temperature=args.temp)

    for transformation in transformation_kernels:
        print('\n---- Selected {} as transformation kernel ----\n'.format(transformation))
        print('---- Initial p_mut {}, p_cross {}, decrease overtime: {} ----\n'.format(args.p_mut, args.p_cross, args.temp))
        results_tf= {}
        results_all[transformation] = {}


        for rep in range(num_repetitions):
            print('Currently at {}/{}'.format(rep, num_repetitions))

            np.random.seed(rep+1)

            model.sample_b()
            model.sample_f()
            population = model.initialize(1000)


            #run the MCMC algorithm
            error = alg.run(population, transformation)

            textfile = open(filename + transformation + '_benchmark.txt', 'a+')
            textfile.write('------------------------------------------------\n')
            textfile.write('Iteration {}\n'.format(rep))
            textfile.write('\n b truth\n')
            textfile.write(str([int(n) for n in model.b_truth]))
            textfile.write('\n best simulated b\n')
            textfile.write(str([int(n) for n in alg.best_b[0]]))
            textfile.write('\n corresponding likelihood : {}'.format(alg.best_b[1]))
            textfile.write('\n\n')

            results_tf['rep'+str(rep)]=error

        #store results
        result_matrix = np.array([results for results in results_tf.values()])
        error_mean = np.mean(result_matrix, axis=0)
        error_std = np.std(result_matrix, axis=0)


        results_all[transformation]['mean']=error_mean
        results_all[transformation]['std'] = error_std


    pkl.dump(results_all, open(filename + 'results_benchmark.pkl', 'wb'))

    # create plots
    create_plot(results_all, filename)



















