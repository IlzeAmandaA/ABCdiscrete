import numpy as np
import pickle as pkl
import os
import sys

"""
Main file to run to obtain the results of the benchmark experiments 
For code to run please define the correct PYTHONPATH to where the repository is located 
"""

PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/experiments'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.benchmark_testbed import BenchmarkStren
from algorithms.mcmc_algorithm import Metropolis
from utils.visualization import create_plot



if __name__ == '__main__':

    #possible transformation kernels
    transformation_kernels = ['mutation'] #, 'mutation_cross']

    #model settings
    diseases = 20
    findings = 80


    #experiment settings
    transformation_p = 0.1
    number_iterations = 100000
    num_repetitions = 40

    results_all = {}

    for transformation in transformation_kernels:
        print('---- Selected {} as transformation kernel ----'.format(transformation))
        results_dir = '../results/'
        results_tf= {}
        results_all[transformation] = {}


        for rep in range(num_repetitions):
            print('Currently at {}/{}'.format(rep, num_repetitions))

            np.random.seed(rep)

            #initialze experiment settings
            model=BenchmarkStren(diseases, findings)
            print(model.b_truth)
            print(model.mu)

            #initialize MCMC settings
            alg = Metropolis(model, number_iterations, transformation, transformation_p)

            #run the MCMC algorithm
            data, error = alg.run()

            results_tf['rep'+str(rep)]=error


        #store results
        result_matrix = np.array([results for results in results_tf.values()])
        error_mean = np.mean(result_matrix, axis=0)
        error_std = np.std(result_matrix, axis=0)


        results_all[transformation]['mean']=error_mean
        results_all[transformation]['std'] = error_std

    pkl.dump(results_all, open(results_dir+'results_benchmark.pkl', 'wb'))

    #create plots
    create_plot(results_all, results_dir)

















