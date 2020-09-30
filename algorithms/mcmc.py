import numpy as np
from algorithms.main_sampling import Sampling_Algorithm
import time


"""
Implementation of Metropolis algorithm
"""

class PB_MCMC(Sampling_Algorithm):

    def __init__(self, sim, settings, pflip, pcross, N):

        super(PB_MCMC, self).__init__(sim, settings, pflip=pflip, pcross=pcross, N=N)

        self.target_population = None

    def compute_fitness(self):
        self.target_population = [self.simulator.neg_log_posterior(chain) for chain in self.population]

    def run(self, method, steps, runid):
        initial_time = time.time()

        #initialize the population
        population = self.population.copy()
        target_population = self.target_population.copy()

        best_target = min(target_population)
        best_params = population[min(range(len(target_population)), key=lambda i: target_population[i])]

        fitHistory = []
        fitDist = []
        error = []
        xlim=[]
        sample = 500 #*20

        n=0
        while n < steps:
            for i in range(len(population)):
                iprime, jprime, j = self.proposal(population, i, method)
                target_iprime = self.simulator.neg_log_posterior(iprime)
                alpha = self.metropolis(target_iprime, i, jprime, j, population)

                n += 1 if jprime is None else 2

                if alpha >= np.random.uniform(0,1):
                    population[i] = iprime
                    target_population[i] = target_iprime

                    if target_iprime <= best_target:
                        best_target = target_iprime
                        best_params = iprime

                if n >= sample:
                    fitHistory.append(self.simulator.hamming(best_params, self.simulator.parameters))
                    fitDist.append(best_target)

                    error.append(self.pop_error(population))
                    xlim.append(n)
                    sample += 500 #*20

        print('final {} for {} time ---- {} minutes ---'.format(runid, method, (time.time() - initial_time) / 60))

        return best_params, fitHistory, fitDist, error, xlim


    def pop_error(self, population):
        error = 0.
        for chain in population:
            error += self.simulator.hamming(chain, self.simulator.parameters)
        return error/len(population)


    def metropolis(self, post_iprime, i, jprime, j, population):
        #based on negative log distribution
        if jprime is None:
            return min(1, np.exp(self.simulator.neg_log_posterior(population[i])-post_iprime))
        else:
            c1 = post_iprime
            c2 = self.simulator.neg_log_posterior(jprime)
            bi = self.simulator.neg_log_posterior(population[i])
            bj = self.simulator.neg_log_posterior(population[j])
            return min(1,  np.exp((bi - c1)+(bj- c2)))


