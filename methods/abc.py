import numpy as np
from methods.proposals import Proposals
import sys
import time


"""
Implementation of Metropolis algorithm
"""


class ABC_Discrete():

    def __init__(self, simulator, pflip, pcross, settings, epsilon, nchains): #12 #24

        self.simulator = simulator
        self.N = nchains

        self.proposals = Proposals(pflip, pcross)
        self.settings =  settings

        self.population = None
        self.tolerance = epsilon


    def initialize_population(self):
        self.population = self.simulator.initialize(self.N)
            # self.model.generate_population(self.N)


    # def sample_chain(self):
    #     return np.random.binomial(1, .5, self.model.D)
    #

    def run(self, method, steps, seed):
        # print('Started the algorihtm')

        #initialize the population
        population = self.population.copy() #{0,1}

        error = []
        xlim=[]
        sample = 250

        n=0
        acceptence_ratio=0.
        start_time = time.time()
        print_t = 500

        while n < steps:

            if seed == 0 and n >= print_t:
                print('for run at {} pop time ---- {} minutes ---'.format(n,(time.time() - start_time) / 60))
                print_t += 1000
                start_time = time.time()

            for i in range(len(population)):
                theta_ = self.proposal(population, i, method)
                # print('proposal obtined')
                # print('theta shape {}'.format(theta_.shape))
                # print('values of theta {}'.format(set(theta_)))
                start_time = time.time()
                x=self.simulator.simulate(theta_)
                print('for run sim time ---- {} minutes ---'.format((time.time() - start_time) / 60))
                # sys.exit()

                error = self.simulator.distance(x)
                tol = np.random.exponential(self.tolerance)
                if error <=tol:
                    print('error and tol'.format(error, tol))
                    alpha = self.metropolis(theta_, population[i])
                    acceptence_ratio += 1 if n <= 10000 else 0

                    if alpha >= np.random.uniform(0,1):
                        population[i] = theta_
                n += 1

                if n >= sample:
                    # print(n)
                    error.append(self.pop_error(population))
                    xlim.append(n)
                    sample += 1000 #1500


        acceptence_ratio = (acceptence_ratio/10000)*100
        return error, xlim, acceptence_ratio, population


    def pop_error(self, chains):
        error = 0.
        for chain in chains:
            x = self.simulator.simulate(chain)
            error += self.simulator.distance(x)
        error /= len(chains)
        return error



    def metropolis(self, theta_, theta):
        return min(1, self.simulator.prior(theta_)/self.simulator.prior(theta))
    #    return min(1, np.exp(self.model.log_prior(theta_)-self.model.log_prior(theta)))


    def proposal(self, population, i, method):

        if method == 'mut+xor':
            if self.settings[method] >= np.random.uniform(0, 1):
                iprime = self.proposals.bit_flip(population[i])
            else:
                j, k = self.sample_idx(i, len(population), 2)
                assert j != k, 'Check proposal xor method {} {}'.format(j, k)
                iprime = self.proposals.xor(population[i], population[j], population[k])

        elif method == 'de-mc':
            j, k = self.sample_idx(i, len(population), 2)
            assert j != k, 'Check proposal xor method {} {}'.format(j, k)
            iprime = self.proposals.de_mc(population[i], population[j], population[k])

        else:
            sys.exit('Incorrect proposal')

        return iprime

    def sample_idx(self, i, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != i], size=size, replace=False)

