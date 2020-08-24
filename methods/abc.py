import numpy as np
from methods.proposals import Proposals
import sys
from tqdm import tqdm


"""
Implementation of Metropolis algorithm
"""


class ABC_Discrete():

    def __init__(self, model, pflip, pcross, settings, info, epsilon, nchains): #12 #24
        self.model = model
        self.N = nchains

        self.exp_id = info
        self.proposals = Proposals(pflip, pcross)
        self.settings =  settings

        self.population = self.model.initialize_pop(self.N)
        self.tolerance = epsilon


    def initialize_chains(self):
        self.population = self.model.generate_population(self.N)


    def sample_chain(self):
        return np.random.binomial(1, .5, self.model.D)


    def run_abc(self, method, steps):

        #initialize the population
        population = self.population.copy()

        error = []
        xlim=[]
        sample = 250

        n=0
        acceptence_ratio=0.

        while n < steps:

            for i in range(len(population)):
                theta_ = self.proposal(population, i, method)
                x=self.model.simulate(theta_)

                #print(self.model.distance(x))
                if self.model.distance(x)<=np.random.exponential(self.tolerance):
                    alpha = self.metropolis(theta_, population[i])
                    acceptence_ratio += 1 if n <= 10000 else 0

                    if alpha >= np.random.uniform(0,1):
                        population[i] = theta_
                n += 1

                if n >= sample:
                    error.append(self.pop_error(population))
                    xlim.append(n)
                    sample += 1500


        acceptence_ratio = (acceptence_ratio/10000)*100
        return error, xlim, acceptence_ratio, population


    def distance(self, y):
        avg_d=0
        for y0 in self.model.data:
            avg_d += self.hamming(y,y0)
        return avg_d * 1/self.model.data.shape[0]

    def hamming(self, y, y0):
        d = 0.
        for idx,yi  in enumerate(y):
            if yi != y0[idx]:
                d += 1
        return d


    def pop_error(self, chains):
        error = 0.
        for chain in chains:
            x = self.model.simulate(chain)
            error += self.model.distance(x)
        error /= len(chains)
        return error



    def metropolis(self, theta_, theta):
        return min(1, self.model.prior(theta_)/self.model.prior(theta))
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

