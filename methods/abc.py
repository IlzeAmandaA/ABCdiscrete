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

        self.population = None
        self.epsilon = epsilon


    def initialize_chains(self):
        self.population = [self.sample_chain() for n in range(self.N)]


    def sample_chain(self):
        return np.random.binomial(1, .5, self.model.m)


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

                if self.distance(x)<=self.epsilon:
               # if self.distance(x) <= self.exp_epsilon():
                    alpha = self.metropolis(theta_,population[i])
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

    def exp_epsilon(self):
        return np.random.exponential(self.epsilon)

    def distance(self, f):
        avg=0
        for x in self.model.data:
            avg += self.hamming(f,x)
        avg /= len(self.model.data)
        return avg


    def pop_error(self, chains):
        error = 0.
        for chain in chains:
            x = self.model.simulate(chain)
            error += self.distance(x)
        error /= len(chains)
        return error

    def hamming(self, x, x0):
        distance = 0.
        for idx,xi  in enumerate(x):
            if xi != x0[idx]:
                distance += 1
        return distance

    def metropolis(self, theta_, theta):
        return min(1, np.exp(self.model.log_prior(theta_)-self.model.log_prior(theta)))


    def proposal(self, population, i, method):

        if method == 'mut+xor':
            if self.settings[method] >= np.random.uniform(0, 1):
                iprime = self.proposals.bit_flip(population[i])
            else:
                j, k = self.sample(i, len(population), 2)
                assert j != k, 'Check proposal xor method {} {}'.format(j, k)
                iprime = self.proposals.xor(population[i], population[j], population[k])

        elif method == 'de-mc':
            j, k = self.sample(i, len(population), 2)
            assert j != k, 'Check proposal xor method {} {}'.format(j, k)
            iprime = self.proposals.de_mc(population[i], population[j], population[k])

        else:
            sys.exit('Incorrect proposal')

        return iprime

    def sample(self, i, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != i], size=size, replace=False)

