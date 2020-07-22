import numpy as np
from methods.proposals import Proposals
from tqdm import tqdm


"""
Implementation of Metropolis algorithm
"""



class ABC_Discrete():

    def __init__(self, model, pflip, pcross, settings, info, nchains=12): #12 #24
        self.model = model
        self.N = nchains

        self.exp_id = info
        self.proposals = Proposals(pflip, pcross)
        self.settings =  settings

        self.population = None #list of nparray
        self.epsilon = SPECIFY

    def initialize_chains(self):
        self.population = [self.model.simulate() for n in range(self.N)]


    def run_abc(self, method, steps):

        #initialize the population
        population = self.population.copy()

        error = []
        xlim=[]
        sample = 500*20

        n=0
        while n < steps:
            for i in range(len(population)):
            # for i in sample_chains:
                theta_ = self.proposal(population, i, method)
                x=self.model.simulate(theta_)

                if self.distance(x)<=self.epsilon:
                    alpha = self.metropolis(theta_,population[i])
                    n += 1

                    if alpha >= np.random.uniform(0,1):
                        population[i] = theta_


                if n >= sample:
                    #WHAT TO PLOT????
                    # xlim.append(n)
                    sample += 500*20


        return WHAT TO RETURN

    def distance(self, f):
        avg=0
        for x in self.model.data:
            avg += self.hamming(f,x)
        avg /= len(self.model.data)
        return avg

    def hamming(self, x, x0):
        distance = 0.
        for idx,xi  in enumerate(x):
            if xi != x0[idx]:
                distance += 1
        return distance

    def metropolis(self, theta_, theta):
        return min(1, np.exp(self.model.log_prior(theta_)-self.model.log_prior(theta)))




    def proposal(self, population, i, method):


        if self.exp_id == 'braak':
            j, k = self.sample(i, len(population), 2)
            assert j != k, 'Check proposal xor method {} {}'.format(j, k)
            if method == 'de-mc':
                iprime = self.proposals.de_mc(population[i], population[j], population[k])
            elif method == 'de-mc1':
                iprime = self.proposals.de_mc1(population[i], population[j], population[k])
            elif method == 'de-mc2':
                iprime = self.proposals.de_mc2(population[i], population[j], population[k])

        else: #stren
            if self.settings[method] >= np.random.uniform(0,1):
                iprime = self.proposals.bit_flip(population[i])

            elif method == 'mut+crx':
                j = self.sample(i, len(population))[0]
                assert j != i, 'Check proposal cross method'
                iprime, jprime = self.proposals.crossover(population[i], population[j])

            elif method == 'mut+xor':
                j, k = self.sample(i, len(population), 2)
                assert j!=k, 'Check proposal xor method {} {}'.format(j,k)
                iprime = self.proposals.xor(population[i], population[j], population[k])

            else:
                print('incorrect proposal')

        return iprime

    def sample(self, i, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != i], size=size, replace=False)



