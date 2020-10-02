from kernels.proposals import Proposals
import sys
import numpy as np

class Sampling_Algorithm():

    def __init__(self, sim, settings, N=24, pflip=0.01, pcross=0.5):

        self.simulator = sim
        self.N = N
        self.settings = settings

        self.proposals = Proposals(pflip, pcross)
        self.population = None

    def initialize_population(self):
        self.population = self.simulator.initialize(self.N)


    def run(self, *args):
        pass

    def pop_error(self, pop):
        pass

    def metropolis(self, *args):
        pass

    def proposal(self, population, i, method):
        jprime=None
        j = None

        if method == 'mut':
            iprime = self.proposals.bit_flip(population[i])

        elif method == 'ind-samp':
            iprime = self.proposals.indepent_sampler(population[i])

        else:
            j, k = self.sample_idx(i, len(population), 2)
            assert j != k, 'Check proposal xor method {} {}'.format(j, k)

            if method == 'mut+xor':
                if self.settings[method] >= np.random.uniform(0, 1):
                    iprime = self.proposals.bit_flip(population[i])
                else:
                    iprime = self.proposals.xor(population[i], population[j], population[k])

            elif method == 'mut+crx':
                if self.settings[method] >= np.random.uniform(0, 1):
                    iprime = self.proposals.bit_flip(population[i])
                else:
                    iprime, jprime = self.proposals.crossover(population[i], population[j])

            elif method == 'dde-mc':
                iprime = self.proposals.dde_mc(population[i], population[j], population[k])

            elif method == 'dde-mc1':
                iprime = self.proposals.dde_mc1(population[i], population[j], population[k])

            elif method == 'dde-mc2':
                iprime = self.proposals.dde_mc2(population[i], population[j], population[k])

            else:
                sys.exit('Invalid proposal selected!')

        return iprime, jprime, j

    def sample_idx(self, i, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != i], size=size, replace=False)

