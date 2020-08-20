import numpy as np
from methods.proposals import Proposals
from tqdm import tqdm


"""
Implementation of Metropolis algorithm
"""

STRENS=False

class DDE_MC():

    def __init__(self, model, pflip, pcross, settings, info, nchains=12): #12 #24
        self.model = model
        self.N = nchains

        self.exp_id = info
        self.proposals = Proposals(pflip, pcross)
        self.settings =  settings

        self.chains = None #list of nparray
        self.target_chains = None #list #i think I can move it here

    def initialize_chains(self):
        self.chains = [self.simulate() for n in range(self.N)]

    def simulate(self):
        return np.random.binomial(1, .5, self.model.m)


    def compute_fitness(self):
        self.target_chains = [self.model.neg_log_posterior(chain) for chain in self.chains]

    def run_mc(self, method, steps):

        #initialize the population
        chains = self.chains.copy()
        target_chains = self.target_chains.copy()

        best_target = min(target_chains)
        best_params = chains[min(range(len(target_chains)), key=lambda i: target_chains[i])]

        fitHistory = []
        fitDist = []
        error = []
        xlim=[]
        sample = 500*20

        n=0
        while n < steps:
        #for n in range(0,steps):

            if STRENS:
                i = np.random.randint(0,len(chains)) # uniform sampling
                iprime, jprime, j = self.proposal(chains, i, method)
                target_iprime = self.model.neg_log_posterior(iprime)
                alpha = self.metropolis_ratio(target_iprime, i, jprime, j)

                if alpha >= np.random.uniform(0, 1):
                    chains[i] = iprime
                    target_chains[i] = target_iprime

                    if target_iprime <= best_target:
                        best_target = target_iprime
                        best_params = iprime

            else:
                # sample_chains = np.random.choice([i for i in range(len(chains))], size=len(chains), replace=False)
                for i in range(len(chains)):
                # for i in sample_chains:
                    iprime, jprime, j = self.proposal(chains, i, method)
                    target_iprime = self.model.neg_log_posterior(iprime)
                    alpha = self.metropolis_ratio(target_iprime, i, jprime, j)

                    n += 1 if jprime is None else 2

                    if alpha >= np.random.uniform(0,1):
                        chains[i] = iprime
                        target_chains[i] = target_iprime

                        if target_iprime <= best_target:
                            best_target = target_iprime
                            best_params = iprime

                    if n >= sample:
                        # % 500*20 == 0 or (n-1)% 500*20 == 0:
                        fitHistory.append(self.model.error(best_params))
                        # fitDist.append(np.exp(best_target))
                        fitDist.append(best_target)

                        error.append(self.pop_error(chains))
                        xlim.append(n)
                        sample += 500*20


        return best_params, fitHistory, fitDist, error, xlim

    def pop_error(self, chains):
        error = 0.
        for chain in chains:
            error += self.model.error(chain)
        return error

    def metropolis_ratio(self, post_iprime, i, jprime, j):
        #based on negative log distribution
        if jprime is None:
            # return np.exp(post_iprime - self.model.log_posterior(self.chains[i]))#changed this from / to -
            return min(1, np.exp(self.model.neg_log_posterior(self.chains[i])-post_iprime))
        else:
            c1 = post_iprime
            c2 = self.model.neg_log_posterior(jprime)
            bi = self.model.neg_log_posterior(self.chains[i])
            bj = self.model.neg_log_posterior(self.chains[j])
            return min(1,  np.exp((bi - c1)+(bj- c2)))


    def proposal(self, population, i, method):
        jprime=None
        j = None


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

        return iprime, jprime, j

    def sample(self, i, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != i], size=size, replace=False)



