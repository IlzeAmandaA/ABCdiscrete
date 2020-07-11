import numpy as np
from methods.proposals import Proposals


"""
Implementation of Metropolis algorithm
"""

STRENS=False

class EvolutionaryMC():

    def __init__(self, model, pflip, pcross, settings, nchains=12):
        self.model = model
        self.N = nchains

        self.proposals = Proposals(pflip, pcross)
        self.settings =  settings   #Strens

        self.chains = None #list of nparray
        self.target_chains = None #list #i think I can move it here

    def initialize_chains(self):
        self.chains = [self.model.simulate() for n in range(self.N)]

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

        for n in range(0,steps):

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
                for i in range(len(chains)):
                    iprime, jprime, j = self.proposal(chains, i, method)
                    target_iprime = self.model.neg_log_posterior(iprime)
                    alpha = self.metropolis_ratio(target_iprime, i, jprime, j)

                    if alpha >= np.random.uniform(0,1):
                        chains[i] = iprime
                        target_chains[i] = target_iprime

                        if target_iprime <= best_target:
                            best_target = target_iprime
                            best_params = iprime

            if n % 500 == 0:
                fitHistory.append(self.model.error(best_params))
                # fitDist.append(np.exp(best_target))
                fitDist.append(np.exp(-(best_target)))

                error.append(self.pop_error(chains))


        return best_params, fitHistory, fitDist, error

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

        if self.settings[method] >= np.random.uniform(0,1):
            iprime = self.proposals.mutation(population[i])  # sample using EA

        if method == 'mut' and self.settings[method] >= np.random.uniform(0,1):
            iprime = self.proposals.mutation(population[i])  # sample using EA

        elif method == 'mut+xor':
            j, k = self.sample(i, len(population), 2)
            assert j!=k, 'Check proposal xor method {} {}'.format(j,k)
            iprime = self.proposals.xor(population[i], population[j], population[k])

        elif method == 'mut+crx':
            j = self.sample(i, len(population))[0]
            assert j!=i, 'Check proposal cross method'
            iprime, jprime = self.proposals.crossover(population[i], population[j])

        elif method == 'braak':
            j, k = self.sample(i, len(population), 2)
            assert j != k, 'Check proposal braak method {} {}'.format(j, k)
            iprime = self.proposals.de_mc(population[i], population[j], population[k])


        return iprime, jprime, j

    def sample(self, i, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != i], size=size, replace=False)



    # def metropolis_ratio(self, post_iprime, i, jprime, j):
    #     if jprime is None:
    #         # return np.exp(post_iprime - self.model.log_posterior(self.chains[i]))#changed this from / to -
    #         return min(1, np.exp(post_iprime-self.model.log_posterior(self.chains[i])))
    #     else:
    #         c1 = post_iprime
    #         c2 = self.model.log_posterior(jprime)
    #         bi = self.model.log_posterior(self.chains[i])
    #         bj = self.model.log_posterior(self.chains[j])
    #         return min(1,  np.exp((c1 + c2) - (bi + bj)))




