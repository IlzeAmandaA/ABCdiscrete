import numpy as np
from tqdm import tqdm

"""
Implementation of Metropolis algorithm
"""

class EvolutionaryMC():

    def __init__(self, model, nchains=50):
        self.model = model
        self.chains = [self.model.simulate() for n in range(nchains)] #list of nparray
        self.target_chains = None #list #i think I can move it here

    # def compute_lh(self): #compute the target distribution
    #     self.lh_chains = [self.model.product_lh(chain) for chain in self.chains]
    #
    def compute_target(self):
        self.target_chains = [self.posterior(chain) for chain in self.chains]

    def run_mc(self, method, steps):
        #initialize the population
        chains = self.chains.copy()
        target_chains = self.target_chains.copy()

        best_target = max(target_chains)
        best_params = chains[max(range(len(target_chains)), key=lambda i: target_chains[i])]
        fitHistory = []
        fitDist = []
        error = []

        #how long to run the chains
        for n in range(0,steps):
            # for i in range(len(self.chains)):
                #change this to uniform sampling method
            i = np.random.randint(0,len(chains))

            # for i in range(len(self.chains)): #this is not exaclty random  but if i update the entrie popluation is does mattter
            iprime, jprime, j = self.model.proposal(chains, i, method)
            # assert not np.array_equal(iprime,self.chains[i]), 'incorrect proposal iter {}  {}:{}'.format(n, iprime, self.chains[i])
            target_iprime = self.posterior(iprime)
            alpha = self.metropolis_ratio(target_iprime, i, jprime, j)

            if alpha >= np.random.uniform(0,1):
                chains[i] = iprime
                target_chains[i] = target_iprime

                if target_iprime >= best_target:
                    best_target = target_iprime
                    best_params = iprime

            if n % 500 == 0:
                fitHistory.append(self.model.error(best_params))
                fitDist.append(best_target)

                error.append(self.pop_error(chains))


        return best_params, fitHistory, fitDist, error

    def pop_error(self, chains):
        error = 0.
        for chain in chains:
            error += self.model.error(chain)
        return error

    def metropolis_ratio(self, post_iprime, i, jprime, j):
        if jprime is None:
            return post_iprime / self.posterior(self.chains[i])
        else:
            c1 = post_iprime
            c2 = self.posterior(jprime)
            bi = self.posterior(self.chains[i])
            bj = self.posterior(self.chains[j])
            if (c1 * c2) >= (bi * bj):
                return 1
            else:
                return (c1 * c2) / (bi * bj)

    def posterior(self, data):
        return self.model.product_lh(data) * np.exp(self.model.prior(data))



