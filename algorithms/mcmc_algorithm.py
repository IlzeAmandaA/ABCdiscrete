import numpy as np
from tqdm import tqdm

"""
Implementation of Metropolis algorithm
"""

class EvolutionaryMC():

    def __init__(self, model, nchains=12):
        self.model = model
        self.chains = [self.model.simulate() for n in range(nchains)] #list of nparray
        self.lh_chains = None #list

    def compute_lh(self):
        self.lh_chains = [self.model.product_lh(chain) for chain in self.chains]

    def run_mc(self, method, steps):
        #initialize the population

        best_lh = max(self.lh_chains)
        best_params = self.chains[max(range(len(self.lh_chains)), key=lambda i: self.lh_chains[i])]
        fitHistory = []

        #how long to run the chains
        for n in tqdm(range(steps)):
            for i in range(len(self.chains)): #this is not exaclty random  but if i update the entrie popluation is does mattter

                iprime, jprime, j = self.model.proposal(self.chains, i, method)
                lh_iprime = self.model.product_lh(iprime)
                alpha = self.metropolis_ratio(iprime, lh_iprime, i, jprime, j)

                if alpha >= np.random.uniform(0,1):
                    self.chains[i] = iprime
                    self.lh_chains[i] = lh_iprime

                    if lh_iprime > best_lh:
                        best_lh = lh_iprime
                        best_params = iprime

            if n % 1000 == 0:
                fitHistory.append(self.model.error(best_params))

        return best_params, fitHistory

    def metropolis_ratio(self, iprime, lh_iprime, i, jprime, j):
        if jprime is None:
            return self.posterior(lh_iprime, iprime) / self.posterior(self.lh_chains[i], self.chains[i])
        else:
            c1 = self.posterior(lh_iprime, iprime)
            c2 = self.posterior(self.model.product_lh(jprime), jprime)
            bi = self.posterior(self.lh_chains[i], self.chains[i])
            bj = self.posterior(self.lh_chains[j], self.chains[j])
            if (c1 * c2) >= (bi * bj):
                return 1
            else:
                return (c1 * c2) / (bi * bj)

    def posterior(self, lh, data):
        return lh * np.exp(self.model.prior(data))



