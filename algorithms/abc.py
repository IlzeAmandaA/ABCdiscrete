import numpy as np
import time
from algorithms.main_sampling import Sampling_Algorithm


"""
Implementation of Metropolis algorithm
"""


class ABC_Discrete(Sampling_Algorithm):

    def __init__(self, sim, settings, epsilon, store=5):
        super(ABC_Discrete, self).__init__(sim, settings)

        self.tolerance = epsilon
        self.ensemble = store


    def run(self, method, steps, runid):
        initial_time = time.time()
        population = self.population.copy() #{0,1}

        #storage
        error_pop = []
        xlim=[]
        parameter_dict = {}

        start_store = steps - ((self.ensemble*self.N)+1)
        sample = 1000
        acceptence_ratio = 0.
        id = 1
        n = 0


        while n < steps:

            if n>=start_store:
                parameter_dict[str(id)]=population
                id+=1
                start_store+=self.N


            for i in range(len(population)):
                theta_, _, _ = self.proposal(population, i, method)
                x=self.simulator.simulate(theta_)

                if self.simulator.distance(x) <= np.random.exponential(self.tolerance):
                    alpha = self.metropolis(theta_, population[i])
                    acceptence_ratio += 1 if n <= 10000 else 0

                    if alpha >= np.random.uniform(0, 1):
                        population[i] = theta_

                n += 1

                if n >= sample:
                    error_pop.append(self.pop_error(population))
                    xlim.append(n)
                    sample += 1000 #1500



        acceptence_ratio = (acceptence_ratio/10000)*100
        print('final {} time ---- {} minutes ---'.format(runid, (time.time() - initial_time) / 60))

        return error_pop, xlim, acceptence_ratio, parameter_dict


    def pop_error(self, population):
        error = 0.
        for chain in population:
            x = self.simulator.simulate(chain)
            error += self.simulator.distance(x, eval=False)
        error /= len(population)
        return error


    def metropolis(self, theta_, theta):
        return min(1, self.simulator.prior(theta_)/self.simulator.prior(theta))
