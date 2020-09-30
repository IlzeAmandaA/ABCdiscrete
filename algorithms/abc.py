import numpy as np
import time
from algorithms.main_sampling import Sampling_Algorithm


"""
Implementation of the ABC Discrete Algorithm 
"""


class ABC_Discrete(Sampling_Algorithm):

    def __init__(self, sim, settings, epsilon, pflip=0.01, store=1):
        super(ABC_Discrete, self).__init__(sim, settings, pflip=pflip)

        self.tolerance = epsilon
        self.ensemble = store


    def run(self, method, steps, runid):
        initial_time = time.time()
        population = self.population.copy()

        #storage
        er_min = np.inf
        error_pop = []
        error_min = []
        xlim=[]
        parameter_dict = {}

        start_store = steps - ((self.ensemble*self.N)+1)
        sample = 1000
        acceptence_ratio = 0.
        id = 1
        n = 0

        while n < steps:

            for i in range(len(population)):
                theta_, _, _ = self.proposal(population, i, method)
                x = self.simulator.simulate(theta_)
                if self.simulator.distance(x) <= np.random.exponential(self.tolerance):
                    alpha = self.metropolis(theta_, population[i])
                    acceptence_ratio += 1 if n <= 10000 else 0

                    if alpha >= np.random.uniform(0, 1):
                        population[i] = theta_

                n += 1

                if n >= sample:
                    er_p, er_min_ = self.pop_error(population)

                    if er_min_ < er_min:
                        er_min = er_min_


                    error_min.append(er_min)
                    error_pop.append(er_p)
                    xlim.append(n)
                    sample += 1000 #1500

            if self.ensemble != 1 and n>=start_store:
                parameter_dict[str(id)]=population
                id+=1
                start_store+=self.N

        if self.ensemble == 1:
            parameter_dict = population

        acceptence_ratio = (acceptence_ratio/10000)*100
        print('final {} {} time ---- {} minutes ---'.format(runid, method, (time.time() - initial_time) / 60))

        return error_pop, error_min, xlim, acceptence_ratio, parameter_dict


    def pop_error(self, population):
        error = np.zeros(len(population))
        for idx,chain in enumerate(population):
            x = self.simulator.simulate(chain)
            error[idx] = self.simulator.distance(x, eval=False)
        return np.mean(error), np.min(error)


    def metropolis(self, theta_, theta):
        return min(1, self.simulator.prior(theta_)/self.simulator.prior(theta))
