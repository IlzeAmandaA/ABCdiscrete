import numpy as np
from experiments.benchmark_model.proposals import Proposals


"""
Main file to run to obtain the results of the benchmark_model experiments 
For code to run please define the correct PYTHONPATH to where the repository is located 
"""


class BenchmarkStren():

    def __init__(self, pflip, pcross):

        self.m = 20
        self.f = 80
        self.association_prob = 0.9
        self.p_l = np.random.uniform(0, 0.5, self.m)  # disease prior
        self.q_i0 = np.random.uniform(0, 1, self.f)  # leak probability
        self.q_il = self.association()  # association between disease l and finding i (finding, disease)
        self.b_truth = None #sample b_truth from disease prior
        self.findings = None #generate findings given b_truth

        self.proposals = Proposals(pflip, pcross)
        self.settings = {'mut': 1., 'mut+crx': 2. / 3., 'mut+xor': 0.5}

    def association(self):
        """
        Function to create matrix between the finding and association probabilities q_il
        :return: matrix(findings,diseases)
        """
        q_il = []
        for f in range(self.f):
            f_association=np.zeros(self.m)
            for b in range(self.m):
                if np.random.uniform(0,1) >= self.association_prob:
                    f_association[b] = np.random.uniform(0,1)
            q_il.append(f_association)

        return np.array(q_il)

    def generate_parameters(self):
        """
        Function to generate test cases based on the disease prior p_l
        :return: b_truth's
        """
        self.b_truth = np.zeros(self.m)
        for l in range(self.m):
            if self.p_l[l] >= np.random.uniform(0,1):
                self.b_truth[l] = 1

    def generate_data(self):
        """
        Function that returns possible findings given the underlying disease state b_truth
        :return: findings
        """
        self.findings = np.zeros(self.f)
        for idx in range(self.f):
            if (1-np.exp(self.llh(self.b_truth, idx)))>= np.random.uniform(0,1):
                self.findings[idx]=1

    def simulate(self):
        return np.random.binomial(1, .5, self.m)


    def prior(self, b):
        """
        Function to estimate the prior probability of a given input string
        :param b: binary vector (np.array)
        :return: probability
        """
        return np.sum(b * np.log(self.p_l) + (1 - b) * np.log(1 - self.p_l))

    def product_lh(self, b):
        product = 0.
        for id, f in enumerate(self.findings):

            if f == 1:
                product += (1-np.exp(self.llh(b,id)))
            else:
                product += np.exp(self.llh(b,id))
        return product


    def llh(self,b,id):
        return np.log(1 - self.q_i0[id]) + np.sum(b * np.log(1 - self.q_il[id]))


    def proposal(self, population, i, method):
        jprime=None
        j = None
        if self.proposals[method] >= np.random.uniform(0,1):
            iprime = self.proposals.mutation(population[i])  # sample using EA

        elif method == 'mut+xor':
            j, k = self.sample(i, len(population), 2)
            iprime = self.proposals.xor(population[i], population[j], population[k])

        elif method == 'mut+crx':
            j = self.sample(i, len(population))
            iprime, jprime = self.proposals.crossover(population[i], population[j])


        return iprime, jprime, j

    def error(self, b):
        """
        Hamming distance to evaluate two strings of bits (b_truth and generated b)
        """
        distance = 0.
        for idx,bl in enumerate(b):
            if bl != self.b_truth[idx]:
                distance+=1
        return distance


    def sample(self, j, max, size=1):
        return np.random.choice([x for x in range(1, max) if x != j], size=size)















