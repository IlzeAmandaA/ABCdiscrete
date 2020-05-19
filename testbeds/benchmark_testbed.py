import numpy as np
import itertools
from utils.distributions import likelihood

"""
Benchmark implementation following the design of Strens' 2003 Paper
"""

class BenchmarkStren():
    def __init__(self, disease, findings, association_p = 0.9):
        self.m = disease
        self.f = findings
        self.association_prob = association_p
        self.p_l = np.random.uniform(0, 0.5, self.m)  # disease prior
        self.q_i0 = np.random.uniform(0, 1, self.f)  # leak probability
        self.q_il = self.association()  # association between disease l and finding i (finding, disease)
        self.b_truth = None #sample b_truth from disease prior
        self.findings = None #generate findings given b_truth

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

    def sample_b(self):
        """
        Function to generate test cases based on the disease prior p_l
        :return: b_truth's
        """
        b_truth = np.zeros(self.m)
        for l in range(self.m):
            if self.p_l[l] >= np.random.uniform(0,1):
                b_truth[l] = 1

        self.b_truth = b_truth

    def sample_f(self):
        """
        Function that returns possible findings given the underlying disease state b_truth
        :return: findings
        """
        findings = np.zeros(self.f)
        for idx in range(self.f):
            if likelihood(self.q_i0, self.q_il, self.b_truth, idx)>= np.random.uniform(0,1):
                findings[idx]=1
        self.findings = findings


    def generate_combinations(self):
        """
        Generate all posssible disease combinations
        :return: 2**disease combinations
        """
        return np.array([list(l) for l in itertools.product([0, 1], repeat=self.m)])


    def loss(self, b):
        """
        Hamming distance to evaluate two strings of bits (b_truth and generated b)
        :param b: np.array
        :return: float, distance
        """
        distance = 0.
        for idx,bl in enumerate(b):
            if bl != self.b_truth[idx]:
                distance+=1
        return distance

    def initialize(self, N):
        """
        Generate a population of data
        :return: np.array
        """
        population = []
        for i in range(N):
            population.append(np.random.binomial(1, .5, self.m))

        return np.array(population)

