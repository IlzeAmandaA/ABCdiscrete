import numpy as np
import itertools
from utils.distributions import likelihood, prior
from tqdm import tqdm

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
        self.b_truth = self.sample_b() #sample b_truth from disease prior
        self.findings = self.sample_f() #generate findings given b_truth
        self.mu = self.estimate_mu()

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
        return b_truth

    def sample_f(self):
        """
        Function that returns possible findings given the underlying disease state b_truth
        :return: findings
        """
        findings = np.zeros(self.f)
        for idx in range(self.f):
            if likelihood(self.q_i0, self.q_il, self.b_truth, idx)>= np.random.uniform(0,1):
                findings[idx]=1
        return findings


    def generate_combinations(self):
        """
        Generate all posssible disease combinations
        :return: 2**disease combinations
        """
        return np.array([list(l) for l in itertools.product([0, 1], repeat=self.m)])


    def estimate_mu(self):
        """
        Function to compute mu_l for every disease l
        :return: np.array, mu, probabilities
        """
        print('Estimating mu by exhaustive evaluation')
        combinations = self.generate_combinations()

        posterior = [] #approximate the posterior for every generated b
        for b in tqdm(combinations):
            p = 0.
            for idx,f in enumerate(self.findings):
                p += np.log(likelihood(self.q_i0, self.q_il, b,idx,f))
            posterior.append(np.exp(p) * prior(b, self.p_l))  # should i better convert this to sum of logs and then exp

        mu = []
        for i in range(self.m):
            bl = combinations[:, i] == 1  # create a mask
            bl_mu = np.array([b for a, b in zip(bl, posterior) if a])
            mu_l = np.mean(bl_mu)
            mu.append(mu_l)

        return np.array(mu)


    def evaluate(self, phi):
        """
        Function to evaluate the difference between the true disease distribution (mu) and simulated samples (phi)
        :param phi: np.array, simulated samples probabilities
        :return: difference (zero when phi=mu, positive otherwise)
        """
        return np.sum((self.mu - phi) * (np.log2(self.mu) - np.log2(phi)))

