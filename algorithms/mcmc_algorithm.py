import numpy as np
from utils.distributions import likelihood,prior
from tqdm import tqdm
from .transformations import Evolution

"""
Implementation of Metropolis algorithm
"""

class Metropolis():
    def __init__(self, model, num_iterations=1000, prob_trans=0.1, transition_type=None, evaluate=500):
        self.iter = num_iterations
        self.prob_t = prob_trans
        self.transition_type = transition_type
        self.evaluate = evaluate

        self.model = model
        self.transition = Evolution(self.transition_type, self.prob_t)

        self.likelihood_b=None
        self.prior_b=None
        self.likelihood_bprime = None
        self.prior_bprime = None

        self.best_b = (None, None)



    def run(self):
        """
        Metropolis algorithm
        :return: simulated b's
        """
        generated =[]
        performance =[]

        b = np.zeros(self.model.m) #b initial

        for i in range(self.iter):
            b_prime = self.transition.recombination(b) #sample b'

            alpha = min(1,self.ratio(b,b_prime))
            if alpha > np.random.uniform(0,1):
                b = b_prime
                self.likelihood_b = self.likelihood_bprime
                self.prior_b = self.prior_bprime

            self.store_best(b)
            generated.append(b)

            if i%self.evaluate==0:
                #estimate phi
                # phi = self.estimate_phi(np.array(generated))
                #compute the difference between phi and mu

                error=self.model.evaluate(b)
                print('Error at iteration {} is {}'.format(i, error))
                performance.append(error)

        return generated, performance


    def ratio(self, b, b_prime):
        """
        Function to estimate the proportion of p(f|b')p(b')/p(f|b)p(b)
        :return: ratio
        """
        if self.likelihood_b == None and self.prior_b==None:
            self.likelihood_b = self.likelihood_f(b)
            self.prior_b = prior(b, self.model.p_l)

        self.likelihood_bprime = self.likelihood_f(b_prime)
        self.prior_bprime = prior(b_prime, self.model.p_l)

        # self.likelihood_b=self.likelihood_findings(b,findings) if self.likelihood_b == None else self.likelihood_b
        # likelihood_b_prime = self.likelihood_findings(b_prime,findings)

        return (self.likelihood_bprime * self.prior_bprime)/(self.likelihood_b * self.prior_b)


    def likelihood_f(self,b):
        """
        Function to compute the likelihood given the simulated findings
        :param b: np.array (disease)
        :return: likelihood probability
        """
        p=0.
        for idx,f in enumerate(self.model.findings):
            p += np.log(likelihood(self.model.q_i0, self.model.q_il, b, idx,f_i=f))
        return np.exp(p)



    def store_best(self, b):
        """
        Function to store the b with the highest likelihood
        :param b: np.array
        :return: b with the highest likelihood
        """
        if self.best_b[0]==None:
            self.best_b = (self.likelihood_b, b)
        elif self.best_b[0]< self.likelihood_b:
            self.best_b = (self.likelihood_b, b)


    # def estimate_phi(self, simulated_data):
    #     return simulated_data.sum(axis=0) / float(simulated_data.shape[0])
