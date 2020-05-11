import numpy as np
from utils.distributions import likelihood,prior
from tqdm import tqdm
from .transformations import Evolution

"""
Implementation of Metropolis algorithm
"""

class Metropolis():
    def __init__(self, model, num_iterations=1000, transition_type=None, evaluate=500,
                 p_flip=0.1, p_cross=0.5):

        self.iter = num_iterations
        self.transition_type = transition_type
        self.evaluate = evaluate

        self.model = model
        self.run = self.run_mutation if transition_type=='mutation' else self.run_combi
        self.transition = Evolution(p_flip=p_flip,p_cross=p_cross)

        self.best_b = (None, None)
        self.b_lh=None



    def run_mutation(self):
        """
        Metropolis algorithm
        :return: simulated b's
        """
        self.b_lh={}
        generated =[]
        performance =[]

        b = np.random.binomial(1,.5,self.model.m) # np.zeros(self.model.m) #b initial

        for i in tqdm(range(self.iter)):
            b_prime = self.transition.mutation(b) #sample b'

            alpha = min(1,self.ratio(b_prime, b))
            if alpha > np.random.uniform(0,1):
                b = b_prime
                self.b_lh['b1']=self.b_lh['b1_']

            self.store_best(b, 'b1')
            generated.append(b)

            if i%self.evaluate==0:
                error=self.model.evaluate(b)
                # print('Error at iteration {} is {}'.format(i, error))
                performance.append(error)

        return generated, performance

    def run_combi(self):
        self.b_lh={}
        b1 = np.random.binomial(1,.5,self.model.m) #zeros(self.model.m)
        b2 = np.random.binomial(1,.5,self.model.m)

        generated = []
        performance = []

        for i in tqdm(range(self.iter)):
            b1_, b2_ = self.transition.combi(b1, b2)

            if self.acceptance_prob(b1_,b2_, b1, b2):
                alpha=1
            else:
                alpha = self.ratio(b1_,b1)*self.ratio(b2_,b2)

            if alpha > np.random.uniform(0, 1):
                #I pass the same variables next, or should I resample
                b1 = b1_
                b2 = b2_
                self.b_lh['b1']=self.b_lh['b1_']
                self.b_lh['b2']=self.b_lh['b2_']

            if self.b_lh['b1'][0]>self.b_lh['b2'][0]:
                self.store_best(b1,'b1')
            else:
                self.store_best(b2,'b2')

            generated.append(b1)
            generated.append(b2)

            if i % self.evaluate == 0:
                error = self.model.evaluate(b1) #maybe change this see
                # print('Error at iteration {} is {}'.format(i, error))
                performance.append(error)

        return generated, performance

    def acceptance_prob(self, b1_, b2_, b1, b2):
        if self.approx_posterior(b1_, 'b1_')*self.approx_posterior(b2_, 'b2_') >= self.approx_posterior(b1, 'b1')*self.approx_posterior(b2, 'b2'):
            return True
        else:
            return False

    def approx_posterior(self, b, id):
        try:
            if id!='b1' or id!='b2':
                self.b_lh[id][0]=self.likelihood_f(b)
                self.b_lh[id][1]=self.b_lh[id][0]*prior(b,self.model.p_l)
            return self.b_lh[id][1]

        except KeyError:
            self.b_lh[id] = []
            self.b_lh[id].append(self.likelihood_f(b))
            self.b_lh[id].append(self.b_lh[id][0] * prior(b, self.model.p_l))
            return self.b_lh[id][1]



    def ratio(self, b_prime, b):
        """
        Function to estimate the proportion of p(f|b')p(b')/p(f|b)p(b)
        :return: ratio
        """
        return self.approx_posterior(b_prime, 'b1_')/self.approx_posterior(b, 'b1')


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



    def store_best(self, b, id):
        """
        Function to store the b with the highest likelihood
        :param b: np.array
        :return: b with the highest likelihood
        """
        if self.best_b[1]==None:
            self.best_b = (b, self.b_lh[id][0])
        elif self.best_b[1] < self.b_lh[id][0]:
            self.best_b = (b, self.b_lh[id][0])

