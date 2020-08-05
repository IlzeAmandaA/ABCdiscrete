import numpy as np

"""
QMR-DT sample problem
"""


class QMR_DT():

    def __init__(self):

        self.m = 10
        self.f = 20
        self.association_prob = 0.9
        self.p_l = np.random.beta(0.15, 0.15, self.m)  # disease prior
        self.q_i0 = np.random.beta(0.15, 0.15, self.f)  # leak probability
        self.q_il = self.association()  # association between disease l and finding i (finding, disease)


        self.b_truth = None #sample b_truth from disease prior
        self.data = None #generate multipl findings given b_truth


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
                    f_association[b] = np.random.beta(0.15,0.15)
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


    def generate_data(self, n=1):
        self.data = np.zeros(shape=(n,self.f))
        for row in range(self.data.shape[0]):
            for idx in range(self.data.shape[1]):
                if (1-np.exp(self.llh(self.b_truth, idx)))>= np.random.uniform(0,1):
                    self.data[row,idx]=1


    def simulate(self, b):
        findings = np.zeros(self.f)
        for idx in range(self.f):
            if (1 - np.exp(self.llh(b, idx))) >= np.random.uniform(0, 1):
                findings[idx] = 1

        return findings


    def prior(self, b):
        """
        :param b: binary vector (np.array)
        :return: log prior
        """
        return np.sum(b * np.log(self.p_l) + (1 - b) * np.log(1 - self.p_l))

    def log_prior(self, b):
        return np.sum(b * np.log(self.p_l) + (1 - b) * np.log(1 - self.p_l))

    def product_lh(self, b):
        product = 1.
        for id, f in enumerate(self.data):
            if f == 1:
                product *= (1-np.exp(self.llh(b,id)))
            else:
                product *= np.exp(self.llh(b,id))
        print(product)
        return product


    def product_llh(self,b):
        product = 0.
        for id, f in enumerate(self.data):
            if f == 1:
                product += np.log(1-np.exp(self.llh(b,id)))
            else:
                product += self.llh(b,id)
        return product

    def product_neg_llh(self,b):
        product = 0.
        for id, f in enumerate(self.data):
            if f == 1:
                product += np.log(1-np.exp(self.llh(b,id)))
            else:
                product += self.llh(b,id)
        return -product

    def llh(self,b,id):
        return np.log(1 - self.q_i0[id]) + np.sum(b * np.log(1 - self.q_il[id]))

    def posterior(self, data):
        return self.product_lh(data) * np.exp(self.prior(data))

    def log_posterior(self,data):
        return self.product_llh(data) + self.prior(data)

    def neg_log_posterior(self,data):
        return -(self.product_llh(data) + self.prior(data))

    def posterior_abc(self,data):
        return self.product_lh_abc(data) * np.exp(self.prior(data))

    def log_posterior_abc(self,data):
        return self.product_llh_abc(data) + self.prior(data)

    def product_lh_abc(self,b):
        avg_lh = []
        for datapoint in self.data:
            product = 1.
            for id, f in enumerate(datapoint):
                if f == 1:
                    product *= (1 - np.exp(self.llh(b, id)))
                else:
                    product *= np.exp(self.llh(b, id))
            avg_lh.append(product)
        return sum(avg_lh)/len(self.data)

    def product_llh_abc(self, b):
        avg_lh = []
        for datapoint in self.data:
            product = 0.
            for id, f in enumerate(datapoint):
                if f == 1:
                    product += np.log(1 - np.exp(self.llh(b, id)))
                else:
                    product += self.llh(b, id)
            avg_lh.append(product)

        return sum(avg_lh)/len(self.data)








    def error(self, b):
        """
        Hamming distance to evaluate two strings of bits (b_truth and generated b)
        """
        distance = 0.
        for idx,bl in enumerate(b):
            if bl != self.b_truth[idx]:
                distance+=1
        return distance
















