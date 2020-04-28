
import numpy as np
from tqdm import tqdm
import itertools



class   Metropolis():
    """
    Assumes that the transition-kernel Q(x'|x) is symmetric
    Thus, simplifies to Metropolis instead of Metropolis-Hastings

    """

    def __init__(self,  diseases=20, findings=80, association_p = 0.9, mutation_p=0.1, iterations=10000):
        self.iter = iterations
        self.m = diseases
        self.f = findings
        self.mutation_p = mutation_p
        self.association_prob = association_p
        self.p_l = np.random.uniform(0, 0.5, self.m) #disease prior
        self.q_i0 = np.random.uniform(0,1, self.f) #leak probability
        self.q_il=self.association() #association between disease l and finding i (finding, disease)
        self.likelihood_b=None
        self.likelihood_bprime = None
        self.prior_b = None
        self.prior_bprime = None
        self.best_b = (None, None)

    def metropolis(self, findings):
        """
        Metropolis algorithm
        :return: simulated b's
        """
        generated =[]

        b = np.ones(self.m) #b initial

        for i in tqdm(range(self.iter)):
            b_prime = self.transition_kernel(b) #sample b'

            alpha = min(1,self.ratio(b,b_prime,findings))
            if alpha > np.random.uniform(0,1):
                b = b_prime
                self.likelihood_b = self.likelihood_bprime
                self.prior_b = self.prior_bprime

            self.store_best(b)
            generated.append(b)
        return np.array(generated)


    def transition_kernel(self, b, mutation=True, combination=False):
        """
        function to construct b_prime
        :param b: list containing bits
        :param mutation: flag to select mutation
        :param combination: flag to select combination
        :return: b_prime based on mjmutatation or mutation + cross-over
        """
        b_prime = b.copy()
        for i,val in enumerate(b_prime):
            if mutation:
                b_prime[i] = self.bit_flip(val)
            elif combination:
            # bitflip + crossval
            #have to make sure that the created value is symmetrical
                pass
            else:
                print('none of the sampling methods is selected')

        # assert not (b == b_prime).all(), 'check transition kernel'
        return b_prime

    def bit_flip(self, val):
        """
        function to perform bit flipping
        :param val: input binary value
        :return: flipped bit based on mutation probability
        """
        return 1-val if np.random.uniform(0,1) <= self.mutation_p else val


    def cross_over(self,b):
        pass



    def prior(self,b):
        """
        Function to estimate the probability of a given input string
        :param b: binary vector (np.array)
        :return: probability of this vector
        """
        return np.exp((np.sum(b*np.log(self.p_l)+(1-b)*np.log(1-self.p_l)))) #added the exp to get back to x


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

    def ratio(self, b, b_prime, findings):
        """
        Function to estimate the proportion of p(f|b')p(b')/p(f|b)p(b)
        :return: ratio
        """
        if self.likelihood_b == None and self.prior_b==None:
            self.likelihood_b = self.likelihood_findings(b,findings)
            self.prior_b = self.prior(b)

        self.likelihood_bprime = self.likelihood_findings(b_prime,findings)
        self.prior_bprime = self.prior(b_prime)

        # self.likelihood_b=self.likelihood_findings(b,findings) if self.likelihood_b == None else self.likelihood_b
        # likelihood_b_prime = self.likelihood_findings(b_prime,findings)

        return (self.likelihood_bprime * self.prior_bprime)/(self.likelihood_b * self.prior_b)


    def likelihood_findings(self,b,findings):
        """
        Function to compute the likelihood given the simulated findings
        :param b: np.array (disease)
        :param findings: np.array (findings)
        :return: likelihood probability
        """
        p=0.
        for idx,f in enumerate(findings):
            p += np.log(self.likelihood(b,f,idx))
        return np.exp(p)


    def likelihood(self, b, present, finding_idx):
        """
        Function to estimate the likelihood P(f_i=1|b)
        :param b: np.array for diseases
        :param present: 1 or 0 indicating whether the finding is present or not
        :return: probability
        """
        if present:
            return 1 - np.exp(np.log(1-self.q_i0[finding_idx])+np.sum(b*np.log(1-self.q_il[finding_idx])))
        else:
            return np.exp(np.log(1-self.q_i0[finding_idx])+np.sum(b*np.log(1-self.q_il[finding_idx])))

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

    def sample_f(self, b):
        """
        Function
        :param b: np.array for diseases
        :return: findings
        """
        findings = np.zeros(self.f)
        for idx in range(self.f):
            if self.likelihood(b,1,idx)>= np.random.uniform(0,1):
                findings[idx]=1
        return findings

def estimate_phi(simulated_data):
    return simulated_data.sum(axis=0)/float(simulated_data.shape[0])

def generate_combinations():
    """
    Generate all posssible disease combinations
    :return: 2**disease combinations
    """
    return np.array([list(l) for l in itertools.product([0, 1], repeat=Metropolis.m)])


def posterior(B, findings):
    """
    Function to estimate P(b|f) for all combinations of b given the findings
    :param findings: the simulated findings based on b_truth
    :param B: all possible combinations of b
    :return: posterior of each combination
    """
    posterior = []
    for b in tqdm(B):
        posterior.append(Metropolis.likelihood_findings(b, findings) * Metropolis.prior(b))
    return posterior


def estimate_mu(findings):
    """
    Function to compute mu_l for every disease l
    :param findings: computed findings
    :return: np.array, mu, probabilities
    """
    combinations = generate_combinations()
    p = posterior(combinations, findings)
    mu = []
    for i in tqdm(range(Metropolis.m)):
        bl = combinations[:100, i] == 1  # create a mask
        bl_mu = np.array([b for a, b in zip(bl, p) if a])
        mu_l = np.mean(bl_mu)
        mu.append(mu_l)
    return np.array(mu)

#evaluate difference
def evaluate(mu, phi):
    """
    Function to evaluate the difference between the true disease distribution (mu) and simulated samples (phi)
    :param mu: np.array, true distribution probabilities
    :param phi: np.array, simulated samples probabilities
    :return: difference (zero when phi=mu, positive otherwise)
    """
    return np.sum((mu-phi)*(np.log2(mu)-np.log2(phi)))


np.random.seed(1)
Metropolis = Metropolis()
test_b = Metropolis.sample_b()
findings = Metropolis.sample_f(test_b)
simulated_b = Metropolis.metropolis(findings)
phi = estimate_phi(simulated_b) #not sure if correct
# print(test_b)
# # print(findings)
# # print(simulated_b.shape)
#
# print(Metropolis.best_b)
# print(phi_b)

mu = estimate_mu(findings) #not sure if correct
error = evaluate(mu, phi) #happes every 500 steps




