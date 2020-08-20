import math
import numpy as np

"""
Network Model based on Boltzman diststribution
"""


class Bolztmann_Net():

    def __init__(self, D=4, N=100):

        self.iter = N
        self.D = D
        self.b = self.bern(0., D * D, 1).squeeze()
        self.parameters= None
        self.data = None


    def bern(self, p, D1, D2):
        return np.random.binomial(1, p, (D1, D2))

    def bern2(self, p, D1, D2):
        return 2. * np.random.binomial(1, p, (D1, D2)) - 1.

    def energy(self, x, J, b):
        e = np.dot(x.T, np.dot(J, x)) + np.dot(b, x)
        return e.squeeze()

    def generate_binary(self, n):
        # 2^(n-1)  2^n - 1 inclusive
        bin_arr = range(0, int(math.pow(2, n)))
        bin_arr = [bin(i)[2:] for i in bin_arr]

        # Prepending 0's to binary strings
        max_len = len(max(bin_arr, key=len))
        bin_arr = [i.zfill(max_len) for i in bin_arr]

        return bin_arr

    def generate_all_x(self, n):
        B = self.generate_binary(n)
        X = np.zeros((len(B), n))

        for i in range(len(B)):
            b = B[i]
            for j in range(n):
                X[i, j] = float(b[j])

        return X

    def compute_energy(self, X, J, b):
        E = np.zeros((X.shape[0],))
        for i in range(E.shape[0]):
            x = X[i]
            E[i] = self.energy(x, J, b)

        E_exp = np.exp(-E)
        E_exp_sum = E_exp.sum()

        return E_exp, E_exp_sum  # real distribution!

    def generate_parameters(self):
        self.parameters = self.bern2(0.5, self.D**2, self.D**2)

    def generate_data(self, n):
        X = 2. * self.generate_all_x(self.D * self.D) - 1
        E_exp, E_exp_sum = self.compute_energy(X, self.parameters, self.b)
        P = E_exp / E_exp_sum
        indexes = np.random.choice(np.arange(X.shape[0]), size=n, replace=True, p=P)

        self.data =  X[indexes]

    def generate_population(self, N):
        population = []
        for i in range(N):
            population.append(self.bern(0.5, self.D ** 2, self.D ** 2))

        return np.stack(population)


    def simulate(self, J_orig, iters=10):
            #convert 0 to -1
            J = np.copy(J_orig)
            J[J == 0] = -1

            D = int(np.sqrt(J.shape[0]))

            x = self.bern2(0.1, D * D, 1).squeeze()
            E = self.energy(x, J, self.b)

            for i in range(iters):
                ind = np.random.randint(D * D)

                x_prime = x.copy()

                x_prime[ind] = x[ind] * -1.

                E_prime = self.energy(x_prime, J, self.b)

                if E_prime < E:
                    x = x_prime
                    E = E_prime

            return x


    def log_prior(self, J):
        #a uniform prior (not really informative)
        return 1/2**(J.shape[0]*J.shape[1])

























