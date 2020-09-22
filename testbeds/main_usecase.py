import numpy as np
import sys

class Testbed():

    def __init__(self):
        self.D = None

    def bern(self, D1, D2, p=0.5):
        return np.random.binomial(1, p, (D1, D2))

    def initialize(self, N):
        if self.D == None:
            sys.exit('Dimensions of the testbed are not specified')
        return self.bern(N, self.D)

    def simulate(self, *args):
        pass

    def distance(self, *args):
        pass

    def prior(self, *args):
        pass