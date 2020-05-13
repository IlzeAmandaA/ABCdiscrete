import numpy as np

"""
Class for transformation-kernel possibilities
"""

class Evolution():

    def __init__(self, p_flip = 0.1, p_cross = 0.5, t=False):
        self.p_flip = p_flip
        self.p_cross = p_cross
        self.temp = t
        self.max_decrease = self.p_flip/10000 if t else None

    def mutation(self,b, iter):
        """
        Mutation
        :param b: string of bits
        :return: mutated b
        """
        b_prime  = b.copy()
        for idx, bit in enumerate(b):
            b_prime[idx] = self.bit_flip(bit, iter)

        return b_prime

    def combi(self, b1, b2, iter):
        """
        Combination (Cross-over and mutation)
        :param b1: string of bits
        :param b2: string of bits
        :return: 2 children, crossed over and mutated
        """
        c1, c2 = self.cross_over(b1,b2)
        c1_prime = self.mutation(c1,iter)
        c2_prime = self.mutation(c2,iter)
        return c1_prime, c2_prime

    def bit_flip(self, val, iter):
        """
        function to perform bit flipping
        :param val: input binary value
        :return: flipped bit based on mutation probability
        """
        bit = 1 - val if np.random.uniform(0, 1) <= self.p_flip else val
        if self.temp and self.p_flip>0.001:
            self.p_flip -= (self.max_decrease * 1/iter)
        return bit

    def cross_over(self, b1, b2):
        c1 = b1.copy()
        c2 = b2.copy()
        for idx,bit in enumerate(b1):
            if np.random.uniform(0,1) <= self.p_cross:
                c2[idx] = bit
                c1[idx] = b2[idx]
        return c1, c2






