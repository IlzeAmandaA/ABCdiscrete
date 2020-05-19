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
        b_prime  = b[0].copy()
        for idx, bit in enumerate(b[0]):
            b_prime[idx] = self.bit_flip(bit, iter)

        return [b_prime]

    def combi(self, b, iter):
        """
        Combination (Cross-over and mutation)
        :param b1: string of bits
        :param b2: string of bits
        :return: 2 children, crossed over and mutated
        """
        c1, c2 = self.cross_over(b)
        c1_prime = self.mutation([c1],iter)
        c2_prime = self.mutation([c2],iter)
        return (c1_prime[0], c2_prime[0])


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

    def cross_over(self, bs):
        c1 = bs[0].copy()
        c2 = bs[1].copy()
        for idx,bit in enumerate(bs[0]):
            if np.random.uniform(0,1) <= self.p_cross:
                c2[idx] = bit
                c1[idx] = bs[1][idx]
        return c1, c2

    def xor(self, bs):
        b1, b2, b3 = bs[0], bs[1], bs[2]
        return b1 * np.logical_or(b2,b3)

    def xom(self, bs, iter):
        return self.mutation([self.xor(bs)], iter)








