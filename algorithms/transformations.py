import numpy as np

"""
Class for transformation-kernel possibilities
"""

class Evolution():

    def __init__(self,type, prob):
        self.type = type
        self.prob = prob

    def recombination(self, b):
        """
        Function for selecting the correct transformation
        :param b: np.array
        :return: modified np.array
        """
        b_prime = b.copy()
        for i, val in enumerate(b_prime):
            if self.type == 'mutation':
                b_prime[i]=self.bit_flip(val)
            elif self.type == 'm_cross':
                pass #to be implemented
        return b_prime

    def bit_flip(self, val):
        """
        function to perform bit flipping
        :param val: input binary value
        :return: flipped bit based on mutation probability
        """
        return 1 - val if np.random.uniform(0, 1) <= self.prob else val



