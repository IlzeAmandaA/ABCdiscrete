import numpy as np

class Proposals():

    def __init__(self, pflip=0.1,pcross=0.5):
        self.pflip = pflip
        self.pcross = pcross


    #Strens
    def mutation(self, chain):
        new = chain.copy()
        for id, bit in enumerate(chain):
            new[id] = self.bit_flip(bit)
        return new

    def crossover(self, chain_i, chain_j):
        c1 = chain_i.copy()
        c2 = chain_j.copy()
        for idx,bit in enumerate(chain_i):
            if self.pcross >= np.random.uniform(0,1):
                c2[idx] = bit
                c1[idx] = chain_j[idx]
        return c1, c2

    def xor(self, chain_i, chain_j, chain_k):
        return np.logical_xor(chain_i,np.logical_xor(chain_j,chain_k)).astype(int)


    def bit_flip(self, val):
        bit = 1 - val if self.pflip >= np.random.uniform(0, 1) else val
        return bit


    #Braak
    def de_mc(self, chain_i, chain_j, chain_k):
        new=self.mutation(np.logical_xor(chain_j, chain_k).astype(int))
        return np.logical_xor(chain_i,new).astype(int)
