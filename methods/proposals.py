import numpy as np

class Proposals():

    def __init__(self, pflip,pcross):
        self.pflip = pflip
        self.pcross = pcross


    #Strens
    def bit_flip(self, chain):
        new = chain.copy()
        for id, bit in enumerate(chain):
            new[id] = self.flip(bit)
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


    def flip(self, val):
        bit = 1 - val if self.pflip >= np.random.uniform(0, 1) else val
        return bit


    #Braak discrete
    def de_mc(self, chain_i, chain_j, chain_k):
        bf_diff=self.bit_flip(np.logical_xor(chain_j, chain_k).astype(int))
        return np.logical_xor(chain_i,bf_diff).astype(int)

    def de_mc1(self, chain_i, chain_j, chain_k):
        xor=np.logical_xor(chain_i, np.logical_xor(chain_j, chain_k).astype(int))
        return self.bit_flip(xor)

    def de_mc2(self, chain_i, chain_j, chain_k):
        diff=np.logical_xor(chain_j, chain_k).astype(int)
        ep_diff =np.logical_xor(diff, np.random.binomial(1, 0.5, len(diff))).astype(int)
        return np.logical_xor(chain_i, ep_diff).astype(int)



