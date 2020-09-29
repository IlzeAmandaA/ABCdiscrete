import numpy as np

class Proposals():

    def __init__(self, pflip,pcross):
        self.pflip = pflip
        self.pcross = pcross

    def bit_flip(self, chain):
        shape = chain.shape
        new = chain.copy().flatten()
        for id, bit in enumerate(chain.flatten()):
            new[id] = self.flip(bit)
        new = np.reshape(new, shape)
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


    def dde_mc(self, i, j, k):
        diff = np.logical_xor(j, k).astype(int)
        mut_diff = self.bit_flip(diff)
        return np.logical_xor(i, mut_diff).astype(int)


    def dde_mc1(self, chain_i, chain_j, chain_k):
        xor=np.logical_xor(chain_i, np.logical_xor(chain_j, chain_k).astype(int))
        return self.bit_flip(xor)

    def dde_mc2(self, chain_i, chain_j, chain_k):
        diff=np.logical_xor(chain_j, chain_k).astype(int)
        ep_diff =np.logical_xor(diff, np.random.binomial(1, 0.5, len(diff))).astype(int)
        return np.logical_xor(chain_i, ep_diff).astype(int)


    def indepent_sampler(self, chain):
        return np.random.binomial(1, 0.5, len(chain))

