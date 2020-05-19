import numpy as np

#Underlying probability distirbution of the data

def likelihood(q_i0, q_il, b, finding_idx, f_i=1):
    """
    Function to estimate the likelihood P(f_i=1|b)
    :param finding_idx
    :return: probability
    """
    if f_i:
        return 1 - np.exp(np.log(1 - q_i0[finding_idx]) + np.sum(b * np.log(1 - q_il[finding_idx])))
    else:
        return np.exp(np.log(1 - q_i0[finding_idx]) + np.sum(b * np.log(1 - q_il[finding_idx])))


def prior(b, p_l):
    """
    Function to estimate the prior probability of a given input string
    :param b: binary vector (np.array)
    :return: probability
    """
    return np.sum(b * np.log(p_l) + (1 - b) * np.log(1 - p_l))

