import pickle as pkl
import numpy as np
import os
import sys
import pandas as pd

PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/evaluate'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.mnist_numpy import MNIST

def eval_test(method_id):
    Y_hat = []
    for runid, data in dict.items():
        for method, method_dict in data.items():
            if method == method_id:
                for cross_eval, pop in method_dict.items():
                    for idx,chain in enumerate(pop):
                        y_hat, _=use_case.simulate(chain, eval=True)
                        Y_hat.append(y_hat)

    Y_df = pd.DataFrame(Y_hat)
    Y_mode = np.array(Y_df.mode(axis=0))[0]
    error = use_case.distance(Y_mode, eval=True)
    report_txt(method_id, error)

def report_txt(method, error):
    textfile = open(res + 'test_results.txt', 'a+')
    textfile.write('Results for proposal {} \n'.format(method))
    textfile.write('Test error obtained: {} \n'.format(error))
    textfile.write('---------------------------------\n')


loc='/home/ilze/PycharmProjects/MasterThesis/ensemble/bnn_mnist/'
res = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/results/abc/bnn_mnist/'
type = '0.04'
dict=pkl.load(open(loc+'pop_store' + type + '.pkl', 'rb'))
image_size = (14, 14)
hidden_units = 20

labels = [0,1]
use_case = MNIST(l1=labels[0], l2=labels[1], image_size=image_size, H=hidden_units, path='internal')

method_id='de-mc'

for method in ['de-mc', 'mut+xor']:
    eval_test(method)


# print('Test error of {} for ensemble approach: {}'.format(method, error))





