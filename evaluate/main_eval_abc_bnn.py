import pickle as pkl
import numpy as np
import os
import sys
import pandas as pd

PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/evaluate'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.mnist_numpy import MNIST

def eval_test(method_id):
    error_argmin=np.inf
    total_error = np.zeros((len(storage),))
    for id,store in enumerate(storage):
        dict = pkl.load(open(store + 'pop_store' + type + '.pkl', 'rb'))
        Y_hat = []
        for runid, data in dict.items():
            for method, method_dict in data.items():
                if method == 'de-mc':
                    method = 'dde-mc'
                if method == method_id:
                    for cross_eval, pop in method_dict.items():
                        for idx,chain in enumerate(pop):
                            y_hat, _= use_case.simulate(chain, eval=True)
                            error_min = use_case.distance(y_hat,eval=True)
                            Y_hat.append(y_hat)
                            if error_min<error_argmin:
                                error_argmin=error_min

        Y_df = pd.DataFrame(Y_hat)
        Y_mode = np.array(Y_df.mode(axis=0))[0]
        error = use_case.distance(Y_mode, eval=True)
        report_txt(method_id,  id, error)

        total_error[id] = error

    avg_error = np.mean(total_error)
    ste = np.std(total_error) / np.sqrt(len(total_error))
    print('method {} avg error {} ste {}, argmin error {}'.format(method_id, avg_error, ste, error_argmin))

def report_txt(method, id, error):
    textfile = open(res + 'test_results.txt', 'a+')
    textfile.write('Model id {} \n'.format(id))
    textfile.write('Results for proposal {} \n'.format(method))
    textfile.write('Test error obtained: {} \n'.format(error))
    textfile.write('---------------------------------\n')


loc='/home/ilze/PycharmProjects/MasterThesis/ensemble/argseed'
storage = [loc+str(i*10)+'/' for i in range(10)]
res = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/results/abc/bnn_mnist/'
type = '0.04'

image_size = (14, 14)
hidden_units = 20

labels = [0,1]
use_case = MNIST(l1=labels[0], l2=labels[1], image_size=image_size, H=hidden_units, path='internal')


for method in ['dde-mc', 'mut+xor']:
    eval_test(method)


#
# best= (np.inf, None)
#
# for method, run in data.items():
#     results = {}
#
#     for i, pop in enumerate(run):
#         pop_test_error = np.zeros((len(pop),))
#
#         for idx, chain in enumerate(pop):
#             #performance
#             Y_hat = use_case.simulate(chain, eval=True)
#             error = use_case.distance(Y_hat, eval=True)
#             pop_test_error[idx] = error
#
#         min_error = (np.min(pop_test_error), pop[np.argmin(pop_test_error)])
#         if min_error[0] < best[0]:
#             best = min_error
#
#         results[str(i)] = (np.mean(pop_test_error), np.std(pop_test_error))
#
#     results['min_error'] = best[0]
#     results['dist'] = (1 - np.mean(best[1]))*100
#
#
#     report_txt(method, results, len(run))
#
#
#


