import pickle as pkl
import numpy as np
import os
import sys

###### CHECK THIS ONCE RE-RUN

PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/evaluate'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.nas import NAS


def report_txt(method, results, N):
    textfile = open(loc + 'results' + type + '.txt', 'a+')
    textfile.write('Results for proposal {} \n'.format(method))
    textfile.write('Minimum error obtain on test set {} \n'.format(results['min_error']))
    textfile.write('With a population chain of 0 distribution of {} \n'.format(results['dist']))
    textfile.write('The avg error (Std) per run across chains: \n')
    for i in range(N):
        textfile.write('run {}: error {}, std {} \n'.format(str(i), results[str(i)][0], results[str(i)][1]))
    textfile.write('---------------------------------\n')


loc = '/home/ilze/PycharmProjects/MasterThesis/NAS/nas/'
type = '0.01'
data=pkl.load(open(loc+'pop_store' + type + '.pkl', 'rb'))

image_size = (14, 14)
hidden_units = 20

labels = [0,1]
use_case = NAS()

best= (np.inf, None)

for method, run in data.items():
    results = {}

    for i, pop in enumerate(run):
        pop_test_error = np.zeros((len(pop),))

        for idx, chain in enumerate(pop):
            #performance
            Y_hat = use_case.simulate(chain, eval=True)
            error = use_case.distance(Y_hat, eval=True)
            pop_test_error[idx] = error

        min_error = (np.min(pop_test_error), pop[np.argmin(pop_test_error)])
        if min_error[0] < best[0]:
            best = min_error

        results[str(i)] = (np.mean(pop_test_error), np.std(pop_test_error))

    results['min_error'] = best[0]
    results['dist'] = (1 - np.mean(best[1]))*100


    report_txt(method, results, len(run))








