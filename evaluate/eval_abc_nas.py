import pickle as pkl
import numpy as np
import os
import sys

PYTHONPATH = 'specify the python path to folder'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.nas import NAS

def compute_argmin(method_id):
    all = []
    best = (np.inf, None)
    results = {}
    for runid, res in data.items():
        for method, population in res.items():
            if method == method_id:
                pop_test_error = np.zeros((len(population),))
                for idx, chain in enumerate(population):
                    Y_hat = use_case.simulate(chain, eval=True)
                    error = use_case.distance(Y_hat, eval=True)
                    pop_test_error[idx] = error

                min_error = (np.min(pop_test_error), population[np.argmin(pop_test_error)])
                all.append(min_error[0])
                if min_error[0] < best[0]:
                    best = min_error

                results[str(runid)] = (np.mean(pop_test_error), np.std(pop_test_error))

    results['min_error'] = best[0]
    results['dist'] = (1 - np.mean(best[1])) * 100
    results['ste'] = np.std(np.array(all))/np.sqrt(len(all))
    report_txt(method_id, results, len(data.keys()))


def report_txt(method, results, N):
    textfile = open(res + 'test_error_' + type + '.txt', 'a+')
    textfile.write('Results for proposal {} \n'.format(method))
    textfile.write('Minimum error obtain on test set {} (ste {}) \n'.format(results['min_error'], results['ste']))
    textfile.write('With a population chain of 0 distribution of {} \n'.format(results['dist']))
    textfile.write('The avg error (Std) per run across chains: \n')
    for i in range(N):
        textfile.write('run {}: error {}, std {} \n'.format(str(i), results[str(i)][0], results[str(i)][1]))
    textfile.write('---------------------------------\n')


DATA_PATH = 'specify the path to where the nasbench_only108.tfrecord is stored'
STORE =  'specify the path where the result files are stored'
type = '0.2' #best epsilon threshold
data=pkl.load(open(STORE+'pop_store' + type + '.pkl', 'rb'))
res = 'specify the path where to store the result'


image_size = (14, 14)
hidden_units = 20

labels = [0,1]
use_case = NAS(path=DATA_PATH)

for method_id in ['mut+xor', 'dde-mc', 'id-samp']:
    print(method_id)
    compute_argmin(method_id)









