import pickle as pkl
import numpy as np
import os
import sys
import pandas as pd

PYTHONPATH = 'specify the python path to folder'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.mnist import MNIST

def eval_test(method_id):
    all_min_err =[]
    theta_best = None
    error_argmin=np.inf
    total_error = np.zeros((len(storage),))
    for id,store in enumerate(storage):
        dict = pkl.load(open(store + 'pop_store' + str(type) + '.pkl', 'rb'))
        Y_hat = []
        for runid, data in dict.items():
            for method, method_dict in data.items():
                if method == method_id:
                    for cross_eval, pop in method_dict.items():
                        for idx,chain in enumerate(pop):
                            y_hat = use_case.simulate(chain, eval=True)
                            error_min = use_case.distance(y_hat,eval=True)
                            Y_hat.append(y_hat)
                            all_min_err.append(error_min)
                            if error_min<error_argmin:
                                error_argmin=error_min
                                theta_best = y_hat

        Y_df = pd.DataFrame(Y_hat)
        Y_mode = np.array(Y_df.mode(axis=0))[0]
        error = use_case.distance(Y_mode, eval=True)
        total_error[id] = error

    avg_error = np.mean(total_error)
    ste = np.std(total_error) / np.sqrt(len(total_error))
    dist = 1 - np.mean(theta_best)
    report_txt(method_id, avg_error, ste, error_argmin, all_min_err, dist)

def report_txt(method_id, avg_error, ste, error_argmin, all_argmin, dist):
    textfile = open(res + 'test_results_' + str(type) + '.txt', 'a+')
    textfile.write('method {}\n'.format(method_id))
    textfile.write('avg error ensemble {} ste {} \n'.format(avg_error, ste))
    textfile.write('min error {} ste {} \n'.format(error_argmin, np.std(np.array(all_argmin))/np.sqrt(len(all_argmin))))
    textfile.write('percentage of 0s: {}'.format(dist))
    textfile.write('---------------------------------\n')

DATA_PATH = 'specify where to store the MNIST dataset'
STORE =  'specify the path where the result files are stored'
n = 5 #how many cross evaluations performed
storage = [STORE+str(i*10)+'/' for i in range(n)]
res = 'specify the path where to store the result'
type = 0.05 #best tolerance threshold value

image_size = (14, 14)
hidden_units = 20

labels = [0,1]
use_case = MNIST(path=DATA_PATH,l1=labels[0], l2=labels[1], image_size=image_size, H=hidden_units)


for method in ['dde-mc', 'mut+xor', 'ind-samp']:
    eval_test(method)



