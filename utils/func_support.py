import numpy as np
import matplotlib.pyplot as plt


def initialze_storage(settings):
    store={}
    for key in settings:
        store[key] = []
    return store

def text_output(method, iter, solution, simulation):
    textfile = open('results/benchmark/'+ method + '_benchmark.txt', 'a+')
    textfile.write('------------------------------------------------\n')
    textfile.write('Iteration {}\n'.format(iter))
    if np.array_equal(solution, simulation.model.b_truth):
        textfile.write('---MATCH---')
    else:
        textfile.write('--MISMATCH --')
    textfile.write('\n b truth\n')
    textfile.write(str([int(n) for n in simulation.model.b_truth]))
    textfile.write('\n best simulated b\n')
    textfile.write(str([int(n) for n in solution]))
    textfile.write('\n\n')

def prepare_data(dict):
    overall={}
    for key, values in dict.items():
        overall[key] = {}
        overall[key]['mean'] = np.mean(np.asarray(values), axis=0)
        overall[key]['std'] = np.std(np.asarray(values), axis=0)

    return overall

def plot(avg_dict, location):

    formats = ['--or',':^g','-.vb']
    formating = {key:formats[id] for id, key in enumerate(avg_dict)}

    plt.figure(figsize=(16, 6))

    for transformation in avg_dict:
        results = avg_dict[transformation]
        y = results['mean']
        std = results['std']
        x = [i * 250 for i in range(len(y))]
        assert len(x) == len(y) == len(std), 'The number of instances fo not match, check create plot function'
        plt.errorbar(x, results['mean'], yerr=results['std'], fmt=formating[transformation], label=transformation)

    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(location + 'benchmark_Stren.png')
    plt.show()


def create_plot(results, location):
    averages = prepare_data(results)
    plot(averages, location)



