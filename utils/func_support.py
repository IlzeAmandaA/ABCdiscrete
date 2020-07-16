import numpy as np
import matplotlib.pyplot as plt

formats = {'mut': '--or', 'mut+crx': ':^g', 'mut+xor': '-.vb',
           'de-mc':'--or', 'de-mc1':':^g', 'de-mc2':'-.vb'} #check
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html

def initialze_storage(settings):
    store={}
    for key in settings:
        store[key] = []
    return store

def text_output(method, iter, solution, simulation, store):
    textfile = open(store+ method + '.txt', 'a+')
    textfile.write('------------------------------------------------\n')
    textfile.write('Iteration {}\n'.format(iter))
    if np.array_equal(solution, simulation.model.b_truth):
        textfile.write('---MATCH---')
    else:
        textfile.write('--MISMATCH of {} --'.format(simulation.model.error(solution)))
    textfile.write('\n b truth\n')
    textfile.write(str([int(n) for n in simulation.model.b_truth]))
    textfile.write('\n target posterior {} '.format(simulation.model.posterior(simulation.model.b_truth)))
    textfile.write('\n target likelihood {} '.format(simulation.model.product_lh(simulation.model.b_truth)))
    textfile.write('\n best simulated b\n')
    textfile.write(str([int(n) for n in solution]))
    textfile.write('\n best posterior {} '.format(simulation.model.posterior(solution)))
    textfile.write('\n best likelihood {} '.format(simulation.model.product_lh(solution)))
    textfile.write('\n\n')

def prepare_data(dict):
    print(dict.keys())
    overall={}
    for key, values in dict.items():
        print(values)
        overall[key] = {}
        overall[key]['mean'] = np.mean(np.asarray(values), axis=0)
        overall[key]['std'] = np.std(np.asarray(values), axis=0)

    return overall

def plot(avg_dict, location, yaxis):

    formating = {key:formats[key] for key in avg_dict}
    plt.figure(figsize=(16, 6))

    for transformation in avg_dict:
        results = avg_dict[transformation]
        y = results['mean']
        std = results['std']
        x = [i * 500*12 for i in range(len(y))]
        assert len(x) == len(y) == len(std), 'The number of instances fo not match, check create plot function'
        plt.errorbar(x, y, yerr=std, fmt=formating[transformation], label=transformation, capsize=10)

    plt.xlabel('evaluations')
    plt.ylabel(yaxis)
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(location + '.png')
    # plt.show()


def create_plot(results, location, yaxis):
    averages = prepare_data(results)
    plot(averages, location, yaxis)

def plot_pop(results, name, true=None):
    formating = {key:formats[key] for key in results}

    plt.figure(figsize=(16, 6))

    for transformation in results:
        y = results[transformation]
        std = [0*i for i in range(len(y))]
        x = [i * 500*12 for i in range(len(y))]
        assert len(x) == len(y) == len(std), 'The number of instances fo not match, check create plot function'
        plt.errorbar(x, y, yerr=std, fmt=formating[transformation], label=transformation)

    # try:
    #     plt.hlines(true, xmin=x[0], xmax=x[-1], colors='c')
    # except ValueError:
    #     pass

    plt.xlabel('evaluations')
    plt.ylabel(name)
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig('results/benchmark/pop_'+ name + '_.png')
    # plt.show()




