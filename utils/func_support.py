import numpy as np
import matplotlib.pyplot as plt
from operator import  add

formats = {'mut': '--or', 'mut+crx': ':^g', 'mut+xor': '-.vb',
           'de-mc':'--or', 'de-mc1':':^g', 'de-mc2':'-.vb'} #check
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html

line = {'mut': '--o', 'mut+crx': ':^', 'mut+xor': '-.v',
        'de-mc':'-<', 'de-mc1':':^', 'de-mc2':'-.v'
        }


color = {'mut': '#fa4224', 'mut+crx': '#388004', 'mut+xor': '#004577',
         'de-mc':'#ff028d', 'de-mc1':'#388004', 'de-mc2':'#004577'
         }

fill = {'mut': '#FF9848', 'mut+crx': '#c7fdb5', 'mut+xor': '#95d0fc',
        'de-mc':'#ffb2d0', 'de-mc1':'#c7fdb5', 'de-mc2':'#95d0fc'
        }


def text_output(method, iter, solution, simulation, store):
    textfile = open(store + '/'+ method + '.txt', 'a+')
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

def report(dict, store):
    textfile = open(store + '.txt', 'a+')
    for method, values in dict.items():
        textfile.write('proposal : {}'.format(method))
        textfile.write('acceptance ratio : mean {} (std {}) \n'.format(values['mean'], values['std']))


def create_plot(results, x, location, yaxis, transform=False, ylim=None, xlim=None, length=16, height=6):
    averages = compute_statistics(results, x, transform)
    plot(averages, location, yaxis, ylim, xlim, length, height)

def compute_statistics(dict, x, transform=False):
    overall={}
    for key, values in dict.items():
        assert len(values[0]) == len(values[1]), 'issue with lenghts'
        overall[key] = {}
        values = np.exp(-(np.asarray(values))) if transform else np.asarray(values)
        overall[key]['mean'] = np.mean(values, axis=0)
        overall[key]['std'] = np.std(values, axis=0)

    for key, values in x.items():
        overall[key]['x'] = np.mean(np.asarray(values), axis=0)

    return overall

def compute_avg(dict):
    overall={}
    for key, values in dict.items():
        overall[key] = {}
        print(values)
        overall[key]['mean'] = np.mean(np.asarray(values))
        overall[key]['std'] = np.std(np.asarray(values))

    return overall


def plot(avg_dict, location, yaxis, ylim, xlim, length=16, height=6):
    plt.figure(figsize=(length, height))

    for transformation, results in avg_dict.items():
        y = results['mean']
        std = results['std']
        x = results['x']
        y_min=y-std
        y_plus=y+std
        assert len(x) == len(y) == len(std), 'The number of instances fo not match, check create plot function'

        plt.plot(x,y, line[transformation], color=color[transformation], label = transformation)
        plt.fill_between(x, y_min, y_plus,
                         alpha=0.5, edgecolor=color[transformation], facecolor=fill[transformation])

    if ylim is not None:
        a,b=ylim
        plt.ylim(a, b)
    if xlim is not None:
        a,b = xlim
        plt.xlim(a,b)
    plt.xlabel('evaluations')
    plt.ylabel(yaxis)
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(location + '.png')
    # plt.show()


def plot_single(results, points, name, location):

    plt.figure(figsize=(16, 6))

    for transformation, data in results.items():
        y = np.asarray(data)
        std = np.std(y)
        y_min = y+std
        y_plus = y-std
        x = points[transformation]
        assert len(x) == len(y), 'The number of instances fo not match, check create plot function'
        plt.plot(x, y, line[transformation], color=color[transformation], label=transformation)
        plt.fill_between(x, y_min, y_plus,
                         alpha=0.5, edgecolor=color[transformation], facecolor=fill[transformation])


    plt.xlabel('evaluations')
    plt.ylabel(name)
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(location+ '.png')





