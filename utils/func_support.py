import numpy as np
import matplotlib.pyplot as plt
from operator import  add
import collections

formats = {'mut': '--or', 'mut+crx': ':^g', 'mut+xor': '-.vb',
           'dde-mc':'--or', 'dde-mc1':':^g', 'dde-mc2':'-.vb',
           'id-samp':':^g', 'ind-samp':':^g',
           'dde-mc*': ':^g', 'mut+xor*': '--or'} #check
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html

line = {'mut': '--o', 'mut+crx': ':^', 'mut+xor': '-.v',
        'dde-mc':'-<', 'dde-mc1':':^', 'dde-mc2':'-.v',
        'id-samp':':^', 'ind-samp':':^',
        'dde-mc*': ':^', 'mut+xor*': '--o'
        }


color = {'mut': '#fa4224', 'mut+crx': '#7e1e9c', 'mut+xor': '#004577',
         'dde-mc':'#388004', 'dde-mc1':'#388004', 'dde-mc2':'#004577',
         'id-samp': '#ff028d', 'ind-samp': '#ff028d',
         'dde-mc*': '#6f7c00', 'mut+xor*': '#13bbaf'
         }

fill = {'mut': '#FF9848', 'mut+crx': '#efc0fe', 'mut+xor': '#95d0fc',
        'dde-mc':'#c7fdb5', 'dde-mc1':'#c7fdb5', 'dde-mc2':'#95d0fc',
        'id-samp': '#ffb2d0', 'ind-samp': '#ffb2d0',
        'dde-mc*': '#d0e429', 'mut+xor*': '#cafffb'
        }


def text_output(method, iter, solution, simulation, store):
    textfile = open(store + '/'+ method + '.txt', 'a+')
    textfile.write('------------------------------------------------\n')
    textfile.write('Iteration {}\n'.format(iter))
    if np.array_equal(solution, simulation.simulator.parameters):
        textfile.write('---MATCH---')
    else:
        textfile.write('--MISMATCH of {} --'.format(simulation.simulator.hamming(solution, simulation.simulator.parameters)))
    textfile.write('\n b truth\n')
    textfile.write(str([int(n) for n in simulation.simulator.parameters]))
    textfile.write('\n target posterior {} '.format(simulation.simulator.posterior(simulation.simulator.parameters)))
    textfile.write('\n target likelihood {} '.format(simulation.simulator.product_lh(simulation.simulator.parameters)))
    textfile.write('\n best simulated b\n')
    textfile.write(str([int(n) for n in solution]))
    textfile.write('\n best posterior {} '.format(simulation.simulator.posterior(solution)))
    textfile.write('\n best likelihood {} '.format(simulation.simulator.product_lh(solution)))
    textfile.write('\n\n')

def report_weight(list_avg, loc):
    textfile = open(loc + '.txt', 'a+')
    textfile.write('Percentage of active wegihts: {} (std {}) \n'.format(np.mean(list_avg), np.std(list_avg)))


def report(dict,epsilon, store):
    textfile = open(store + '.txt', 'a+')
    textfile.write('epsilon value: {} \n'.format(epsilon))
    for method, values in dict.items():
        textfile.write('proposal : {} \n'.format(method))
        textfile.write('acceptance ratio : mean {} (std {}) \n'.format(values['mean'], values['std']))
    # textfile.write('--------------------- \n\n')

def report_variablitity(data, store):
    textfile = open(store + '.txt', 'a+')
    for runid, info in data.items():
        for method, var in info.items():
            mean = np.mean(np.asarray(var))
            std = np.std(np.asarray(var))
        textfile.write('for run {} \n'.format(runid))
        textfile.write('avg variability : {}  (std {})  '.format(mean, std))
        textfile.write('--------------------- \n\n')

def report_posterior(sim, run, method, pops, store):
    posterior_list = []
    textfile = open(store + '.txt', 'a+')
    textfile.write('\nRun: {} \n'.format(run))
    textfile.write('Method: {} \n'.format(method))

    for chain in pops:
        posterior_list.append(sim.simulator.log_posterior_abc(chain))
    post_avg = np.mean(np.asarray(posterior_list))
    post_std = np.std(np.asarray(posterior_list))
    textfile.write('avg post : {}  (std {})  \n'.format(post_avg, post_std))

    true_post = sim.simulator.log_posterior_abc(sim.simulator.parameters)
    textfile.write('true post : {}  '.format(true_post))
    textfile.write('--------------------- \n\n')

    return ([post_avg, post_std], true_post, posterior_list)


def create_plot(results, x, location, yaxis, transform=False, ylim=None, xlim=None, length=16, height=6):
    averages = compute_statistics(results, x, transform)
    plot(averages, location, yaxis, ylim, xlim, length, height)

def compute_statistics(dict, x, transform=False):
    overall={}

    for key, values in dict.items():
        # assert len(values[0]) == len(values[1]), 'issue with lenghts'
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
        overall[key]['mean'] = np.mean(np.asarray(values))
        overall[key]['std'] = np.std(np.asarray(values)) # / len(values)

    return overall


def plot(avg_dict, location, yaxis, ylim, xlim, length=16, height=6):
    plt.figure(figsize=(length, height))
    # order = ['dde-mc*', 'mut+xor*', 'dde-mc', 'mut+xor']
    #
    # for transformation in order:
    #     results = avg_dict[transformation]
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

def plot_bnn(list, location, yaxis, ylim=None, xlim=None, length=16, height=6):
    plt.figure(figsize=(length, height))

    y = np.mean(np.asarray(list), axis=0)
    std = np.std(np.asarray(list), axis=0)
    x = [i for i in range(1, len(list[0])+1)]
    y_min=y-std
    y_plus=y+std
    assert len(x) == len(y) == len(std), 'The number of instances fo not match, check create plot function'

    plt.plot(x,y, ':^g', color='#004577', label = 'test error')
    plt.fill_between(x, y_min, y_plus,
                     alpha=0.5, edgecolor='#004577', facecolor='#95d0fc')

    if ylim is not None:
        a,b=ylim
        plt.ylim(a, b)
    if xlim is not None:
        a,b = xlim
        plt.xlim(a,b)
    plt.xlabel('epochs')
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

def plot_dist(dict_res, dict_true, location):
    format = {'mut+xor':['o','green','lightgreen'], 'dde-mc':['v','blue','lightblue']}
    od_res = collections.OrderedDict(sorted(dict_res.items()))
    od_true = collections.OrderedDict(sorted(dict_true.items()))
    move = 0.08

    fig, ax = plt.subplots(figsize=(16, 6))

    for selection in ['mut+xor', 'dde-mc']:
        x = []
        y = []
        std = []

        for id in od_res:
            y.append(od_res[id][selection][0])
            std.append(od_res[id][selection][1])
            if selection == 'mut+xor':
                x.append(float(id)+move)
            else:
                x.append(float(id)-move)

        ax.errorbar(x, y, yerr = std, label = selection,
                    fmt = format[selection][0], color = format[selection][1], ecolor = format[selection][2],
                    elinewidth=1, capsize=1)

    x = [int(k) for k in od_true.keys()]
    ax.scatter(x, [v['dde-mc'] for v in od_true.values()], color='magenta', label='true posterior', marker='*')

    # Set plot title and axes labels
    ax.set(xlabel="Run",
           ylabel="Posterior")

    plt.xticks(x, [str(id) for id in x])
    #plt.ylim(-25, 15)
    plt.legend()
    plt.savefig(location + '.png')






