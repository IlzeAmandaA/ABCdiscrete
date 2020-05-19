import matplotlib.pyplot as plt

"""
Functions for data visualization purposes
"""

def create_plot(data_dict, store):

    formating=[('--or'), (':^g'), ('-.vb')]

    for i,transformation in enumerate(data_dict):
        results = data_dict[transformation]
        y = results['mean']
        std = results['std']
        x = [i*250 for i in range(len(y))]
        assert len(x)==len(y)==len(std), 'The number of instances fo not match, check create plot function'
        plt.errorbar(x, results['mean'], yerr=results['std'], fmt=formating[i],label=transformation)

    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(store+'benchmark_Stren.png')
    plt.show()
