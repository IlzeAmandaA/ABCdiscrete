import matplotlib.pyplot as plt

"""
Functions for data visualization purposes
"""

def create_plot(data_dict, store):

    for transformation in data_dict:
        results = data_dict[transformation]
        y = results['mean']
        std = results['std']
        x = [i*500 for i in range(len(y))]
        assert len(x)==len(y)==len(std), 'The number of instances fo not match, check create plot function'
        plt.errorbar(x, results['mean'], results['std'])
        #plt.savefig(store+'name.png')
        plt.show()
