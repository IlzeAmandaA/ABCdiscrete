import pickle as pkl
import matplotlib.pyplot as plt
"""
Code for loading the pickle file to create different version of the plots 
"""

data = pkl.load(open('results/benchmark/_p_mut_0.1_p_cross_0.5_results_benchmark.pkl', 'rb'))

formating={'mutation': '--or', 'cross': ':^g', 'xor':'-.vb'}
plt.figure(figsize=(16,6))

for transformation in data:
    print(transformation)
    results = data[transformation]
    y = list(results['mean'])[::2]
    std = list(results['std'])[::2]
    x = [ i*250*2 for i in range(len(y))]
    assert len(x)==len(y )==len(std), 'The number of instances fo not match, check create plot function'
    plt.errorbar(x, y, yerr=std,
                 fmt=formating[transformation],
                 label=transformation)


plt.xlabel('iterations')
plt.ylabel('error')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('/home/ilze/MasterThesis/results/benchmarks/may20/' +'_p_mut_0.1_p_cross_0.5.png')
plt.show()
