import pickle as pkl
import matplotlib.pyplot as plt
"""
Code for loading the pickle file to create different version of the plots 
"""

data = pkl.load(open('results/benchmark/results_benchmark.pkl', 'rb'))

formating =[('o', 'dotted'), ('^' ,'dashed')]
plt.figure(figsize=(16,6))

for i ,transformation in enumerate(data):
    results = data[transformation]
    y = list(results['mean'])[0:15]
    std = list(results['std'])[0:15]
    x = [ i*500 for i in range(len(y))]
    assert len(x)==len(y )==len(std), 'The number of instances fo not match, check create plot function'
    plt.errorbar(x, y, yerr=std,
                 fmt=formating[i][0], linestyle=formating[i][1],
                 label=transformation)


plt.xlabel('iterations')
plt.ylabel('error')
plt.grid(True)
plt.legend(loc=0)
# plt.savefig('results/benchmark/' +'benchmark_Stren_closeup.png')
plt.show()
