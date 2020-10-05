from torch.utils.data import DataLoader
import torch.utils.data
import sys
import os

PYTHONPATH = 'specify the python path to folder'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from testbeds.mnist_torch import *
from utils.func_support import *

def test():
    clf.eval()
    tb_error = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            error, predicted = clf.calculate_classification_error(inputs, targets)
            tb_error.append(error)

        avg_terror = np.mean(np.array(tb_error))

    return avg_terror


models = ['model'+str(i) + str('.pt') for i in range(10)]
path = 'specify the path where the model is stored'

rescale = 14
hidden_units = 20
output = 1

testloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=False,path='internal'),
                        batch_size=128, shuffle=True)

total_error = np.zeros((len(models),))

for idx,model in enumerate(models):
    clf = torch.load(path+model)
    error=test()
    total_error[idx]=error

avg_error = np.mean(total_error)
ste = np.std(total_error)/np.sqrt(len(total_error))
print('avg error {} ste {}'.format(avg_error, ste))
