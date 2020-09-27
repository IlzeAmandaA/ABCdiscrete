import pickle as pkl
from torch.utils.data import DataLoader
import torch.utils.data
import torch.optim as optim
import sys
import os

PYTHONPATH = '/home/ilze/PycharmProjects/MasterThesis/ABCdiscrete/experiments'
sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from algorithms.bnn import Network
from testbeds.mnist_torch import *
from utils.func_support import *


print('Loading Data')
rescale = 14
trainloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=True,path='internal'),
                         batch_size=128, shuffle=True)
testloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=False,path='internal'),
                        batch_size=128, shuffle=True)

evaluate = 5
epochs = 50
early_stop = False
cross_tr_loss = []
cross_te_loss = []
cross_w = []

# for eval in range(evaluate):
#     print('Evaluation {} \n'.format(eval))
#     cuda_available = torch.cuda.is_available()
#     torch.manual_seed = (0)
#     if cuda_available:
#         torch.cuda.manual_seed(0)
#         print('running on GPU')
#
#     hidden_units = 20
#     output = 1
#     clf = Network(rescale * rescale, output, hidden_units)
#     if cuda_available:
#         clf = clf.cuda()
#
#     optimizer = optim.Adam(clf.parameters(), lr=0.0001)
#
#     min_loss = np.inf
#     n_epochs_stop = 10
#     epochs_no_improve = 0
#     best_params = None
#
#     # e=[]
#     train_loss = []
#     test_error = []
#
#     for epoch in range(1,epochs+1):
#         # e.append(epoch)
#         train(epoch)
#         error = test()
#         test_error.append(error)
#         if error < min_loss:
#             epochs_no_improve = 0
#             min_loss = error
#             best_params = clf.parameters()
#             # print('Epoch %d, Test Error: %.3f' % (epoch, error))
#             # print('--------------------------------------------------------------')
#
#         else:
#             epochs_no_improve += 1
#
#         if early_stop and epoch > 5 and epochs_no_improve == n_epochs_stop:
#             print('--------------------------------------------------------------')
#             print('Early Stopping')
#             print('Best error obtain on training set %.3f'%(min_loss))
#             print('--------------------------------------------------------------')
#             break
#
#     cross_tr_loss.append(train_loss)
#     cross_te_loss.append(test_error)
#     cross_w.append(content(best_params))
