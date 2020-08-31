from methods.cnn import Binary_CNN
from experiments.mnist_torch import *
from utils.func_support import *
import pickle as pkl
from torch.utils.data import DataLoader
import torch.utils.data
import torch.optim as optim
import numpy as np
import torch.nn as nn
import os
import argparse

parser = argparse.ArgumentParser(description='ABC models for discrete data')
parser.add_argument('--lr', type=float, default=0.01, metavar='float',
                    help='evaluation steps') #600000
parser.add_argument('--N', type=float, default=60000, metavar='float',
                    help='evaluation steps') #600000


args = parser.parse_args()


def train(epoch):
    batch_loss = []
    clf.train()
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.LongTensor)

        optimizer.zero_grad()
        output = clf(inputs)
        loss = criterion(output, targets)

        loss.backward()

        for p in list(clf.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)

        optimizer.step()

        for p in list(clf.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

        batch_loss.append(loss.item())

    avg_loss = np.mean(batch_loss)
    train_loss.append(avg_loss)

    # if epoch%5==0:
    print('Train Epoch: %d Training Loss %.3f' % (epoch,  avg_loss))
# print('Training Loss : %.3f Time : %.3f seconds ' % (np.mean(avg_loss)), end - start))

def test(epoch):
    clf.eval()
    test_error = []
    test_loss = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # inputs = inputs.type(torch.FloatTensor)
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.LongTensor)

            Y_out, error = clf.calculate_classification_error(inputs, targets)
            loss = criterion(Y_out, targets)
            test_error.append(error)
            test_loss.append(loss)

        avg_error = np.mean(np.array(test_error))
        avg_loss = np.mean(np.array(test_loss))

        # if epoch % 10 == 0:
        print('Test Error: %.3f' % (avg_error))
        print('Test Loss: %.3f' % (avg_loss))
        print('--------------------------------------------------------------')

    return avg_error

def content(data):
    weights = []
    for param in list(data):
        w = param.data.numpy()
        transform_binary(w)
        weights.append(w)

    w = np.concatenate((weights[0].flatten(), weights[1].flatten()), axis=None)
    return np.mean(w)


print('Loading Data')
rescale = 14

trainloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=True,  binary=False, train_size=args.N),
                         batch_size=64, shuffle=True)
testloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=False,  binary=False),
                        batch_size=64, shuffle=True)

evaluate = 1
epochs = 100
cross_tr_loss = []
cross_te_loss = []
cross_w = []

for eval in range(evaluate):
    print('Evaluation {} \n'.format(eval))
    cuda_available = torch.cuda.is_available()
    torch.manual_seed = (0)
    if cuda_available:
        torch.cuda.manual_seed(0)
        print('running on GPU')

    clf = Binary_CNN(1,10)

    if cuda_available:
        clf = clf.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr)

    min_loss = np.inf
    n_epochs_stop = 5
    epochs_no_improve = 0
    best_params = None
    early_stop = True

    # e=[]
    train_loss = []
    test_error = []

    for epoch in range(1,epochs+1):
        # e.append(epoch)
        train(epoch)
        error = test(epoch)
        test_error.append(error)
        if error < min_loss:
            epochs_no_improve = 0
            min_loss = error
            best_params = clf.parameters()
            # print('Epoch %d, Test Error: %.3f' % (epoch, error))
            # print('--------------------------------------------------------------')

        else:
            epochs_no_improve += 1

        if early_stop and epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('--------------------------------------------------------------')
            print('Early Stopping')
            print('Best error obtain on training set %.3f'%(min_loss))
            print('--------------------------------------------------------------')
            break

    cross_tr_loss.append(train_loss)
    cross_te_loss.append(test_error)
    cross_w.append(content(best_params))

store = 'results/' + 'cbnn'
if not os.path.exists(store):
    os.makedirs(store)

plot_bnn(cross_te_loss, store + '/test' + str(args.lr), 'error')
plot_bnn(cross_tr_loss, store + '/train' + str(args.lr), 'loss')
report_weight(cross_w, store + '/weight' + str(args.lr))
pkl.dump(cross_te_loss, open(store + '/test' + str(args.lr)+'.pkl', 'wb'))
pkl.dump(cross_tr_loss, open(store + '/train'  + str(args.lr)+  '.pkl', 'wb'))
pkl.dump(cross_w, open(store + '/avg_w' + str(args.lr)+ '.pkl', 'wb'))







