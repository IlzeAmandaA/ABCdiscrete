from methods.bnn import Network
from experiments.mnist_torch import *
from utils.func_support import *
import pickle as pkl
from torch.utils.data import DataLoader
import torch.utils.data
import torch.optim as optim
import numpy as np



def train(epoch):
    batch_loss = []
    clf.train()
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        optimizer.zero_grad()
        loss = clf.calculate_objective(inputs, targets)
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

    if epoch%10==0:
        print('Train Epoch: %d Training Loss %.3f' % (epoch,  avg_loss))
# print('Training Loss : %.3f Time : %.3f seconds ' % (np.mean(avg_loss)), end - start))

def test():
    clf.eval()
    tb_error = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # inputs = inputs.type(torch.FloatTensor)
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)

            error, predicted = clf.calculate_classification_error(inputs, targets)
            tb_error.append(error)

        avg_terror = np.mean(np.array(tb_error))
        # print('Test Error: %.3f' % (avg_terror))
        # print('--------------------------------------------------------------')

    return avg_terror

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
trainloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=True),
                         batch_size=128, shuffle=True)
testloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=False),
                        batch_size=128, shuffle=True)

evaluate = 10
epochs = 50
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

    hidden_units = 20
    output = 1
    clf = Network(rescale * rescale, output, hidden_units)
    if cuda_available:
        clf = clf.cuda()

    optimizer = optim.Adam(clf.parameters(), lr=0.0001)

    min_loss = np.inf
    n_epochs_stop = 10
    epochs_no_improve = 0
    best_params = None
    early_stop = False

    # e=[]
    train_loss = []
    test_error = []

    for epoch in range(1,epochs+1):
        # e.append(epoch)
        train(epoch)
        error = test()
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

store = 'results/' + 'bnn'
if not os.path.exists(store):
    os.makedirs(store)

plot_bnn(cross_te_loss, store + '/test', 'error')
plot_bnn(cross_tr_loss, store + '/train', 'loss')
report_weight(cross_w, store + '/weight')
pkl.dump(cross_te_loss, open(store + '/test' + '.pkl', 'wb'))
pkl.dump(cross_tr_loss, open(store + '/train'  + '.pkl', 'wb'))
pkl.dump(cross_w, open(store + '/avg_w' + '.pkl', 'wb'))







