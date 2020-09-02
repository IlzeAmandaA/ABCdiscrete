import numpy as np
from experiments.mnist_torch import MNIST
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data
from methods.cnn import *
import sys

class RandomKitchenSinks():

    def __init__(self, inD=1, outD=10, C1=32, C2=64, F1=5, F2=3,
                 rescale=14, N_data=10000, lr=0.01, batch_size=1000,
                 path = 'external'):

        #c1=6, c2=32

        self.inD= inD
        self.c1=C1
        self.c2=C2
        self.f1= F1
        self.f2= F2
        self.outD = outD
        self.N = N_data
        self.batch_size = batch_size
        self.loss = None



        self.cuda_available = torch.cuda.is_available()

        self.trainloader = DataLoader(MNIST(image_size=(rescale, rescale), train=True, binary=False, train_size=N_data, path=path),
                                 batch_size=self.batch_size, shuffle=True)
        # self.testloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=False),
        #                         batch_size=128, shuffle=True)

        self.cnn = Forward_CNN()
        self.nn = FFNN(self.c2, self.outD)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)

        if self.cuda_available:
            self.clf = self.clf.cuda()
            torch.cuda.manual_seed(0)
            print('running on GPU')


    def initialize(self, N):
        D = self.inD*self.f1*self.f1*self.c1+\
        self.c1*self.f2*self.f2*self.c2
        # 4*4*self.c2 * self.outD
        return self.bern(0.5, N, D)

    def bern(self, p, D1, D2):
        return np.random.binomial(1, p, (D1, D2))

    def simulate(self, w_orig):  # objective

        self.cnn.eval()

        with torch.no_grad():
            w = np.copy(w_orig)
            w[w == 0] = -1

            w1 = w[0: (self.f1 * self.f1 * self.c1)]
            w2 = w[(self.f1 * self.f1 * self.c1):]

            w1 = np.reshape(w1, (self.c1, self.inD, self.f1, self.f1))
            w2 = np.reshape(w2, (self.c2, self.c1, self.f2, self.f2))

            w1 = torch.from_numpy(w1).type(torch.float32)
            w2 = torch.from_numpy(w2).type(torch.float32)

            z = []
            y_true = []

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                # inputs = inputs.type(torch.FloatTensor)
                if self.cuda_available:
                    inputs, targets = inputs.cuda(), targets.cuda()

                inputs = inputs.type(torch.float32)

                y_true.append(targets.type(torch.LongTensor))
                z.append(self.cnn(inputs,(w1,w2)))

            z = torch.cat(z, dim=0)
            y_true = torch.cat(y_true, dim=0)

            return (z, y_true)

    def distance(self, sim_output, run=1, eval=False):
        z, y_true = sim_output

        if not eval:
            self.nn.train()
        else:
            self.nn.eval()


        # error_avg = 0
        # Y_hat = []

        # for i in range(int(y_true.shape[0]/self.batch_size)):
        #     x = z[i*self.batch_size : (i+1)*self.batch_size]
        #     y = y_true[i*self.batch_size : (i+1)*self.batch_size]

        self.optimizer.zero_grad()
        output, y_hat = self.nn.objective(z)


        if not eval:
            loss = self.criterion(output, y_true)
            loss.backward()
            self.optimizer.step()

        # if run==0:
        #     self.loss.backward()
        #     self.optimizer.step()
        # Y_hat.append(y_hat)

        # Y_hat = torch.cat(Y_hat, dim=0)
        error = 1. - y_hat.eq(y_true).cpu().float().mean().item()
        return error

    def prior(self,theta):
        return np.exp(-np.mean(theta))





