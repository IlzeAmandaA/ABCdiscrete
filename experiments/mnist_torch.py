import os
import gzip
import pickle
import numpy as np
from torch.utils.data import DataLoader
from urllib import request
import sys
from skimage.transform import resize
import torch
import torch.utils.data
from torch.utils.data import Dataset
from methods.cnn import Forward_CNN



PYTHONPATH = '/home/iaa510'
#PYTHONPATH = '/home/ilze/MasterThesis/mnist'

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def transform_polar(image):
    image[image<0.5]=-1
    image[image>=0.5]=1
    # return image

def transform_binary(data):
    data[data >= 0] = 1
    data[data<0] = 0
    # return data


class MNIST(Dataset):

    def __init__(self, l1=0, l2=1, image_size=(14, 14), train=True, binary=True, train_size=5000):
       # train_size = 5000

        self.train=train
        self.train_size = train_size
        self.binary = binary

        if not(os.path.isfile(PYTHONPATH + '/data/' + 'mnist.pkl')):
            self.download_mnist(location=PYTHONPATH + '/data/')
            self.save_mnist(location=PYTHONPATH +'/data/')

        x_train, y_train, x_test, y_test = self.load(location=PYTHONPATH + '/data/')

        x_train = x_train / 255.
        x_test = x_test / 255.

        if self.train:
            if self.binary:
                # TRAIN DATA
                idx_l1 = np.where(y_train == l1)[0]
                idx_l2 = np.where(y_train == l2)[0]
                idx = np.sort(np.concatenate((idx_l1, idx_l2), axis=None))
                self.train_size = len(idx)
                x_train = np.reshape(x_train[idx], (self.train_size, 28, 28))
                self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
                self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
                for i in range(x_train.shape[0]):
                    self.x_train[i] = resize(x_train[i], image_size, anti_aliasing=True)

                self.x_train = np.reshape(self.x_train, (train_size, image_size[0] * image_size[1]))
                self.y_train = y_train[idx]


            else:
                x_train = np.reshape(x_train[0:self.train_size], (self.train_size, 28, 28))
                self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
                for i in range(x_train.shape[0]):
                    self.x_train[i] = resize(x_train[i], image_size, anti_aliasing=True)
                # self.x_train = x_train
                self.y_train = y_train[0:self.train_size]

            assert self.x_train.shape[0] == self.y_train.shape[0], 'incorrect dim xtrain and ytrain'

            transform_polar(self.x_train)
            print('Shape of train data {}'.format(self.x_train.shape))

        else:
            # TEST DATA
            if self.binary:
                idx_l1 = np.where(y_test == l1)[0]
                idx_l2 = np.where(y_test == l2)[0]
                idx = np.sort(np.concatenate((idx_l1, idx_l2), axis=None))
                x_test = np.reshape(x_test[idx], (len(idx), 28, 28))
                self.x_test = np.zeros((x_test.shape[0], image_size[0], image_size[1]))
                for i in range(x_test.shape[0]):
                    self.x_test[i] = resize(x_test[i], image_size, anti_aliasing=True)
                self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], image_size[0] * image_size[1]))
                self.y_test = y_test[idx]

            else:
                x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
                self.x_test = np.zeros((x_test.shape[0], image_size[0], image_size[1]))
                for i in range(x_test.shape[0]):
                    self.x_test[i] = resize(x_test[i], image_size, anti_aliasing=True)
                # self.x_test = x_test
                self.y_test = y_test

            assert self.x_test.shape[0] == self.y_test.shape[0], 'incorrect dim xtest and ytest'

            transform_polar(self.x_test)
            print('Shape of test data {}'.format(self.x_test.shape))

    def __len__(self):
        if self.train:
            return len(self.y_train)
        else:
            return len(self.y_test)


    def __getitem__(self, item):
        if self.train:
            image = self.x_train[item]
            label = self.y_train[item]
        else:
            image = self.x_test[item]
            label = self.y_test[item]

        return image, label



    @staticmethod
    def download_mnist(location):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], location + name[1])
        print("Download complete.")

    @staticmethod
    def save_mnist(location):
        mnist = {}
        for name in filename[:2]:
            with gzip.open(location + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in filename[-2:]:
            with gzip.open(location + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(location + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    @staticmethod
    def load(location):
        with open(location + "mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


class HighDim():

    def __init__(self, inD=1, outD=10, rescale=14):


        self.cuda_available = torch.cuda.is_available()

        self.trainloader = DataLoader(MNIST(image_size=(rescale, rescale), train=True, binary=False),
                                 batch_size=1000, shuffle=True)
        # self.testloader = DataLoader(MNIST(l1=0, l2=1, image_size=(rescale, rescale), train=False),
        #                         batch_size=128, shuffle=True)

        self.clf = Forward_CNN(inD,outD)

        if self.cuda_available:
            self.clf = self.clf.cuda()
            torch.cuda.manual_seed(0)
            print('running on GPU')

        self.distancev  = None


    def initialize_pop(self, N):
        D = self.clf.inD*self.clf.F*self.clf.F*self.clf.K1+\
        self.clf.K1*self.clf.F*self.clf.F*self.clf.K2 + \
        4*4*self.clf.K2 * self.clf.outD
        return self.bern(0.5, N, D)

    def bern(self, p, D1, D2):
        return np.random.binomial(1, p, (D1, D2))

    def simulate(self, w_orig):  # objective

        self.clf.eval()

        with torch.no_grad():
            w = np.copy(w_orig)
            w[w == 0] = -1

            w1 = w[0: (self.clf.F * self.clf.F * self.clf.K1)]
            w2 = w[(self.clf.F * self.clf.F * self.clf.K1):(self.clf.F * self.clf.F * self.clf.K1) * 2]
            w3 = w[(self.clf.F * self.clf.F * self.clf.K1) * 2:]

            w1 = np.reshape(w1, (self.clf.K1, self.clf.inD, self.clf.F, self.clf.F))
            w2 = np.reshape(w2, (self.clf.K2, self.clf.K1, self.clf.F, self.clf.F))
            w3 = np.reshape(w3, (self.clf.outD, 4 * 4 * 1))

            w1 = torch.from_numpy(w1)
            w2 = torch.from_numpy(w2)
            w3 = torch.from_numpy(w3)

            # self.clf.conv1.weight.copy_(W1)
            # self.clf.conv2.weight.copy_(W2)
            # self.clf.fc.weight.copy_(W3)
            tb_error = []

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                # inputs = inputs.type(torch.FloatTensor)
                if self.cuda_available:
                    inputs, targets = inputs.cuda(), targets.cuda()

                inputs = inputs.type(torch.FloatTensor)
                targets = targets.type(torch.FloatTensor)

                error, predicted = self.clf.calculate_classification_error(inputs, targets, (w1,w2,w3))
                tb_error.append(error)

            avg_error = np.mean(np.array(tb_error))
            self.distancev= avg_error

            return None

    def distance(self, fake):
        return self.distancev

    def prior(self,theta):
        return np.exp(-np.mean(theta))





