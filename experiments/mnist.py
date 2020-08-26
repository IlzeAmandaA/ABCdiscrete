import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from urllib import request
from scipy.special import expit
import sys
from skimage.transform import resize

PYTHONPATH = '/home/iaa510/'

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

class MNIST():
    def __init__(self, l1=0, l2=1, H=20, name='mnist', image_size=(14, 14), batch_size=1000):
        super().__init__()
        self.name = name
        self.image_size = image_size
        self.H=H
        self.batch_size = batch_size

        if not(os.path.isfile(PYTHONPATH + '/data/' + 'mnist.pkl')):
            self.download_mnist(location=PYTHONPATH + '/data/')
            self.save_mnist(location=PYTHONPATH +'/data/')

        x_train, y_train, x_test, y_test = self.load(location=PYTHONPATH + '/data/')

        x_train = x_train / 255.
        x_test = x_test / 255.

        # TRAIN DATA
        idx_l1 = np.where(y_train == l1)[0]
        idx_l2 = np.where(y_train == l2)[0]
        idx = np.sort(np.concatenate((idx_l1, idx_l2), axis=None))
        train_size = len(idx)
        # print(len(idx1))
        # idx = np.sort(np.concatenate((idx_l1[0:int(train_size / 2)], idx_l2[0:int(train_size / 2)]), axis=None))

        x_train = np.reshape(x_train[idx], (train_size, 28, 28))
        self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
        for i in range(x_train.shape[0]):
            self.x_train[i] = resize(x_train[i], image_size, anti_aliasing=True)
        self.x_train = np.reshape(self.x_train, (train_size, image_size[0] * image_size[1]))
        self.y_train = y_train[idx]  # [0:train_size]
        assert self.x_train.shape[0] == self.y_train.shape[0], 'incorrect dim xtrain and ytrain'
        # self.x_train = transform_polar(self.x_train)
        transform_polar(self.x_train)

        print('Shape of train data {}'.format(self.x_train.shape))


        # TEST DATA
        idx_l1 = np.where(y_test == l1)[0]
        idx_l2 = np.where(y_test == l2)[0]
        idx = np.sort(np.concatenate((idx_l1, idx_l2), axis=None))

        x_test = np.reshape(x_test[idx], (len(idx), 28, 28))
        self.x_test = np.zeros((x_test.shape[0], image_size[0], image_size[1]))
        for i in range(x_test.shape[0]):
            self.x_test[i] = resize(x_test[i], image_size, anti_aliasing=True)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], image_size[0] * image_size[1]))
        self.y_test = y_test[idx]
        assert self.x_test.shape[0] == self.y_test.shape[0], 'incorrect dim xtest and ytest'
        # self.x_test = transform_polar(self.x_test)
        transform_polar(self.x_test)

        print('Shape of test data {}'.format(self.x_test.shape))



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

    def transform(self):
        self.x_train[self.x_train<0.5]=-1
        self.x_train[self.x_train>0.5]=1
        self.x_test[self.x_test < 0.5] = -1
        self.x_test[self.x_test > 0.5] = 1


    def bern2(self, p, D1, D2):
        return 2. * np.random.binomial(1, p, (D1, D2)) - 1.

    def bern(self, p, D1, D2):
        return np.random.binomial(1, p, (D1, D2))


    def initialize_pop(self, N):
        D=self.image_size[0] * self.image_size[1] * self.H + self.H * 2
        return self.bern(0.5,N,D)

    def hardtanh(self,data):
        data[data > 1] = 1
        data[data < -1] = -1

    def binary_hardtanh(self,data):
        self.hardtanh(data)
        data[data>0]=1
        data[data==0]=-1


    def simulate(self, w_orig): #objective
        #change 0 to -1
        w = np.copy(w_orig)
        w[w==0]=-1
        # print('updated theta values {}'.format(set(w)))

        im_shape = self.image_size[0] * self.image_size[1]
        data_x = self.x_train
        data_y = self.y_train

        y_pred = np.zeros((data_y.shape[0],))

        for i in range(data_x.shape[0] // self.batch_size):
            # print(i)
            w1 = w[0: im_shape * self.H]
            w2 = w[im_shape * self.H:]

            W1 = np.reshape(w1, (im_shape, self.H))
            W2 = np.reshape(w2, (self.H, 1))

          #  First layer
            h = np.dot(data_x[i * self.batch_size: (i + 1) * self.batch_size], W1)
            # print('h', h)
            # tanh
            self.binary_hardtanh(h)
            # print('h tanh', h)
            # Second layer
            logits = np.dot(h, W2)
            print('logts')
            print(logits.shape)
            # print(logits)
             # sigmoid
            prob = expit(logits)
            print('prob')
            print(prob.shape)
            # print(prob)

            y_pred[i * self.batch_size: (i + 1) * self.batch_size] = np.argmax(prob, -1)

        print(y_pred)
        return y_pred

    def binarize(self, x):
        x[x>0]=1
        x[x<=0]=-1
        return x.astype(int)

    def distance(self, y):
        y = y.astype(int)
        return 1/self.y_train.shape[0] * sum(np.invert(np.logical_xor(self.y_train, y)))

    def prior(self, theta):
        #define a pseduo-bolztman distribution
        return np.exp(-np.mean(theta))

