import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Function
import sys

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input>=0]=1
        output[input<0]=-1
        output = output.type(torch.FloatTensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

binarize = BinarizeF.apply

class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output


class Network_CNN(nn.Module):
    def __init__(self, in_features, out_features, K1=32, K2=1, F=3):
        super(Network_CNN, self).__init__()


        self.F = F
        self.inD = in_features
        self.outD = out_features
        self.K1 = K1
        self.K2 = K2

        self.conv1 = nn.Conv2d(self.inD, self.K1, kernel_size=self.F, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.K1,self.K2, kernel_size=self.F, padding=2, bias=False)
        self.btanh = BinaryTanh()
        self.fc = nn.Linear(4*4*1, self.outD, bias=False)
        self.soft = nn.Softmax(dim=0)


    def forward(self, x):
        out = self.conv1(x)
        out = self.mp(out)
        out = self.btanh(out)
        out = self.conv2(out)
        out = self.mp(out)
        out = self.btanh(out)
        out = self.fc(out.view(-1,4*4*1))
        prob = self.soft(out)
        y_hat = torch.argmax(prob, dim=1)

        return y_hat


    def calculate_classification_error(self, X,Y):
        # print(X.shape)
        X = torch.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
        Y_hat = self.forward(X)
        Y = Y.int()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        print(error)
        return error, Y_hat




