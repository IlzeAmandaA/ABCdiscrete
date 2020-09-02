import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Function
import torch.nn.functional as F
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

class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)



class Forward_CNN(nn.Module):
    def __init__(self):
        super(Forward_CNN, self).__init__()

    def forward(self, x, W):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        w1, w2 = W
        out = F.conv2d(x, w1, padding=2, bias=None)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.hardtanh(out)
        out = binarize(out)
        out = F.conv2d(out,w2, padding=1, bias=None)
        out = F.max_pool2d(out, kernel_size=2, ceil_mode=True)
        out = F.hardtanh(out)
        out = binarize(out)

        return out

class FFNN(nn.Module):
    def __init__(self, K2, outD):
        super(FFNN, self).__init__()

        self.fc = nn.Linear(4 * 4 * K2, outD)
        self.sm= nn.Softmax(dim=0)

    def forward(self, z):
        out = self.fc(z)
        return out

    def objective(self,z,Y):
        a = self.forward(z.view(z.size(0),-1))
        Y_prob = self.sm(a)
        Y_hat = torch.argmax(Y_prob, dim=1)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return a, error


"""""
"""""


class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        bw = binarize(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)


class Binary_CNN(nn.Module):
    def __init__(self, in_features=1, out_features=10, K1=16, K2=64, F1=5, F2=3):
        # in_features=1, out_features=10, K1=6, K2=32, F1=5, F2=3):
        super(Binary_CNN, self).__init__()

        self.F1 = F1
        self.F2 = F2
        self.inD = in_features
        self.outD = out_features
        self.K1 = K1
        self.K2 = K2

        # 14 smaller net
        self.layer1 = nn.Sequential(
            BinaryConv2d(self.inD, self.K1, kernel_size=self.F1, padding=2),
            nn.MaxPool2d(2),
            BinaryTanh()
            )
        self.layer2 = nn.Sequential(
            BinaryConv2d(self.K1, self.K2, kernel_size=self.F2, padding=1),
            nn.MaxPool2d(2, ceil_mode=True),
            BinaryTanh()
            )

        self.fc = BinaryLinear(4 * 4 * self.K2, self.outD)


    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out.view(out.size(0),-1))
        return out


    def calculate_classification_error(self, X, Y):
        # print(X.shape)
        Y = Y.int()
        Y_out = self.forward(X)
        prob = F.softmax(Y_out, dim=0)
        Y_hat = torch.argmax(prob, dim=1)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return Y_out, error

