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
    def __init__(self, in_features, out_features, K1=32, K2=1, F=3):
        super(Forward_CNN, self).__init__()


        self.F = F
        self.inD = in_features
        self.outD = out_features
        self.K1 = K1
        self.K2 = K2


        # self.conv1 = nn.Conv2d(self.inD, self.K1, kernel_size=self.F, padding=1, bias=False)
        # self.mp = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(self.K1,self.K2, kernel_size=self.F, padding=2, bias=False)
        # self.btanh = BinaryTanh()
        # self.fc = nn.Linear(4*4*1, self.outD, bias=False)
        # self.soft = nn.Softmax(dim=0)


    def forward(self, x, W):
        w1, w2, w3 = W
        out = F.conv2d(x, w1, padding=1, bias=None)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.hardtanh(out)
        out = binarize(out)
        out = F.conv2d(out,w2, padding=2, bias=None)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.hardtanh(out)
        out = binarize(out)
        out = F.linear(out.view(-1, 4*4*1), w3, bias=None)
        prob = F.softmax(out, dim=0)
        y_hat = torch.argmax(prob, dim=1)

        #
        # out = self.conv1(x)
        # out = self.mp(out)
        # out = self.btanh(out)
        # out = self.conv2(out)
        # out = self.mp(out)
        # out = self.btanh(out)
        # out = self.fc(out.view(-1,4*4*1))
        # prob = self.soft(out)
        # y_hat = torch.argmax(prob, dim=1)

        return y_hat


    def calculate_classification_error(self, X,Y,W):
        # print(X.shape)
        X = torch.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
        Y_hat = self.forward(X, W)
        Y = Y.int()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        # print(error)
        return error, Y_hat

"""""
"""""


class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        bw = binarize(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)


class Binary_CNN(nn.Module):
    def __init__(self, in_features, out_features, K1=6, K2=16, F=5):
        super(Binary_CNN, self).__init__()

        self.F = F
        self.inD = in_features
        self.outD = out_features
        self.K1 = K1
        self.K2 = K2

        #14
        # self.layer1 = nn.Sequential(
        #     BinaryConv2d(1, 16, kernel_size=5, padding=2),
        #     # nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
        #     nn.MaxPool2d(2),
        #     BinaryTanh())
        # self.layer2 = nn.Sequential(
        #     BinaryConv2d(16, 32, kernel_size=5, padding=2),
        #     # nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps),
        #     nn.MaxPool2d(2, ceil_mode=True),
        #     BinaryTanh())
        #
        # self.fc = BinaryLinear(4 * 4 * 32, 10)

        # 14 smaller net
        # self.layer1 = nn.Sequential(
        #     BinaryConv2d(1, 6, kernel_size=5, padding=2),
        #     # nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
        #     nn.MaxPool2d(2),
        #     BinaryTanh())
        # self.layer2 = nn.Sequential(
        #     BinaryConv2d(6, 16, kernel_size=5, padding=2),
        #     # nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps),
        #     nn.MaxPool2d(2, ceil_mode=True),
        #     BinaryTanh())
        #
        # self.fc = BinaryLinear(4 * 4 * 16, 10)
        #

        #simple cnn plus fnn
        self.layer1 = nn.Sequential(
            BinaryConv2d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
            nn.MaxPool2d(2),
            BinaryTanh()
        )

        self.layer2 = nn.Sequential(
            BinaryLinear(7*7*32, 100, bias=False),
            BinaryTanh(),
            BinaryLinear(100, 10, bias=False)
        )


        # self.fc1 = nn.Sequential(BinaryLinear(4 * 4 * 16, 20, bias=False),
        #                          BinaryTanh()
        #                          )
        # self.fc2 = BinaryLinear(20, 10, bias=False)
        #
        # #28
        # self.layer1 = nn.Seque   ntial(
        #     BinaryConv2d(1, 16, kernel_size=5, padding=2),
        #     # nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
        #     nn.MaxPool2d(2),
        #     BinaryTanh())
        # self.layer2 = nn.Sequential(
        #     BinaryConv2d(16, 32, kernel_size=5, padding=2),
        #     # nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps),
        #     nn.MaxPool2d(2),
        #     BinaryTanh())
        # self.fc = BinaryLinear(7*7*32, 10)

        # self.layer1 = nn.Sequential(
        #     BinaryConv2d(self.inD, self.K1, kernel_size=3, stride=1, padding=1),
        #     BinaryTanh(),
        #     nn.MaxPool2d(2)
        # )
        #
        # self.layer2 = nn.Sequential(
        #     BinaryConv2d(self.K1, self.K2, kernel_size=3, stride=1, padding=1),
        #     BinaryTanh(),
        #     nn.MaxPool2d(2, ceil_mode=True)
        # )
        #
        # self.layer3 = nn.Sequential(
        #     BinaryLinear(4 * 4 * self.K2, 64, bias=False), #2x2
        #     BinaryTanh(),
        #     # BinaryLinear(120, 84, bias=False),
        #     # BinaryTanh(),
        #     BinaryLinear(64, self.outD, bias=False)
        # )


        # self.layer1 = nn.Sequential(
        #     BinaryConv2d(self.inD, self.K1, kernel_size=5, padding=2, bias=False),
        #     BinaryTanh(),
        #     BinaryConv2d(self.K1, 1, kernel_size=1, padding=0, bias=False),
        #     BinaryTanh(),
        #     nn.MaxPool2d(2)
        #   #
        # )
        # self.layer2 = nn.Sequential(
        #     BinaryConv2d(1, self.K2, kernel_size=3, padding=1, bias=False),
        #     # BinaryConv2d(self.K2, 1, kernel_size=1, padding=0, bias=False),
        #     BinaryTanh(),
        #     nn.MaxPool2d(2, ceil_mode=True)
        # )
        # self.layer2 = nn.Sequential(
        #     BinaryConv2d(self.K1, self.K2, kernel_size=3, padding=1, bias=False),
        #     BinaryTanh(),
        #     nn.MaxPool2d(2),
        #     BinaryTanh()
        # )

        # self.fc = BinaryLinear(3 * 3 * self.K2, self.outD, bias=False)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
       # out = self.layer2(out)
        out = self.layer2(out.view(out.size(0),-1))
       # print(out.shape)
       #  out = self.fc1(out.view(out.size(0),-1))
       #  out = self.fc2(out)
        return out


    def calculate_classification_error(self, X, Y):
        # print(X.shape)
        Y = Y.int()
        Y_out = self.forward(X)
        prob = F.softmax(Y_out, dim=0)
        Y_hat = torch.argmax(prob, dim=1)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        # print(error)
        return Y_out, error

