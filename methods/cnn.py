import torch
import torch.nn as nn
import torch.utils.data



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
        self.fc = nn.Linear(4*4*1, self.outD, bias=False)
        self.soft = nn.Softmax(dim=0)

    def forward(self, x):
        out = self.conv1(x)
        print('con1', out.shape)
        out = self.mp(out)
        out = self.conv2(out)
        print('con2', out.shape)
        out = self.mp(out)
        out = self.fc(out)
        print('fc', out.shape)
        prob = self.soft(out)
        print('prob', prob.shape)
        y_hat = torch.argmax(prob, dim=1)
        print('yhay', y_hat.shape)

        return y_hat


    def calculate_classification_error(self, X,Y):
        print(X.shape)
        print('calculating error')
        Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat




