import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Function
import torch.nn.functional as F


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



class Network(nn.Module):
    def __init__(self, in_features, out_features, num_units):
        super(Network, self).__init__()

        self.fc1 = nn.Sequential(BinaryLinear(in_features, num_units, bias=False),
                                 BinaryTanh()
                                 )
        self.fc2 = nn.Sequential(BinaryLinear(num_units, out_features, bias=False),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        y_hat = torch.ge(out, 0.5).float().squeeze()

        return out, y_hat

    def calculate_objective(self, X,Y):
        Y = Y.float()
        Y_prob, Y_hat = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5).squeeze()
        neg_log_likelihood = torch.mean(-1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))) # negative log bernoulli
        return neg_log_likelihood

    def calculate_classification_error(self, X,Y):
        _, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

