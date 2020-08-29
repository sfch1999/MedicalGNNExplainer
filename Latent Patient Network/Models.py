import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEBUG = False


class LatentNet(nn.Module):
    def __init__(self, num_classes):
        super(LatentNet, self).__init__()
        self.fc1 = nn.Linear(60, 32)
        # self.bn1=nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.temp = torch.empty(1)
        torch.nn.init.trunc_normal_(self.temp, std=1e-1)
        self.temp = nn.Parameter(self.temp[0] + 1)
        self.theta = torch.empty(1)
        torch.nn.init.trunc_normal_(self.theta, std=1e-1)
        self.theta = nn.Parameter(self.theta[0] + 1)
        self.sig = nn.Sigmoid()
        self.fc3 = nn.Linear(60, 8)
        # self.fc4 = nn.Linear(32, 16)
        # self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, num_classes)
        self.sm = nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        # torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        # print('x',x.shape)
        net = self.fc1(x)
        # if DEBUG:
        #     print('fc1', net)

        net = self.fc2(net)
        # if DEBUG:
        #     print('fc2', net)

        A = -1 * self.pairwise_distances(net)
        # if DEBUG:
        #     print(A)
        #     print('bef sig', self.temp * A + self.theta)
        A = self.sig(self.temp * A + self.theta)
        # if DEBUG:
        #     print(A)

        # print('adj',A)

        A = A * (1 - torch.eye(A.shape[-1])) + torch.eye(A.shape[-1])
        # if DEBUG:
        #     print(A)

        A = torch.true_divide(A, torch.sum(A, dim=1)[:, None])
        # if DEBUG:
        #     print(A)
        x = A.matmul(x)
        if DEBUG:
            print('after matmul',x)
        x = self.fc3(x)
        # x = A.matmul(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        x = self.sm(self.fc6(x))

        return x

    @staticmethod
    def pairwise_distances(x, y=None):
        """
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return torch.clamp(dist, 0.0, np.inf)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(60, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_classes)
        self.sm = nn.Softmax(dim=1)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        net = self.fc1(x)
        # print(net)
        net = self.fc2(net)
        # print(net)
        net = self.fc3(net)
        # print(net)
        net = self.sm(self.fc4(net))

        return net

