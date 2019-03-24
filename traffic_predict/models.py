import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from pygcn.layers import GraphConvolution
import pandas as pd


class GCN(nn.Module):
    def __init__(self, dropout, input_dim = 1, vertex_num = 194, k = 3):
        super(GCN, self).__init__()

        self.vertex_num = vertex_num
        # self.gc1 = ChebConv(in_channels = 1, out_channels = 16, K = 3)
        # self.gc2 = ChebConv(in_channels = 16, out_channels = 32, K = 3)
        # self.gc3 = ChebConv(in_channels = 32, out_channels = 16, K = 3)
        self.gc1 = GraphConvolution(input_dim, 16, vertex_num, k)
        self.gc2 = GraphConvolution(16, 32, vertex_num, k)
        self.gc3 = GraphConvolution(32, 64, vertex_num, k)

        self.gru = nn.GRU(input_size = 64, hidden_size = 64, batch_first = True).cuda()

        self.dropout = dropout

        self.fc1 = nn.Linear(64, 1).cuda()

    def forward(self, X, adj):
        x_list = []
        for i in range(len(X)):
            x = F.relu(self.gc1(X[i].cuda(), adj), inplace = True)
            x = F.relu(self.gc2(x, adj), inplace = True)
            x = F.relu(self.gc3(x, adj), inplace = True)
            x_list.append(x.reshape(x.shape[0] * x.shape[1], x.shape[2]))
        H = torch.stack(x_list, dim = 1).cuda()

        _, output = self.gru(H)

        output = output.squeeze(0)
        outputs = self.fc1(output).reshape(X[0].shape[0], self.vertex_num)
        return outputs


class LSTM(nn.Module):
    def __init__(self, dropout, batch_size = 16, vertex_num = 194):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.vertex_num = vertex_num

        self.gru = nn.GRU(input_size = 1, hidden_size = 1024, batch_first = True).cuda()

        self.dropout = dropout

        self.fc1 = nn.Linear(1024, 1).cuda()

    def forward(self, X):
        output, _ = self.gru(X.cuda())
        output = output[:, -1, :]
        output = F.relu(self.fc1(output))
        return output


class HA:
    @ classmethod
    def evaluate(cls, input_x, timestep):
        res = []
        count = 0
        for i in range(timestep, input_x.shape[0], 1):
            feature = input_x[i - timestep: i]
            output_y = np.mean(feature, axis = 0)
            true_y = input_x[i]
            res.append(np.square(output_y - true_y).mean())
            count += 1
        res = np.array(res)
        return res
        # print("predict mse loss is: %s" % (res / count))


def HA_test():
    with open("F:/DATA/dataset/v1/HA.dat", 'rb') as f:
        data = pickle.load(f)
    data = data.toarray()
    model = HA()
    res = {'mean': [], 'var': [], 'max': [], 'min': [], 'timesteps': []}
    for i in range(1, 24):
        loss = model.evaluate(data, i)
        res['timesteps'].append(i)
        res['mean'].append(loss.mean())
        res['var'].append(loss.var())
        res['max'].append(loss.max())
        res['min'].append(loss.min())
    res = pd.DataFrame(res)
    res.to_csv("F:/DATA/dataset/v1/HA_result.csv", index = False)