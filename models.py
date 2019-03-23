import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, dropout, input_dim = 1, batch_size = 2, vertex_num = 4148, k = 3):
        super(GCN, self).__init__()

        self.batch_size = batch_size
        self.vertex_num = vertex_num

        self.gc1 = GraphConvolution(input_dim, 16, batch_size, vertex_num, k)
        self.gc2 = GraphConvolution(16, 32, batch_size, vertex_num, k)
        self.gc3 = GraphConvolution(32, 16, batch_size, vertex_num, k)

        self.gru = nn.GRU(input_size = 16, hidden_size = 16, batch_first = True).cuda()
        self.gru_batch_size = 4148
        self.gru_batch_num = batch_size

        self.dropout = dropout

        self.fc1 = nn.Linear(18, 1).cuda()

    def forward(self, X, T_k):
        x_list = []
        for i in range(len(X)):
            x = F.relu(self.gc1(X[i], T_k))
            x = F.relu(self.gc2(x, T_k))
            x = F.relu(self.gc3(x, T_k))
            x_list.append(x.reshape(x.shape[0] * x.shape[1], x.shape[2]))
        H = torch.stack(x_list, dim = 1)

        outputs = []
        for i in range(self.gru_batch_num):
            _, output = self.gru(H[self.gru_batch_size * i: self.gru_batch_size * (i + 1)])

            output = output.squeeze(0).cpu()
            outputs.append(output)
        outputs = torch.cat(outputs, dim = 0).cuda()
        outputs = self.fc1(outputs).reshape(self.batch_size, self.vertex_num)
        return outputs


class HA:
    @ classmethod
    def evaluate(cls, input_x):
        res = 0
        i = 0
        for i in range(24, input_x.shape[0], 1):
            feature = input_x[i - 24: i]
            output_y = np.mean(feature, axis = 0)
            true_y = input_x[i]
            res += np.square(output_y - true_y).mean()
            i += 1
        print("predict mse loss is: %s" % (res / i))


if __name__ == '__main__':
    with open("F:/DATA/dataset/v1/HA.dat", 'rb') as f:
        data = pickle.load(f)
    model = HA()
    model.evaluate(data)


