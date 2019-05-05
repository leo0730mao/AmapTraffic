import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from traffic_predict.layers import GraphConvolution
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

        self.encoder = nn.GRU(input_size = 64, hidden_size = 128, batch_first = True).cuda()
        self.decoder = nn.GRU(input_size = 128, hidden_size = 128, batch_first = False).cuda()

        self.dropout = dropout

        self.fc1 = nn.Linear(134, 1).cuda()

    def forward(self, X, adj, time_feature):  # X[(batch_size, vertex_num, 1)] * 24
        x_list = []
        for i in range(len(X)):
            x = F.relu(self.gc1(X[i].cuda(), adj), inplace = True)
            x = F.relu(self.gc2(x, adj), inplace = True)
            x = F.relu(self.gc3(x, adj), inplace = True)  # x(batch_size, vertex_num, dim)
            x_list.append(x.reshape(x.shape[0] * x.shape[1], x.shape[2]))
        H = torch.stack(x_list, dim = 1).cuda()  # H(batch_size * vertex_num, 24, dim)

        _, ctx = self.encoder(H.cuda())
        ctx = ctx.repeat(24, 1, 1)
        out, _ = self.decoder(ctx)
        out = out.permute(1, 0, 2)  # out(batch_size * vertex_num, 24, 128)
        out = torch.cat((out, time_feature), 2)
        out = self.fc1(out)
        out = out.squeeze(-1)  # out(batch_size * vertex_num, 24, 1)
        return out


class LSTM(nn.Module):
    def __init__(self, dropout, batch_size = 1024, vertex_num = 194):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.vertex_num = vertex_num

        self.gru = nn.GRU(input_size = 1, hidden_size = 128, batch_first = True).cuda()
        self.dropout = dropout

        self.fc1 = nn.Linear(128, 1).cuda()
        self.weight = torch.Tensor([1])

    def forward(self, x):
        out, _ = self.gru(x.cuda())
        out = self.fc1(out[:, -1, :])
        return out


class Seq2seq(nn.Module):
    def __init__(self, dropout, batch_size = 1024, vertex_num = 194):
        super(Seq2seq, self).__init__()

        self.batch_size = batch_size
        self.vertex_num = vertex_num

        self.encoder = nn.GRU(input_size = 1, hidden_size = 128, batch_first = True).cuda()
        self.decoder = nn.GRU(input_size = 128, hidden_size = 128, batch_first = False).cuda()
        self.dropout = dropout

        self.fc1 = nn.Linear(134, 1).cuda()
        self.weight = torch.Tensor([1])

    def forward(self, x, time_feature):
        _, ctx = self.encoder(x.cuda())
        ctx = ctx.repeat(24, 1, 1)
        out, _ = self.decoder(ctx)
        out = out.permute(1, 0, 2)
        out = torch.cat((out, time_feature), 2)
        out = self.fc1(out)
        out = out.squeeze(-1)
        return out


class HA:
    @ classmethod
    def evaluate(cls, input_x, timestep):
        res = []
        count = 0
        for i in range(timestep, input_x.shape[0], 1):
            feature = input_x[i - timestep: i]
            output_y = np.mean(feature, axis = 0)
            true_y = input_x[i]
            res.append(np.sqrt(np.square(output_y - true_y).mean()))
            count += 1
        res = np.array(res)
        return res
        # print("predict mse loss is: %s" % (res / count))


def HA_test():
    with open("F:/DATA/dataset/v1/HA.dat", 'rb') as f:
        data = pickle.load(f)
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


if __name__ == '__main__':
    HA_test()
