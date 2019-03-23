import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    input_size: (batch_size, vertex_num, dim)
    output_size: (batch_size, vertex_num, dim)
    """

    def __init__(self, input_dim, output_dim, vertex_num, K, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vertex_num = vertex_num
        self.K = K
        self.weight = Parameter(torch.Tensor(self.K, input_dim, output_dim).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, T_k):
        inputs = torch.split(input, 1, 0)
        outputs = []
        for x in inputs:  # x's size: (1, vertex_num, dim)
            x = x.squeeze(0)
            out = None
            for i in range(len(T_k)):
                temp = torch.spmm(T_k[i].cuda(), x)
                if out is None:
                    out = torch.mm(temp, self.weight[i])
                else:
                    out += torch.mm(temp, self.weight[i])
            if self.bias is not None:
                out += self.bias
            outputs.append(out)
        outputs = torch.stack(outputs, dim = 0)  # outputs' size: (batch_size, vertex_num, output_dim)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
