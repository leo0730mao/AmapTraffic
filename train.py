from __future__ import division
from __future__ import print_function

import math
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.nn import MSELoss
from utils import new_load_data
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
train_X,  train_y, test_X, test_y, T_k = new_load_data("F:/DATA/dataset/v1")

# Model and optimizer
model = GCN(dropout = args.dropout)
loss_func = MSELoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
    # model.cuda()
    # train_X = [x.cuda() for x in train_X]
    # train_y = train_y.cuda()
    # test_X = [x.cuda() for x in test_X]
    # test_y = test_y.cuda()
    # T_k = [T.cuda() for T in T_k]


def train(epoch, X, y, i, batch_num):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(X, T_k)
    loss_train = loss_func(output, y.cuda())
    loss_train.backward()
    optimizer.step()
    print('[Train  {:d}/{:d}]\tEpoch: {:04d}'.format(i + 1, batch_num, epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train.item()


def validation(epoch):
    t = time.time()
    model.eval()
    output = model(test_X, T_k)

    loss_val = loss_func(output, test_y.cuda())
    print('[Valid]\tEpoch: {:04d}'.format(epoch + 1),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(test_y, T_k)
    loss_test = loss_func(output, test_y.cuda())
    print("[Test]\tTest set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Train model
t_total = time.time()
batch_size = 2
batch_num = int(math.ceil(train_X[0].shape[0] / batch_size))
for epoch in range(args.epochs):
    loss = []
    t = time.time()
    for i in range(batch_num):
        X = [x[batch_size * i: batch_size * (i + 1)] for x in train_X]
        y = train_y[batch_size * i: batch_size * (i + 1)]
        loss.append(train(epoch, X, y, i, batch_num))
    time_cost = time.time() - t
    mean_loss = sum(loss) / len(loss)
    info_string = "epoch: {:04d} loss_train: {:.4f} time: {:.4f}s\n".format(epoch + 1, mean_loss, time_cost)
    with open("F:/DATA/dataset/v1/log.txt", 'a') as f:
        f.write(info_string)
    print("--------------------------------------------------------------")
    print(info_string)
    print("--------------------------------------------------------------")
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
