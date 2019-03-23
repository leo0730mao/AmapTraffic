from __future__ import division
from __future__ import print_function

import math
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.nn import MSELoss
from pygcn.generate_dataset import DataReader
from pygcn.models import GCN


def config_model():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help = 'Disables CUDA training.')
	parser.add_argument('--src', action = 'store_true', default = "F:/DATA/dataset/v1",
						help = "data set's path.")
	parser.add_argument('--fastmode', action='store_true', default=True,
						help = 'Validate during training pass.')
	parser.add_argument('--seed', type=int, default=42, help='Random seed.')
	parser.add_argument('--epochs', type=int, default=20,
						help = 'Number of epochs to train.')
	parser.add_argument('--batch_size', type = int, default = 16,
						help = 'size of mini batch')
	parser.add_argument('--lr', type=float, default=0.01,
						help = 'Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
						help = 'Weight decay (L2 loss on parameters).')
	parser.add_argument('--hidden', type=int, default=16,
						help = 'Number of hidden units.')
	parser.add_argument('--dropout', type=float, default=0.5,
						help = 'Dropout rate (1 - keep probability).')

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	return args


def run(epoch, x, y, adj, args, mode = "train"):
	batch_num = int(math.ceil(x[0].shape[0] / args.batch_size))
	loss = []
	start_time = time.time()
	for i in range(batch_num):
		batch_start_time = time.time()

		# preparing needed data
		input_x = [x[args.batch_size * i: args.batch_size * (i + 1)] for x in x]
		input_y = y[args.batch_size * i: args.batch_size * (i + 1)]

		if mode == "train":
			model.train()
			optimizer.zero_grad()
		else:
			model.eval()

		output = model(input_x, adj)
		batch_loss = loss_func(output, input_y.cuda())

		if mode == "train":
			batch_loss.backward()
			optimizer.step()

		print('[{:s}  {:d}/{:d}]\tEpoch: {:04d}'.format(mode, i + 1, batch_num, epoch + 1),
				'loss_train: {:.4f}'.format(batch_loss.item()),
				'time: {:.4f}s'.format(time.time() - batch_start_time))
		loss.append(batch_loss.item())

	time_cost = time.time() - start_time
	mean_loss = sum(loss) / len(loss)
	return time_cost, mean_loss


if __name__ == '__main__':
	args = config_model()
	model = GCN(dropout = args.dropout)
	loss_func = MSELoss()
	optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
	train_x, train_y, test_x, test_y, adj = DataReader(args.src).load_data()
	start_time = time.time()
	date_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(start_time))
	for epoch in range(args.epochs):
		time_cost, mean_loss = run(epoch, train_x, train_y, adj, args, mode = "train")
		info_string = "epoch: {:04d} loss_train: {:.4f} time: {:.4f}s\n".format(epoch + 1, mean_loss, time_cost)
		with open("F:/DATA/dataset/v1/%s_report.txt" % date_time, 'a') as f:
			f.write(info_string)
		print("--------------------------------------------------------------")
		print(info_string)
		print("--------------------------------------------------------------")


