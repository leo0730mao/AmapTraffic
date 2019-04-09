from __future__ import division
from __future__ import print_function

import math
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from traffic_predict.generate_dataset import DataReader
from traffic_predict.models import GCN, LSTM, Seq2seq


def config_model():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help = 'Disables CUDA training.')
	parser.add_argument('--src', action = 'store_true', default = "F:/DATA/dataset/v1",
						help = "data set's path.")
	parser.add_argument('--fastmode', action='store_true', default=True,
						help = 'Validate during training pass.')
	parser.add_argument('--seed', type=int, default=42, help='Random seed.')
	parser.add_argument('--epochs', type=int, default=50,
						help = 'Number of epochs to train.')
	parser.add_argument('--batch_size', type = int, default = 1024,
						help = 'size of mini batch')
	parser.add_argument('--lr', type=float, default=0.001,
						help = 'Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
						help = 'Weight decay (L2 loss on parameters).')
	parser.add_argument('--hidden', type=int, default=16,
						help = 'Number of hidden units.')
	parser.add_argument('--dropout', type=float, default=0.1,
						help = 'Dropout rate (1 - keep probability).')

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	return args


def RMSELoss(yhat, y):
	return torch.sqrt(torch.mean((yhat-y)**2))


def MAPELoss(yhat, y):
	temp = torch.Tensor([0.0001]).cuda()
	return torch.mean(torch.abs(yhat - (y + temp)) / torch.abs(y))


def run(x, y, adj, mode = "train"):
	batch_num = int(math.ceil(x.shape[0] / args.batch_size))
	loss = []
	mape_loss = []
	start_time = time.time()
	for i in range(batch_num):
		batch_start_time = time.time()

		# preparing needed data
		# input_x = [item[args.batch_size * i: args.batch_size * (i + 1)] for item in x]
		input_x = x[args.batch_size * i: args.batch_size * (i + 1)]
		input_y = y[args.batch_size * i: args.batch_size * (i + 1)]

		if mode == 'train':
			model.train(True)
		else:
			model.eval()

		output = model(input_x)
		# output = output.reshape(output.shape[0])
		batch_loss = loss_func(output, input_y.cuda())
		batch_mape_loss = MAPELoss(output, input_y.cuda())

		if mode == 'train':
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		print('[{:s}  {:d}/{:d}]\tEpoch: {:04d}'.format(mode, i + 1, batch_num, epoch + 1),
				'loss_train: {:.4f}'.format(batch_loss.item()),
				'loss_mape: {:.4f}'.format(batch_mape_loss.item() * 100),
				'time: {:.4f}s'.format(time.time() - batch_start_time))
		loss.append(batch_loss.item())
		mape_loss.append(batch_mape_loss.item() * 100)
	time_cost = time.time() - start_time
	mean_loss = sum(loss) / len(loss)
	mean_mape_loss = sum(mape_loss) / len(mape_loss)
	return time_cost, mean_loss, mean_mape_loss


if __name__ == '__main__':
	args = config_model()
	model = Seq2seq(dropout = args.dropout)
	loss_func = MSELoss()
	optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
	train_x, train_y = DataReader(args.src).load_data_for_seq("train")
	test_x, test_y = DataReader(args.src).load_data_for_seq("test")
	# train_x, train_y, test_x, test_y, adj = DataReader(args.src).load_data()
	train_start_time = time.time()
	date_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(train_start_time))
	for epoch in range(args.epochs):
		t, loss, mape_loss = run(train_x, train_y, None)
		info_string = "epoch: {:04d} loss_train: {:.4f} loss_mape: {:.4f} time: {:.4f}s\n".format(epoch + 1, loss, mape_loss, t)
		with open("F:/DATA/dataset/v1/%s_report.txt" % date_time, 'a') as f:
			f.write(info_string)
		print("--------------------------------------------------------------")
		print(info_string)
		print("--------------------------------------------------------------")

	t, loss, mape_loss = run(train_x, train_y, None, mode = "eval")
	info_string = "eval loss_test: {:.4f} loss_mape: {:.4f} time: {:.4f}s\n".format(loss, mape_loss, t)
	with open("F:/DATA/dataset/v1/%s_report.txt" % date_time, 'a') as f:
		f.write(info_string)
	print("--------------------------------------------------------------")
	print(info_string)
	print("--------------------------------------------------------------")

	torch.save(model, "F:/DATA/dataset/v1/model.pt")
