from __future__ import division
from __future__ import print_function

import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.nn import MSELoss
import seaborn as sns
from traffic_predict.generate_dataset import DataReader
from traffic_predict.models import GCN, LSTM, Seq2seq


def config_model():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help = 'Disables CUDA training.')
	parser.add_argument('--src', action = 'store_true', default = "F:/DATA/dataset/v2",
						help = "data set's path.")
	parser.add_argument('--fastmode', action='store_true', default=True,
						help = 'Validate during training pass.')
	parser.add_argument('--seed', type=int, default=42, help='Random seed.')
	parser.add_argument('--epochs', type=int, default=50,
						help = 'Number of epochs to train.')
	parser.add_argument('--batch_size', type = int, default = 2,
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


def run_seq(x, y, ctx_y, mode = "train"):
	batch_num = int(math.ceil(x.shape[0] / args.batch_size))
	loss = []
	mape_loss = []
	start_time = time.time()
	y = y.reshape(y.shape[0], y.shape[1])
	for i in range(batch_num):
		batch_start_time = time.time()

		# preparing needed data
		input_x = x[args.batch_size * i: args.batch_size * (i + 1)].cuda()
		input_y = y[args.batch_size * i: args.batch_size * (i + 1)].cuda()

		time_feature = ctx_y[args.batch_size * i: args.batch_size * (i + 1)].cuda()

		if mode == 'train':
			model.train(True)
		else:
			model.eval()

		output = model(input_x, time_feature)
		batch_loss = loss_func(output, input_y)
		batch_mape_loss = MAPELoss(output, input_y)

		if mode == 'train':
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

			print('[{:s}  {:d}/{:d}]\tEpoch: {:04d}'.format(mode, i + 1, batch_num, epoch + 1),
					'loss_train: {:.4f}'.format(batch_loss.item()),
					'loss_mape: {:.4f}'.format(batch_mape_loss.item() * 100),
					'time: {:.4f}s'.format(time.time() - batch_start_time))
		else:
			print('[{:s}  {:d}/{:d}]\t'.format(mode, i + 1, batch_num),
					'loss_test: {:.4f}'.format(batch_loss.item()),
					'loss_mape: {:.4f}'.format(batch_mape_loss.item() * 100),
					'time: {:.4f}s'.format(time.time() - batch_start_time))
		loss.append(batch_loss.item())
		mape_loss.append(batch_mape_loss.item() * 100)
	time_cost = time.time() - start_time
	mean_loss = sum(loss) / len(loss)
	mean_mape_loss = sum(mape_loss) / len(mape_loss)
	return time_cost, mean_loss, mean_mape_loss


def run_gcn(x, y, ctx_y, adj, mode = "train"):
	batch_num = int(math.ceil(x[0].shape[0] / args.batch_size))
	loss = []
	mape_loss = []
	start_time = time.time()
	y = [item.reshape(item.shape[0] * item.shape[1], item.shape[2]) for item in y]
	y = torch.stack(y, dim = 1)
	y = y.reshape(y.shape[0], y.shape[1])
	loss_func = MSELoss()
	for i in range(batch_num):
		batch_start_time = time.time()

		# preparing gcn needed data
		input_x = [item[args.batch_size * i: args.batch_size * (i + 1)] for item in x]
		input_y = y[args.batch_size * i * x[0].shape[1]: args.batch_size * (i + 1) * x[0].shape[1]].cuda()

		time_feature = ctx_y[args.batch_size * i * x[0].shape[1]: args.batch_size * (i + 1) * x[0].shape[1]].cuda()
		if mode == "train":
			model.train(True)
		else:
			model.eval()

		output = model(input_x, adj, time_feature)
		batch_loss = loss_func(output, input_y)
		batch_mape_loss = MAPELoss(output, input_y)

		if mode == 'train':
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

			print('[{:s}  {:d}/{:d}]\tEpoch: {:04d}'.format(mode, i + 1, batch_num, epoch + 1),
					'loss_train: {:.4f}'.format(batch_loss.item()),
					'loss_mape: {:.4f}'.format(batch_mape_loss.item() * 100),
					'time: {:.4f}s'.format(time.time() - batch_start_time))
		else:
			print('[{:s}  {:d}/{:d}]\t'.format(mode, i + 1, batch_num),
					'loss_test: {:.4f}'.format(batch_loss.item()),
					'loss_mape: {:.4f}'.format(batch_mape_loss.item() * 100),
					'time: {:.4f}s'.format(time.time() - batch_start_time))
		loss.append(batch_loss.item())
		mape_loss.append(batch_mape_loss.item() * 100)
	time_cost = time.time() - start_time
	mean_loss = sum(loss) / len(loss)
	mean_mape_loss = sum(mape_loss) / len(mape_loss)
	return time_cost, mean_loss, mean_mape_loss


def draw_train_history(train_loss, test_loss, path):
	sns.set(color_codes = True)
	plt.xlabel("epochs")
	plt.ylabel("MSE loss")
	plt.title("train history of baseline")
	plt.plot(train_loss, label = "train loss")
	plt.plot(test_loss, label = "test loss")
	plt.savefig("%s/%s.jpg" % (args.src, path))


if __name__ == '__main__':
	args = config_model()
	model = GCN(dropout = args.dropout)
	# model = torch.load("F:/DATA/dataset/v1/gcn_model_with_time_feature.pt")
	loss_func = MSELoss()
	optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

	train_x, train_y, train_ctx_x, train_ctx_y = DataReader(args.src).load_data_for_gcn("train")
	valid_x, valid_y, valid_ctx_x, valid_ctx_y = DataReader(args.src).load_data_for_gcn("valid")
	test_x, test_y, test_ctx_x, test_ctx_y = DataReader(args.src).load_data_for_gcn("test")
	adj = DataReader(args.src).load_adj()

	train_start_time = time.time()
	date_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(train_start_time))
	train_loss_history = []
	valid_loss_history = []

	for epoch in range(args.epochs):
		t, loss, mape_loss = run_gcn(train_x, train_y, train_ctx_y, adj)
		train_loss_history.append(loss)
		info_string = "epoch: {:04d} loss_train: {:.4f} loss_mape: {:.4f} time: {:.4f}s\n".format(epoch + 1, loss, mape_loss, t)
		with open("%s/%s_report.txt" % (args.src, date_time), 'a') as f:
			f.write(info_string)
		print("--------------------------------------------------------------")
		print(info_string)
		print("--------------------------------------------------------------")
		_, loss, _ = run_gcn(valid_x, valid_y, valid_ctx_y, adj, mode = "eval")
		valid_loss_history.append(loss)

	torch.save(model, "%s/gcn_model_with_time_feature.pt" % args.src)
	draw_train_history(train_loss_history, valid_loss_history, "train_gcn")

	t, loss, mape_loss = run_gcn(test_x, test_y, test_ctx_y, adj, mode = "eval")
	info_string = "eval loss_test: {:.4f} loss_mape: {:.4f} time: {:.4f}s\n".format(loss, mape_loss, t)
	with open("%s/%s_report.txt" % (args.src, date_time), 'a') as f:
		f.write(info_string)
	print("--------------------------------------------------------------")
	print(info_string)
	print("--------------------------------------------------------------")

