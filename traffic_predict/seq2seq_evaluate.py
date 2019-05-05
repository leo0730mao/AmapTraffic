import torch
import time
import numpy as np
import math
from torch.nn import MSELoss
import pandas as pd
from torch.nn import MSELoss
from traffic_predict.Analysis import getTimeSeries
from traffic_predict.utils import *
from traffic_predict.models import Seq2seq
import matplotlib.pyplot as plt
import pickle
from traffic_predict.generate_dataset import DataReader


def MAPELoss(yhat, y):
	temp = torch.Tensor([0.0001]).cuda()
	return torch.mean(torch.abs(yhat - (y + temp)) / torch.abs(y))


def get_data(date):
	holi_info = pd.read_csv("F:/DATA/dataset/v1/holiday_info.csv")
	d = pd.DataFrame(getTimeSeries(["F:/DATA/Amap_csv/%s.csv" % date], center = [121.490857, 31.243738]))
	data = d['speed'].tolist()
	ts = d['ts'].tolist()
	temp_data = []
	temp_time_feature = []
	time_features = []
	for t in ts:
		day_type = holi_info[holi_info['date'] == t[:10]]['type'].tolist()[0]
		temp = [0] * 6
		temp[day_type] = 1
		temp[3] = int(t[11:13])
		temp[4] = float(t[14:16]) / 10
		if (temp[3] in range(6, 9)) or (temp[3] in range(16, 19)):
			temp[5] = 1
		temp_time_feature.append(temp)
	i = 0
	while i < len(data):
		temp_data.append(data[i: i + 24])
		time_features.append(temp_time_feature[i:i + 24])
		i += 24
	temp_data = np.array(temp_data)
	time_features = np.array(time_features)
	x = temp_data[:-1]
	x = x.reshape(x.shape[0], x.shape[1], 1)
	y = temp_data[1:]
	y = y.reshape(y.shape[0], y.shape[1], 1)

	ctx_y = time_features[1:]

	x = torch.Tensor(x).cuda()
	y = torch.Tensor(y).cuda()
	ctx_y = torch.Tensor(ctx_y).cuda()
	return x, y, ctx_y


def predict(x, y, ctx_y, date, mode, need_pic = False):
	model = torch.load("F:/DATA/dataset/v1/model_with_time_feature.pt")
	model.eval()
	output = model(x, ctx_y)
	y = y.reshape(y.shape[0], y.shape[1])
	loss_func = MSELoss()
	loss = loss_func(output, y)
	mape_loss = MAPELoss(output, y)
	print("loss: %s\tmape_loss: %s" % (loss.item(), mape_loss.item()))

	if need_pic is True:
		y = y.cpu().detach().numpy()
		output = output.cpu().detach().numpy()
		y = y.flatten()
		output = output.flatten()

		plt.title("%s NanJing East Road Seq2Seq_4h_4h" % date)
		plt.plot(output, label = "predict")
		plt.plot(y, label = "true")
		plt.legend()
		plt.savefig("baseline_%s.jpg" % mode)
	return loss.item(), mape_loss.item()


def get_predict_result_for_one_region(pred_y, v_id):
	pred_y = pred_y.reshape(-1, 194, 24)
	return pred_y[:, v_id, :]


def test_baseline():
	dates = dict()
	dates['new_year'] = ["2018-12-31", "2019-01-01", "2019-01-02"]
	dates['weekend'] = ["2019-02-16", "2019-02-17"]
	dates['normal'] = ["2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15"]
	dates['chinese_new_year'] = ["2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10"]

	for k in dates.keys():
		print("------------------%s----------------------------" % k)
		loss_all = []
		mape_loss_all = []
		for date in dates[k]:
			x, y, ctx_y = get_data(date)
			loss, mape_loss = predict(x, y, ctx_y, date = date, mode = k, need_pic = True)
			loss_all.append(loss)
			mape_loss_all.append(mape_loss * 100)
		loss_all = np.array(loss_all)
		mape_loss_all = np.array(mape_loss_all)
		print("-------------loss-------------")
		print(loss_all.mean())
		print(loss_all.max())
		print(loss_all.min())
		print(loss_all.var())
		print("-------------mape-------------")
		print(mape_loss_all.mean())
		print(mape_loss_all.max())
		print(mape_loss_all.min())
		print(mape_loss_all.var())
		print("----------------------------------------------------------------------")


def predict_seq(x, y, ctx_y, model, batch_size = 2):
	batch_num = int(math.ceil(x.shape[0] / batch_size))
	loss = []
	mape_loss = []
	start_time = time.time()
	y = y.reshape(y.shape[0], y.shape[1])
	loss_func = MSELoss()
	for i in range(batch_num):
		input_x = x[batch_size * i: batch_size * (i + 1)].cuda()
		input_y = y[batch_size * i: batch_size * (i + 1)].cuda()

		time_feature = ctx_y[batch_size * i: batch_size * (i + 1)].cuda()

		model.eval()

		output = model(input_x, time_feature)
		batch_loss = loss_func(output, input_y)
		batch_mape_loss = MAPELoss(output, input_y)

		loss.append(batch_loss.item())
		mape_loss.append(batch_mape_loss.item() * 100)
	time_cost = time.time() - start_time
	return time_cost, loss, mape_loss


def predict_gcn(x, y, adj, ctx_y, model, batch_size = 2):
	batch_num = int(math.ceil(x[0].shape[0] / batch_size))
	loss = []
	mape_loss = []
	start_time = time.time()
	y = [item.reshape(item.shape[0] * item.shape[1], item.shape[2]) for item in y]
	y = torch.stack(y, dim = 1)
	y = y.reshape(y.shape[0], y.shape[1])
	loss_func = MSELoss()
	for i in range(batch_num):
		# preparing gcn needed data
		input_x = [item[batch_size * i: batch_size * (i + 1)] for item in x]
		input_y = y[batch_size * i * x[0].shape[1]: batch_size * (i + 1) * x[0].shape[1]].cuda()

		time_feature = ctx_y[batch_size * i * x[0].shape[1]: batch_size * (i + 1) * x[0].shape[1]].cuda()

		model.eval()

		output = model(input_x, adj, time_feature)
		batch_loss = loss_func(output, input_y)
		batch_mape_loss = MAPELoss(output, input_y)

		"""print('[{:s}  {:d}/{:d}]\t'.format("test", i + 1, batch_num),
				'loss_test: {:.4f}'.format(batch_loss.item()),
				'loss_mape: {:.4f}'.format(batch_mape_loss.item() * 100),
				'time: {:.4f}s'.format(time.time() - batch_start_time))"""
		loss.append(batch_loss.item())
		mape_loss.append(batch_mape_loss.item() * 100)
	time_cost = time.time() - start_time
	return time_cost, loss, mape_loss


def stat(mse, mape):
	mse = np.array(mse)
	mape = np.array(mape)
	print("-------------loss-------------")
	print(mse.mean())
	print(mse.max())
	print(mse.min())
	print(mse.var())
	print("-------------mape-------------")
	print(mape.mean())
	print(mape.max())
	print(mape.min())
	print(mape.var())
	print("----------------------------------------------------------------------")


def test_seq():
	# adj = DataReader("F:/DATA/dataset/v1").load_data()
	model = torch.load("F:/DATA/dataset/v2/model_with_time_feature.pt")

	weekend_x, weekend_y, weekend_ctx_x, weekend_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_seq("weekend")
	_, loss, mape_loss = predict_seq(weekend_x, weekend_y, weekend_ctx_y, model)
	print("-----------------weekend-----------------------")
	stat(loss, mape_loss)

	holiday_x, holiday_y, holiday_ctx_x, holiday_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_seq("holiday")
	_, loss, mape_loss = predict_seq(holiday_x, holiday_y, holiday_ctx_y, model)
	print("-----------------holiday-----------------------")
	stat(loss, mape_loss)

	weekday_x, weekday_y, weekday_ctx_x, weekday_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_seq("weekday")
	_, loss, mape_loss = predict_seq(weekday_x, weekday_y, weekday_ctx_y, model)
	print("-----------------weekday-----------------------")
	stat(loss, mape_loss)

	test_x, test_y, test_ctx_x, test_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_seq("test")
	_, loss, mape_loss = predict_seq(test_x, test_y, test_ctx_y, model)
	print("-----------------test-----------------------")
	stat(loss, mape_loss)


def test_gcn():
	adj = DataReader("F:/DATA/dataset/v2").load_adj()
	model = torch.load("F:/DATA/dataset/v2/gcn_model_with_time_feature.pt")

	weekend_x, weekend_y, weekend_ctx_x, weekend_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_gcn("weekend")
	_, loss, mape_loss = predict_gcn(weekend_x, weekend_y, adj, weekend_ctx_y, model)
	print("-----------------weekend-----------------------")
	stat(loss, mape_loss)

	holiday_x, holiday_y, holiday_ctx_x, holiday_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_gcn("holiday")
	_, loss, mape_loss = predict_gcn(holiday_x, holiday_y, adj, holiday_ctx_y, model)
	print("-----------------holiday-----------------------")
	stat(loss, mape_loss)

	weekday_x, weekday_y, weekday_ctx_x, weekday_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_gcn("weekday")
	_, loss, mape_loss = predict_gcn(weekday_x, weekday_y, adj, weekday_ctx_y, model)
	print("-----------------weekday-----------------------")
	stat(loss, mape_loss)

	test_x, test_y, test_ctx_x, test_ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_gcn("test")
	_, loss, mape_loss = predict_gcn(test_x, test_y, adj, test_ctx_y, model)
	print("-----------------test-----------------------")
	stat(loss, mape_loss)


def test_seq_in_one_region(v_id):
	model = torch.load("F:/DATA/dataset/v1/model_with_time_feature.pt")

	x, y, _, ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_seq("newyear")

	x = x.reshape(-1, 194, 24, 1)
	y = y.reshape(-1, 194, 24, 1)
	ctx_y = ctx_y.reshape(-1, 194, 24, 6)

	x = x[:, v_id, :, :]
	y = y[:, v_id, :, :]
	ctx_y = ctx_y[:, v_id, :, :]

	_, loss, mape_loss = predict_seq(x, y, ctx_y, model)
	print("-----------------weekend-----------------------")
	stat(loss, mape_loss)


def test_gcn_in_one_region(v_id):
	model = torch.load("F:/DATA/dataset/v1/gcn_model_with_time_feature.pt")

	x, y, _, ctx_y = DataReader("F:/DATA/dataset/v2").load_data_for_gcn("newyear")
	adj = DataReader("F:/DATA/dataset/v2").load_adj()

	ctx_y = ctx_y.reshape(-1, 194, 24, 6)

	x = [item[:, v_id, :] for item in x]
	y = [item[:, v_id, :] for item in y]
	ctx_y = ctx_y[:, v_id, :, :]

	ctx_y = ctx_y.reshape(ctx_y.shape[0] * ctx_y.shape[1], 24, 1)

	_, loss, mape_loss = predict_gcn(x, y, adj, ctx_y, model)
	print("-----------------weekend-----------------------")
	stat(loss, mape_loss)


if __name__ == '__main__':
	test_seq()#2153#83
