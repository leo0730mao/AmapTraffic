from traffic_predict.utils.Analysis import getTimeSeries
from traffic_predict.utils.generate_dataset import DataReader
import pandas as pd
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt


def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_data(predate, date):
	d1 = pd.DataFrame(getTimeSeries(["F:/DATA/Amap_csv/%s.csv" % predate], center = [121.490857, 31.243738]))
	d2 = pd.DataFrame(getTimeSeries(["F:/DATA/Amap_csv/%s.csv" % date], center = [121.490857, 31.243738]))
	d1 = d1['speed'].tolist()
	d2 = d2['speed'].tolist()
	data = d1[-24:] + d2
	temp = []
	i = 0
	while i < len(data):
		temp.append(data[i: i + 24])
		i += 24
	temp = np.array(temp)
	# for i in range(24, len(data)):
		# x.append(data[i - 24:i])
		# y.append(data[i])
	x = temp[:-1]
	x = x.reshape(x.shape[0], x.shape[1], 1)
	y = temp[1:]
	y = y.reshape(y.shape[0], y.shape[1], 1)
	return x, y


def predict_one_day(predate, date):
	x, y = get_data(predate, date)
	model = load_model("lstm_model.h5", custom_objects = {'root_mean_squared_error': root_mean_squared_error})
	loss = model.evaluate(x, y)
	print(loss)

	pred_y = model.predict(x)
	pred_y = list(pred_y.flatten())
	y = list(y.flatten())
	# pred_y = np.mean(x, axis = 1)
	# loss = root_mean_squared_error(y, pred_y)
	# print(loss.numpy().mean())
	plt.title("2019-02-17 NanJing East Road Seq2Seq_4h_4h")
	plt.plot(pred_y, label="predict")
	plt.plot(y, label="true")
	plt.legend()
	plt.savefig("test_seq2seq_4h.jpg")


if __name__ == '__main__':
	predict_one_day("2019-02-16", "2019-02-17")
