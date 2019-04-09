import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, RepeatVector
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
import keras

from traffic_predict.generate_dataset import DataReader


def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def my_model():
	model = Sequential()
	model.add(LSTM(128, input_shape = (24, 1)))
	model.add(Dense(1))
	model.add(Activation('linear'))
	model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = [root_mean_squared_error])
	return model


def seq2seq(input_size, max_out_seq_len = 24, hidden_size = 128):
	model = Sequential()

	# Encoder(第一个 LSTM)
	model.add(LSTM(input_shape = (24, 1), units = hidden_size, return_sequences = False))
	# 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
	model.add(RepeatVector(max_out_seq_len))

	# Decoder(第二个 LSTM)
	model.add(LSTM(units = hidden_size, return_sequences=True))

	# TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
	model.add(TimeDistributed(Dense(units = 1, activation="linear")))

	model.compile(loss="mse", optimizer='adam', metrics = [root_mean_squared_error])

	return model


if __name__ == '__main__':
	model = seq2seq(24)
	train_x, train_y = DataReader("F:/DATA/dataset/v1").load_data_for_seq("train")
	test_x, test_y = DataReader("F:/DATA/dataset/v1").load_data_for_seq("test")
	model.fit(train_x, train_y, batch_size = 1024, epochs = 20, validation_data = (test_x, test_y), verbose = 1)
	model.save('lstm_model.h5')
