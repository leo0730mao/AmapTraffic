import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
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


if __name__ == '__main__':
	model = my_model()
	train_x, train_y = DataReader("F:/DATA/dataset/v1").load_data_for_lstm("train")
	test_x, test_y = DataReader("F:/DATA/dataset/v1").load_data_for_lstm("test")
	model.fit(train_x, train_y, batch_size = 1024, nb_epoch = 20, validation_data = (test_x, test_y), verbose = 1)
	model.save('lstm_model.h5')
