from pygcn.MyGraph import Graph
from pygcn.utils import *
from pygcn.config import *
import os
import pickle
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp


class DataBuilder:
	def __init__(self, path, length = 1000, time_steps = 24, if_need_graph = False):
		self.path = path
		self.length = length
		self.time_steps = time_steps
		self.if_need_graph = if_need_graph

		for mode in ['train', 'test']:
			if if_need_graph:  # 若尚未将原始数据划分到区域
				# 创建每个时间段对应的graph文件
				# 产生文件./mode/graph/time_slot.dat
				self.build_data_graph(mode = mode, length = self.length)

			# 构建matrix格式的全市各区域平均车速, size: (time_slot_num, vertex_num)
			# 产生文件./mode/X.dat
			self.build_feature_as_matrix(mode = mode)

			# 构建目标模型的合法输入X, y, size: (time_slot_num, vertex_num, 1), X为长time_step的list
			# 产生文件./mode_purposed.dat
			self.build_data_for_purposed_model(mode = mode)

		# 创建训练集中出现的所有道路及其平均车速，用于目标模型GCN层的输入adj
		# 产生文件./road_set.csv
		self.build_road_set()

		# 创建用于目标模型GCN层的输入adj
		# 产生文件./adj.dat
		self.build_road_graph()

	def build_road_set(self):
		print("building road set...")
		res_path = self.path + "/road_set.csv"
		if os.path.exists(res_path):
			return
		else:
			csv_path = "%s/train/graph" % self.path
			files = os.listdir(csv_path)
			res = {}
			for file in files:
				file_path = csv_path + "/" + file
				data = pd.read_csv(file_path)
				roads_list = data['road'].tolist()
				speed_list = data['speed'].tolist()
				for road, speed in zip(roads_list, speed_list):
					if road in res.keys():
						res[road].append(speed)
					else:
						res[road] = [speed]
			road_set = {'road': [], 'speed': []}
			for road in res.keys():
				road_set['road'].append(road)
				road_set['speed'].append(sum(res[road]) / len(res[road]))
			road_set = pd.DataFrame(road_set)
			print("contains %s road" % len(road_set))
			road_set.to_csv(res_path, index = False)

	def build_feature_as_matrix(self, mode = "train"):
		print("build %s data's matrix..." % mode)
		res_path = "%s/%s/X.dat" % (self.path, mode)
		if os.path.exists(res_path):
			return
		else:
			graph_path = "%s/%s/graph" % (self.path, mode)
			dirs = ["%s/%s" % (graph_path, d) for d in os.listdir(graph_path)]
			files = []
			for d in dirs:
				files += ["%s/%s" % (d, file) for file in os.listdir(d)]
			res = None
			ctx = None
			holiday_info = pd.read_csv("%s/holiday_info.csv" % self.path)
			for file in files:
				with open(file, 'rb') as f:
					data = pickle.load(f)
				print("%s read success" % file)
				vertex = data.v_to_vector()
				date = os.path.split(file)[1].split(".")[0][:10]
				t = [int(i) for i in os.path.split(file)[1].split(".")[0][11:].split("-")]
				date_type = holiday_info[holiday_info['date'] == date]['type'].tolist()[0]
				minute_num = t[0] * 6 + t[1] / 10
				if res is None:
					res = vertex
					ctx = np.array([date_type, minute_num])
				else:
					res = np.row_stack((res, vertex))
					ctx = np.row_stack((res, np.array([date_type, minute_num])))
			res = sp.csr_matrix(res, shape = res.shape, dtype = np.float32, copy = False)
			ctx = np.tile(np.expand_dims(ctx, axis=1), (1, res.shape[1], 1))
			print("result's size: (%s, %s)" % res.shape)
			with open(res_path, 'wb') as f:
				pickle.dump(res, f)
			with open("%s/%s/ctx.dat" % (self.path, mode)) as f:
				pickle.dump(ctx, f)

	def build_data_for_purposed_model(self, mode = "train"):
		print("building %s data for purposed model..." % mode)
		res_path = "%s/%s_purposed.dat" % (self.path, mode)
		ctx_path = "%s/%s_ctx_purposed.dat" % (self.path, mode)
		if os.path.exists(res_path):
			return
		else:
			with open("%s/%s/X.dat" % (self.path, mode), 'rb') as f:
				feature = pickle.load(f)
			with open("%s/%s/ctx.dat" % (self.path, mode), 'rb') as f:
				raw_ctx = pickle.load(f)
			features = []
			ctx = []
			res = {}
			ctx_res = {}
			for i in range(self.time_steps + 1):
				features.append(feature[i:])
				ctx.append(raw_ctx[i:])
			for i in range(self.time_steps):
				features[i] = features[i][:i - self.time_steps]
				ctx[i] = ctx[i][:i - self.time_steps]
				features[i] = np.reshape(features[i], (features[i].shape[0], features[i].shape[1], 1))
			res['X'] = features[:-1]
			res['y'] = features[-1]
			ctx_res['X'] = ctx[:-1]
			ctx_res['y'] = ctx[-1]
			print("X(%s, %s, %s, %s) y(%s, %s)" % tuple((len(res['X'],)) + res['X'][0].shape + res['y'].shape))
			print("context X(%s, %s, %s) y(%s, %s)" % tuple((len(ctx_res['X'], )) + ctx_res['X'][0].shape + ctx_res['y'].shape))
			with open(res_path, 'wb') as f:
				pickle.dump(res, f)
			with open(ctx_path, 'wb') as f:
				pickle.dump(ctx_res, f)

	def build_data_graph(self, mode = "train", length = 1000):
		print("build %s data's graph with size %s..." % (mode, length))
		csv_path = "%s/%s/csv" % (self.path, mode)
		dst_path = "%s/%s/graph" % (self.path, mode)
		if not os.path.exists(dst_path):
			os.mkdir(dst_path)
		files = os.listdir(csv_path)
		res = None
		for fileName in files:
			print(fileName)
			file = csv_path + '/' + fileName
			date = fileName.split(".")[0]
			date_path = "%s/%s" % (csv_path, date)
			if not os.path.exists(date_path):
				os.mkdir(date_path)
			data = pd.read_csv(file, encoding = "gbk")
			ts = pd.date_range(start = date + " 00:00", end = date + " 23:50", freq = '10T')
			for t in ts:
				data_selected = data[data['time'] == t.strftime("%Y-%m-%d %H:%M")]
				if len(data_selected) > 0:
					res = Graph(length, data_selected)
				with open("%s/%s.dat" % (date_path, t.strftime("%Y-%m-%d-%H-%M")), 'wb') as f:
					pickle.dump(res, f)

	def build_road_graph(self):
		print("building adj for GCN...")
		res_path = "%s/adj.dat" % self.path
		if os.path.exists(res_path):
			return
		else:
			raw_adj_path = "%s/raw_adj.dat" % self.path
			if os.path.exists(raw_adj_path):
				with open(raw_adj_path, 'rb') as f:
					raw_adj = pickle.load(f)
			else:
				csv_file = "%s/road_set.csv" % self.path
				road_data = pd.read_csv(csv_file)
				raw_adj = Graph(self.length, data = road_data, need_edge = True).e_to_matrix()
				with open(raw_adj_path, 'wb') as f:
					pickle.dump(raw_adj, f)
			norm_adj = normalized_laplacian(raw_adj, SYM_NORM)
			scaled_adj = rescale_laplacian(norm_adj)
			adj = chebyshev_polynomial(scaled_adj, MAX_DEGREE)

			with open(res_path, 'wb') as f:
				pickle.dump(adj, f)

	def build_data_for_avg_model(self):
		print("building data for model HA...")
		res_path = "%s/HA.dat" % self.path
		if os.path.exists(res_path):
			return
		else:
			with open("%s/train/X.dat" % self.path, 'rb') as f:
				train_data = pickle.load(f)
			with open("%s/test/X.dat" % self.path, 'rb') as f:
				test_data = pickle.load(f)
			print(train_data.shape)
			data = np.concatenate((train_data.toarray(), test_data.toarray()), axis = 0)
			print(data.shape)
			with open(res_path, 'wb') as f:
				pickle.dump(data, f)


class DataReader:
	def __init__(self, path, model_type = "purposed"):
		self.path = path
		self.model_type = model_type

	def load_data(self):
		print("load data set for model %s..." % self.model_type)
		with open("%s/train_%s.dat" % (self.path, self.model_type), 'rb') as f:
			data = pickle.load(f)
		train_x, train_y = data['X'], data['y']
		print("size: X (%s,%s,%s), y (%s,%s)" % tuple(train_x[0].shape + train_y.shape))

		with open("%s/test_%s.dat" % (self.path, self.model_type), 'rb') as f:
			data = pickle.load(f)
		test_x, test_y = data['X'], data['y']
		print("size: X (%s,%s,%s), y (%s,%s)" % tuple(test_x[0].shape + test_y.shape))

		with open("%s/train_ctx_%s.dat" % (self.path, self.model_type), 'rb') as f:
			data = pickle.load(f)
		train_ctx_x, train_ctx_y = data['X'], data['y']
		print("size: X (%s,%s,%s), y (%s,%s)" % tuple(train_x[0].shape + train_y.shape))

		with open("%s/test_ctx_%s.dat" % (self.path, self.model_type), 'rb') as f:
			data = pickle.load(f)
		test_ctx_x, test_ctx_y = data['X'], data['y']
		print("size: X (%s,%s,%s), y (%s,%s)" % tuple(test_x[0].shape + test_y.shape))

		train_x = [torch.Tensor(x) for x in train_x]
		train_y = torch.Tensor(train_y)
		test_x = [torch.Tensor(x) for x in test_x]
		test_y = torch.Tensor(test_y)

		train_ctx_x = [torch.Tensor(x) for x in train_ctx_x]
		train_ctx_y = torch.Tensor(train_ctx_y)
		test_ctx_x = [torch.Tensor(x) for x in test_ctx_x]
		test_ctx_y = torch.Tensor(test_ctx_y)

		with open("%s/adj.dat" % self.path, 'rb') as f:
			adj = pickle.load(f)
		adj = [sparse_mx_to_torch_sparse_tensor(m) for m in adj]
		return train_x, train_y, test_x, test_y, adj, train_ctx_x, train_ctx_y, test_ctx_x, test_ctx_y


if __name__ == '__main__':
	db = DataBuilder("F:/DATA/dataset/v1")
	db.build_data_for_avg_model()
