from pygcn.MyGraph import Graph
from pygcn.utils import *
from pygcn.config import *
import os
import pickle
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx


class DataBuilder:
	def __init__(self, path, length = 1000, time_steps = 24, if_need_graph = False):
		self.path = path
		self.length = length
		self.time_steps = time_steps
		self.if_need_graph = if_need_graph

		# 创建训练集中出现的所有道路及其平均车速，用于目标模型GCN层的输入adj
		# 产生文件./road_set.csv
		self.build_road_set()

		# 创建用于目标模型GCN层的输入adj
		# 产生文件./adj.dat
		self.build_raw_adj()

		#筛选区域
		with open("%s/selected_vertex.dat" % self.path, 'rb') as f:
			self.selected_vertex = pickle.load(f)
		self.selected_vertex = [int(i) for i in self.selected_vertex]
		self.build_adj()

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
		res_path = "%s/%s/X.csv" % (self.path, mode)
		if os.path.exists(res_path):
			return
		else:
			graph_path = "%s/%s/graph" % (self.path, mode)
			dirs = ["%s/%s" % (graph_path, d) for d in os.listdir(graph_path)]
			files = []
			for d in dirs:
				files += ["%s/%s" % (d, file) for file in os.listdir(d)]
			res = {}
			for i in range(4148):
				res[i] = []
			res['year'] = []
			res['month'] = []
			res['day'] = []
			res['hour'] = []
			res['minute'] = []
			res['ifholiday'] = []
			holiday_info = pd.read_csv("%s/holiday_info.csv" % self.path)
			for file in files:
				with open(file, 'rb') as f:
					data = pickle.load(f)
				print("%s read success" % file)
				vertex = data.v_to_vector()
				for i in range(len(vertex)):
					res[i].append(vertex[i])
				date_info = os.path.split(file)[1].split(".")[0].split("-")
				date = "-".join(date_info)[:10]
				res['year'].append(int(date_info[0]))
				res['month'].append(int(date_info[1]))
				res['day'].append(int(date_info[2]))
				res['hour'].append(int(date_info[3]))
				res['minute'].append(int(date_info[4]))
				res['ifholiday'].append(int(holiday_info[holiday_info['date'] == date]['type'].tolist()[0]))
			res = pd.DataFrame(res)

			res.to_csv(res_path, index = False)

	def build_data_for_purposed_model(self, mode = "train"):
		print("building %s data for purposed model..." % mode)
		res_path = "%s/%s_purposed.dat" % (self.path, mode)
		ctx_path = "%s/%s_ctx_purposed.dat" % (self.path, mode)
		if os.path.exists(res_path):
			return
		else:
			with open("%s/%s/X.dat" % (self.path, mode), 'rb') as f:
				feature = pickle.load(f).toarray()
			with open("%s/%s/ctx.dat" % (self.path, mode), 'rb') as f:
				raw_ctx = pickle.load(f)
			feature = self.compression_data(feature)
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
			print("X(%s, %s, %s, %s) y(%s, %s)" % tuple((len(res['X']),) + res['X'][0].shape + res['y'].shape))
			print(ctx_res['X'][0].shape)
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

	def build_raw_adj(self):
		print("building raw adj...")
		res_path = "%s/raw_adj.dat" % self.path
		if os.path.exists(res_path):
			return
		else:
			csv_file = "%s/road_set.csv" % self.path
			road_data = pd.read_csv(csv_file)
			raw_adj = Graph(self.length, data = road_data, need_edge = True).e_to_matrix()
			with open(res_path, 'wb') as f:
				pickle.dump(raw_adj, f)

	def build_adj(self):
		print("building adj...")
		res_path = "%s/adj.dat" % self.path
		if os.path.exists(res_path):
			pass
		else:
			with open("%s/raw_adj.dat" % self.path, 'rb') as f:
				raw_adj = pickle.load(f)
			raw_adj = self.compression_data(raw_adj.todense(), if_adj = True)
			raw_adj = sp.csr_matrix(raw_adj, shape = raw_adj.shape, dtype = np.float32, copy = False)
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
				data = pickle.load(f)
			data = self.compression_data(data)
			print(data.shape)
			with open(res_path, 'wb') as f:
				pickle.dump(data, f)

	def build_network(self):
		print("building network...")
		res_path = "%s/network.dat" % self.path
		if os.path.exists(res_path):
			return
		else:
			raw_adj_path = "%s/raw_adj.dat" % self.path
			if os.path.exists(raw_adj_path):
				with open(raw_adj_path, 'rb') as f:
					raw_adj = pickle.load(f).tocoo()
			else:
				return
			res = nx.Graph()
			edges = []
			for i in range(len(raw_adj.data)):
				edges.append((raw_adj.row[i], raw_adj.col[i], raw_adj.data[i]))
			res.add_weighted_edges_from(edges)
			with open(res_path, 'wb') as f:
				pickle.dump(res, f)

	def select_vertex(self):
		network_path = "%s/network.dat" % self.path
		with open(network_path, 'rb') as f:
			net = pickle.load(f)
		pr_net = nx.pagerank(net)
		pr_net = sorted(pr_net.items(), key = lambda d: d[1])[-500:]
		selected_vertex = [item[0] for item in pr_net]
		path = "%s/selected_vertex.dat" % self.path
		with open(path, 'wb') as f:
			pickle.dump(selected_vertex, f)
		return selected_vertex

	def compression_data(self, x, if_adj = False):
		x = x[:, self.selected_vertex]
		if if_adj:
			x = x[self.selected_vertex, :]
		return x


class DataReader:
	def __init__(self, path, model_type = "purposed"):
		self.path = path
		self.model_type = model_type
		with open("%s/selected_vertex.dat" % self.path, 'rb') as f:
			self.selected_vertex = pickle.load(f)

	def load_data(self, need_ctx = False):
		print("load data set for model %s..." % self.model_type)
		with open("%s/train_%s.dat" % (self.path, self.model_type), 'rb') as f:
			data = pickle.load(f)
		train_x, train_y = data['X'], data['y']
		print("size: X (%s,%s,%s), y (%s,%s)" % tuple(train_x[0].shape + train_y.shape))

		with open("%s/test_%s.dat" % (self.path, self.model_type), 'rb') as f:
			data = pickle.load(f)
		test_x, test_y = data['X'], data['y']
		print("size: X (%s,%s,%s), y (%s,%s)" % tuple(test_x[0].shape + test_y.shape))

		with open("%s/adj.dat" % self.path, 'rb') as f:
			adj = pickle.load(f)
		# adj = sp.csr_matrix(self.compression_data(adj.toarray(), if_adj = True))
		# adj = sp.coo_matrix(adj, dtype = np.float32)
		# edge_index = torch.Tensor(np.row_stack((adj.row, adj.col)))
		# edge_weight = torch.Tensor(adj.data)

		adj = [sparse_mx_to_torch_sparse_tensor(m) for m in adj]

		"""train_x = [self.compression_data(x) for x in train_x]
		train_y = self.compression_data(train_y)
		test_x = [self.compression_data(x) for x in test_x]
		test_y = self.compression_data(test_y)"""

		train_x = [torch.Tensor(x) for x in train_x]
		train_y = torch.Tensor(train_y)
		test_x = [torch.Tensor(x) for x in test_x]
		test_y = torch.Tensor(test_y)

		if need_ctx:
			with open("%s/train_ctx_%s.dat" % (self.path, self.model_type), 'rb') as f:
				data = pickle.load(f)
			train_ctx_x, train_ctx_y = data['X'], data['y']
			train_ctx_x = [np.tile(np.expand_dims(x, axis = 1), (1, train_x[0].shape[1], 1)) for x in train_ctx_x]
			train_ctx_y = np.tile(np.expand_dims(train_ctx_y, axis = 1), (1, train_x[0].shape[1], 1))

			with open("%s/test_ctx_%s.dat" % (self.path, self.model_type), 'rb') as f:
				data = pickle.load(f)
			test_ctx_x, test_ctx_y = data['X'], data['y']
			test_ctx_x = [np.tile(np.expand_dims(x, axis = 1), (1, train_x[0].shape[1], 1)) for x in test_ctx_x]
			test_ctx_y = np.tile(np.expand_dims(test_ctx_y, axis = 1), (1, train_x[0].shape[1], 1))

			train_ctx_x = [torch.Tensor(x) for x in train_ctx_x]
			train_ctx_y = torch.Tensor(train_ctx_y)
			test_ctx_x = [torch.Tensor(x) for x in test_ctx_x]
			test_ctx_y = torch.Tensor(test_ctx_y)

			return train_x, train_y, test_x, test_y, adj, train_ctx_x, train_ctx_y, test_ctx_x, test_ctx_y
		else:
			return train_x, train_y, test_x, test_y, adj

	def compression_data(self, x, if_adj = False):
		x = x[:, self.selected_vertex]
		if if_adj:
			x = x[self.selected_vertex, :]
		return x


if __name__ == '__main__':
	db = DataBuilder("F:/DATA/dataset/v1")
