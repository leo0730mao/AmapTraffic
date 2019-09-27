import os
import pickle
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def cluster_spatial(file_path):
	files = ["%s/%s" % (file_path, f) for f in os.listdir(file_path)]
	mat = []
	for file in files:
		with open(file, 'rb') as f:
			g = pickle.load(f)
		v = g.v_to_vector()
		mat.append(v)
	mat = np.stack(mat, axis = 1)
	print(mat.shape)
	s = np.sum(mat, axis = 1)
	coord = np.where(s != 0)
	mat = mat[coord[0], :]
	print(mat.shape)
	bandwidth = estimate_bandwidth(mat, quantile = 0.2)

	ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
	ms.fit(mat)  # 训练模型
	with open("./spatial_cluster_model.dat", 'wb') as f:
		pickle.dump(ms, f)


def cluster_result():

if __name__ == '__main__':
	cluster_spatial("F:/DATA/dataset/v2/train/graph/2019-01-20")
