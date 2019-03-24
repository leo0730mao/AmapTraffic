import math
from geopy.distance import vincenty
import numpy as np
import scipy.sparse as sp
from math import exp
import pickle
import pandas as pd


class GeoComputer(object):
	a = 6378137
	b = 6356752.3142
	f = 1 / 298.2572236

	@classmethod
	def rad(cls, d):
		return d * math.pi / 180.0

	@classmethod
	def deg(cls, x):
		return x * 180 / math.pi

	@classmethod
	def computer_that_lonlat(cls, lon, lat, brng, dist):  # 北：0；东：90；南：180；西：270
		alpha1 = cls.rad(brng)
		sinAlpha1 = math.sin(alpha1)
		cosAlpha1 = math.cos(alpha1)

		tanU1 = (1 - cls.f) * math.tan(cls.rad(lat))
		cosU1 = 1 / math.sqrt((1 + tanU1 * tanU1))
		sinU1 = tanU1 * cosU1
		sigma1 = math.atan2(tanU1, cosAlpha1)
		sinAlpha = cosU1 * sinAlpha1
		cosSqAlpha = 1 - sinAlpha * sinAlpha
		uSq = cosSqAlpha * (cls.a * cls.a - cls.b * cls.b) / (cls.b * cls.b)
		A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
		B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

		cos2SigmaM = 0
		sinSigma = 0
		cosSigma = 0
		sigma = dist / (cls.b * A)
		sigmaP = 2 * math.pi

		while abs(sigma - sigmaP) > 1e-12:
			cos2SigmaM = math.cos(2 * sigma1 + sigma)
			sinSigma = math.sin(sigma)
			cosSigma = math.cos(sigma)
			deltaSigma = B * sinSigma * (
					cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)))
			sigmaP = sigma
			sigma = dist / (cls.b * A) + deltaSigma

		tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
		lat2 = math.atan2(sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1, (1 - cls.f) * math.sqrt(sinAlpha * sinAlpha + tmp * tmp))
		lam = math.atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1)
		C = cls.f / 16 * cosSqAlpha * (4 + cls.f * (4 - 3 * cosSqAlpha))
		L = lam - (1 - C) * cls.f * sinAlpha * (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))

		return [lon + cls.deg(L), cls.deg(lat2)]


class Graph(object):
	def __init__(self, length, data, need_edge = False, sigma = 10, top_left = (121.119238, 31.411676), right_bottom = (121.829403, 30.866891)):
		self.base = top_left
		self.len = length
		self.sigma = sigma
		self.need_edge = need_edge
		self.latDelta = GeoComputer.computer_that_lonlat(top_left[0], top_left[1], 180, length)[1] - top_left[1]
		self.lngDelta = GeoComputer.computer_that_lonlat(top_left[0], top_left[1], 90, length)[0] - top_left[0]
		self.lngNum = int((right_bottom[0] - top_left[0]) / self.lngDelta)
		self.latNum = int((right_bottom[1] - top_left[1]) / self.latDelta)
		self.edges = {}
		self.vertex = {}
		for i in range(self.lngNum + 1):
			for j in range(self.latNum + 1):
				self.edges["%_s%s" % (i, j)] = {}
				self.vertex["%s_%s" % (i, j)] = [0, 0]
		self.data = data
		self.generate()

	def box_distance(self, box1, box2):
		box1 = self.v_string2latlng(box1)
		box1.reverse()
		box2 = self.v_string2latlng(box2)
		box2.reverse()
		return vincenty(box1, box2).m

	def v_string2latlng(self, v_id, mode = 1):
		v_id = [int(i) for i in v_id.split("_")]
		if v_id[0] < 0 or v_id[0] > self.lngNum or v_id[1] < 0 or v_id[1] > self.latNum:
			return None
		lng = v_id[0] * self.lngDelta + self.base[0]
		lat = v_id[1] * self.latDelta + self.base[1]
		if mode == 1:
			return [lat, lng]
		else:
			return [lng, lat]

	def v_string2int(self, v_id):
		temp = [int(i) for i in v_id.split("_")]
		return temp[0] * (self.latNum + 1) + temp[1]

	def v_int2string(self, v_id):
		i = int(v_id / (self.latNum + 1))
		j = v_id % (self.latNum + 1)
		return "%s_%s" % (i, j)

	def in_box(self, point):
		lng = int((float(point[0]) - self.base[0]) / self.lngDelta)
		lat = int((float(point[1]) - self.base[1]) / self.latDelta)
		if lng < 0 or lng > self.lngNum or lat < 0 or lat > self.latNum:
			return None
		else:
			return [lng, lat]

	def line_in_box(self, x, y):
		x = [float(i) for i in x]
		y = [float(i) for i in y]
		box1 = self.in_box(x)
		box2 = self.in_box(y)
		box_list = []
		if box1 is None and box2 is None:
			return []
		elif box1 is not None and box2 is None:
			return ["%s_%s" % tuple(box1)]
		elif box2 is not None and box1 is None:
			return ["%s_%s" % tuple(box2)]
		else:
			if box1[1] < box2[1]:
				y_min, y_max = box1[1] + 1, box2[1] + 1
			else:
				y_min, y_max = box2[1] + 1, box1[1] + 1
			if box1 == box2:
				box_list.append("%s_%s" % tuple(box2))
				return box_list
			elif box1[0] == box2[0]:
				j = box1[1]
				d = 1 if(box1[1] < box2[1]) else -1
				while j != box2[1]:
					box_list.append("%s_%s" % (box1[0], j))
					j += d
					box_list.append("%s_%s" % (box1[0], j))
				return box_list
			elif box1[1] == box2[1]:
				for i in range(box1[0], box2[0] + 1):
					box_list.append("%s_%s" % (i, box1[1]))
				return box_list
			else:
				intersections = []
				k = (y[1] - x[1]) / (y[0] - x[0])
				for i in range(box1[0] + 1, box2[0] + 1):
					intersections.append([i, ((x[1] + k * (self.base[0] + (i * self.lngDelta) - x[0])) - self.base[1]) / self.latDelta])
				for j in range(y_min, y_max):
					intersections.append([((x[0] + (1 / k) * ((self.base[1] + j * self.latDelta) - x[1])) - self.base[0]) / self.lngDelta, j])
				intersections.sort(key = lambda dd: dd[0])
				current_box = box1
				box_list.append("%s_%s" % tuple(current_box))
				for point in intersections:
					point = [int(point[0]), int(point[1])]
					if point == current_box:
						current_box = [point[0], point[1] - 1]
					else:
						current_box = point
					box_list.append("%s_%s" % tuple(current_box))
				return box_list

	def generate(self):
		for index, row in self.data.iterrows():
			road = row['road']
			speed = float(row['speed'])
			points = [p.split(",") for p in road.split(";")]
			temp_box_list = []
			for i in range(len(points) - 1):
				if points[i][0] < points[i + 1][0]:
					temp = self.line_in_box(points[i], points[i + 1])
				else:
					temp = self.line_in_box(points[i + 1], points[i])
					temp.reverse()
				temp_box_list = temp_box_list[:-1] + temp
			if len(temp_box_list) > 0:
				box_list = list(set(temp_box_list))
				box_list.sort(key = temp_box_list.index)
				box_distance = [0]
				pre_box = box_list[0]
				for i in range(len(box_list)):
					self.vertex[box_list[i]][0] += speed
					self.vertex[box_list[i]][1] += 1
					if self.need_edge:
						box_distance.append(vincenty(self.v_string2latlng(pre_box), self.v_string2latlng(box_list[i])).km + box_distance[-1])
						j = 0
						while j < i:
							dis = box_distance[i + 1] - box_distance[j + 1]
							temp = exp(-(dis * 3600 / speed) / self.sigma)
							if box_list[j] == box_list[i]:
								print(road)
								print(box_list[i])
								print(box_list[j])
							if box_list[j] in self.edges[box_list[i]].keys():
								self.edges[box_list[i]][box_list[j]].append(temp)
							else:
								self.edges[box_list[i]][box_list[j]] = [temp]
							j += 1
						pre_box = box_list[i]
		if self.need_edge:
			adj = np.zeros(((self.lngNum + 1) * (self.latNum + 1), (self.lngNum + 1) * (self.latNum + 1)))
			for box1 in self.edges.keys():
				for box2 in self.edges[box1].keys():
					temp = sum(self.edges[box1][box2]) / len(self.edges[box1][box2])
					adj[self.v_string2int(box1)][self.v_string2int(box2)] = temp
					adj[self.v_string2int(box2)][self.v_string2int(box1)] = temp
			self.edges = sp.csr_matrix(adj, shape = adj.shape, dtype = np.float32, copy = False)

	def v_to_vector(self):
		vertex = []
		for i in range(self.lngNum + 1):
			for j in range(self.latNum + 1):
				box = "%s_%s" % (i, j)
				if self.vertex[box][1] != 0:
					vertex.append(self.vertex[box][0] / self.vertex[box][1])
				else:
					vertex.append(0)
		return np.array(vertex)

	def e_to_matrix(self):
		return self.edges

	def filt_with_rect(self, rect):
		selected_vertex = []
		for i in range((self.latNum + 1) * (self.lngNum + 1)):
			box = self.v_int2string(i)
			pos = self.v_string2latlng(box, mode = 0)
			if rect['leftLng'] < pos[0] < rect['rightLng'] and rect['bottomLat'] < pos[1] < rect['topLat'] and self.edges[i].sum() != 0:
				selected_vertex.append(i)
		res = self.edges.toarray()
		res = res[:, selected_vertex]
		res = res[selected_vertex, :]
		res = sp.csr_matrix(res, shape = res.shape, dtype = np.float32, copy = False)
		return res, selected_vertex


if __name__ == '__main__':
	data = pd.read_csv("F:/DATA/dataset/v1/road_set.csv")
	m = Graph(1000, data, need_edge = True)
	a, selected_vertex = m.filt_with_rect(rect = {"leftLng": 121.414316, "rightLng": 121.581042, "topLat": 31.295972, "bottomLat": 31.182597})
	count = 0
	for i in range(a.shape[0]):
		if a[i, i] != 0:
			count += 1
	print(count)
	with open("F:/DATA/dataset/v1/selected_vertex.dat", 'wb') as f:
		pickle.dump(selected_vertex, f)
