import os
import pandas as pd
import geohash
import matplotlib.pyplot as plt
from geopy.distance import vincenty
from analysis.Visualization import draw_speed_time_series, draw_one_week_speed
import seaborn as sns
import pickle
import numpy as np
import dtw as dt
from analysis.MyGraph import Graph
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import vincenty


def road_length(data):
	segment_length = []
	roads_length = []
	for index, row in data.iterrows():
		road = row['road']
		points = [p.split(",") for p in road.split(";")]
		temp_len = 0
		for i in range(len(points) - 1):
			temp = vincenty([points[i][1], points[i][0]], [points[i + 1][1], points[i + 1][0]]).m
			segment_length.append(temp)
			temp_len += temp
		roads_length.append(temp_len)
	return segment_length, roads_length


def road_to_js(data):
	res_str = "roads = ["
	for index, row in data.iterrows():
		road = row['road']
		points = [p.split(",") for p in road.split(";")]
		road_str = "["
		for p in points:
			road_str += "[%s, %s],\n" % tuple(p)
		road_str += "],\n"
		res_str += road_str
	res_str += "];"

	with open("./road_set.js", 'w') as f:
		f.write(res_str)


def get_one_date_speed(files, center = None):
	res = {}
	for file in files:
		data = pd.read_csv(file)
		date = str(os.path.split(file)[1].split(".")[0])
		if center is not None:
			center_hash = geohash.encode(center[1], center[0], 8)
			selected_data = data[data['geohash'].str.contains(";%s.*" % center_hash[:5])]
		else:
			selected_data = data
		ts = pd.date_range(start = date + " 00:00", end = date + " 23:50", freq = '10T')
		ts = [t.strftime("%Y-%m-%d %H:%M") for t in list(ts)]
		res[date] = []
		for i in range(len(ts)):
			temp = selected_data[selected_data['time'] == ts[i]]
			if len(temp) > 0:
				res[date].append(temp['speed'].mean())
			elif len(res[date]) > 0:
				res[date].append(res[date][-1])
			else:
				res[date].append(0)
		res[date] = np.array(res[date])
	return res


def length_statistic():
	d = pd.read_csv("F:/DATA/dataset/v1/road_set.csv")
	segment_length, roads_length = road_length(d)
	print("------------segment---------------")
	print(sum(segment_length) / len(segment_length))
	print(max(segment_length))
	print(min(segment_length))

	print("------------road------------------")
	print(sum(roads_length) / len(roads_length))
	print(max(roads_length))
	print(min(roads_length))
	sns.set(color_codes = True)
	sns.distplot(segment_length, hist = True)
	plt.xlabel("segment length(m)")
	plt.ylabel("probability")
	plt.title("PDF of road segments' length")
	plt.savefig("./segment_length_CDF.jpg")


def get_speed_series_for_compare():
	zjgk = [121.594061, 31.207879]
	njer = [121.490857, 31.243738]
	xjh = [121.442314, 31.201202]
	data = {}

	weekday = ["%s/%s" % ("F:/DATA/Amap_csv/weekday", f) for f in os.listdir("F:/DATA/Amap_csv/weekday")]
	weekend = ["%s/%s" % ("F:/DATA/Amap_csv/weekend", f) for f in os.listdir("F:/DATA/Amap_csv/weekend")]
	holiday = ["%s/%s" % ("F:/DATA/Amap_csv/holiday", f) for f in os.listdir("F:/DATA/Amap_csv/holiday")]

	data['weekday'] = get_one_date_speed(weekday)
	data['weekend'] = get_one_date_speed(weekend)
	data['holiday'] = get_one_date_speed(holiday)

	for k in data.keys():
		temp = []
		for date in data[k].keys():
			temp.append(data[k][date])
		data[k] = list(np.mean(temp, axis = 0))
	draw_speed_time_series(data)
	return data


def get_speed_in_same_time_slot(files, t, center = None):
	res = []
	for file in files:
		data = pd.read_csv(file)
		date = str(os.path.split(file)[1].split(".")[0])
		target = "%s %s" % (date, t)
		data = data[data['time'] == target]
		if center is not None:
			center_hash = geohash.encode(center[1], center[0], 8)
			selected_data = data[data['geohash'].str.contains(";%s.*" % center_hash[:5])]
		else:
			selected_data = data
		if len(selected_data) > 0:
			res.append(selected_data['speed'].mean())
		elif len(res) > 0:
			res.append(res[-1])
		else:
			res.append(0)
	return res


def draw_speed_pdf():
	data = get_speed_series_for_compare()
	sns.set(color_codes = True)
	sns.distplot(data['weekday'], hist = False, label = 'weekday')
	sns.distplot(data['weekend'], hist = False, label = 'weekend')
	sns.distplot(data['new year'], hist = False, label = 'new year')
	plt.title("speed PDF in different kinds of days")
	plt.xlabel("speed(km/h)")
	plt.ylabel("probability")
	plt.legend()
	plt.savefig("./speed_PDF.jpg")


def spatial_top_k(files_path):
	mat = None
	with open("F:/DATA/dataset/v2/selected_vertex.dat", 'rb') as f:
		selected_vertex = pickle.load(f)
	selected_vertex = [int(i) for i in selected_vertex]

	if os.path.exists("./temp_dtw.dat"):
		with open("./temp_dtw.dat", 'rb') as f:
			mat = pickle.load(f)
	else:
		files = ["%s/%s" % (files_path, f) for f in os.listdir(files_path)]
		data = []
		for file in files:
			with open(file, 'rb') as f:
				g = pickle.load(f)
			data.append(g.v_to_vector())
		data = np.stack(data, axis = 1)

		data = data[selected_vertex, :]
		mat = np.zeros((data.shape[0], data.shape[0]))
		for i in range(data.shape[0]):
			for j in range(data.shape[0]):
				mat[i, j], _, _, _ = dt.accelerated_dtw(data[i], data[j], 'euclidean')
		with open("./temp_dtw.dat", 'wb') as f:
			pickle.dump(mat, f)
	sorted_mat = []
	for i in range(mat.shape[0]):
		sorted_mat.append(np.argsort(-mat[i]))
	g = Graph(1000, None)
	dist_mat = []
	adjacent_mat = []
	for i in range(len(sorted_mat)):
		box_target = g.v_int2string(selected_vertex[i])
		box_target = g.v_string2latlng(box_target)
		temp = []
		adjacent_num = 0
		for j in range(10):
			box = g.v_int2string(selected_vertex[sorted_mat[i][j]])
			box = g.v_string2latlng(box)
			if abs(box[0] - box_target[0]) == g.latDelta or abs(box[1] - box_target[1]) == g.lngDelta:
				adjacent_num += 1
			dist = vincenty(box_target, box).km
			temp.append(dist)
		dist_mat.append(temp)
		adjacent_mat.append(adjacent_num)

	adjacent_mat = np.array(adjacent_mat)
	dist_mat = np.array(dist_mat)
	dist_mat = dist_mat.flatten()

	print("max: %s" % dist_mat.max())
	print("min: %s" % dist_mat.min())
	print("mean: %s" % dist_mat.mean())
	print("var: %s" % dist_mat.var())

	print("max: %s" % adjacent_mat.max())
	print("min: %s" % adjacent_mat.min())
	print("mean: %s" % adjacent_mat.mean())
	print("var: %s" % adjacent_mat.var())

	sns.set(color_codes = True)
	plt.figure(0)
	sns.distplot(adjacent_mat, hist = False)
	plt.xlabel("adjacent num")
	plt.ylabel("probability")
	plt.title("number of adjacent region in top10 most similar regions")
	plt.savefig("./top10_similar_region_adjacent_pdf.jpg")

	plt.close(0)

	sns.set(color_codes = True)
	sns.distplot(dist_mat, hist = False)
	plt.xlabel("distance(km)")
	plt.ylabel("probability")
	plt.title("distance between top10 most similar regions")
	plt.savefig("./top10_similar_region_distance_pdf.jpg")


def similar_in_one_group(data):
	res = 0
	k = list(data.keys())
	count = 0
	for i in range(len(k)):
		j = 0
		while j < i:
			# simi = pearsonr(data[dt[i]], data[dt[j]])[0]
			#simi = np.linalg.norm(data[dt[i]] - data[dt[j]])
			simi, _, _, _ = dt.accelerated_dtw(data[k[i]], data[k[j]], 'euclidean')
			res += simi
			count += 1
			j += 1
	res /= count
	return res


def similar_in_two_group(data1, data2):
	res = 0
	count = 0
	for i in data1.keys():
		for j in data2.keys():
			# simi = pearsonr(data[i], data[j])[0]
			simi = np.linalg.norm(data1[i] - data2[j])
			simi, _, _, _ = dt.accelerated_dtw(data1[i], data2[j], 'euclidean')
			res += simi
			count += 1
	res /= count
	return res


def cos_similarity(x, y = None):
	array_x = []
	for k in x.keys():
		array_x.append(x[k])
	array_x = np.stack(array_x, axis = 0)

	if y is not None:
		array_y = []
		for k in y.keys():
			array_y.append(y[k])
		array_y = np.stack(array_y, axis = 0)
	else:
		array_y = None

	return cosine_similarity(array_x, array_y).mean()


def compare_speed_in_different_group():
	weekday = ["%s/%s" % ("F:/DATA/Amap_csv/weekday", f) for f in os.listdir("F:/DATA/Amap_csv/weekday")]
	weekend = ["%s/%s" % ("F:/DATA/Amap_csv/weekend", f) for f in os.listdir("F:/DATA/Amap_csv/weekend")]
	holiday = ["%s/%s" % ("F:/DATA/Amap_csv/holiday", f) for f in os.listdir("F:/DATA/Amap_csv/holiday")]

	weekday = get_one_date_speed(weekday)
	weekend = get_one_date_speed(weekend)
	holiday = get_one_date_speed(holiday)

	print("weekday: %s" % (similar_in_one_group(weekday)))
	print("weekend: %s" % (similar_in_one_group(weekend)))
	print("holiday: %s" % (similar_in_one_group(holiday)))

	print("weekday-weekend: %s" % (similar_in_two_group(weekday, weekend)))
	print("weekday-holiday: %s" % (similar_in_two_group(weekday, holiday)))
	print("holiday-weekend: %s" % (similar_in_two_group(holiday, weekend)))


def one_week_speed():
	date_list = ["2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17"]
	file_list = ["F:/DATA/Amap_csv/%s.csv" % f for f in date_list]
	data = get_one_date_speed(file_list)
	draw_one_week_speed(data)


if __name__ == '__main__':
	spatial_top_k("F:/DATA/dataset/v2/train/graph/2019-01-20")
