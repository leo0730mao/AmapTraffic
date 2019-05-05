import numpy as np
import pandas as pd
import geohash
import pickle
from geopy.distance import vincenty
from scipy.stats import pearsonr
from traffic_predict.MyGraph import Graph
import os
import geopandas as gpd


def getTimeSeries(files, center = None):
	res = {'speed': [], 'ts': []}
	for file in files:
		data = pd.read_csv(file)
		date = str(os.path.split(file)[1].split(".")[0])
		if center is not None:
			centerHash = geohash.encode(center[1], center[0], 8)
			dataSelected = data[data['geohash'].str.contains(";%s.*" % centerHash[:5])]
		else:
			dataSelected = data
		ts = pd.date_range(start = date + " 00:00", end = date + " 23:50", freq = '10T')
		ts = [t.strftime("%Y-%m-%d %H:%M") for t in list(ts)]
		for i in range(len(ts)):
			temp = dataSelected[dataSelected['time'] == ts[i]]
			if len(temp) > 0:
				res['speed'].append(temp['speed'].mean())
			elif len(res['speed']) > 0:
				res['speed'].append(res['speed'][-1])
			else:
				res['speed'].append(0)
		res['ts'] += ts
	return res


def getTimeSeriesFromGraph(dates):
	vList = []
	res = {}
	for i in range(68):
		for j in range(61):
			vList.append("%s_%s" % (i, j))
			res["%s_%s" % (i, j)] = {}
	for date in dates:
		ts = pd.date_range(start = date + " 00:00", end = date + " 23:50", freq = '10T')
		ts = [t.strftime("%Y-%m-%d %H:%M") for t in list(ts)]
		with open("F:/DATA/Amap_graph/%s.dat" % date, 'rb') as f:
			data = pickle.load(f)
		print(len(data[ts[0]].vertex.keys()))
		for v in vList:
			res[v][date] = []
			for t in ts:
				if data[t] is not None:
					if data[t].vertex[v][1] > 0:
						temp = data[t].vertex[v][0] / data[t].vertex[v][1]
					else:
						temp = 0
					res[v][date].append(temp)
				else:
					if len(res[v][date]) > 0:
						res[v][date].append(res[v][date][-1])
					else:
						res[v][date].append(0)
			res[v][date] = np.array(res[v][date])
	return res, vList


def BasicAnalysis(data, vList, period):
	res = {'vertex': [], 'avg': [], 'max': [], 'min': [], 'var': [], 'median': [], 'date': [], 'datalen': []}
	for v in vList:
		for date in data[v].keys():
			temp = data[v][date][np.where(data[v][date] != 0)]
			if len(temp) == 0:
				temp = np.zeros(1)
				res['datalen'].append(0)
			else:
				res['datalen'].append(len(temp))
			res['vertex'].append(v)
			res['avg'].append(temp.mean())
			res['max'].append(temp.max())
			res['min'].append(temp.min())
			res['var'].append(temp.var())
			res['median'].append(np.median(temp))
			res['date'].append(date)
	res = pd.DataFrame(res)
	res.to_csv("result/%s.csv" % period, encoding = "gbk", index = False)


def Similar(data, vList):
	res = {}
	for v in vList:
		dt = list(data[v].keys())
		count = 0
		res[v] = 0
		for i in range(len(dt)):
			j = 0
			while j < i:
				# pear = pearsonr(data[dt[i]], data[dt[j]])[0]
				res[v] += np.linalg.norm(data[v][dt[i]] - data[v][dt[j]])
				count += 1
				j += 1
		res[v] /= count
	return res


def AvgRoad(files):
	road_num = 0
	ts = 0
	for file in files:
		data = pd.read_csv(file)
		road_num += len(data['road'])
		ts += len(data['time'].drop_duplicates())
	print(road_num / ts)


def coveringRate():
	roads = pd.read_csv("F:/DATA/dataset/v1/road_set.csv")['road'].tolist()
	len_sum = 0
	for road in roads:
		points = [p.split(",") for p in road.split(";")]
		for i in range(len(points)):
			points[i].reverse()
		for i in range(len(points) - 1):
			len_sum += vincenty([float(j) for j in points[i]], [float(j) for j in points[i + 1]]).m
	del roads
	roads = gpd.read_file("H:/MyPythonWorkSpace/GaoDeng/data/planet_121.119,30.867_121.829,31.412.osm.geojson")
	len_sum_osm = 0
	for road_type in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']:
		temp = roads[roads['highway'] == road_type]['geometry'].tolist()
		for road in temp:
			points = road.coords[:]
			for i in range(len(points) - 1):
				len_sum_osm += vincenty(points[i], points[i + 1]).m
	print(len_sum / len_sum_osm)


if __name__ == '__main__':
	coveringRate()

