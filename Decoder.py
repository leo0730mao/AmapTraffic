import os
import time
import json
import shutil
import pandas as pd
import geohash
import pickle
import numpy as np
from MyGraph import Graph


def TimeDecoder(file, mode = "old"):
	if mode == "old":
		t = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(file)))
		d = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(file)))
	else:
		timestamp = int(os.path.split(file)[1].split(".")[0])
		t = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
		d = time.strftime("%Y-%m-%d", time.localtime(timestamp))
	return t, d


def decoder(file, type = "point"):
	print(file)
	flag = 0
	data = None
	with open(file, "r", encoding = "gbk") as F_in:
		try:
			data = json.load(F_in)
		except Exception as e:
			flag = 1
	if flag == 1:
		with open(file, "r", encoding = "utf-8") as F_in:
			try:
				data = json.load(F_in)
			except Exception as e:
				print(e)
				return
	t, _ = TimeDecoder(file, mode = "new")
	t = t[:-1] + "0"
	res = {}
	if data is not None:
		for k in data.keys():
			if data[k] is not None and data[k]['info'] == "OK":
				for road in data[k]["trafficinfo"]["roads"]:
					if "speed" in road.keys():
						if road['polyline'] in res.keys():
							res[road['polyline']]['speed'].append(float(road['speed']))
						else:
							res[road['polyline']] = {'speed': [float(road['speed'])],
													'angle': float(road['angle'])}
	del data
	return res, t


def getGeoHash(road):
	points = road.split(";")
	res = ";"
	for p in points:
		lng, lat = p.split(",")
		res += ("%s;" % geohash.encode(float(lat), float(lng), 8))
	return res


def generateJS_road():
	with open("NY_road.dat", 'rb') as f:
		data = pickle.load(f)
	a = set(data[list(data.keys())[0]])
	b = set(data[list(data.keys())[1]])
	c = list(b - a)
	F_out = open("Heatmap_b-a.js", "w")
	F_out.write('data = [')
	for road in c:
		F_out.write('[')
		points = road.split(";")
		for p in points[:-1]:
			F_out.write("[%s],\n" % p)
		F_out.write("[%s]],\n" % points[-1])
	F_out.write('];')


def generateJS(date):
	path = "F:/DATA/%s" % date
	files = os.listdir(path)
	for item in files:
		filename = path + '/' + item
		decoder(filename)
	i = 0
	F_out = open("Heatmap_%s.js" % date, "w")
	F_out.write('data = [')
	for t in sorted(DATA):
		print(t)
		DATA[t] = list(set(DATA[t]))
		# DATA[t] = random.sample(DATA[t], int(len(DATA[t]) / 10))
		if len(DATA[t]) > 0:
			F_out.write('{time: "%s", data: [\n' % t)
			for item in DATA[t][:-1]:
				F_out.write("{location: new google.maps.LatLng(%s, %s), weight: %s},\n" % (item[0], item[1], item[2]))
			F_out.write("{location: new google.maps.LatLng(%s, %s), weight: %s}]},\n" % (DATA[t][-1][0], DATA[t][-1][1], DATA[t][-1][2]))
	F_out.write('];')
	F_out.close()


def saveCSV(dst, data, date):
	print("saving %s" % date)
	res = pd.DataFrame(data)
	res.to_csv("%s/%s.csv" % (dst, date), index = False, encoding = "utf-8")


def generateCSV(src, dst):
	global DATA
	files = os.listdir(src)
	data = {'road': [], 'geohash': [], 'speed': [], 'angle': [], 'time': []}
	preDate = None
	curDate = None
	for fileName in files:
		file = src + '/' + fileName
		_, curDate = TimeDecoder(file, mode = "new")
		if preDate is None:
			preDate = curDate
		if curDate != preDate:
			saveCSV(dst, data, preDate)
			preDate = curDate
			data = {'road': [], 'geohash': [], 'speed': [], 'angle': [], 'time': []}
		DATA, t = decoder(file, type = "line")
		for road in DATA.keys():
			data['road'].append(road)
			data['geohash'].append(getGeoHash(road))
			data['speed'].append(np.array(DATA[road]['speed']).mean())
			data['angle'].append(DATA[road]['angle'])
			data['time'].append(t)
		del DATA
	if curDate is not None:
		saveCSV(dst, data, curDate)


def generateGraph(src, dst):
	files = os.listdir(src)
	for fileName in files:
		print(fileName)
		file = src + '/' + fileName
		date = fileName.split(".")[0]
		data = pd.read_csv(file, encoding = "gbk")
		ts = pd.date_range(start = date + " 00:00", end = date + " 23:50", freq = '10T')
		# os.mkdir("%s/%s" % (dst, date))
		for t in ts:
			dataSelected = data[data['time'] == t.strftime("%Y-%m-%d %H:%M")]
			if len(dataSelected) > 0:
				res = Graph(1000, dataSelected)
			else:
				res = None
			with open("%s/%s/%s.dat" % (dst, date, t.strftime("%Y-%m-%d-%H-%M")), 'wb') as f:
				pickle.dump(res, f)
				del res


def arrangeFile(src):
	dirList = os.listdir(src)
	for dir in dirList:
		path = src + "/" + dir
		if os.path.isfile(path):
			t, date = TimeDecoder(path)
			newPath = "%s/%s" % (src, date)
			if not os.path.exists(newPath):
				os.mkdir(newPath)
			shutil.copyfile(path, "%s/%s" % (newPath, dir))


if __name__ == '__main__':
	generateCSV("F:/DATA/Amap/new", "F:/DATA/Amap_csv")
