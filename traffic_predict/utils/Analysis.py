import pandas as pd
import geohash
import os


def getTimeSeries(files, center = None):
	res = {'speed': [], 'ts': []}
	for file in files:
		data = pd.read_csv(file, encoding = "gbk")
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