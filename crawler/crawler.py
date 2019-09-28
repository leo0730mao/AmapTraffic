import json
from urllib.parse import urlencode
from urllib.request import urlopen
import time
import os
from ResourceManager import ResourceManager
import logging
import Logging


class Crawler:
	def __init__(self, city, dst_dir, log_path, time_slot):
		self.res = ResourceManager("./resource", city)
		self.city = city
		self.try_times = 3
		self.base_url = "http://restapi.amap.com/v3/traffic/status/rectangle"
		self.params = {
			'key':        None,
			'rectangle':  None,
			'extensions': 'all',
			'level':      6
		}
		self.dst_dir = dst_dir
		self.time_slot = time_slot
		self.data = None
		Logging.init_log(log_path)

	def get_request_url(self, boxID):
		box = self.res.get_city_box(boxID)
		box = "%s,%s;%s,%s" % tuple(box)
		self.params['rectangle'] = box
		self.params['key'] = self.res.get_key()
		str_params = urlencode(self.params)
		url = '%s?%s' % (self.base_url, str_params)
		return url

	@classmethod
	def __request__(cls, url):
		try:
			f = urlopen(url, timeout = 30)
		except:
			return None
		try:
			data = f.read()
			data = data.decode("utf-8")
			data = json.loads(data)
		except:
			return None
		return data

	def request(self, boxID):
		url = self.get_request_url(boxID)
		data = None
		res = 0
		for i in range(self.try_times):
			data = self.__request__(url)
			if data is not None and data['info'] == "OK":
				break
			else:
				logging.warning("fail to get %s data in region %s_%s, try again..." % (self.city, boxID[0], boxID[1]))
		if data is None or data['info'] != "OK":
			res = 1
			logging.error("fail to get %s data in region %s_%s after trying %s times..." % (self.city, boxID[0], boxID[1], self.try_times))
		return data, res

	def save(self, t):
		data_path = os.path.join(self.dst_dir, self.city)
		if not os.path.isdir(data_path):
			os.mkdir(data_path)
		data_path = os.path.join(data_path, "%s.json" % t)
		with open(data_path, 'w', encoding = 'utf-8') as f:
			try:
				json.dump(self.data, f, ensure_ascii = False, indent = 4)
				logging.info("save %s data success" % self.city)
			except:
				logging.warning("save %s data fail" % self.city)

	def run(self):
		lngNum, latNum = self.res.get_lat_lng_num()
		self.data = {}
		miss = 0
		start_time = time.time()
		for i in range(lngNum):
			for j in range(latNum):
				boxID = [i, j]
				self.data["%s_%s" % tuple(boxID)], tmp = self.request(boxID)
				miss += tmp
		self.save(start_time)

		elapse_time = time.time() - start_time
		return elapse_time, miss