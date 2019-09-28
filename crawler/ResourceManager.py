import os
import configparser


class ResourceManager:
	def __init__(self, path, city):
		self.keys = []
		self.city_attr = {}
		self.idx = 0

		with open(os.path.join(path, "keys.txt"), 'r') as f:
			self.keys = f.read().strip().split("\n")

		cfg = configparser.ConfigParser()
		cfg.read(os.path.join(path, "city.txt"))

		for option in cfg.options(city):
			self.city_attr[option] = cfg.getfloat(city, option)


		self.lngDelta = (float(self.city_attr['right_lng']) - float(self.city_attr['left_lng']))
		self.lngDelta = self.lngDelta / int(self.city_attr['lng_num'])

		self.latDelta = (float(self.city_attr['bottom_lat']) - float(self.city_attr['top_lat']))
		self.latDelta = self.latDelta / int(self.city_attr['lat_num'])

	def get_city_box(self, boxID):
		tmp_left_lng = self.city_attr['left_lng'] + boxID[0] * self.lngDelta
		tmp_right_lng = tmp_left_lng + self.lngDelta

		tmp_top_lat = self.city_attr['top_lat'] + boxID[1] * self.latDelta
		tmp_bot_lat = tmp_top_lat + self.latDelta

		return [tmp_left_lng, tmp_top_lat, tmp_right_lng, tmp_bot_lat]

	def get_key(self):
		key = self.keys[self.idx]
		self.idx += 1
		self.idx = self.idx % len(self.keys)
		return key

	def get_lat_lng_num(self):
		return int(self.city_attr['lng_num']), int(self.city_attr['lat_num'])