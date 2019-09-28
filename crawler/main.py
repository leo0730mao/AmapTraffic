import multiprocessing
from crawler import Crawler
import time
import logging
import Logging
import os


def request_city(city):
	time_slot = 600
	log_path = "./log/%s/" % city
	if not os.path.isdir(log_path):
		os.mkdir(log_path)

	Logging.init_log(log_path = log_path)
	crawler = Crawler(city = city, dst_dir = "./data", log_path = log_path, time_slot = time_slot)

	while 1:
		t, miss = crawler.run()

		logging.info("%s complete, miss %s region, cost %ss, sleep %ss." % (city, miss, t, time_slot - t))
		if t < time_slot:
			time.sleep(time_slot - t)

def main():
	pool = multiprocessing.Pool(processes = 10)

	for city in ['BJ', 'SZ', 'SY', 'SH', 'XM', 'XA', 'NJ', 'JX', 'TY']:
		logging.info("create process to get %s data" % city)
		pool.apply_async(request_city, (city,))
	print("begin to get data")
	pool.close()
	pool.join()

if __name__ == '__main__':
	main()