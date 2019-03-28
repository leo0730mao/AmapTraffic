import pandas as pd


def filter(src, dst):
	d = pd.read_csv(src, encoding = "utf-8")
	d = d.dropna()
	temp = {}
	for index, row in d.iterrows():
		name = row['name']
		num = row['num']
		if len(num) > 4 and (num[0] == 'c' or num[0] == 'C' or num[-4] == '6'):
			if name in temp.keys():
				temp[name] += 1
			else:
				temp[name] = 1
	res = {'name': [], 'num': []}
	for k in temp.keys():
		res['name'].append(k)
		res['num'].append(temp[k])
	res = pd.DataFrame(res)
	res.to_csv(dst, encoding = "utf-8", index = False)


if __name__ == '__main__':
	filter("H:/data.csv", "H:/result.csv")
