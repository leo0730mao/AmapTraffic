# Abstrct

This is a traffic speed prediction system. I get the data from Amap, do some statistic and built a model based on Pytorch. The model will predict the traffic speed of each region in the future 4 hours according to previous 4 hours data.



# Data

The data's format see here: https://lbs.amap.com/api/webservice/guide/api/trafficstatus

Data is stored in CSV format.



# File Structure

- ./analysis: some statistic function
- ./traffic_predict: the traffic speed prediction system. 
- ./crawler: crawler for getting data from Amap

