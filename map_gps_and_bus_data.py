import glob
import time
import os

from math import radians, cos, sin, asin, sqrt
from operator import itemgetter

#核心想法是，dataframe和list都很慢，dict是最快的
#对bus和gps分别构建dictionary,plate和line为keys，其他为value（其实只用plate就行了）

bus_data_dict=dict()
with open("20151101bus.txt", 'r') as f:
    for lines in f:
        [card_id, time, line_no, plate_no] = lines.strip().split(',')    
        [H,M,S]=time.split(':')
        time_numeric=int(H)*3600+int(M)*60+int(S)#时间处理成秒，方便后面比较
        if bus_data_dict.has_key((line_no,plate_no)):  
            bus_data_dict[line_no,plate_no].append([card_id, time, time_numeric])
        else:
            bus_data_dict[line_no,plate_no] = [[card_id, time, time_numeric]]

print len(bus_data_dict)


for key in bus_data_dict.keys():
    this_bus_line_no=key[0]
    this_bus_plate_no=key[1]            
    
    with open("bus_data\\"+this_bus_line_no+"_"+this_bus_plate_no, 'w') as wf:#按照线路和车牌分别写入文件，其实没必要，直接循环dict就好了
        for entry_no in range(0,len(bus_data_dict[key])):
             wf.write(str(bus_data_dict[key][entry_no][0]) + "," + str(bus_data_dict[key][entry_no][1]) + "," + str(bus_data_dict[key][entry_no][2])+ "\n")
                                


gps_data_dict=dict()
with open("20151101gps.txt", 'r') as f:
    for lines in f:
        [plate_no, line_no, lon, lat, time] = lines.strip().split(',')    
        [H,M,S]=time.split(':')
        time_numeric=int(H)*3600+int(M)*60+int(S)
        if gps_data_dict.has_key((line_no,plate_no)):  # not the first record
            gps_data_dict[line_no,plate_no].append([lon, lat, time, time_numeric])
        else:
            gps_data_dict[line_no,plate_no] = [[lon, lat, time, time_numeric]]
print len(gps_data_dict)    

for key in gps_data_dict.keys():
    this_gps_line_no=key[0]
    this_gps_plate_no=key[1]            
    
    with open("gps_data\\"+this_gps_line_no+"_"+this_gps_plate_no, 'w') as wf:
        for entry_no in range(0,len(gps_data_dict[key])):
             wf.write(str(gps_data_dict[key][entry_no][0]) + "," + str(gps_data_dict[key][entry_no][1]) + "," + str(gps_data_dict[key][entry_no][2])+ "," + str(gps_data_dict[key][entry_no][3]) + "\n")


#后面的思路应该是，对于字典每个车牌循环，然后每个车牌相同的bus和gps记录待匹配
#bus和gps按时间排序，bus的第一个找到时间差最接近的gps后返回行号，然后bus第二个沿着行号往下找

#或者说，把字典数据中相同id的bus插入gps记录里面排序  bus可以是【时间，0,0】gps是【时间，经度，纬度】 排序很快
#numpy里面有where 只要找出第二列是0的bus记录，比较上下两条哪一条时间最接近，把经纬度赋给bus写出就行了
#时间转化为秒来比较可能更快
