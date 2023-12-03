# -*- coding: utf-8 -*-
from collections import defaultdict
import random
import sys
import os
import csv
import numpy as np
import pandas as pd
sys.setrecursionlimit(20000000)

def split(full_list,shuffle=False,ratio = 0.5):
	n_total = len(full_list)
	offset = int(n_total * ratio)
	if n_total==0 or offset<1:
		return [],full_list
	if shuffle:
		random.shuffle(full_list)
	sublist_1 = full_list[:offset]
	sublist_2 = full_list[offset:]
	return sublist_1,sublist_2
if __name__ == "__main__":
	relation_sum_3 = []
	aaa = []
	graph_3 = []
	graph_4 = []
	train = []
	test = []
	dev = []
	count = 0
	sum1 = 0
	dataPath = ''
	graphPath = dataPath + 'graph.txt'
	with open(dataPath+'train.txt',"r",encoding = 'UTF-8') as raw_file:
		csv_file = csv.reader(raw_file, delimiter='\t')
		for line in csv_file:
			e1,r,e2,tim= line
			data = e1 + "\t" +r+"\t"+e2+"\t"+tim+"\n"
			relation_sum_3.append(data)
			sum1 = sum1+1
		for data in relation_sum_3:
			if data not in aaa:
				aaa.append(data)
				count= count+1
	print(sum1-count)
'''
		with open(root_dir+f,"r", encoding='UTF-8') as raw_file:
			csv_file = csv.reader(raw_file, delimiter='\t')
			for line in csv_file:
				e1,r,e2, tim = line
				graph_i = e1.rstrip() + "\t" +r.rstrip()+"\t"+e2.rstrip()+"\t"+tim.rstrip()+"\n"
				graph_4.append(graph_i)
				data_g = e2.rstrip() + "\t" +r.rstrip()+"_inv"+"\t"+e1.rstrip()+"\t"+tim.rstrip()+"\n"
				graph_4.append(data_g)
				

'''
for data in aaa:
	with open(graphPath,"a") as f:
		f.write(data)