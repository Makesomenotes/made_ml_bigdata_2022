#!/usr/bin/env python3

import sys
import csv

cum_sum, count = 0, 0
price_lst = []
line_gen = csv.reader(sys.stdin, delimiter=",")
for line in line_gen:
	cum_sum += float(line[9]) 
	count += 1  # no nans in csv
    price_lst.append(float(line[9]))
mean = cum_sum / count
var_lst = [(cur_price - mean) ** 2 for cur_price in in price_lst]
var = sum(var_lst) / count

print(count, mean, var, sep=",")
