#!/usr/bin/env python3

import sys
import csv

cum_sum, count = 0, 0
price_lst = []
line_gen = csv.reader(sys.stdin, delimiter=",")
for line in line_gen:
	number = line[9]
	if not number.isdigit():
		continue
	cum_sum += float(number)
	count += 1
	price_lst.append(float(number))

mean = cum_sum / count
var_lst = [(cur_price - mean) ** 2 for cur_price in price_lst]
var = sum(var_lst) / count

print(count, mean, var, sep=",")
