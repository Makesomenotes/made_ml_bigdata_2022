#!/usr/bin/env python3

import sys
import csv

cum_sum, count = 0, 0
line_gen = csv.reader(sys.stdin, delimiter=",")
for line in line_gen:
	cum_sum += float(line[9]) 
	count += 1  # no nans in csv

print(count, cum_sum / count, sep=",")
