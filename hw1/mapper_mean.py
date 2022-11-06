#!/usr/bin/env python3

import sys
import csv

cum_sum, count = 0, 0
line_gen = csv.reader(sys.stdin, delimiter=",")
for line in line_gen:
	number = line[9]
	if not number.isdigit():
		continue
	cum_sum += float(number)
	count += 1 

print(count, cum_sum / count, sep=",")
