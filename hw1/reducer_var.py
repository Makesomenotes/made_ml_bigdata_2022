#!/usr/bin/env python3

import sys

count, mean, var = 0, 0, 0
for line in sys.stdin:
    cur_count, cur_mean, cur_var = [float(elem) for elem in line.split(',')]
    var = (var * cur_count + cur_var * cur_count) / (cur_count + count)
    var += count * cur_count * ((mean - cur_mean) / (cur_count + count)) ** 2
    avg_price = (mean * count + cur_mean * cur_count) / (cur_count + count)
    count += cur_count

print(var)