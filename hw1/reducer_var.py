#!/usr/bin/env python3

import sys

count, mean, var = 0, 0, 0
for line in sys.stdin:
    cur_count, cur_mean, cur_var = [float(elem) for elem in line.split(',')]
    var1 = (var * cur_count + cur_var * cur_count) / (cur_count + count)
    var2 = count * cur_count * ((mean - cur_mean) / (cur_count + count)) ** 2
    var = var1 + var2
    avg_price = (mean * count + cur_mean * cur_count) / (cur_count + count)
    count += cur_cnt

print(var)