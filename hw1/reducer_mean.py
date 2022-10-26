#!/usr/bin/env python3

import sys

total_cnt, total_mean = 0, 0
for line in sys.stdin:
    batch_count, batch_mean = map(float, line.split(","))
    total_mean = ((total_count * total_mean) + (batch_count * batch_mean)) / (total_count + batch_count)
    total_count += batch_count

print(total_mean)