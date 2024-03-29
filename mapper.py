#!/usr/bin/env python3
import sys

sys.stdin.readline()

for line in sys.stdin:
    line = line.strip()
    ID,text=line.split(',', 1)
    print('%s,%s' % (ID, text))
