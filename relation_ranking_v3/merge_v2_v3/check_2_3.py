#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-01-22

res = [[0,0],[0,0]]
with open('../relation_ranking_v2/results/valid-rel_results.txt') as r2, open('./results/valid-rel_results.txt') as r3:
    for i, j in zip(r2, r3):
        i = i.strip()
        j = j.strip()
        if i == '1' and j == '1':
            res[0][0] += 1
        elif i == '1':
            res[1][0] += 1
        elif j == '1':
            res[0][1] += 1
        else:
            res[1][1] += 1

print(res)
