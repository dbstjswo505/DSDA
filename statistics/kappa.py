import numpy as np
import pdb



a = np.load('test_result_a.npy')
b = np.load('test_result_b.npy')
num = len(a)
tmp = np.zeros((2,2))
#
#               b
#           true    false
#a true
#  false
#
for i in range(num):
    if b[i] == a[i]:
        if a[i] == True:
            tmp[0,0] += 1
        else:
            tmp[1,1] += 1
    else:
        if a[i] == True:
            tmp[0,1] += 1
        else:
            tmp[1,0] += 1

tmp[0,0] = 1082
tmp[0,1] = 246
tmp[1,0] = 40
tmp[1,1] = 122
total = tmp.sum()
Pa = (tmp[0,0] + tmp[1,1])/total

pat = (tmp[0,0] + tmp[0,1])/total
paf = (tmp[1,0] + tmp[1,1])/total
pbt = (tmp[0,0] + tmp[1,0])/total
pbf = (tmp[0,1] + tmp[1,1])/total
pt = pat*pbt
pf = paf*pbf
Pc = pt + pf
K = (Pa - Pc)/(1 - Pc)
pdb.set_trace()
z=1
