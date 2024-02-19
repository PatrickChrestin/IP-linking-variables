#############################################################
#
#   permuted coefficient matrix version of passonvars_exk2.py
#   TODO: keep original indices for output
#
#############################################################

from gurobipy import *
import staircasedetection

# input coefficient matrix
a = [[0, 0, 1, 1, 1, 0, 0], 
     [0, 0, 1, 1, 0, 1, 0], 
     [1, 0, 0, 0, 0, 1, 1], 
     [0, 0, 1, 0, 1, 0, 0], 
     [1, 1, 0, 0, 0, 1, 1]]

r_sum = []
for i in range(len(a)):
    r_sum.append(sum(a[i]))
N = range(1,len(a[0])+1)
M = range(1,len(a)+1)

# set k
k = 2

model = staircasedetection.solve(N, M, k, a, r_sum)
# model.display()
model.write('staircasedetection.lp')
if model.Status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write('iismodel.ilp')
