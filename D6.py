#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:35:01 2024

@author: rebecca
"""

import numpy as np
from scipy import stats
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies = data[0,:400]

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[:,:400]

kb1Index = 0
for i in range(400):
    if "Kill Bill: Vol. 1" in movies[i]:
        kb1Index = i
kb2Index = 0
for i in range(400):
    if "Kill Bill: Vol. 2" in movies[i]:
        kb2Index = i
pfIndex = 0
for i in range(400):
    if "Pulp Fiction (1994)" in movies[i]:
        pfIndex = i

kb1 = data[:,kb1Index]
kb2 = data[:,kb2Index]
pf = data[:,pfIndex]
temp = np.array([np.isnan(kb1),np.isnan(kb2),np.isnan(pf)],dtype=bool) 
temp2 = temp*1
temp3 = sum(temp2)
missingData = np.where(temp3>0) 
kb1 = np.delete(kb1,missingData) 
kb2 = np.delete(kb2,missingData) 
pf = np.delete(pf,missingData) 

print('total reviewers: ',len(kb1))

kb1Mean = np.mean(kb1)
kb1Med = np.median(kb1)

kb2Mean = np.mean(kb2)
kb2Med = np.median(kb2)

pfMean = np.mean(pf)
pfMed = np.median(pf)

print('pf mean: ',pfMean)

kb1kb2t,kb1kb2p = stats.ttest_ind(kb1,kb2)
print("kb 1 + 2 independent samples t-test t: ",kb1kb2t)
print("kb 1 + 2 independent samples t-test p: ",kb1kb2p)

kb2pft,kb2pfp = stats.ttest_ind(kb2,pf)
print("kb 2 + pf independent samples t-test t: ",kb2pft)
print("kb 2 + pf independent samples t-test p: ",kb2pfp)

kb1pft,kb1pfp = stats.ttest_ind(kb1,pf)
print("kb 1 + pf independent samples t-test t: ",kb1pft)
print("kb 1 + pf independent samples t-test p: ",kb1pfp)


kb1kb2tp,kb1kb2pp = stats.ttest_rel(kb1,kb2)
print("kb 1 + 2 paired samples t-test t: ",kb1kb2tp)
print("kb 1 + 2 paired samples t-test p: ",kb1kb2pp)

kb2pftp,kb2pfpp = stats.ttest_rel(kb2,pf)
print("kb 2 + pf paired samples t-test t: ",kb2pftp)
print("kb 2 + pf paired samples t-test p: ",kb2pfpp)

kb1pftp,kb1pfpp = stats.ttest_rel(kb1,pf)
print("kb 1 + pf paired samples t-test t: ",kb1pftp)
print("kb 1 + pf paired samples t-test p: ",kb1pfpp)






