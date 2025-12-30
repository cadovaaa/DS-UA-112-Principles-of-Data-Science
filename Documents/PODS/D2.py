#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:22:11 2024

@author: rebecca
"""

import numpy as np
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies = data[0,:400]

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[:,:400]

columnSDNoNan = np.nanstd(data,axis=0)
#print(len(columnSDNoNan))

columnMeanNoNan = np.nanmean(data,axis=0)
columnMADNoNan = []

for i in range(len(data[1])):
    column = [x for x in data[:,i] if np.isnan(x)==False]
    column = [np.abs(x-columnMeanNoNan[i]) for x in column]
    columnMADNoNan.append(np.mean(column))
#print(len(columnMADNoNan))
        
SDMean = np.mean(columnSDNoNan)
SDMedian = np.median(columnSDNoNan)
print(SDMean)
print(SDMedian)

MADMean = np.mean(columnMADNoNan)
MADMedian = np.median(columnMADNoNan)
print(MADMean)
print(MADMedian)

import pandas as pd
df = pd.DataFrame(data, index=data[:,0])
corrMatrix = df.corr()

corrCopy = corrMatrix.copy()
corrCopy.values[np.tril_indices_from(corrCopy)] = np.nan
print(corrCopy.unstack().mean())
print(corrCopy.unstack().std())



