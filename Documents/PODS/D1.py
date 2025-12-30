#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:30:28 2024

@author: rebecca
"""

import numpy as np
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies = data[0,:400]

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[:,:400]

columnMeanNoNan = np.nanmean(data,axis=0)
#print(columnMeanNoNan)

minMean = np.min(columnMeanNoNan)
minMeanIndex = 0
maxMean = np.max(columnMeanNoNan)
maxMeanIndex = 0

#print(len(columnMeanNoNan))

for i in range(400):
    if columnMeanNoNan[i] == minMean:
        minMeanIndex = i
    elif columnMeanNoNan[i] == maxMean:
        maxMeanIndex = i
    
print(movies[minMeanIndex])
print(movies[maxMeanIndex])
print(np.mean(columnMeanNoNan))

columnMedNoNan = np.nanmedian(data,axis=0)

minMedList = []
maxMed = np.max(columnMedNoNan)
maxMedIndex = 0

for i in range(400):
    if movies[i] in ["Halloween (1978)", 'Black Swan (2010)', 'Downfall (2004)', 'Battlefield Earth (2000)', 'Harry Potter and the Chamber of Secrets (2002)']:
        minMedList.append(movies[i])
        minMedList.append(columnMedNoNan[i])
    if columnMedNoNan[i] == maxMed:
        maxMedIndex = i
   
minMedChoice = 1
for i in range(1,len(minMedList),2):
    if minMedList[i] < minMedList[minMedChoice]:
        minMedChoice = i
    
print(minMedList[minMedChoice-1])
print(movies[maxMedIndex])

from scipy import stats

independenceDayIndex = 0
for i in range(400):
    if movies[i] == ["Independence Day (1996)"]:
        independenceDayIndex = i
        
columnModes = stats.mode(data,nan_policy='omit',axis=0) 
modalValues = columnModes.mode 

print(modalValues[independenceDayIndex])
