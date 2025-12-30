#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:09:00 2024

@author: rebecca
"""

import numpy as np
from scipy import stats
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies = data[0,:400]

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[:,:400]

cocktailIndex = 0
mcarthurIndex = 0
stirIndex = 0
billyIndex = 0
savingIndex = 0
angerIndex = 0
lifeIndex = 0
mementoIndex = 0
hellIndex = 0
trafficIndex = 0
for i in range(400):
    if "Cocktail (1988)"  == movies[i]:
        cocktailIndex = i
    if "MacArthur (1977)"  == movies[i]:
        mcarthurIndex = i
    if "Stir Crazy (1980)"  == movies[i]:
        stirIndex = i
    if "Billy Madison (1995)"  == movies[i]:
        billyIndex = i
    if "Saving Private Ryan (1998)"  == movies[i]:
        savingIndex = i    
    if "Anger Management (2002)"  == movies[i]:
        angerIndex = i   
    if "The Life of David Gale (2003)"  == movies[i]:
        lifeIndex = i    
    if "Memento (2000)"  == movies[i]:
        mementoIndex = i
    if "From Hell (2001)"  == movies[i]:
        hellIndex = i    
    if "Traffic (2000)"  == movies[i]:
        trafficIndex = i    

meanList = []
ciList95 = []
ciList99 = []

for i in range(400):
    temp = data[:,i]
    tempmask = np.isnan(temp)
    temp = temp[~tempmask]
    temp = (temp,)
    meanList.append(np.mean(temp))
    bootstrapRes = stats.bootstrap(temp, np.mean, n_resamples = 1e4, confidence_level=0.95)
    ciList95.append(bootstrapRes.confidence_interval)
    ciList99.append(stats.bootstrap(temp, np.mean, confidence_level=0.99, bootstrap_result=bootstrapRes).confidence_interval)

sorted_value_index = np.argsort(meanList)
moviesByMean = [str(movies[i]) for i in sorted_value_index]
#print(moviesByMean)

ciWidthList = [(ciList95[i].high - ciList95[i].low) for i in range(400)]
sorted_value_index = np.argsort(ciWidthList)
moviesByCIWidth = [str(movies[i]) for i in sorted_value_index]
#print(moviesByCIWidth)

meanMean = np.mean(meanList)
print('mean of means: ', meanMean)

print('cocktail: ', ciList99[cocktailIndex])
print('mcarthur: ', ciList95[mcarthurIndex])
print('stir: ', ciList95[stirIndex])
print('billy: ', ciList99[billyIndex])
print('saving: ', ciList95[savingIndex])
print('anger: ', ciList95[angerIndex])
print('life: ', ciList95[lifeIndex])
print('memento: ', ciList99[mementoIndex])
print('hell: ', ciList95[hellIndex])
print('traffic: ', ciList99[trafficIndex])





