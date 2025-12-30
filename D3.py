#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:39:00 2024

@author: rebecca
"""

import numpy as np
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies = data[0,:400]

starWars1 = 0
for i in range(400):
    if "Star Wars: Episode 1" in movies[i]:
        starWars1 = i
starWars2 = 0
for i in range(400):
    if "Star Wars: Episode II" in movies[i]:
        starWars2 = i

#print(movies[starWars1])
#print(movies[starWars2])

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[:,:400]

starWars1Rating = []
starWars2Rating = []
for i in range(len(data)):
    if np.isnan(data[i, starWars1])==False and np.isnan(data[i, starWars2])==False:
        starWars1Rating.append(data[i, starWars1])
        starWars2Rating.append(data[i, starWars2])
        
print(len(starWars1Rating))

x = np.array(starWars2Rating).reshape(len(starWars2Rating),1)
y = np.array(starWars1Rating)

from sklearn.linear_model import LinearRegression
starWarsSLR = LinearRegression().fit(x, y)

print(round(starWarsSLR.coef_[0],3))
#print(round(starWarsSLR.intercept_,3))

titantic = 0
for i in range(400):
    if "Titanic" in movies[i]:
        titanic = i

#print(movies[titanic])

starWars1Rating_t = []
titanicRating = []
for i in range(len(data)):
    if np.isnan(data[i, starWars1])==False and np.isnan(data[i, titanic])==False:
        starWars1Rating_t.append(data[i, starWars1])
        titanicRating.append(data[i, titanic])
    
print(len(titanicRating))

x = np.array(starWars1Rating_t).reshape(len(starWars1Rating_t),1)
y = np.array(titanicRating)

titanicSLR = LinearRegression().fit(x, y)

print(round(titanicSLR.coef_[0],3))

count = 0
for i in range(len(data)):
    if np.isnan(data[i, starWars1])==False and np.isnan(data[i, titanic])==False and np.isnan(data[i, starWars2])==False:
        count+= 1
print(count)
    
