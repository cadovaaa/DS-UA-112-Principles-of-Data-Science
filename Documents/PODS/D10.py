#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:55:41 2024

@author: rebecca
"""
#-0.19005361  0.52097674  0.51950907  0.03057696 -0.17889566  0.48316574
# -0.22126911 -0.30935527 -0.09175057  0.05573668
import numpy as np
from scipy import stats
data1 = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies1 = data1[0,:400]

data1 = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data1 = data1[:,:400]

kb1Index = 0
for i in range(400):
    if "Saw (2004)" in movies1[i]:
        kb1Index = i
print(movies1[kb1Index])

from sklearn.decomposition import PCA
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
sensation = data[1:,400:420]
person = data[1:20:464]
movie = data[1:,464:474]
print(movie)

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
sensation_data = data[0:,400:420]
person_data = data[0:,420:464]
movie_data = data[0:,464:474]

zscoredData = stats.zscore(sensation_data,nan_policy='omit')
mask = ~np.isnan(zscoredData).any(axis=1)
arr_no_nan = zscoredData[mask]

pca = PCA().fit(arr_no_nan)
eigVals = pca.explained_variance_
#print(eigVals)

zscoredData = stats.zscore(person_data,nan_policy='omit')
mask = ~np.isnan(zscoredData).any(axis=1)
arr_no_nan = zscoredData[mask]

pca = PCA().fit(arr_no_nan)
eigVals = pca.explained_variance_
#print(eigVals)

zscoredData = stats.zscore(movie_data,nan_policy='omit')
mask = ~np.isnan(zscoredData).any(axis=1)
arr_no_nan = zscoredData[mask]

pca = PCA().fit(arr_no_nan)
eigVals = pca.explained_variance_
#print(eigVals)
loadings = pca.components_
#print(loadings)

saws = data1[:,kb1Index]

zscoredData = stats.zscore(sensation_data[np.isnan(saws) == False],nan_policy='omit')
mask = ~np.isnan(zscoredData).any(axis=1)
arr_no_nan = zscoredData[mask]

pca = PCA().fit(arr_no_nan)
rotatedData = pca.fit_transform(arr_no_nan)

print(len(mask))
nanlesssaw = saws[np.isnan(saws) == False]
print(len(nanlesssaw))
nanlesssaw = nanlesssaw[mask]

saws = data1[:,kb1Index]
sawmed = np.median(nanlesssaw)
print(sawmed)
nanlesssaw[nanlesssaw==sawmed] = np.nan
nanlesssaw[nanlesssaw<sawmed] = 0
nanlesssaw[nanlesssaw>sawmed]= 1

print(nanlesssaw)

from sklearn.linear_model import LogisticRegression
mask = ~np.isnan(nanlesssaw).any(axis=0)
arr_no_nan = zscoredData[mask]
x = rotatedData[:,0].reshape(len(rotatedData[:,0]),1) 
y = nanlesssaw
x = x[mask]
y=y[mask]
model = LogisticRegression().fit(x,y)