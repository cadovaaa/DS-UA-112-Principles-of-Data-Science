#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:15:05 2024

@author: rebecca
"""

import numpy as np
from scipy import stats
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
movies = data[0,:400]

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[:,:400]

lostArkIndex = 0
for i in range(400):
    if "Indiana Jones and the Raiders of the Lost Ark (1981)"  == movies[i]:
        lostArkIndex = i
lostArk = data[:,lostArkIndex]
lastCrusadeIndex = 0
for i in range(400):
    if "Indiana Jones and the Last Crusade (1989)" == movies[i]:
        lastCrusadeIndex = i
lastCrusade = data[:,lastCrusadeIndex]
crystalSkullIndex = 0
for i in range(400):
    if "Indiana Jones and the Kingdom of the Crystal Skull (2008)" == movies[i]:
        crystalSkullIndex = i
crystalSkull = data[:,crystalSkullIndex]
ghostIndex = 0
for i in range(400):
    if "Ghostbusters (2016)" == movies[i]:
        ghostIndex = i
ghost = data[:,ghostIndex]
wallIndex = 0
for i in range(400):
    if "Wolf of Wall Street" in movies[i]:
        wallIndex = i
wall = data[:,wallIndex]
interIndex = 0
for i in range(400):
    if "Interstellar (2014)" == movies[i]:
        interIndex = i
inter = data[:,interIndex]
nemoIndex = 0
for i in range(400):
    if "Finding Nemo (2003)" == movies[i]:
        nemoIndex = i
nemo = data[:,nemoIndex]

temp = np.array([np.isnan(lostArk),np.isnan(lastCrusade)],dtype=bool) 
temp2 = temp*1
temp3 = sum(temp2)
missingData = np.where(temp3>0) 
test1 = np.delete(lostArk,missingData) 
test2 = np.delete(lastCrusade,missingData) 
u,p = stats.mannwhitneyu(test1,test2)
print("lost ark/last crusade mann-whitney u: ",u)
print("lost ark/last crusade mann-whitney p: ",p)

temp = np.array([np.isnan(lastCrusade),np.isnan(crystalSkull)],dtype=bool) 
temp2 = temp*1
temp3 = sum(temp2)
missingData = np.where(temp3>0) 
test1 = np.delete(lastCrusade,missingData) 
test2 = np.delete(crystalSkull,missingData) 
u,p = stats.mannwhitneyu(test1,test2)
print("last crusade/crystal skull mann-whitney u: ",u)
print("last crusade/crystal skull mann-whitney p: ",round(p,5))

temp = np.array([np.isnan(crystalSkull),np.isnan(ghost)],dtype=bool) 
temp2 = temp*1
temp3 = sum(temp2)
missingData = np.where(temp3>0) 
test1 = np.delete(crystalSkull,missingData) 
test2 = np.delete(ghost,missingData) 
u,p = stats.mannwhitneyu(test1,test2)
print("crystal skull/ghost mann-whitney u: ",u)
print("crystal skull/ghost mann-whitney p: ",p)

temp = np.array([np.isnan(inter),np.isnan(wall)],dtype=bool) 
temp2 = temp*1
temp3 = sum(temp2)
missingData = np.where(temp3>0) 
test1 = np.delete(inter,missingData) 
test2 = np.delete(wall,missingData) 
u,p = stats.mannwhitneyu(test1,test2)
print("inter/wall mann-whitney u: ",u)
print("inter/wall skull/ghost mann-whitney p: ",p)

k,p = stats.ks_2samp(ghost,nemo,nan_policy = 'omit')
print("ghost/nemo mann-whitney p: ",round(p,5))

k,p = stats.ks_2samp(inter,nemo,nan_policy = 'omit')
print("inter/nemo mann-whitney p: ",round(p,4))

k,p = stats.ks_2samp(inter,wall,nan_policy = 'omit')
print("inter/wall mann-whitney p: ",round(p,4))
