#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:53:43 2024

@author: rebecca
"""

import numpy as np
data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',',dtype=str)
headers = data[0,400:]

print(headers[-5:-3])

data = np.genfromtxt('/Users/rebecca/Downloads/movieDataReplicationSet.csv',delimiter=',')
data = data[1:,400:]

from sklearn.linear_model import LinearRegression

#%%
incomeEduCorr = np.corrcoef(data[:,-4],data[:,-2])
print(incomeEduCorr)

#%% 
x = data[:,-5].reshape(len(data),1)
y = data[:,-2]
incomeModel = LinearRegression().fit(x, y)

incomeSlope = incomeModel.coef_ 
incomeIntercept = incomeModel.intercept_ 
yHatIncome = incomeSlope * x + incomeIntercept
incomeResiduals = y - yHatIncome.flatten()

y = data[:,-4]
eduModel = LinearRegression().fit(x, y)

eduSlope = eduModel.coef_ 
eduIntercept = eduModel.intercept_ 
yHatEdu = eduSlope * x + eduIntercept
eduResiduals = y - yHatEdu.flatten()

partCorr = np.corrcoef(incomeResiduals,eduResiduals) 
print(partCorr)

#%%
X = data[:,-5:-3]
y = data[:,-2]
twoFactorModel = LinearRegression().fit(X,y)
rSqrTwo = twoFactorModel.score(X,y)
print(twoFactorModel.coef_)
print(rSqrTwo)
print(rSqrTwo**0.5)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y,twoFactorModel.predict(X))
print(rmse)