#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:56:01 2024

@author: rebecca
"""

#%% 0 - Data Cleaning

import numpy as np 
data = np.genfromtxt('/Users/rebecca/Downloads/rmpCapstoneNum.csv',delimiter=',') 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from scipy.stats import bootstrap
#from scipy.stats import permutation_test

np.random.seed(18422761)
data = data[np.where(data[:,2]>5)]

#%% 1 - Positive Male Bias in Ratings

print("\n problem 1:")
temp_data1 = data[np.where(np.isnan(data[:,0])==False)]
male_data = temp_data1[np.where(temp_data1[:,6]>0)]
nonmale_data = temp_data1[np.where(temp_data1[:,6]==0)]

q1 = np.quantile(temp_data1[:,2], 0.25)
q2 = np.quantile(temp_data1[:,2], 0.5)
q3 = np.quantile(temp_data1[:,2], 0.75)
q4 = np.quantile(temp_data1[:,2], 1)

male_q1_data = male_data[np.where(male_data[:,2]<=q1)]
nonmale_q1_data = nonmale_data[np.where(nonmale_data[:,2]<=q1)]

male_q2_data = male_data[np.where(male_data[:,2]>q1)]
male_q2_data = male_q2_data[np.where(male_q2_data[:,2]<=q2)]
nonmale_q2_data = nonmale_data[np.where(nonmale_data[:,2]>q1)]
nonmale_q2_data = nonmale_q2_data[np.where(nonmale_q2_data[:,2]<=q2)]

male_q3_data = male_data[np.where(male_data[:,2]>q2)]
male_q3_data = male_q3_data[np.where(male_q3_data[:,2]<=q3)]
nonmale_q3_data = nonmale_data[np.where(nonmale_data[:,2]>q2)]
nonmale_q3_data = nonmale_q3_data[np.where(nonmale_q3_data[:,2]<=q3)]

male_q4_data = male_data[np.where(male_data[:,2]>q3)]
male_q4_data = male_q4_data[np.where(male_q4_data[:,2]<=q4)]
nonmale_q4_data = nonmale_data[np.where(nonmale_data[:,2]>q3)]
nonmale_q4_data = nonmale_q4_data[np.where(nonmale_q4_data[:,2]<=q4)]

'''
t,p = stats.ttest_ind(male_q1_data[:,0],nonmale_q1_data[:,0])
print("q1: ",t,round(p,6))
t,p = stats.ttest_ind(male_q2_data[:,0],nonmale_q2_data[:,0])
print("q2: ",t,round(p,6))
t,p = stats.ttest_ind(male_q3_data[:,0],nonmale_q3_data[:,0])
print("q3: ",t,round(p,6))
t,p = stats.ttest_ind(male_q4_data[:,0],nonmale_q4_data[:,0])
print("q4: ",t,round(p,6))
'''

plt.figure()
plt.xlabel('Average rating')
plt.hist(male_data[:,0], bins=[i for i in range(0,6)])
plt.ylim(0,7500)
plt.xlim(1,5)
plt.xticks([1,2,3,4,5])
plt.title('Male professors')
plt.show()

plt.figure()
plt.xlabel('Average rating')
plt.hist(nonmale_data[:,0], bins=[i for i in range(0,6)], color='orange')
plt.ylim(0,7500)
plt.xlim(1,5)
plt.xticks([1,2,3,4,5])
plt.title('Non-male professors')
plt.show()

temp = (nonmale_q1_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('q1 nonmale mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('q1 male mean:', round(np.mean(male_q1_data[:,0]),5))

temp = (nonmale_q2_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('q2 nonmale mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('q2 male mean:', round(np.mean(male_q2_data[:,0]),5))

temp = (nonmale_q3_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('q3 nonmale mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('q3 male mean:', round(np.mean(male_q3_data[:,0]),5))

temp = (nonmale_q4_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('q4 nonmale mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('q4 male mean:', round(np.mean(male_q4_data[:,0]),5))

#for some reason when I run these, I get the same p-value for everything (divided by 2 
#for the one-sided tests). not sure why.
'''
def ourTestStatistic(x,y):
    return np.median(x) - np.median(y)

pTest = permutation_test((male_data[:,0],nonmale_data[:,0]),ourTestStatistic,n_resamples=1000,alternative='greater')    
print('Overall test statistic:', pTest.statistic)
print('Overall exact p-value:',pTest.pvalue)

q1pTest = permutation_test((male_q1_data[:,0],nonmale_q1_data[:,0]),ourTestStatistic,n_resamples=1000,alternative='greater')    
print('q1 Test statistic:', q1pTest.statistic)
print('q1 exact p-value:',q1pTest.pvalue)

pTest = permutation_test((male_q2_data[:,0],nonmale_q2_data[:,0]),ourTestStatistic,n_resamples=1000,alternative='greater')    
print('q2 Test statistic:', pTest.statistic)
print('q2 exact q2 p-value:',pTest.pvalue)

pTest = permutation_test((male_q3_data[:,0],nonmale_q3_data[:,0]),ourTestStatistic,n_resamples=1000,alternative='greater')    
print('q3 Test statistic:', pTest.statistic)
print('q3 exact q3 p-value:',pTest.pvalue)

pTest = permutation_test((male_q4_data[:,0],nonmale_q4_data[:,0]),ourTestStatistic,n_resamples=1000,alternative='greater')    
print('q4 Test statistic:', pTest.statistic)
print('q4 exact q4 p-value:',pTest.pvalue)
'''

#%% 2 - Effect of Experience on Quality of Teaching

print("\n problem 2:")
temp_data2 = data[np.where(np.isnan(data[:,0])==False)]
temp_data2 = temp_data2[np.where(np.isnan(temp_data2[:,2])==False)]

median_rating_number = np.median(temp_data2[:,2])
inexperienced_data = temp_data2[np.where(temp_data2[:,2]<=median_rating_number)]
experienced_data = temp_data2[np.where(temp_data2[:,2]>median_rating_number)]

plt.figure()
plt.xlabel('Number of ratings')
plt.ylabel('Average rating')
plt.plot(experienced_data[:,2],experienced_data[:,0],'o')
plt.plot(inexperienced_data[:,2],inexperienced_data[:,0],'o')
plt.yticks([1,2,3,4,5])
plt.show()

'''
pTest = permutation_test((inexperienced_data[:,0],experienced_data[:,0]),ourTestStatistic,n_resamples=1000)    
print('test statistic:', pTest.statistic)
print('p-value:',pTest.pvalue)

t,p = stats.ttest_ind(inexperienced_data[:,0],experienced_data[:,0])
print(t,round(p,8))
'''

temp = (inexperienced_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('inexperienced mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('experienced mean:', round(np.mean(experienced_data[:,0]),5))


#%% 3 - Relationship Between Average Rating and Difficulty

print("\n problem 3:")
temp_data3 = data[np.where(np.isnan(data[:,0])==False)]
temp_data3 = temp_data3[np.where(np.isnan(temp_data3[:,1])==False)]

plt.figure()
plt.xlabel('Difficulty')
plt.ylabel('Average rating')
plt.plot(temp_data3[:,1],temp_data3[:,0],'o')
plt.show()

print("correlation:",(np.corrcoef(temp_data3[:,1], temp_data3[:,0]))[0,1])

#%% 4 - Effect of Amount of Online Classes on Ratings

print("\n problem 4:")
temp_data4 = data[np.where(np.isnan(data[:,0])==False)]
temp_data4 = temp_data4[np.where(np.isnan(temp_data4[:,5])==False)]

median_online_rating_number = np.median(temp_data4[:,5])
online_data = temp_data4[np.where(temp_data4[:,5]>median_online_rating_number)]
nononline_data = temp_data4[np.where(temp_data4[:,5]<=median_online_rating_number)]

plt.figure()
plt.xlabel('Number of online reviews')
plt.ylabel('Average rating')
plt.plot(online_data[:,5],online_data[:,0],'o')
plt.plot(nononline_data[:,5],nononline_data[:,0],'o')
plt.show()

temp = (online_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('online mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('non-online mean:', round(np.mean(nononline_data[:,0]),5))

'''
t,p = stats.ttest_ind(online_data[:,0],nononline_data[:,0])
print(t,round(p,5))

pTest = permutation_test((online_data[:,0],nononline_data[:,0]),ourTestStatistic,n_resamples=1000)    
print('test statistic:', pTest.statistic)
print('p-value:',pTest.pvalue)
'''

#%% 5 - Relationship Between Average Rating and Desire to Retake Class

print("\n problem 5:")
temp_data5 = data[np.where(np.isnan(data[:,0])==False)]
temp_data5 = temp_data5[np.where(np.isnan(temp_data5[:,4])==False)]

plt.figure()
plt.xlabel('Average rating')
plt.ylabel('Proportion who would retake')
plt.plot(temp_data5[:,0],temp_data5[:,4],'o')
plt.show()

print("correlation:",(np.corrcoef(temp_data5[:,0], temp_data5[:,4]))[0,1])

#%% 6 - Positive Attractive Bias in Ratings

print("\n problem 6:")
temp_data6 = data[np.where(np.isnan(data[:,0])==False)]
hot_data = temp_data6[np.where(temp_data6[:,3]>0)]
nonhot_data = temp_data6[np.where(temp_data6[:,3]==0)]

'''
t,p = stats.ttest_ind(hot_data[:,0],nonhot_data[:,0])
print(t,round(p,6))
'''

plt.figure()
plt.hist(hot_data[:,0], bins=[i for i in range(0,6)])
plt.ylim(0,7500)
plt.xlim(1,5)
plt.xticks([1,2,3,4,5])
plt.xlabel('Average rating')
plt.title('"Hot" professors')
plt.show()

plt.figure()
plt.hist(nonhot_data[:,0], bins=[i for i in range(0,6)], color='orange')
plt.ylim(0,7500)
plt.xlim(1,5)
plt.xticks([1,2,3,4,5])
plt.xlabel('Average rating')
plt.title('Non-"hot" professors')
plt.show()

'''
pTest = permutation_test((hot_data[:,0],nonhot_data[:,0]),ourTestStatistic,n_resamples=1000)    
print('test statistic:', pTest.statistic)
print('p-value:',pTest.pvalue)
'''
temp = (nonhot_data[:,0],)
bootstrapCI = bootstrap(temp, np.mean, n_resamples = 1000, confidence_level=0.995)
print('non-hot mean 99.5% CI: (',round(bootstrapCI.confidence_interval.low,5),',', round(bootstrapCI.confidence_interval.high,5),')')
print('hot mean:', round(np.mean(hot_data[:,0]),5))

#%% 7 - Regression Model Predicting Average Rating from Difficulty

print("\n problem 7:")
temp_data7 = data[np.where(np.isnan(data[:,0])==False)]
temp_data7 = temp_data7[np.where(np.isnan(temp_data7[:,1])==False)]

x7 = temp_data7[:,1].reshape(len(temp_data7),1)
y7 = temp_data7[:,0]
x_train7, x_test7, y_train7, y_test7 = train_test_split(x7, y7, test_size=0.4, random_state=0)
model7 = LinearRegression().fit(x_train7,y_train7)

rSq = model7.score(x_test7,y_test7)
slope = model7.coef_ 
intercept = model7.intercept_ 
yHat7 = slope * x7 + intercept
rmse = (mean_squared_error(y7, yHat7))**0.5
print("R^2: ",rSq)
print("slope: ",slope)
print('intercept: ',intercept)
print('RMSE: ',rmse)

plt.figure()
plt.xlabel('Difficulty')
plt.ylabel('Average rating')
plt.ylim(0.9,5.1)
plt.xticks([1,2,3,4,5])
plt.yticks([1,2,3,4,5])
plt.plot(x7,y7,'o')
plt.plot(x7,yHat7,linewidth=3)
plt.show()

#%% 8 - Multiple Regression Model Predicting Average Rating

print("\n problem 8:")
temp_data8 = data[np.where(np.isnan(data[:,0])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,1])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,2])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,3])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,4])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,5])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,6])==False)]
temp_data8 = temp_data8[np.where(np.isnan(temp_data8[:,7])==False)]

X8 = temp_data8[:,1:]
y8 = temp_data8[:,0]
X_train8, X_test8, y_train8, y_test8 = train_test_split(X8, y8, test_size=0.4, random_state=0)
model8 = Lasso().fit(X_train8,y_train8)

rSq = model8.score(X_test8,y_test8)
slope = model8.coef_
intercept = model8.intercept_
yHat8 = model8.predict(X8)
rmse = (mean_squared_error(y8, yHat8))**0.5
print("R^2: ",rSq)
print("slopes: ",slope)
print('intercept: ',intercept)
print('RMSE: ',rmse)


#%% 9 - Classification Model Predicting Pepper from Average Rating 

print("\n problem 9:")
temp_data9 = data[np.where(np.isnan(data[:,0])==False)]
temp_data9 = temp_data9[np.where(np.isnan(temp_data9[:,3])==False)]

x9 = temp_data9[:,0].reshape(len(temp_data9),1) 
y9 = temp_data9[:,3]
x_train9, x_test9, y_train9, y_test9 = train_test_split(x9, y9, test_size=0.4, random_state=0)
model9 = LogisticRegression(class_weight='balanced').fit(x_train9,y_train9)

x9_1 = np.linspace(1,5,100)
y9_1 = x9_1 * model9.coef_ + model9.intercept_
sigmoid = expit(y9_1)

plt.figure()
plt.xlabel('Average rating')
plt.ylabel('Probability of being "hot"')
plt.xticks([1,2,3,4,5])
plt.scatter(data[:,0],data[:,3])
plt.plot(x9_1,sigmoid.ravel(),color='orange',linewidth=3) # the ravel function returns a flattened array
plt.show()

RocCurveDisplay.from_estimator(model9, x_test9, y_test9)
print('mean accuracy: ',model9.score(x_test9,y_test9))

#%% 10 - Multinomial Classification Model Predicting Pepper

print("\n problem 10:")
temp_data10 = data[np.where(np.isnan(data[:,0])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,1])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,2])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,3])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,4])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,5])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,6])==False)]
temp_data10 = temp_data10[np.where(np.isnan(temp_data10[:,7])==False)]

X10 = np.delete(temp_data10,3,1)
y10 = temp_data10[:,3]
X_train10, X_test10, y_train10, y_test10 = train_test_split(X10, y10, test_size=0.4, random_state=0)
model10 = LogisticRegression(class_weight='balanced',max_iter=1000).fit(X_train10,y_train10)

RocCurveDisplay.from_estimator(model10, X_test10, y_test10)
print('mean accuracy: ',model10.score(X_test10,y_test10))
