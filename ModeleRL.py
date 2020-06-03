#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:49:03 2020

@author: oussama
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('filecsv/lycee.csv')

data.insert(0, 'Ones', 1)
x = data.iloc[:,1]
X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.plot(data['min_rev'],data['note_%'],'ro',markersize=4)
plt.show()



X = np.matrix([np.ones(data.shape[0]), data['min_rev'].values]).T
y = np.matrix(data['note_%']).T

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

plt.xlabel('min_rev')
plt.ylabel('note_%')

plt.plot(data['min_rev'], data['note_%'], 'ro', markersize = 4)









X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.random.randn(2,1)
print(theta.T)

def model(X,theta):
    return(X.dot(theta))
    
    
model(X,theta)

plt.scatter(x,model(X,theta), c='r')

def costFunction(X,theta,Y):
    return  (1/2*len(Y)) * ( np.sum(model(X,theta) - Y) ** 2)


costFunction(X,theta,Y)