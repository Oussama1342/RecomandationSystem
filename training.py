#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:28:48 2020

@author: oussama
"""
import numpy as np
 
from sklearn.datasets import make_regression

from sklearn.datasets import load_iris

from mpl_toolkits.mplot3d import Axes3D



import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import SGDRegressor

import requests

import pandas as pd



Data =pd.read_csv("filecsv/data3last.csv")


#Data['orderItems']=Data['orderItems'].replace({'product':''}, regex=True)
#Data['orderItems']= Data['orderItems'].map(lambda x: re.sub(r'\W+', '', x))
#
#Data = Data.drop(['customer'], axis = 1)
#Data = Data.drop(['resto'], axis = 1)
#Data = Data.drop(['total'], axis = 1)
#Data = Data.drop(['createdAt'], axis = 1)

Data.info()
Data.describe()



X = Data.iloc[:,[0,1,2]].values
Y=X[:,2]
#X= np.delete(X, 2,1)



Y = Data.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


labEncr_X = LabelEncoder()
X[:,0] = labEncr_X.fit_transform(X[:,0])
X[:,2] = labEncr_X.fit_transform(X[:,2])

onehotEncr = OneHotEncoder(categorical_features=[0])
onehotEncr = OneHotEncoder(categorical_features=[2])

#onehotEncr = OneHotEncoder(categorical_features=[1])

X = onehotEncr.fit_transform(X).toarray()

labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)



Data.hist(bins=50, figsize=(20,15))
plt.show()


Data.sort_values(by=['age'], ascending=True)[:10]


Data.corr()
print(type(X))
print(type(Data))

from pandas.plotting import scatter_matrix

attrs = ['age','gender','orderItems']

scatter_matrix(Data[attrs], figsize=(12,8))


from sklearn.preprocessing import StandardScaler
std = StandardScaler()

X = std.fit_transform(X)


"""Lineaire Regression"""
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, Y)
"""Test Result Lineaire Regression"""
from sklearn.model_selection import cross_val_score
linreg_score = cross_val_score(linreg, X, Y, scoring="neg_mean_squared_error", cv=15)
linreg_rmse = np.sqrt(-linreg_score)

print(linreg_rmse)
print("Moyenne", linreg_rmse.mean())
print("Ecart-type", linreg_rmse.std())

print('Intercept: \n', linreg.intercept_)
print('Coefficients: \n', linreg.coef_)


"""Decision Tree"""
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor()
treereg.fit(X, Y)

treereg_score = cross_val_score(linreg, X, Y, scoring="neg_mean_squared_error", cv=20)
treereg_rmse = np.sqrt(-treereg_score)

print(treereg_rmse)
print("Moyenne", treereg_rmse.mean())
print("Ecart-type", treereg_rmse.std())



"""RandomForestRegression  """

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(X, Y)
"""Test Random"""
forest_score = cross_val_score(forest, X, Y, scoring="neg_mean_squared_error", cv=5)
forest_rmse = np.sqrt(-forest_score)

print(forest_rmse)
print("Moyenne", forest_rmse.mean())
print("Ecart-type", forest_rmse.std())



"""Statsmodels"""
import statsmodels.api as sm

X1 = sm.add_constant(X)

#data =pd.read_csv("filecsv/lycee.csv")
###Visualisation
plt.scatter(Data['age'],Data['orderItems'],color='red')
model = sm.OLS(Y, X1).fit()
predictions = model.predict(X1) 

print_model = model.summary()
print(print_model)














#def _download(url: str, dest_path: str):
#
#    req = requests.get(url, stream=True)
#    req.raise_for_status()
#
#    with open(dest_path, "wb") as fd:
#        for chunk in req.iter_content(chunk_size=2 ** 20):
#            fd.write(chunk)
#            
#            
#            
#rating_url = "https://www.facebook.com/oussama.amari.121"
#
#path = "filecsv/datatraining.csv"
#_download(rating_url,  path)



