#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:19:52 2020

@author: oussama
"""

import numpy as np 
import pandas as pd

import re
from csv import writer
from csv import reader
import csv
import matplotlib.pyplot as plt
import seaborn as sbrn
import statsmodels as stat
from sklearn.preprocessing import PolynomialFeatures 



##Preparing Data
Data =pd.read_csv("filecsv/data3last.csv")


Data['orderItems']=Data['orderItems'].replace({'product':''}, regex=True)
Data['orderItems']= Data['orderItems'].map(lambda x: re.sub(r'\W+', '', x))


Data = Data.drop(['customer'], axis = 1)
Data = Data.drop(['resto'], axis = 1)
Data = Data.drop(['total'], axis = 1)
Data = Data.drop(['createdAt'], axis = 1)




X=Data[["age","gender"]].values
Y=Data["orderItems"].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labEncr_X = LabelEncoder()
X[:,1] = labEncr_X.fit_transform(X[:,1])

onehotEncr = OneHotEncoder(categorical_features=[1])

X = onehotEncr.fit_transform(X).toarray()

labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

## Jeu de donnees

print(Data.columns)

plt.hist(Data["age"])
plt.show()
Data = Data.dropna(axis=0)
data1 = [Data["age"]== 0]


###Classification avec K-means

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=5, random_state=1)

good_columns = Data._get_numeric_data()
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
plt.show()

columns = Data.columns.tolist()
















































np.random.seed(0)
Y = Y+abs(Y/2)

plt.scatter(X, Y)

print(X.shape)
print(Y.shape)
Y = Y.reshape(Y.shape[0], 1)

X = np.hstack((X, np.ones(X.shape)))
X = np.hstack((X**2, X))

theta = np.random.randn(3, 1)
def model(X, theta):
    return X.dot(theta)

plt.scatter(X, Y)
plt.scatter(X, model(X, theta), c='r')







