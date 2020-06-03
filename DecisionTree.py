#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:45:28 2020

@author: oussama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbrn
import statsmodels as stat

dataset = pd.read_csv("solvablClient.csv")


from sklearn.preprocessing import Imputer

#Decoupage DataSet
X = dataset.iloc[:,-9:-1].values 
Y = dataset.iloc[:,-1].values

#Transformation du valeur NaN
imptr = Imputer(missing_values="NaN", strategy="mean", axis = 0)
imptr.fit(X[:, 0:1])
imptr.fit(X[:, 7:8])

X[:, 0:1] = imptr.transform(X[:, 0:1])
X[:, 7:8] = imptr.transform(X[:, 7:8]) 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


labEncr_X = LabelEncoder()
X[:,2] = labEncr_X.fit_transform(X[:,2])
X[:,5] = labEncr_X.fit_transform(X[:,5])
onehotEncr = OneHotEncoder(categorical_features=[2])
onehotEncr = OneHotEncoder(categorical_features=[5])
X = onehotEncr.fit_transform(X).toarray()


#Codage de valeur a predit
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

#base d'apprentissag e de test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,  test_size = 0.2, random_state = 0)


