#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:34:00 2020

@author: oussama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##Import Data

data = pd.read_csv("filecsv/Advertising.csv",index_col=0)

data.head()

from sklearn.linear_model import LinearRegression

##Create Object Regression

modeleReg=LinearRegression()


cols = data.shape[1] 
X = data.iloc[0: ,0:cols-1]
Y = data.iloc[0: ,cols-1:cols]

modeleReg.fit(X,Y)

print(modeleReg.intercept_)     ##THetha0
print(modeleReg.coef_)   ###Theta 1

modeleReg.score(X,Y)


RMSE=np.sqrt(((Y-modeleReg.predict(X))**2).sum()/len(Y))

plt.plot(Y, modeleReg.predict(X),'.')


plt.plot(Y, Y-modeleReg.predict(X),'.')


