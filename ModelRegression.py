#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:14:18 2020

@author: oussama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbrn
import statsmodels as stat


#Creation DataSet
dataset = pd.read_csv("filecsv/lycee.csv")


# Decoupage de donnees
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

##Construire un echantionn de donnees pour apprentissage et un autre pour test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,  test_size = 0.2, random_state = 0)


#Construction de modele de regression Simple
from sklearn.linear_model import LinearRegression

regresseur = LinearRegression()
regresseur.fit(X_train,Y_train)

##Creation d'une variable de prediction
y_prediction = regresseur.predict(X_test)


#Prediction sur nouvel data
#regresseur.predict(180)

mplt.scatter(X_train,Y_train, color = 'red')
mplt.plot(X_train, regresseur.predict(X_train), color='green')
mplt.title(' Rendment note sur munite de revision')
mplt.xlabel('minute passe a revizer ' )
mplt.ylabel('note en pourcentage')
mplt.show()

mplt.scatter(X_test,Y_test, color = 'yellow')
mplt.plot(X_test, regresseur.predict(X_test), color='black')
mplt.title(' Rendment note sur munite de revision')
mplt.xlabel('minute passe a revizer ' )
mplt.ylabel('note en pourcentage')
mplt.show()