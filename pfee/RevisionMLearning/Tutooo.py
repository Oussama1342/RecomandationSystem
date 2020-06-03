#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:42:40 2020

@author: oussama
"""

import numpy as np
 
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt  ##Visualiser les donnees a travers un graphe



X,y = make_regression()
















x,y = make_regression(n_samples=100, n_features=1, noise=10)



plt.scatter(x,y)

"""Verifier les dimensions de matrice"""
x.shape
y.shape

""""Vrifier rows of y"""
y = y.reshape(x.shape[0],1)


""""Generation de Matrix X """

X = np.hstack((x, np.ones(x.shape)))

theta = np.random.randn(2,1)


#"""2 - Modele Machine Learning """"
  
def model(X, theta):
    return X.dot(theta)   #"""ax+b"""



plt.scatter(x,y)

plt.plot(x,model(X,theta), c='r')


#"""Creation COst function""""

def costfunction(X,y,theta):
    return 1/(2*len(y)) * np.sum((model(X, theta)-y)**2)
    

costfunction(X,y,theta)

#"""Gradien Descendent""""


def grad(X,y,theta):
    m=len(y)
    return  1/m * X.T.dot(model(X,theta)-y)


def gradien_descent(X, y ,theta ,learning_rate, n_iteration):
    
    cost_history = np.zeros(n_iteration)
    
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X,y,theta)
        cost_history[i] = costfunction(X, y , theta)
        
    return theta,cost_history




#"""Entrainerment de modele"""
theta_final,cost_history =  gradien_descent(X, y ,theta ,learning_rate=0.001, n_iteration=1000)


prediction = model(X, theta_final)

pred1 = model(X,theta)

plt.scatter(x,y)
plt.plot(x,prediction, c='r')


""""Courbe d'apprentissage"""

plt.plot(range(1000), cost_history, c='r')
 

"""OPtimisation du modele par le coeficient"""

def coef_termination(y, pred):
    u =((y - pred)**2).sum()
    v =((y -y.mean())**2).sum()
    return 1- u/v


coef_termination(y, prediction)
    














#
###DataSet
#x,y = make_regression(n_samples=100,n_features=1, noise=10)
#
#y=y.reshape(y.shape[0],1)
#
#plt.scatter(x,y)
#
#x.shape
#y.shape
#
###Matrice X
#X = np.hstack((x, np.ones(x.shape)))
#
##Initialise theta
#
#theta = np.random.randn(2,1)
#
####Realisation du modele
#
#def model(X, theta):
#    return X.dot(theta)   ###==> Retourne le produit matricielle de X par thet
#
#
#
#model(X,theta)
#
#plt.plot(x,y)
#plt.plot(x, model(X,theta), c='r')
#
#
###Function Cout
#
#def functioncout(X,y,theta):
#    m= len(y)
#    return 1/(2*m) * np.sum((model(X, theta) -y)**2)




















































#A = np.array([[1,2],[3,4],[5,6]])
#
#A.shape
#
#C =A.T
#C.shape
#
#
#B = np.ones([1,2])
