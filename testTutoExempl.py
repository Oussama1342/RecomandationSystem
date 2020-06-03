#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:30:48 2020

@author: oussama
"""


"""
** ND Array
-Attribut ND Array :
    chape== > Dimension du tableau
   
** Matplotlib:
    
    plt.scatter()
    
"""



import numpy as np
 
from sklearn.datasets import make_regression

from sklearn.datasets import load_iris

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import SGDRegressor 



np.random.seed(0)
###########Iteration 1
x, y = make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x, y) 

model = SGDRegressor(max_iter=100, eta0=0.0001) 
model.fit(x,y) 

model.score(x,y)

plt.scatter(x, y) 

plt.plot(x, model.predict(x), c='red', lw = 3)


###########Iteration 2
model = SGDRegressor(max_iter=1000, eta0=0.001) 
model.fit(x,y) 


plt.scatter(x, y) 
plt.plot(x, model.predict(x), c='red', lw = 3)

model.score(x,y)


fig,ax = plt.subplots(figsize=(12,8))

ax.set_ylabel("Theta")
ax.set_xlabel("Iterations")
#_=ax.plot(range(iterations))


#_=ax.plot(x,y, 'b.')

    











# création du Dataset
x, y = make_regression(n_samples=100, n_features=1, noise=10)
y = y**2 # y ne varie plus linéairement selon x !


# On ajoute des variables polynômiales dans notre dataset
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x = poly_features.fit_transform(x)


plt.scatter(x[:,0], y)
x.shape # la dimension de x: 100 lignes et 2 colonnes 

model = SGDRegressor(max_iter=1000, eta0=0.001) 
model.fit(x,y)

model.score(x,y)
plt.scatter(x[:,0], y, marker='o') 
plt.scatter(x[:,0], model.predict(x), c='red', marker='+') 




##Import Data
iris = load_iris()
x = iris.data
y = iris.target

plt.scatter(x[:,0],x[:,1], c=y, alpha=0.75, s=100)
plt.xlabel("Longuer sepla")
plt.ylabel("Largeur Sepal")

Dataset =pd.read_csv("filecsv/data3last.csv")

Dataset['orderItems']=Dataset['orderItems'].replace({'product':''}, regex=True)
Dataset['orderItems']= Dataset['orderItems'].map(lambda x: re.sub(r'\W+', '', x))



ax = plt.axes(projection='3d')

ax.scatter(x[:,0],x[:,1],x[:,2], c=y)

f= lambda x,y: np.sin(x) + np.cos(x+y)

X = np.linspace(0, 5, 100)
Y = np.linspace(0,5,100)
X,Y = np.meshgrid(X, Y) 
Z = f(X,Y)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')


""" Histogramme  """

plt.hist(x[:,0], bins=20)
plt.hist(x[:,1], bins=20)




plt.hist2d(x[:,0], x[:,1])
plt.xlabel('longueur sepal')
plt.ylabel('largeur sepal')
plt.colorbar()







