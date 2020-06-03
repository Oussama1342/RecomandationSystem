#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:32:47 2020

@author: oussama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):              ####### Hatha wathe7 y7seb lcost ==> Mesure d'erreur
    z = np.power(((X * theta.T) - y), 2)  ### =>(X* theta) == h(x)
    #print('z \n',z)
   # print('m ' ,len(X))
    return np.sum(z) / (2 * len(X))






def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape)) ##met theta.shape en 0 ==> Initialise Theta en 0
    print('temp:' ,temp)
 
    parameters = int(theta.ravel().shape[1])  # recupere la valeur de teta 1


    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
         
        for j in range(parameters):
            term = np.multiply(error, X[:,j]) ## Multiolication de 1 et de input de X
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
       # print('theta:' ,theta)

        cost[i] = computeCost(X, y, theta)
        
    return theta, cost





















path = 'filecsv/tesMultivariabl.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms','Price'])

print('data = ')
print(data2.head(10) )
print()
print('data.describe = ')
print(data2.describe())

# rescaling data
data2 = (data2 - data2.mean()) / data2.std()

print()
print('data after normalization = ')
print(data2.head(10) )


# add ones column
data2.insert(0, 'Ones', 1)


# separate X (training data) from y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]


print('**************************************')
print('X2 data = \n' ,X2.head(10) )
print('y2 data = \n' ,y2.head(10) )
print('**************************************')


# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))


#print('X2 \n',X2)
#print('X2.shape = ' , X2.shape)
#print('**************************************')
#print('theta2 \n',theta2)
#print('theta2.shape = ' , theta2.shape)
#print('**************************************')
#print('y2 \n',y2)
#print('y2.shape = ' , y2.shape)
#print('**************************************')


# initialize variables for learning rate and iterations
alpha = 0.1
iters = 1000

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
thiscost = computeCost(X2, y2, g2)


print('g2 = ' , g2)
print('cost2  = ' , cost2[0:50] )
print('computeCost = ' , thiscost)
print('**************************************')


# get best fit line for Size vs. Price

x = np.linspace(data2.Size.min(), data2.Size.max(), 100)
print('x \n',x)
print('g \n',g2)

f = g2[0, 0] + (g2[0, 1] * x)
print('f \n',f)

# draw the line for Size vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Size, data2.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


# get best fit line for Bedrooms vs. Price

x = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)
print('x \n',x)
print('g \n',g2)

f = g2[0, 0] + (g2[0, 1] * x)
print('f \n',f)

# draw the line  for Bedrooms vs. Price
from mpl_toolkits.mplot3d import Axes3D



fig, ax = fig.add_subplot(111, projection='3d')
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Bedrooms, data2.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')



# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
