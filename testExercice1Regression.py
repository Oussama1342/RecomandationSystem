#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:47:48 2020

@author: oussama
"""


"""Regression lineaire """
import numpy as np


#####Part 1
def warm_up_exercice():
    A = np.eye(5, dtype=float)
    return A




print ('5x5 Identity Matrix: ')
print(warm_up_exercice())


## Part 2


data = np.loadtxt(open("ex1data1.txt", "r"), delimiter=",")
x = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
print(len(X))

import matplotlib.pyplot as plt

def plot_data(x,y):
    plt.plot(x, y, linestyle='', marker='x', color='r', label='Training Data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    

plt.figure()
plot_data(x,y)
plt.show()


##Part 3: Gradient descent

x = np.hstack((np.ones((m, 1)), x.reshape(m, 1)))
theta = np.zeros(2)

iterations = 1500
alpha = 0.01

