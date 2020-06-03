#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:50:40 2020

@author: oussama
"""

lst = [1,2,4]

lst = lst*2

for i , val in enumerate(lst):
    lst[i] = val *2


"""
On trouve dans le module NumPy de nombreux 
outils de manipulation des tableaux pour le calcul numérique"""

import numpy as np

##Creation array a partir d'un liste
a = np.array(lst)
b= np.array([1,3,4,5,6,7,8])


a = a+2
a = a* 2

np.arange(4,10,10.5)

np.linspace(4, 10, 5)
np.zeros(5)
np.zeros((2,4))
np.ones((2,4),dtype=int)

z = np.empty((2,4))
z.fill(5)
np.diag([1,2,3,4])

"""Indexing"""
a = np.array([[1,2,3],[4,5,6]])

##dimensuion 
a.ndim
a.shape
a.size

"""Slicing"""
a[:,2]

"""Boleean"""
a>4

"""Boolean Indexing"""
a[a>4]


"""Soustraction array"""
#array de même forme
a = np.array([11, 12, 13, 14])
b = np.array([1, 2, 3, 4])

c= a-b
 
#array de forme différente (taille différente ici)
a = np.array([11, 12, 13, 14])
b = np.array([1, 2, 3])
c= a-b

"""Fonction trignometrique"""
from math import pi, cos

theta = np.array([pi, pi/2,pi/4,pi/6,0])
d = np.cos(theta)
"""MAtrice"""
M1 = np.array([[1,2],[3,4]])
M2 = np.array([[0,4],[0,2]])

prod = np.dot(M1,M2)
#Définition avec des matrix

M1 = np.matrix([[1,2],[3,4]])

M1 = np.matrix([[0,4],[0,2]])
prod = M1 * M2

M1.T   ##Transpose
M1**-1 ## inverse
M1.diagonal()
#Trace d'un matrice (somme des coefficients diagonaux)
M1.trace()
####Autres opérations utiles sur les arrays (et les matrices)
a = np.array([[1,2,3],[4,5,6]])

a.shape = (3,2)

#Aplatissement

a.ravel()


























