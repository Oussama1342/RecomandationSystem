#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:51:36 2020

@author: oussama
"""

import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(1,10,2000)
plt.plot(t,np.cos(t))
plt.savefig('image1')

"""Figure , axe et subplots"""


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0,2.0)



y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)

y2 = np.cos(2 * np.pi * x2)

fig, axs = plt.subplots(2,1, sharex = True)

ax1, ax2 = axs

ax1.plot(x1,y1,  'yo-')
ax1.set_title('Une histoire en 2 morceaux')

ax1.set_ylabel('Oscolisaion amorite')

ax2.plot(x2, y2, 'r.-') # r=red .=point -=trait continu
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Non amortie')

plt.show()


"""Scatter plot"""
x = np.random.random(1000)
y = np.random.random(1000)
plt.scatter(x,y)
plt.savefig(plt.scatter(x,y))













