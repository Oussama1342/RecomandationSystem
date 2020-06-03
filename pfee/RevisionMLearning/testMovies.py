#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:49:13 2020

@author: oussama
"""

import pandas as pd
from scipy import spatial

dataset = pd.read_csv('movie.csv')

une = [1,2]
b = [2,5,4]

spatial.distance.cosine(b,une)