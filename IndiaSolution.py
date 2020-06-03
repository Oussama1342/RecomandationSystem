#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:59:20 2020

@author: oussama
"""

import csv, json
import numpy as np
import pandas as pd
import re
from collections import Counter
import seaborn as sns

from pandasticsearch import Select


from operator import itemgetter 
from elasticsearch import Elasticsearch
import math
from scipy.spatial.distance import cosine

data = pd.read_csv('Groupby/result.csv')
data = pd.DataFrame(data)
#data.groupby('customer')['product'].count()
#
#data.groupby('product').mean()
#data['product'].head(5)
#data = data.fillna(np.nan)
#data.sample(5)
dataPlats = pd.pivot_table(data, values='quantity', index=['customer'],
                           
                           columns=['product'], aggfunc=np.sum)


"""Netoyage de donnees ==> Elimination de valeur NAN"""
#dataPlatVal = pd.DataFrame(dataPlats).values

dataPlats.isnull()
dataPlats=dataPlats.fillna(0)
for i in dataPlats.index:
    for j in dataPlats.columns:
        if dataPlats.ix[i][j]>1:
            dataPlats.ix[i][j] =1
"""Collaborativ filtering Item Based"""
dataplat_ibs = pd.DataFrame(index = dataPlats.columns, columns=dataPlats.columns)

for i in range(0,len(dataplat_ibs.columns)):
    for j in range(0,len(dataplat_ibs.columns)):
        dataplat_ibs.ix[i,j] = 1 - cosine(dataPlats.ix[:,i], dataPlats.ix[:,j])
    

data_neighbours = pd.DataFrame(index= dataplat_ibs.columns , columns=range(1,11))


for i in range(0,len(dataplat_ibs.columns)):
    data_neighbours.ix[i,:10] = dataplat_ibs.ix[0:,i].sort_values(ascending=True)[:10].index

data_neighbours.head(6).ix[:6,2:4]
 
"""User BAsed Filtering"""












        


from sklearn.preprocessing import Imputer
imptr = Imputer(missing_values="NaN", strategy="mean", axis = 0)
imptr.fit(dataPlats[:, 0:1])
imptr.fit(dataPlats[:, 2:3])

dataPlatVal[:, 0:1]  = imptr.transform(dataPlats[:, 0:1])
dataPlatVal[:, 0:1]  = imptr.transform(dataPlatVal[:, 2:3])
#dataPlats['Pizza Cheese'] = [x for x in dataPlats['Pizza Cheese'] if not math.isnan(x)]
"""Filtrage Collaboratif"""
from scipy import spatial