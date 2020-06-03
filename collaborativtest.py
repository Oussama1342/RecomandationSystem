#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:16:58 2020

@author: oussama
"""
import pandas as pd
from scipy.spatial.distance import cosine

data= pd.read_csv("pfee/lastfm-matrix-germany.csv")
data.head(10)

data_germany = data.drop('user',1)
data_ibs = pd.DataFrame(index=data_germany.columns,columns=data_germany.columns)


#---remplir les simulitudes
for i in range(0,len(data_ibs.columns)):
    # On boucle a travers les colones pour chaque colone
    for j in range(0,len(data_ibs.columns)):
        data_ibs.iloc[i,j]=1 - cosine(data_germany.iloc[:,i],data_germany.iloc[:,j])
        
#Creer des articles d'emplacement pour fermer le voisin d'un article
data_voisins= pd.DataFrame(index=data_ibs.columns,columns=[(1,11)])

data_ibs.describe()

for i in range(0,len(data_ibs.columns)):
    
    data_voisins.iloc[i,:10]=data_ibs.iloc[0:,i].sort_values(ascendinf=False)[:10].index 