#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:57:57 2020

@author: oussama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbrn
import statsmodels as stat

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

Dataset = pd.read_csv("CommandeTest2.csv")  

##Decoupage de donnes
X = Dataset.iloc[:,-4:-1]

#lentghDataSet = len(Dataset.index)
#
#lentghDataSet = lentghDataSet -1


##Nettoyage de donnees
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labEncr_X = LabelEncoder()
X[:,1] = labEncr_X.fit_transform(X[:,1])
X[:,2] = labEncr_X.fit_transform(X[:,2])

onehotEnc = OneHotEncoder(categorical_features=[1])

onehotEnc = OneHotEncoder(categorical_features=[2])

X = onehotEnc.fit_transform(X).toarray()



#kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10, random_state=0)
#pred_y = kmeans.fit_predict(X)
#mplt.scatter(X[:,0], X[:,1])
#mplt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
#mplt.show()


###########K-Means############

     kmeans = KMeans(n_clusters=100, init='k-means++', n_init=10, random_state=0)
    #kmeans = KMeans(n_clusters= 20, init = 'k-means++', random_state = 0)

    y_kmeans = kmeans.fit_predict(X)
    #mplt.scatter(X[:,0], X[:,1])
    mplt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='green')
   
    

##########K-Means##############
    
#cluster = y_kmeans[lentghDataSet]











#X = X.to_numpy()


#dt = X.values
#dt = dt.astype('float32')
#train_size = int(len(dt) * 0.67)
#train_dataset = dt[0:train_size,:]


###
#from sklearn.preprocessing import Imputer
#imptr = Imputer(missing_values="NaN", strategy="mean", axis = 0)
#imptr.fit(X[:, 0:1])
#
#X[:, 0:1] = imptr.transform(X[:, 0:1])
#
#
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##

