#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:16:53 2020

@author: oussama
"""

##########12##########

##LIste de librairie necessaires
"""
    numpy : pour des clculs mathematique
    matpllotib : pour faire de graphes
    Maplotlib,pyplot : Fare des graphes 2D
    Pandas : Pour traiter les dataset
    seaborn : Visualisation de donnees statistique
    Statsmodels : est un module Python qui fournit des classes et de fonctions de nombreux modele statistiques 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbrn
import statsmodels as stat

##########Session de mise en place de ML par l'import de donnees##########
dtst =  pd.read_csv("solvablClient.csv")  ## ==> import file csv


X = dtst.iloc[:,-9:-1].values     ###==> elimination de colone et montre les variables explicatives et les variables a expliques

Y = dtst.iloc[:,-1].values

#15 -Nettoyage : Gestion du valeur Null ===> Data Cleaning

from sklearn.preprocessing import Imputer
#
####LEs valeurs NaN soit on les remplace par le moyen soit par le mediane
#
#
imptr = Imputer(missing_values="NaN", strategy="mean", axis = 0)
imptr.fit(X[:, 0:1])
imptr.fit(X[:, 7:8])
# 
#
###Transformation du valeur Nan
#
X[:, 0:1] = imptr.transform(X[:, 0:1])
X[:, 7:8] = imptr.transform(X[:, 7:8])   ###==>Elimination de valeurs NAN


##############dummy variable: Traitement##########33
     #Donnees CAtegorique : LabelEncoder, OneHotEncoder de sqlearn
    
    #dummyvariable==> Traduit une presence ou une absence
    #demy variable de trap==>
    
##COdage de la variable indepandante
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#

labEncr_X = LabelEncoder()
X[:,2] = labEncr_X.fit_transform(X[:,2])
X[:,5] = labEncr_X.fit_transform(X[:,5])
onehotEncr = OneHotEncoder(categorical_features=[2])
onehotEncr = OneHotEncoder(categorical_features=[5])
#
X = onehotEncr.fit_transform(X).toarray()
#
#### Ma variable DEP
#
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

####Decoupage de donnees en echantillonnage de d'apprentissage et de test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,  test_size = 0.2, random_state = 0)
#X_train = onehotEncr.fit_transform(X_train).toarray()


##Phase de preparation de donnees

#==> Normalisation de donnees
#==> Standardisation
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.fit_transform(X_test)


X_train = normalize(X_train)
X_test = normalize(X_test)

### Section 5 ==> Modele de regression lineaire Multiple

from sklearn.linear_model import LinearRegression

regrisseur = LinearRegression()
regrisseur.fit(X_train,Y_train)

## on va faire de test de nouvelle prediction

Y_prediction = regrisseur.predict(X_test)



###


"""
def kmeabsAlgo(X,nn):
    
    lengthDataset = len(X.index) -1
    
    
        print("alog k-means nombre clusteur = ",nn)
"""
#X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#plt.scatter(X[:,0], X[:,1])
##
##wcss = []
##for i in range(1, 11):
##    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
##    kmeans.fit(X)
##    wcss.append(kmeans.inertia_)
##plt.plot(range(1, 11), wcss)
##plt.title('Elbow Method')
##plt.xlabel('Number of clusters')
##plt.ylabel('WCSS')
##plt.show()
#
#kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10, random_state=0)
#pred_y = kmeans.fit_predict(X)
#plt.scatter(X[:,0], X[:,1])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
#plt.show()



#yi = @+B(BETA) * xi + random error




    