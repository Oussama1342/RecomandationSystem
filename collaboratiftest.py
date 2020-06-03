#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:37:40 2020

@author: oussama
"""

import numpy as np 
import pandas as pd

import re
from csv import writer
from csv import reader
import csv
import matplotlib.pyplot as plt
import seaborn as sbrn
import statsmodels as stat
from sklearn.preprocessing import PolynomialFeatures 
import sys, json

from sklearn.linear_model import SGDRegressor 


data = pd.read_csv("filecsv/datatestElastic.csv")
data['orderItems']=data['orderItems'].replace({'product':''}, regex=True)
data['orderItems']= data['orderItems'].map(lambda x: re.sub(r'\W+', '', x))


data.head()

X=data.iloc[:,1].values
##Codage de doata

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labEncr_X = LabelEncoder()
X[:,0] = labEncr_X.fit_transform(X[:,0])

data[:,0] = labEncr_X.fit_transform(data[:,0])
data[:,0] = labEncr_X.fit_transform(data[:,1])
#X[:,1] = labEncr_X.fit_transform(X[:,1])

onehotEncr = OneHotEncoder(categorical_features=[0])
onehotEncr = OneHotEncoder(categorical_features=[1])
X = onehotEncr.fit_transform(data).toarray()


#################33Test 2


from datetime import datetime
from elasticsearch import Elasticsearch
from pandas import DataFrame, Series
import pandas as pd
import matplotlib.pyplot as plt

es = Elasticsearch([{u'host': u'127.0.0.1', u'port': 9200}])
response = es.search( index='oreders', body={} )



columns = ['Home','Car','Sport','Food']
index= data
df = pd.DataFrame(data=data,index=index,columns=columns)


