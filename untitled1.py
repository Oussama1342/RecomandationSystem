#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:24:21 2020

@author: oussama
"""

Dataset =pd.read_csv("filecsv/data3last.csv")
Dataset['orderItems']=Dataset['orderItems'].replace({'product':''}, regex=True)
Dataset['orderItems']= Dataset['orderItems'].map(lambda x: re.sub(r'\W+', '', x))


Dataset["age"] = Dataset["age"].fillna(-0.3)
bins = [18, 27, 35, 60, np.inf]
labels = [ 'Student', 'employee', 'Adult', 'Senior']
Dataset['status'] = pd.cut(Dataset["age"], bins, labels = labels)

Dataset = Dataset.drop(['customer'], axis = 1)
Dataset = Dataset.drop(['resto'], axis = 1)
Dataset = Dataset.drop(['total'], axis = 1)
Dataset = Dataset.drop(['createdAt'], axis = 1)

from sklearn.model_selection import train_test_split

