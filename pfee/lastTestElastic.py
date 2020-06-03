#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:38:19 2020

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


"""Import Data drom ELasticsearch"""
es  = Elasticsearch(["localhost:9200"])

order = es.search(index='commandes',body={},size=1000)
customer = es.search(index='custommer',body={},size=1000)
order_df = Select.from_dict(order).to_pandas()
customer_df = Select.from_dict(customer).to_pandas()


"""Prepare DataSet With  GroupBy"""
#order_df.groupby(order_df.version.apply(lambda x: x['major'])).size()
"""Essay 1"""
def productReturn():
    for i in order_df.orderItems:
        for j in i:
            return j['product']


productReturn()
#cd = order_df.groupby('orderItems').apply(lambda x : x)
""""""
dataCustom = order_df.groupby(by='customer')['orderItems'].count()
dataCustom = order_df.groupby(['customer'])[['orderItems']].count()



order_df.to_csv(r'order.csv', index = False)


"""Test Knoledge"""
order_df.orderItems.value_counts()
order_df.groupby(['orderItems']).mean()
order_df.applymap(lambda x: isinstance(x,list)).all()
order_df.applymap(lambda x: isinstance(x,list)).all()
order_df['orderItems'].astype('str').value_counts()
cd = order_df[order_df.orderItems.notna()].astype('str').groupby(['orderItems']).count()
""""""
data = pd.DataFrame(order_df.orderItems.sum())
data['customer']=order_df.customer
orderPLats = data.groupby('product').count()
dataplas= data.groupby('product').count()
dataGroup = data.groupby('product')['customer'].count()

import itertools
from operator import itemgetter
#ordersProduct = sorted(ordertest.orderItems, key=itemgetter('product'))
for key, value in itertools.groupby(order_df.orderItems, key=itemgetter('product')):
    print (key)


dataCustom = order_df.groupby('customer')['orderItems'].apply(list).reset_index(name='new')
order_df.groupby('orderItems').apply(lambda x:np.sum(x))
for i in order_df.orderItems:
    for j in i :
        print (j.keys())
        #print(j)
        #cd=dataCustom.groupby('customer')[j['product']].count()
            



#order_df.groupby('resto')(order_df.orderItems.apply(lambda x: x)).count()


grouped = order_df.groupby(order_df.orderItems.sum())
order_df.orderItems.sum()[50]['product']

#order_df.groupby(order_df.orderItems.sum()[len(order_df)]['product'])

newDF = order_df.append(customer_df)

DFNeww = order_df.append(customer_df)

nmbrResto = orderTest = order_df.groupby('resto')['customer','orderItems'].count()

from collections import defaultdict

v = defaultdict(list)

for key, value in sorted (order_df.orderItems.items()):
    v[value].append(key)

for i in order_df.orderItems.sum():



""""""



#df1 = order_df.groupby('resto')(order_df.orderItems.apply([0]['product']))



for i in order_df.orderItems.sum():
    print(i['product'])
df1 = order_df.groupby('resto')(order_df.orderItems.sum()).product


testOrder = order_df.groupby


#order_df.orderItems.get_group('product')
#order_df.groupby('resto')(order_df.orderItems.apply(lambda x: x)).count()
