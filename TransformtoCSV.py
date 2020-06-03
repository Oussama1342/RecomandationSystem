#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:42:48 2020

@author: oussama
"""

import csv, json
import numpy as np
import pandas as pd
import re


from operator import itemgetter 
from elasticsearch import Elasticsearch


elastic_client = Elasticsearch()

result = elastic_client.search(index='oreders', body={}, size=99)

elastic_docs = result["hits"]["hits"]

docs = pandas.DataFrame()


for num, doc in enumerate(elastic_docs):
    source_data = doc["_source"]
    _id = doc["_id"]

doc_data = pandas.Series(source_data, name = _id)
docs = docs.append(doc_data)

docs.to_json("objectrocket.json")
json_export = docs.to_json() # return JSON data
print ("\nJSON data:", json_export)



##export to csv file


doc_data.to_csv("objectrocket.csv")
csv_export = docs.to_csv(sep=",") # CSV delimited by commas
print ("\nCSV data:", csv_export)


es  = Elasticsearch(["localhost:9200"])

es.indices.stats(index='oreders')

     # returns dict object of the index _mapping schema




"""Interessant custommer Insex"""
response = elastic_client.search( index='custommer', body={} )

for key,val in response.items():
    print(key, "/n")


for key,val in response["hits"].items():
    print(key, "/n")

elastic_docs = response["hits"]["hits"]

print ("documents returned:", len(response["hits"]["hits"]))

for key, val in response["hits"].items():
    if key == "hits":
        for num, doc in enumerate(val):
            print(num, '-->',doc, "\n")

fields={}

for num, doc in enumerate(elastic_docs):
    pass



source_data = doc["_source"]


for key, val in source_data.items():
    try:
        fields[key] = np.append(fields[key],val)
    except KeyError:
        fields[key] = np.array([val])


for key, val in fields.items():
    print (key, "--->", val)
    print ("NumPy array len:", len(val), "\n")


elastic_df = pandas.DataFrame(fields)
print ('elastic_df:', type(elastic_df), "\n")




"""test4444"""


from pandasticsearch import Select


dftest = pd.DataFrame(columns=["num1","num2"])
gb = order_df.apply('orderItems')


order_df['orderItems'].groupby([order_df['customer'],order_df['total']]).describe()


#from pandasticsearch import DataFrame
#
#df = DataFrame.from_es(url='http://localhost:9200', index='commandes', doc_type='properties')
#
#df.print_schema()
#
#
#
#import requests
#res = requests.get('http://localhost:9200')
#print(res)


tabplats =pd.DataFrame(columns=['plats'])
tabPlats = []
"""Extract Data par GroupBy"""
print(order_df.orderItems)
for i in order_df.orderItems:
    for j in i :
        tabPlats.append(j['product'])


from collections import Counter

NumberPlat = Counter(tabPlats)

tabplats = pd.DataFrame(tabplats)

print(tabplats[0])
    
   # print(i['product'])

order = es.search(index='commandes',body={},size=1000)
custommer = es.search(index='custommer',body={},size=1000)

order_df = Select.from_dict(order).to_pandas()
custommer_df = Select.from_dict(custommer).to_pandas()

print(order_df.orderItems)
for i in order_df.orderItems:
         for j in i :
        print(j['product'])

print(order_df.orderItems)



dataCOmnd = order_df.orderItems


df = pd.DataFrame({'a': [1,2,3], 
                   'b': [4,5,6], 
                   'version': [{'major': 7, 'minor':1}, 
                               {'major':8, 'minor': 5},
                               {'major':7, 'minor':2}] })


#    df =pd.DataFrame(response)

#df.groupby(df.version.apply(lambda x: x['major'])).size()
#df.groupby(df.version.apply(lambda x: x['major']))[['a', 'b']].sum()

#pd.DataFrame(order_df)
#order_df.index
#g = order_df.groupby(['customer','orderItems']).count()
#pd.Series(index=order_df).groupby(level=0).count()



#testOrde.groupby(testOrde.orderItems.apply(lambda x: x['product'])).size()
#
#
#order_df.groupby(order_df.orderItems.apply(lambda x: x['product'])).sum()
#order_df.groupby(order_df.orderItems.apply(lambda x: x['product'])).size()

#test2 =order_df.groupby(order_df.orderItems.apply(lambda x: x[0]))[['customer']].sum()
#
#data1 = order_df.groupby('customer')(order_df.orderItems.apply(lambda x: x[0]))

order_df.groupby('resto')(order_df.orderItems.apply(lambda x: x)).count()
oeder_df.groupby(')


group2 = order_df['resto'].groupby(order_df['orderItems']).count()


order_df['orderItems'].groupby(order_df['resto']).count()





#order_df['orderItems']= order_df['orderItems'].map(lambda x: re.sub(r'\W+', '', x))


for gname, data in group2:
    print(gname)
    print('-------')
    print (data)

g = order_df['customer'].groupby([order_df.orderItems.apply([' product'])]).mean





#Dicrionary
#for i in order_df.orderItems:
#    for j in i:
#        print(j.keys())

#grouped_data = order_df.groupby(by=mapping,axis=1).sum()


orderdftest.groupby(orderdftest.orderItems.apply(lambda x: x['product'])).size()
mask = order_df.groupby(['orderItems']).apply(lambda x : (x['product']))

dataPlats = order_df.groupby(['orderItems']).apply(lambda x :x )


   
order_df.groupby(order_df.orderItems.apply(pd.Series).product)[['customer']].sum()



#print(order_df.orderItems['product'])    

#print(i[0])
#number_cmd = order_df.groupby('customer')[datCmd()].count()


#number_cmd4 = order_df.groupby('orderItems')['product'].apply(list),['customer ']
#
#number_cmd2 = order_df.groupby('customer')['orderItems'].apply(list
    



order_df['customer'].count()


order_df = order_df.drop(['_score'], axis = 1)
gk = order.groupby('customer')


test1 = order_df.iloc[:,7]



order_df.groupby(['order_df']).mean()




order_df.orderItems[0]

for c in number_cmd2:
    print(c)
    
    
print(c[0]

for x  in order_df.orderItems:
    print(x[0])
    
    
    

#number_cmd3 = order_df.groupby('customer')['ordersss']['product']




















































	__v	_id	_index	_type	createdAt	customer	orderItems	resto	total	updatedAt
0	0	5da71cb3275210218f1469e8	commandes	orders	2019-10-16T14:35:47.244+01:00	5d499c467cdaeefdb118637a	[{'_id': '5da71cb32752108bf71469ea', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}, {'_id': '5da71cb3275210aa0b1469e9', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	9.0	2019-10-16T14:35:47.244+01:00
1	0	5dc04a5873bfc341b54d1ab4	commandes	orders	2019-11-04T16:57:12.43+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc04a5873bfc34b684d1ab5', 'price': 4.7, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.7	2019-11-04T16:57:12.43+01:00
2	0	5dc04b7373bfc3b6304d1ab8	commandes	orders	2019-11-04T17:01:55.145+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc04b7373bfc3e9124d1ab9', 'price': 4.7, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.7	2019-11-04T17:01:55.145+01:00
3	0	5dc28ca7e17ef5373f8568da	commandes	orders	2019-11-06T10:04:39.676+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc28ca7e17ef533188568db', 'price': 4.9, 'product': 'Sandwich Alaska', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.9	2019-11-06T10:04:39.676+01:00
4	0	5dc28fc0e17ef5650a8568de	commandes	orders	2019-11-06T10:17:52.848+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc28fc0e17ef537b88568df', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	5.0	2019-11-06T10:17:52.848+01:00
5	0	5dc29474e17ef5a3ee8568e2	commandes	orders	2019-11-06T10:37:56.211+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29474e17ef5b0438568e3', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	5.0	2019-11-06T10:37:56.211+01:00
6	0	5dc29559e17ef5d68d8568e6	commandes	orders	2019-11-06T10:41:45.515+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29559e17ef5818d8568e7', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.0	2019-11-06T10:41:45.515+01:00
7	0	5dc296e3e17ef537588568ea	commandes	orders	2019-11-06T10:48:19.914+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc296e3e17ef537138568eb', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.0	2019-11-06T10:48:19.914+01:00
8	0	5dc29a50e17ef5d4bf8568ee	commandes	orders	2019-11-06T11:02:56.789+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29a50e17ef57a068568ef', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:02:56.789+01:00
9	0	5dc29c75e17ef57b818568f2	commandes	orders	2019-11-06T11:12:05.915+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29c75e17ef5952b8568f3', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:12:05.915+01:00
10	0	5dc29ee7e17ef57e458568f6	commandes	orders	2019-11-06T11:22:31.276+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29ee7e17ef576e58568f7', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:22:31.276+01:00
11	0	5dc29f9de17ef53a6f8568fa	commandes	orders	2019-11-06T11:25:33.607+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29f9de17ef592f48568fb', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:25:33.607+01:00
12	0	5dc2c1a4e17ef517148568fe	commandes	orders	2019-11-06T13:50:44.418+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc2c1a4e17ef53e178568ff', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.0	2019-11-06T13:50:44.418+01:00
13	0	5dc2e8e4e17ef557af856902	commandes	orders	2019-11-06T16:38:12.452+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc2e8e4e17ef56550856903', 'price': 4.7, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.7	2019-11-06T16:38:12.452+01:00
14	0	5dc3f53ee17ef5327f856906	commandes	orders	2019-11-07T11:43:10.902+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc3f53ee17ef540de856907', 'price': 4.9, 'product': 'Sandwich Alaska', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.9	2019-11-07T11:43:10.902+01:00
15	0	5dc3f8b8e17ef564d385690a	commandes	orders	2019-11-07T11:58:00.921+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc3f8b8e17ef5650a85690b', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-07T11:58:00.921+01:00
16	0	5dc3f992e17ef51c9a85690e	commandes	orders	2019-11-07T12:01:38.439+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc3f992e17ef54fed85690f', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-07T12:01:38.439+01:00
17	0	5dc40f71e17ef50570856912	commandes	orders	2019-11-07T13:34:57.432+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc40f71e17ef59cb8856913', 'price': 9, 'product': 'Salade Maison', 'quantity': 1}]	5d495147b25fcd46b5da1f1e	9.0	2019-11-07T13:34:57.432+01:00
18	0	5dc41224e17ef50322856916	commandes	orders	2019-11-07T13:46:28.253+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc41224e17ef56820856917', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	5.0	2019-11-07T13:46:28.253+01:00
19	0	5dc4163de17ef5a57085691a	commandes	orders	2019-11-07T14:03:57.072+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc4163de17ef5621a85691b', 'price': 10, 'product': 'Plat Escalop', 'quantity': 1}]	5d4999f45631f25421ca0052	10.0	2019-11-07T14:03:57.072+01:00
20	0	5dc418eee17ef50be185691e	commandes	orders	2019-11-07T14:15:26.275+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc418eee17ef57e1885691f', 'price': 10, 'product': 'Plat Escalop', 'quantity': 1}]	5d4999f45631f25421ca0052	10.0	2019-11-07T14:15:26.275+01:00
21	0	5dc419bee17ef5be05856920	commandes	orders	2019-11-07T14:18:54.966+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc419bee17ef528c1856921', 'price': 10, 'product': 'Plat Escalop', 'quantity': 1}]	5d4999f45631f25421ca0052	10.0	2019-11-07T14:18:54.966+01:00
22	0	5dc4227de17ef533d7856927	commandes	orders	2019-11-07T14:56:13.17+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc4227de17ef56c19856928', 'price': 10.9, 'product': 'Pizza Garden Special', 'quantity': 1}]	5d495029b25fcd39a8da1f19	10.9	2019-11-07T14:56:13.17+01:00
23	0	5dc9226ef7bf0195d3e113cb	commandes	orders	2019-11-11T09:57:18.803+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc9226ef7bf017cd8e113cc', 'price': 10, 'product': 'Plat Escalop', 'quantity': 1}]	5d4999f45631f25421ca0052	10.0	2019-11-11T09:57:18.803+01:00
24	0	5de133406b9f99eef914125d	commandes	orders	2019-11-29T16:03:28.691+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5de133406b9f995cfc14125e', 'comment': 'LES SAUCES ::Avec harissa  SOUHAITEZ VOUS RAJOUTER UN SUPPLEMENT ::Mais  a:a  ', 'price': 15, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	15.0	2019-11-29T16:03:28.691+01:00
25	0	5de7684cbd9e52fdcff91a3f	commandes	orders	2019-12-04T09:03:24.381+01:00	5d499c467cdaeefdb118637a	[{'_id': '5de7684cbd9e527954f91a40', 'price': 10, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	10.0	2019-12-04T09:03:24.381+01:00
26	0	5df7551e8586fd67417f7ce4	commandes	orders	2019-12-16T10:57:50.064+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5df7551e8586fd70027f7ce5', 'price': 10, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	10.0	2019-12-16T10:57:50.064+01:00
27	0	5df75dab8586fd50ce7f7ce6	commandes	orders	2019-12-16T11:34:19.711+01:00	5d499c467cdaeefdb118637a	[{'_id': '5df75dab8586fd40fc7f7ce7', 'comment': 'LES SAUCES ::Avec harissa Sans Tomate  SOUHAITEZ VOUS RAJOUTER UN SUPPLEMENT :: a: ', 'price': 10, 'product': 'Sandwich Zinger', 'quantity': 2}]	5d494f91b25fcd0b10da1f15	20.0	2019-12-16T11:34:19.711+01:00
28	0	5e45498bda0180223e0c17f7	commandes	orders	2020-02-13T14:05:15.863+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5e45498bda018038650c17f8', 'comment': 'choix:Avec harissa,    ', 'price': 10, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	10.0	2020-02-13T14:05:15.863+01:00
29	0	5e4e5d01da0180895b0c17fb	commandes	orders	2020-02-20T11:18:41.738+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5e4e5d01da0180cb8e0c17fc', 'comment': 'choix: Jambon  coca  ', 'price': 13, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	13.0	2020-02-20T11:18:41.738+01:00
30	0	5e4ff630da0180a7b20c17ff	commandes	orders	2020-02-21T16:24:32.592+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5e4ff630da018021bf0c1800', 'comment': 'choix:   ', 'price': 10, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	10.0	2020-02-21T16:24:32.592+01:00
31	0	5da718eb2752101e751469e6	commandes	orders	2019-10-16T14:19:39.913+01:00	5d499c467cdaeefdb118637a	[{'_id': '5da718eb275210e6351469e7', 'price': 4.8, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.8	2019-10-16T14:19:39.913+01:00
32	0	5da71d17275210efa51469eb	commandes	orders	2019-10-16T14:37:27.273+01:00	5d499c467cdaeefdb118637a	[{'_id': '5da71d17275210dc111469ec', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	5.0	2019-10-16T14:37:27.273+01:00
33	0	5dc04b1773bfc3d6f44d1ab6	commandes	orders	2019-11-04T17:00:23.531+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc04b1773bfc3738d4d1ab7', 'price': 4.7, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.7	2019-11-04T17:00:23.531+01:00
34	0	5dc146d173bfc344014d1aba	commandes	orders	2019-11-05T10:54:25.462+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc146d173bfc342094d1abb', 'price': 4.9, 'product': 'Sandwich Alaska', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.9	2019-11-05T10:54:25.462+01:00
35	0	5dc28db3e17ef55d928568dc	commandes	orders	2019-11-06T10:09:07.346+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc28db3e17ef570fd8568dd', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	5.0	2019-11-06T10:09:07.346+01:00
36	0	5dc28fdde17ef51e068568e0	commandes	orders	2019-11-06T10:18:21.993+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc28fdde17ef5da228568e1', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.0	2019-11-06T10:18:21.993+01:00
37	0	5dc294dfe17ef53ff28568e4	commandes	orders	2019-11-06T10:39:43.863+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc294dfe17ef56b8c8568e5', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	5.0	2019-11-06T10:39:43.863+01:00
38	0	5dc296bde17ef5356a8568e8	commandes	orders	2019-11-06T10:47:41.584+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc296bde17ef5c4d38568e9', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.0	2019-11-06T10:47:41.584+01:00
39	0	5dc29792e17ef586598568ec	commandes	orders	2019-11-06T10:51:14.466+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29792e17ef5b61c8568ed', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T10:51:14.466+01:00
40	0	5dc29c4fe17ef545d78568f0	commandes	orders	2019-11-06T11:11:27.081+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29c4fe17ef525c18568f1', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:11:27.081+01:00
41	0	5dc29d5fe17ef596fa8568f4	commandes	orders	2019-11-06T11:15:59.979+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29d5fe17ef5f6b98568f5', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:15:59.979+01:00
42	0	5dc29f0ae17ef569ac8568f8	commandes	orders	2019-11-06T11:23:06.967+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc29f0ae17ef5ba8c8568f9', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T11:23:06.967+01:00
43	0	5dc2a003e17ef5af618568fc	commandes	orders	2019-11-06T11:27:15.069+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc2a003e17ef576038568fd', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.0	2019-11-06T11:27:15.069+01:00
44	0	5dc2e863e17ef50ddf856900	commandes	orders	2019-11-06T16:36:03.807+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc2e863e17ef534cb856901', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-06T16:36:03.807+01:00
45	0	5dc3e5a7e17ef51d85856904	commandes	orders	2019-11-07T10:36:39.107+01:00	5dbc1c7db5ef1b7e87f99207	[{'_id': '5dc3e5a7e17ef5907b856905', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-07T10:36:39.107+01:00
46	0	5dc3f68ae17ef59728856908	commandes	orders	2019-11-07T11:48:42.919+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc3f68ae17ef570cf856909', 'price': 4.7, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.7	2019-11-07T11:48:42.919+01:00
47	0	5dc3f919e17ef50c9a85690c	commandes	orders	2019-11-07T11:59:37.721+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc3f919e17ef5606185690d', 'price': 4.7, 'product': 'Sandwich Tunisien', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	4.7	2019-11-07T11:59:37.721+01:00
48	0	5dc40497e17ef57e82856910	commandes	orders	2019-11-07T12:48:39.226+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc40497e17ef5cf2e856911', 'price': 10.9, 'product': 'Pizza The Hawaiian', 'quantity': 1}]	5d495029b25fcd39a8da1f19	10.9	2019-11-07T12:48:39.226+01:00
49	0	5dc4101ae17ef52bc5856914	commandes	orders	2019-11-07T13:37:46.555+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc4101ae17ef51785856915', 'price': 10, 'product': 'Plat Escalop', 'quantity': 1}]	5d4999f45631f25421ca0052	10.0	2019-11-07T13:37:46.555+01:00
50	0	5dc41290e17ef52deb856918	commandes	orders	2019-11-07T13:48:16.907+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc41290e17ef57c41856919', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	6.2	2019-11-07T13:48:16.907+01:00
51	0	5dc41702e17ef5123685691c	commandes	orders	2019-11-07T14:07:14.307+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc41702e17ef53b3c85691d', 'price': 4.5, 'product': 'Plat Tunisien', 'quantity': 1}]	5d4999f45631f25421ca0052	4.5	2019-11-07T14:07:14.307+01:00
52	0	5dc41b6ce17ef57987856922	commandes	orders	2019-11-07T14:26:04.627+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc41b6ce17ef59aca856923', 'price': 10, 'product': 'Plat Escalop', 'quantity': 1}]	5d4999f45631f25421ca0052	10.0	2019-11-07T14:26:04.627+01:00
53	0	5dc420d2e17ef5eef4856924	commandes	orders	2019-11-07T14:49:06.838+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc420d2e17ef50691856926', 'price': 10.9, 'product': 'Pizza The Hawaiian', 'quantity': 1}, {'_id': '5dc420d2e17ef55d32856925', 'price': 8.5, 'product': 'Pizza Cheese', 'quantity': 1}]	5d495029b25fcd39a8da1f19	19.4	2019-11-07T14:49:06.838+01:00
54	0	5dc92386f7bf014c2ee113cd	commandes	orders	2019-11-11T10:01:58.9+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dc92386f7bf01d033e113ce', 'price': 10, 'product': 'Pancake Fruits Secsn', 'quantity': 1}]	5d499a885631f22ea7ca0057	10.0	2019-11-11T10:01:58.9+01:00
55	0	5dcd73eaf7bf014aa0e113cf	commandes	orders	2019-11-14T16:34:02.971+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5dcd73eaf7bf011731e113d1', 'price': 5, 'product': 'Sandwich Zinger', 'quantity': 1}, {'_id': '5dcd73eaf7bf0175b8e113d0', 'price': 4, 'product': 'Sandwich Baguette XL', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	9.0	2019-11-14T16:34:02.971+01:00
56	0	5df7453c8586fdd72d7f7cdf	commandes	orders	2019-12-16T09:50:04.103+01:00	5d499c467cdaeefdb118637a	[{'_id': '5df7453c8586fd403d7f7ce1', 'comment': '', 'price': 1, 'product': 't', 'quantity': 17}, {'_id': '5df7453c8586fd95957f7ce0', 'comment': 'LES SAUCES ::Avec harissa Sans Tomate  SOUHAITEZ VOUS RAJOUTER UN SUPPLEMENT ::Jambon Mais  a:a  ', 'price': 21.5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	38.5	2019-12-16T09:50:04.103+01:00
57	0	5df746bd8586fdf9187f7ce2	commandes	orders	2019-12-16T09:56:29.033+01:00	5d499c467cdaeefdb118637a	[{'_id': '5df746bd8586fdc82a7f7ce3', 'comment': 'LES SAUCES ::Sans Tomate Sans laitue  SOUHAITEZ VOUS RAJOUTER UN SUPPLEMENT ::Jambon Mais  a: ', 'price': 11.5, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	11.5	2019-12-16T09:56:29.033+01:00
58	0	5e29acc29cb1c9382cfdb076	commandes	orders	2020-01-23T15:25:06.657+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5e29acc29cb1c93d33fdb077', 'comment': 'choix:Avec harissa,  Jambon,  coca,  ', 'price': 13, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	13.0	2020-01-23T15:25:06.657+01:00
59	0	5e45b819da018053be0c17f9	commandes	orders	2020-02-13T21:56:57.9+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5e45b819da0180e7730c17fa', 'comment': 'choix:Avec harissa    ', 'price': 10, 'product': 'Sandwich Zinger', 'quantity': 1}]	5d494f91b25fcd0b10da1f15	10.0	2020-02-13T21:56:57.9+01:00
60	0	5e4e9279da018076ff0c17fd	commandes	orders	2020-02-20T15:06:49.656+01:00	5db99cf0b4606a3c1cb0e4ac	[{'_id': '5e4e9279da0180c0300c17fe', 'comment': 'choix:   ', 'price': 6.2, 'product': 'Sandwich Philly Steak', 'quantity': 2}]	5d494f91b25fcd0b10da1f15	12.4	2020-02-20T15:06:49.656+01:00

order_df.groupby('customer').count()[['total']]


#order_df.groupby('customer').aggregate(lambda tdf: tdf.unique().tolist())



#dtaTest = order_df.groupby(order_df.version.apply(lambda x: x['product'])).size()

custommer_df.count()
custommer_df[['email']].count

order_df.groupby('customer').size()


#number_cmd2=order_df.groupby('customer').apply(datCmd())


#froupCustom = custommer_df.groupby("_id")['firstName']
#froupCustom.head()


order_df.orderItems[0]


#custommer_df.dtypes

from collections import defaultdict

from collections import Counter
from itertools import chain

results = defaultdict(lambda: defaultdict(dict))

#order_df.groupby('customer').agg({
#        'orderItems.product':sum
#        
#        })


    
    
dframeNew = order_df.append(custommer_df)


#dframeNew = order_df.groupby('orderItems')['customer']

print(dframeNew)
#
#c = Counter(chain.from_iterable(dataPlats))
#
#for k,v in c.items():
#    print(k, v)

      #  print(j['product'])



result = pandas.concat([order_df, custommer_df], axis=1, join='inner')



S = {x**2 for x  in range(10)}