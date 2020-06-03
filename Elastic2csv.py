#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:19:29 2020

@author: oussama
"""

#
#import pandas as pd
#from pymongo import MongoClient
#import json
#
#def mongoimport(csv_path, db_name, coll_name, db_url='localhost', db_port=27000):
#    """ Imports a csv file at path csv_name to a mongo colection
#    returns: count of the documants in the new collection
#    """
#    client = MongoClient(db_url, db_port)
#    db = client[db_name]
#    coll = db[coll_name]
#    data = pd.read_csv(csv_path)
#    payload = json.loads(data.to_json(orient='records'))
#    coll.remove()
#    coll.insert(payload)
#    return coll.count()
#
#
#
#
#
#
#mongoimport('data.csv', testCSV, stages, localhost, 27017)
from elasticsearch import Elasticsearch
import csv

es = Elasticsearch([{u'host': u'127.0.0.1', u'port': 9200}])


res = es.search(index="oreders", body=
                {
                    "_source": ["createdAt", "customer","age","gender", "resto", "total","orderItems.product"],
                     "query": {
                      "match_all": {}
      }               
}, size=1000)

with open('filecsv/data100.csv', 'w') as f:  # Just use 'w' mode in 3.x
    header_present  = False
    for doc in res['hits']['hits']:
        my_dict = doc['_source'] 
        if not header_present:
            w = csv.DictWriter(f, my_dict.keys())
            w.writeheader()
            header_present = True


        w.writerow(my_dict)







#exporter = ElasticSearchExporter()
#exporter.export(dataFilePath + "results2.csv",
#                "localhost",
#                "performtracking",
#                "default",
#                "loadingTime:[10000 TO 100000] AND referrer:logon",
#                None,
#                100,
#                "all",
#                ",")