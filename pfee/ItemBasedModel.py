#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:04:40 2020

@author: oussama
"""

import numpy as np
from flask import jsonify

import pandas as pd
from scipy.spatial.distance import cosine

from flask import Flask
from flask_cors import CORS
app  = Flask(__name__)
cors = CORS(app, resources={r"/items/*": {"origins": "*"}})


import json

"""User in the first time"""
@app.route('/items', methods=['GET'])
def ItemBasedColl():
    liste :[]
    dataPlats = pd.read_csv('dataPlats.csv')
    dataPlatstest = dataPlats.drop(['customer'], axis = 1)
    
    dataplat_ibs = pd.DataFrame(index = dataPlatstest.columns, columns=dataPlatstest.columns)

    for i in range(0,len(dataplat_ibs.columns)):
        for j in range(0,len(dataplat_ibs.columns)):
            dataplat_ibs.ix[i,j] = 1 - cosine(dataPlatstest.ix[:,i], dataPlatstest.ix[:,j])
    data_neighbours = pd.DataFrame(index= dataplat_ibs.columns , columns=range(1,7))
    data_neighbours[1] = dataPlatstest.columns


    for i in range(0,len(dataplat_ibs.columns)):
        data_neighbours.ix[i,2:6] = dataplat_ibs.ix[0:,i].sort_values(ascending=True)[:5].index
    for i in data_neighbours : 
        data_neighbours[i] = data_neighbours[i].astype(np.int64)
    df = data_neighbours.head(6).ix[:6,1:7]
    ds = df.to_json(orient='records')
    #ds =df.to_json(orient='table')
    
    return ds
   # dfMat = data_neighbours.to_numpy()
   
    
    #l1 = df.values.tolist()


    #f= data_neighbours.head(1).ix[:6,1:5]
    #f =  data_neighbours.head(6).ix[:1,1:5]
    #return json.dumps(dfMat)
    #return df.to_json(orient='records')[1:-1]
    #return df 
    #return json.dumps(df)

   # return jsonify(username= ds.Name)
   # return json.dumps(l1)
    # return f.to_json(orient='records')[1:-1]


if __name__ == "__main__":
    print('==Running in debug mode ==')
    app.run(host='localhost',port=7000, debug=True)
    

        
    
    
"""User hase already reserved"""



    
