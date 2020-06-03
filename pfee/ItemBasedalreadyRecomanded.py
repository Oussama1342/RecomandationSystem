#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:04:02 2020

@author: oussama
"""

import pandas as pd
from scipy.spatial.distance import cosine

from flask import Flask
app  = Flask(__name__)

@app.route('/its/<string:a>/')
def ItemBasedCollfirsttime(a):
    dataPlats = pd.read_csv('dataPlats.csv')
    dataPlatstest = dataPlats.drop(['customer'], axis = 1)
    
    dataplat_ibs = pd.DataFrame(index = dataPlatstest.columns, columns=dataPlatstest.columns)

    for i in range(0,len(dataplat_ibs.columns)):
        for j in range(0,len(dataplat_ibs.columns)):
            dataplat_ibs.ix[i,j] = 1 - cosine(dataPlatstest.ix[:,i], dataPlatstest.ix[:,j])
    data_neighbours = pd.DataFrame(index= dataplat_ibs.columns , columns=range(1,6))

    for i in range(0,len(dataplat_ibs.columns)):
        data_neighbours.ix[i,:5] = dataplat_ibs.ix[0:,i].sort_values(ascending=True)[:5].index
    #f= data_neighbours.head(1).ix[:6,1:5]
    f =  data_neighbours.loc[a,:]
    return f.to_json(orient='records')[1:-1].replace('},{', '} {')

if __name__ == "__main__":
	app.run()