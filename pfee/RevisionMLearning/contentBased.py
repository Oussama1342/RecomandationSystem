#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:18:22 2020

@author: oussama
"""

import pandas as pd
from scipy.spatial.distance import cosine

from flask import Flask
app  = Flask(__name__)


@app.route('/content/<string:x>/')
def contentFiltring(x):
    listp = []
    dataPlats = pd.read_csv('dataPlats.csv')
    for i in dataPlats.columns:
        if x in i :
            listp.append(i)
    return listp[0]

if __name__ == "__main__":
	app.run()