#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:14:18 2020

@author: oussama
"""
import pandas as pd
from flask import Flask
from flask_cors import CORS
app  = Flask(__name__)
cors = CORS(app, resources={r"/justTEst/*": {"origins": "*"}})

@app.route('/justTEst', methods=['GET'])
def tedtService():
    cars = {'idPlat': [1,45,6,32,6],}
    df = pd.DataFrame(cars, columns = ['idPlat'])

    return df.to_json(orient='records')[1:-1]

if __name__ == "__main__":
	app.run()