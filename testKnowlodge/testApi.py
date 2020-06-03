#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:04:26 2020

@author: oussama
"""
from flask.ext.api import FlaskAPI
from flask import request

app = FlaskAPI(__name__)

@app.route('/getData/')
def getData():
    return {'name':'roy'}

if __name__=="__main__":
    app.run(debug=True)

    
    
