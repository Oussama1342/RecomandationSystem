#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 08:59:47 2020

@author: oussama
"""

from flask import Flask

app  = Flask(__name__)

@app.route('/')
def hellow():
    return 'Hellow World'









#
#
#df = pd.DataFrame([['A231', 'Book', 5, 3, 150], 
#                ],
#                   columns = ['Code', 'Name', 'Price', 'Net', 'Sales'])
#
#
#images = ['scalop.jpg']
#df['image'] = images
#
#def path_to_image_html(path):
#    return '<img src="'+ path + '" width="60" >'
#
#pd.set_option('display.max_colwidth', -1)
#
#HTML(df.to_html(escape=False ,formatters=dict(image=path_to_image_html)))
#import requests
#r = requests.get('http://api06.dev.openstreetmap.org/api/0.6/map?bbox=0.2,46.5,0.4,46.7')
#print(r)
