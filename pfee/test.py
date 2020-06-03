#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:58:29 2020

@author: oussama
"""
import pandas as pd
data = {'Name':['nv','hy'],
        'Number':[1,2]}
ds = pd.DataFrame(data)


def get_attributes(obj):
    return {
        attr: getattr(obj, attr) for attr in dir(obj) if 
        not attr.startswith('_') and not callable(getattr(obj, attr))
    }

print([
    get_attributes(row)
    for row in ds.itertuples()
])
ds.to_json(orient='records')
ds.to_json(orient='table')