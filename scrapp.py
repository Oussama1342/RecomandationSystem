# -*- coding: utf-8 -*-

#from elasticsearch import Elasticsearch
#es = Elasticsearch(HOST="http://localhost", PORT=9200)
#es = Elasticsearch()
#es.indices.create(index="firstPython", ignore=400)
import pandas as pd
a=1
    
    
    
if a > 0:
        print("a est positivs")
elif a < 0:
        print("a est negatives")
else:
        print("a est null")
        
age =21
majeur = False
        
    
if age >21:
    majeur = True
print(majeur)
print("************************************")
if a >=2:
    if a <=8:
        print("a dans l'intervalle")
    else:
        print("a n'est pas dans linterval")
else:
            print("a n'est pas dans linterval")
print("&&&&&&&&&&&&&&&&&&&&&&&&")

if a >=2 and a<=8:
    print("a est dans l'interval")
else:
    print("a n'est pas dans l'interval")
print("&&&&&&&&&&&&&&&&&&&&&&&&")

if a >=2 or a<=8:
    print("a est dans l'interval")
else:
    print("a n'est pas dans l'interval")