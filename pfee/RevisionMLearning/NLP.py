#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:34:05 2020

@author: oussama
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset = pd.read_csv("filecsv/NLP_MLP_last.csv")

from collections import Counter

titres = [
        "PretzelBros, airbnb for people who like pretzels,raises  $2 million",
          "Top 10 reasons why Go is better than whatever language you use.",
          "Why working at apple stole my soul (I still love it though)",
          "80 things I think you should do immediately if you use python.",
          "Show HN: carjack.me -- Uber meets GTA"]

# Trouve les mots uniques de ce titre

mot_unique = list(set(" ".join(titres).split(" ")))


counter =Counter()
def make_matricx(titres, vocab):
    matrix = []
    for titre in titres : 
        #Compter chaque mot dans le titre et faites un dictionnaire
        counter = Counter(titre)
        # Transformez la dictionnaire en une ligne matricielle en utilisant le vocab
        row  = [counter.get(w,0) for w in vocab]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = mot_unique
    return df



print(make_matricx(titres,mot_unique))

import re

nv_titre = [re.sub(r'[^\w\s\d]','',h.lower()) for h in titres]

nv_titre = [re.sub("\s+", " ",h)for h in titres]

mot_unique = list(set(" ".join(nv_titre).split(" ")))

print(make_matricx(nv_titre, mot_unique))



#Lisez et dvisez le fichier de mots vides

with open("stop_words.txt", "r") as f :
    stopwords = f.read().split("\n")
    
    
    
##Faire remplacement

stopwords = [re.sub(r'[^\w\s\d]','',s.lower()) for s in stopwords]

mot_unique = list(set(" ".join(nv_titre).split(" ")))
# Supportez les mots vides de vocabulaire

mot_unique = [w for w in mot_unique if w not in stopwords]


print(make_matricx(nv_titre, mot_unique))










