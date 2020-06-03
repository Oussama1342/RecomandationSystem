#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:42:54 2020

@author: oussama
"""
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()


text = ['London Paris London','Paris Paris London ']

count_matrix = cv.fit_transform(text)

print(cv.get_feature_names())
print(count_matrix.toarray())

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(count_matrix)


print(similarity_scores)

df = pd.DataFrame({'A': [1.1,2.4,5.9], 'B':[45,87,9], 'S':[12,0,3]})

df1 = pd.DataFrame({'id_Item':[12,43,543], 'A': [1.1,2.4,5.9]})

dfs = pd.merge(df,df1,on='A')
