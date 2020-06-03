#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:39:33 2020

@author: oussama
"""

"""
Pandas est une bibliotheque python pour manipuler efficacement des donnes structures

-Series
-DAtaFrame
-Panel

"""

"""Series"""
import pandas as pd
import numpy as np

s = pd.Series(np.random.randn(5))
#proprietes
s.index
s.values
s.abs
s.all
s.value_counts
s.dtype
s.dtypes
s.isnull
np.zeros(s)
type(s)
a = np.array([[1,2,3],[i for i in range(3)]])
asd = pd.Series(np.random.randn(3), index=['row 1','row2','row3'])
s.name = 'firsSerie'

d = {'A':200,'B':400,'C':600}
SerieDictionary = pd.Series(d)

"""Création d'une série temporelle"""
N = 100
s = pd.Series(np.random.randn(N), index=pd.date_range('2015-09-01', periods=N, freq='1D'))
"""Limitation de l'affichage"""
pd.set_option('max_rows', 6)

"""Indexing"""

s.iloc[0]
s[1]
"""Slicing"""

SerieDictionary.iloc[1:2]
SerieDictionary.loc['B':'C']
##2 premier valeur
SerieDictionary.iloc[:2]
SerieDictionary.iloc[-2:]
"""Tracé"""
s.plot()
s.plot(kind='bar')
s.plot(kind='scatter')


"""DataFrame"""
#Création à partir d'un dictionnaire de Series

d = {'c0': pd.Series(['A','B','C']),
     'c1': pd.Series([1,2,3,4])
     }

df = pd.DataFrame(d)
#Creation a partire de dictionnaire de liste
d = {
    'c0':['A','B','C','D'],
    'c1':[1,2,3,4]
     
     }
df1 = pd.DataFrame(d)

df1.info()

#Propriétés
df1.index

df1.columns

df1 = df1.rename(columns = {
        
        'c0':'c10',
        'c1':'c11'
        
        })

df.dtypes

#Indexing

df1['c10'][2]
df1.loc[1, 'c10']
df1.iloc[2,0]
#Pour sélectionner tous les colonnes d'une ligne donnée, plusieurs solutions sont possibles :
#ttranspose
df1.transpose()[2]
#loc
df1.loc[2, :]

#Les indexes sont sur l'axe nommé axis=0. C'est l'axe par défaut du DataFrame
#Les colonnes sont sur l'axe nommé axis=1.

"""Entre / Sortie"""
filename = 'jjjj'
df1.to_csv(filename)

##Fichier excel

df1.read_excel(filename)

df1.to_excel(filename)

with pd.ExcelWriter(filename) as writer :
    df1.to_excel(writer, 'sheet')

"""Fichiers JSON"""

df  = pd.read_json(filename)

df1.to_json


"""sort_index"""

df1.sort_index(0, ascending=True)
df1.sort_index(1, ascending=False)

"""sort_values"""
df1.sort_values(['c10','c11'], ascending=[False,True])
"""describe"""
#Afficher les stats (nombre d'éléments, moyenne, écart type (std=standard deviation), ...)


df1.T.describe()
df1.describe()


"""set_index"""
idx = pd.date_range("2010-01-01", "2011-12-31")
df10 = pd.DataFrame(np.random.randn(len(idx), 3), columns=list('ABC'))
df10['Datetime'] = idx
df10 =df10.set_index('Datetime')


"""shift"""
#Permet de décaler une série d'un certain nombre d'échantillons
idx1 = pd.date_range('2010-01-01', '2010-01-05')
idx = pd.date_range("2010-01-01", "2010-01-05")

df = pd.DataFrame({"E": [20.2,22.3,24.4,28.2,30]}, index=idx)

df['E_2'] = df['E'].shift()

"""value_counts"""
s = pd.Series(['a','b','a','c','d','d','d','d'])
s.value_counts()

indexs  = pd.DatetimeIndex(["2010-01-1", "2010-01-02", "2010-01-04", "2010-01-05", "2010-01-06", "2010-01-10"])
s = pd.Series(100*np.random.rand(len(indexs)), indexs)

s.name='value'; s.index.name = 'DateTime'

#ecart
ecarts = pd.Series(s.index) - pd.Series(s.index).shift()

ecarts.value_counts()

"""fillna"""
idx1 = pd.date_range("2010-01-01", "2010-01-05")
s = pd.Series([20.2,22.3,None,28.2,30], idx)

s.fillna(method='backfill')
s.fillna(method='ffill')

s = s.fillna('oussama')

np.random.seed(0)

"""resample"""
dx = pd.date_range("2010-01-01", "2010-01-05", freq="1H")

s = pd.Series(np.random.randn(len(idx)), index=idx).cumsum() + 100

#En prenant la valeur moyenne par jour
s.resample('1d').mean()

"""Fuseaux horaire"""
s = pd.Series(100*np.random.rand(len(idx1)), idx1)

s_Paris = s.tz_localize('Europe/Paris')
s_Paris.tz_convert('UTC')



"""concat / append"""

idx = pd.date_range('2000-01-01', '2015-12-31')
df = pd.DataFrame(index=idx)
df['Annee'] = df.index.map(lambda dt : dt.year)
df['Mois'] = df.index.map(lambda dt : dt.month)
df['Jour'] = df.index.map(lambda dt : dt.day)
df['JourSemaine'] = df.index.map(lambda dt: dt.weekday)

df_ven_13 = df[(df['JourSemaine'] == 4) & (df['Jour'] == 13)] 

df_ven_13.groupby('Annee')['jour'].count()




"""Panel"""
#Le Panel est une structure ressemblant à un DataFrame mais qui possède une 3ième dimension











