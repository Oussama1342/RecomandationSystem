#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:53:10 2020

@author: oussama
"""

from FonctionModuChap5 import *

annee = input("Saisissez une anne:")
annee = int(annee)
bissextile= False

if annee % 400 == 0:
    bissextile = True
elif  annee % 100 == 0 :
    bissextile = False
elif annee % 4 == 0:
   bissextile = True
else:
   bissextile = False

if bissextile:
   print("L'annee saisi est bosecstle") 
else:
   print("n'est pas bisextile")



#def table_par_7():
#nb= 7 
#i=0
#while i< 10:
#     print(i+1,"*",nb,"=",(i+1) * nb)
#     i+=1
   
