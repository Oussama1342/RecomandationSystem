#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:02:49 2020

@author: oussama
"""

import numpy as np


###Commencer avec Python #######

"""1/30  ==> Variables et fonctions"""
X, y  = 1,2
y = 2.5
##Arithmetique 
print(X+y)
print(X)
print(X-y)
print(X**y)
print(X*y)
###Comparaison 
print(X <= y)
print(X >= y)
print(X == y)
print(X != y)

##LOgique 
print(False & True)  #==> AND
print(False | True)  #==> OR
print(False ^ True)  #==> XOR

# String
prenon = "amari"


####Fonction 

f = lambda x, y: x**2 +y   ##==> Fonction mathematique
print(f(3, 5))

def e_potentiel( masse, hauteur,e_limit, g):
    E  = (masse * hauteur * g)## Variable interne
    return(E< e_limit)

   
    #print(E, 'Joules')
    
    
    
print('bonjour')

e_potentiel(80, 5,5000, 9.81)
"""3/30 ==> IF/ELSE, FOR ,While"""

x=1
y=2
def signe(x):
    if (x > 0)  :
        print(x,'positif')
    elif x==0 : 
        print(x, 'nul')
    
    else:
        print(x, 'negativ')

signe(0)

###Boucle For
for element in range(10,-10,-1):
    signe(element)
    
###Boucle While
x=0
while x<10:
    print(x)
    x+=1
 
#Practice

def fibonacc(n):
    i=0
    while i< n :
        print(i)
        i+=1
#    for i in range(n):
#        print(i)  
fibonacc(50)



"""4/30 ==> Structure de donnees"""
list_1 = [1,2,3,4,5,6,7]
ville = ['Paris', 'Berline', 'Londers']
list3 = [list_1,ville]

##tupple
tupl_1=(1,2,3,4,5,6)   ##On peut pas le modifier

##String
prenon = 'oussama'

#Indexing
print(ville[-2])

##Slicing   ===> list[debut:fin:pas]
print(ville[0:3:2])
print(ville[::-1])

###Action sur list

ville.append('Dublin')  ##Ajout un element a la fin de ,liste
ville.insert(2, 'Madrid') ##Insert un element dans un indice de liste
ville2 = ['Amestrdam','Rome']

ville.extend(ville2) ## Insert list a ala fin de liste actual
len(ville)
ville.sort()
ville.sort(reverse=True)
list_1.sort(reverse=True)### Trie d'un list

ville[3]='Paris'
ville.count('Paris')

#####Liste ave structure de controle
if 'Tunis' in ville:
    print('Oui')
else:
    print('Non')
    
for index, valeur in enumerate( ville):  ###enumerata ==>donne index e valeur d'un list
    print(index, valeur)
    

for a, b in zip(list_1, ville):   #### traiter 2 listes
    print(a,b)





fibnnacci2(10)


"""5/30 ==> Dictionnaire"""
traduction = {
        "chien":"dog",
        "chat": "cat",
        "souri": "mouse",
        "oiseau":"bird"
        }

inventaire = {
        "banane":5000,
        "pomme":2094,
        "pores":42930
        }

dictionnaire_3 ={
        "dict1":traduction,
        "dict2": inventaire
        }

parametrs = {
        "w1": np.random.randn(10,100),
        "b1": np.random.randn(10,1),
        "w2": np.random.randn(10,10),
        "b2": np.random.randn(10,1)
        
        }

inventaire.values()
inventaire.keys()
inventaire["abricots"] = 4902
inventaire.get('cerise',1)
inventaire.fromkeys(ville, 'defaut')

inventaire.pop('abricots')  ##Extraire vleur du dictionnaire
for k,v in inventaire.items():
    print(k, v)
##Exercice 
    
classeu={
        "positiv":[],
        "negative":[]
        }

def trier(classeu,nombre):
    if nombre > 0:
        classeu['positiv'].append(nombre)
    else:
        classeu['negative'].append(nombre)


trier(classeu,-2)



##Reponse practice
def fibnnacci2(n):
    a=0
    b=1
    listab = [a]
    while a<n :
        listab.append(a) 
        a,b = b,a+b
    return listab

print(fibnnacci2(1000))

##Practice dictionnaire
classeur = {
        "positif":[],
        "negatif":[]
        }

"""6/30 ==>LIste comprehension && Dictionnaire comprehension"""

#Liste Comprehension
list_2 = [i**2 for i in range(10)]

list3 = [[i+j for i in range(3)] for j in range(3)]

##Dictionnaire Comprehension

prename = ['oussama','Jean','Julie','Sophie']

dico = {k:v for k, v in enumerate(prename)}

dico.values()
dico.keys()


for k,v in dico.items():
    print(k, v)


ages =[24,63,45,23,]

dico_2={prenom:age for prenom, age in zip(prename, ages) if age> 20}
dico_2


#dictTest = {k:v for k in range(20)}
##Tuple Comprehension
tuple_1 = tuple((i**2 for i in range(10)))

tuple_1


"""Fonctions en python utilise en ML"""
x = -3
abs(x)
round(x)
list_1 = [0,2,39,5,-32]
max(list_1)
len(list_1)
lisr_2 = [True,True,False]
all(lisr_2)
any(lisr_2)


