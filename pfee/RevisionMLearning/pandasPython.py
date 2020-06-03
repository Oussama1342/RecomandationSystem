#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:12:05 2020

@author: oussama
"""

"""
est une bibliothèque Python pour manipuler
simplement et efficacement des données structurées.

Pandas possede essentielement 3 structures de donnees :
    -Series
    -DataFrame
    -Panel
**Fonctionalites principales :
    -Recuperer les donnees de fichier CSV,EXCEL,des pages web....
    -Grouper ,decouper,alleger, deplacer, ecrire les donnees
    -Manipuler ces donnes qui peuvent etre a une ou 2 dimensions
    

"""
"""Python Tuto """
width = 10
width +=10
height =5 * 9
x = y  = z =0
tax = 0.125
ht = 100.50
ttc = ht *(1+tax)
round(ttc) #===> Transforme en decimal
a,b =5,4

###TYpe bolean 
bresult = width <10

###Comparaison de type d'un vatiable a untype donnee
isinstance(width, int)

isinstance(width, float)
###Comparaison de type d'un variable a un 
isinstance(width, (bool,float))

print('dosen\'t')   #### \ pour l'echapement
print("\"Yes,\" he said.")
print("""
    -Oussama
    -Amari      
""")


###Mode par defaut et mode brute
##Par defaut Mode
print('a\nb')
##Brut Mode   ==> r = raw
print(r'a\nb')


### Concatenation et duplication
#==>Concatenation with +, ===>Multiplication with *
word = 'Help '+'A'
phrase = word * 5


##Access a un caractere par index
print(word[5])
for i in word :
    print(i)
print(word[-3])

## Access par tranche 
    
print(word[0:2]) 
print(word[:2]) 
print(word[2:4]) 
print(word[3:]) 
########################Test d'apartenance

s = 'abcd'
print('a' in s)
print('d' in s)


#### les chaines sont immeutables ===> Impossible de modifier la chaine sans reconstruire la cgaine
""""""""""""""""""""""""""""""""""""""""Definition ==== >Liste"""""""""""""""""""""""""""""""""""""""
a = ['spam', 'eggs', 100, 1234]
##Acces par indice ou par tranche
print(a[1])
print(a[0:2])
print(a[1:])
print(a[1:3])
print(a[-4])

##Affectation de clonage
a= [0,1]
b = a    ####a et b sont lie a la meme objet
b[0] = 9
a[0] = 'oussama'  ## ===> Tout les modification se fait sur a et b
a[1]+=23

"""TRansformation """
a = ["O rage", "ô désespoir", "ô vieillesse ennemie"]
print('.'.join(a))

"""Decoupage"""
s = "La cigale et la fourmi"
s.split()
s = "et un ! et deux ! et trois zéros !"
s.split('!')
"""Liste imbriquees"""
q = [2,3]
p = [1,q,4]
print(p[1][0])
"""Ajout de liste """
q.append('xtra')
q[0]=4
"""Insertion """
q.insert(1,'python')
"""""""""""""""""""""""""""""""""""""""""""""""Tuple"""""""""""""""""""""""""""""""""
##Comme une liste mais immeable ==>pas de modification

t=(1,2,4,5,'a')
t[1]
t[1]=2
""""""""""""""""""""""""""""""""""""""""""""""""'Premier pas de programation Python'""""""""""""
i = 256*256
print('La valeur de i est ',i )

##########Condition If, elif, else
x  = input('SAisisser un nombre entier : ')
x= int(x)

if x < 0 :
    print(x,'est negative')
elif x > 0 :
    print(x,'est positive')
else:
    print(x,'est neutre')
###############Boucle For #############33
a = ['cat','window','snake']
for x in a:
    print(x,len(x))
################3##Boucle while############
mot_de_passe = 'python'
while(True):
    mot_saisi = input('Saisissez votre mot de passe :')
    if mot_saisi == mot_de_passe:
        break

"""###Iterateur"""
####################Range Function

for i in range(5):
    print(i)
    
#rang(stop), range(start, stop[,step])
for i in range(len(a)):
    print(a[i], len(a[i]))

for i in range(1,5,2):
    print(i)

##Continue
for num in range(0,6):
    if num % 2 ==0:
        continue
    print(num, 'est un nombre impaire')
##break et else dans un boucle
    for n in range(0,10):
        for x in range(2,n):
            if n % x == 0:
                print(n, '=', x,'*', n/x )
                break
        else:  ##Nombre premier
            print(n ,'est un nombre premier')
    

"""""""""""""""""Function """""""""
def soustrac(a,b):
    return a-b


soustrac(10,4) 
##Argument par defaut

def f(x,y=4):
    return x+y

f(2)
i = 5
def fuct(y=i):   ### Le variable par defaut e calcule au moment de declaration pas au moment de l'appel
    print(y)
i=6
fuct()

def functuinlist(a,L=[]):
    L.append(a)
    return L
functuinlist(6)

"""args et kwargs"""

def fargs(*args):
    print(args)   #### Print un tuple

fargs(1,2,'oussama',8,0)

#############Appel de fonction par 'depuillage de tuple'
def f(x,y):
    return x + y

t = (3,7)

f(*t)
###################3Appel de fonction par 'depuillage de dictionnaire'
d= {'x':'pissama','y':' amari'}

f(**d)



"""""""""""""""""""""""""""""""""""""""""""""Liste"""""""""""""""""""""
### Création par list comprehensions (listes en extension)

[x for x in range(5)]
[x**2 for x in range(5)]
[x*2 for x in range(5) if x != 3]

##Imbrication de list comprehensions
matrix = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],      
        ]

[[row[i] for row in matrix ] for i in range(4)]

###Suppression d'element
a = [-1, 1, 66.25, 333, 333, 1234.5]
del a[0]
del a[2:4]
del a[:]
#### tuple

t = 1,2   ### Le virgule qui creer un tuple pas le parenthese

"""""""""""""""Set"""""""""""

panier = {'pomme', 'orange', 'pomme', 'poire', 'orange', 'banane'}

###ou création par set comprehensions

a = {x for x in 'abracadabra' if x not in 'abc'}
panier.add('degle')


#Intersection d'ensembles
couleurs = {'bleu','jaune','orange'}
c= panier & couleurs
d =panier | couleurs ##union

"""""""""""""""""""""dictionnaire"""""""""""""""
## Creation
tel = {'jack': 4098, 'sape': 4139}
##Ajout

tel['guido']=1235
tel['jack']
##Suppression

del tel['jack']

'guido' in tel

tel.keys()
tel.values()

for key, value in tel.items():
    print(key , "his number is", value)
    
for key in tel :
    print(key)

{x:x**2 for x in (2,4,6)}
"""""""""""""""""""enumerate"""""""""""""""""""
###enumerate pour itérer sur le tuple (rang, valeur)
for i ,x in enumerate(['bordeaux','rennes','kkk']):
    print(i,x)
"""""""Zip"""""""
#Pour iterer sur plusieurs sequences a la fois
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']

for q,a in zip(questions,answers):
    print('whats your '+q+'? It is '+a)

"""Affectation multiple
"""
a,b,c =4,5,8
a,b = {3:9,10:20}
a,b,c = range(3)
"""Script"""
"""
Avec Python, il est possible de travailler de deux manières:
en mode intéractif (avec l'interpréteur Python)
en mode script

"""
"""Creation d'un module"""


""""""
"""Packages"""

"""str.format() avec {}"""
#Permet de mettre en forme une chaîne de caractères selon certains paramètres

print('We are th {} who say "{}!"'.format('engeiner', 'Python'))

##Specificaion de rang des argument a inserer

print('{0} and {1}'.format('spam','eggs'))

#str.format() avec {keyword}

print('the {food} is {adjective}'.format(
        food='spam', adjective='absolutely horrible'))

#et mélanger rang et mots-clés
print('the story of {0} et {1}, and {other} is {2}'.format('oussama', 'amari','famous',
      other = 'maissa'))

##str.format() et nombre de décimales
import math

print('{0:.3f}'.format(3.4973524738))
print('{0:.10f}'.format(math.pi))

#on peut specifier et le nombre minimal de caractères
table ={'oussama':4127,'amari':7836,'uirt':1234}
table['amari'] =26649644

for name, phone in table.items():
    print('{0:30}==>{1:30d}'.format(name, phone))


"""Erreurs et exceptions"""
"""
les erreurs : détectées dans la phase d'analyse du programme
 -erreurs de syntaxe
 -erreur d'indentation
les exceptions : surviennent pendant l'exécution d'un programme

"""
#Gestion des exceptions
while True:
    try:
        x =int(input('Veillez saisir un nombre : '))
        break
    except :
        print('Ce nest pas un nombre valide')
#else et le traitement des exceptions
try:
    x = 1
except : 
    print('erreure')
else:
    print('Pas grave')

"""as ====>pour recuperer l'objet exception"""

x = [6]
try:
    x[2]
except IndexError as exc:
    print('erreur index',exc)

## L'objet exc est une instance de la classe IndexError

"""raise"""
#raise pour déclencher des exceptions

raise NameError('HiThree')
"""finally"""
#finally pour exécuter un code qu'il y ait eu exception ou pas
#Si une exception n'a pas été gérée, elle est déclenchée après finally
x=[5]
try:
    print(x[2])
finally:
    print('Goodby, world')


##Exempl except practice
    
def devise(x,y):
    try:
        result= x/y
    except ZeroDivisionError:
        print('Division by Zero !!!')
    else:
        print('result is : ',result)
    finally:
        print('Executing finally clause')


devise(3,0)


"""Espaces de noms"""
#Permet de définir et d'utiliser le même nom dans des contextes différents
## Variable local et global
a,b = 1,2 ## ==>L'espace de nom global
def  f ():
    global b #===> b est global
    global a
    a= 3     ##=====> a est local dans f
    b = 99
    print(a)


f()

##Exempl 1f()
x = 0
def f(x):
    import Z
    
import X


from Y import A
###################3Exempl 2
a = 6
def f():
    b= 8
    def g(n):
        print(n)
        print(a)
        print(b)
    g(8)


####Bibliotheque Standard
    
###math
math.cos(math.pi/0.4)
math.log(1024, 2)

###random  
##pour string
fruit = (['apple','orange','banane'])
import random as r
r.sample(range(30,100),2)

r.random() #3Random float

r.randrange((7))  ##random integer chosen from range 6


########333dattime
import datetime
now = datetime.date.today()

now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B.")

birthday = date(1964, 7, 31)

nows  = datetime.datetime.now()

"""Function Lamba"""

#Fonctions anonymes, sur une seule ligne

f = lambda x : 2*x   ### ====> x == argument , 2*x contenu
gg = lambda x,y :(x+y)**2

"""tri la liste t"""
t = list('azerty')
t.sort(reverse=True)


t = list('AzeRty')
t.sort(key=str.lower)


##Tri d'une liste d'objets


class Z:
    def __init__(self, nom, prenom):
        self.nom = nom
        self.prenom = prenom
        
        
    def ident(self):
        






