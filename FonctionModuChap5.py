

import math
     
def table_par_7():
    nb = 7
    i = 0 # Notre compteur ! L'auriez-vous oublié ?
    while i < 10: # Tant que i est strictement inférieure à 10,
        print(i + 1, "*", nb, "=", (i + 1) * nb)
        i += 1 # On incrémente i de 1 à chaque tour de boucle.
        
        
###########################


#def table(nb):
#    i=0
#    while i<10:
#        print(i + 1, "*", nb, "=", (i + 1) * nb)
#        i+=1
        
################Exempl3#########
def table(nb, max):
    i=0
    while i<max:
        print(i + 1, "*", nb, "=", (i + 1) * nb)
        i+=1
################Exempl4#########
def table(nb, max=20):
    i=0
    while i<max:
        print(i + 1, "*", nb, "=", (i + 1) * nb)
        i+=1
###################Exrmple5######
def fonc(a=1, b=2, c=3, d=4, e=5):
    print("a =", a, "b =", b, "c =", c, "d =", d, "e =", e) 

##########Surcharge#######333
def exemple():
    print("Un exemple d'une fonction sans paramètre")

def exemple(): # On redéfinit la fonction exemple
    print("Un autre exemple de fonction sans paramètre")   
    
#################Return###########33
def carre(valeur):
    return valeur * valeur    

a= carre(5)

val = lambda x:x*x

test = math.sqrt(5)

