#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:37:53 2020

@author: oussama
"""
#monStagaire = input("Quel est Votre nom: ")
#
#print("Merci ",monStagaire," D'avoire prendre cet formation" )

########Operateur video############
#Il existe 4 Operateurs en Python

#calcul = 11/2
#calcul = float(calcul)
#print("REsultat: ",calcul)
#
#etudiant = input("Quel est ta moyenne sur 20 : ")
#etudiant = int(etudiant)
#etudiant=(etudiant*100)/20
#
#
#print("Votre pourcentage est de ",etudiant," 1%")
##############Structure Conditionnele


############3Boucle en Python#######3

"""print("Je suis un boucle")
print("Je suis un boucle")
print("Je suis un boucle")
print("Je suis un boucle")
print("Je suis un boucle")

cpt = 0 

while cpt < 5:
    print("je suis un boucle")
    cpt+=1 """
    
#######33Gestion des erreus#######
"""numTel = input("Quel est votre numero du telephone : ")
try:
    numTel = int(numTel)
except:
    print("L e numero saisie est incorrecte ")
else:
    print("L e numero saisie est : ",numTel)
finally:
    print("FErneture de prgramme....")"""
    
    
################## Les fonctions et les methodes ################
    #Il existe 3 type de methodes : 
       #-Methode Standard : Fonction sur instance
       #-Methode Statique: Fonction independant mais attache a une classe
       #-Methode de classe: Fonction sur une classe
       
       
       #######1- Methode Standard==> Basee sur l'instance###########
class world:
    
    nour_prefere = "Formage"
    def __init__(self,pays,continent):   #==>self : Pour envoyer les informations sur le methode et dire estun methode statique
        self.pays = pays
        self.continent = continent
    def caracteristique(self,temperature):
        print("La {} est un pays : {}".format(self.pays,temperature))
      ################3Methode de classe##########3
    def change_nomprefer(cls,new_nour_pref):
         world.nour_prefere = new_nour_pref
    
    change_nomprefer = classmethod(change_nomprefer)


vwld = world("Tunisia","Afrique")
vwld.caracteristique("Froid")
print("@@@@@@@@@@@@@@@@@@@@")
print("Nourriture de francais est : {} ".format(world.nour_prefere))
world.change_nomprefer("Poulet")      
print("Nourriture de francais est : {} ".format(world.nour_prefere))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")

##############Lecture et ecriture de donnees en Python##############
             #1-Mode de traitement : r (lecture)
             #2 W : Ecriture avec emplacement ==> a (ecrire apres)

"""fichier = open("PAPA.txt","r")
contenu = fichier.read()
print(contenu)
ligne = fichier.readline()
print(ligne)
ligne = fichier.readline()
print(ligne)

fichier.close()"""

###############3Write Now##################
with open("PAPA.txt","w") as fichier:
    chif =15
    fichier.write(str(chif))
    fichier.write("\nJe suis un debutant")










