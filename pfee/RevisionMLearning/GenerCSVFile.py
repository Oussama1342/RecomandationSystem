#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:21:19 2020

@author: oussama
"""
import random
import csv
#datalist =['value %d'% i for i in range(1,4)]
#
#
#for n in range(10):
#  print(random.randint(1,101))
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import secrets
from ast import literal_eval
from sklearn.cluster import KMeans

from scipy.spatial.distance import cosine
import re


## Recommandation Librairi

from scipy.spatial.distance import cdist, pdist


from surprise import SVD
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import math

from flask import Flask,render_template
import json
from flask import Flask
from flask_cors import CORS
app  = Flask(__name__)
cors = CORS(app, resources={r"/items/*": {"origins": "*"}})

#POur connaitre l'id de produits
#product = pd.DataFrame({'id_plat':[1,5,3,64,50,74,45,83,76,54,90,101,81,183,98],
#                        'name_plat':['Plat Escalope',' Plat Tunisien','Pancake Fruit Secsn','Sandwich Ayari','Salade Maison','Pizza The Hawaiian','Sandwich Zinger','SAndwich Philly Steak','Lablebi','ma9loub','Sandwich Baguette XL','Pizza Chees','Pizza Garden Specieal','sandwich tunisien','sandwich scalop']})
#
#




prodcut_list = pd.DataFrame(columns= ['customer',product['id_plat']])
producttest = [
               [{"product":"Plat Escalope", "quantity":1},{"product":"Plat Plat Tunisien","quantity":1}],
               [{"product" :"Pancake Fruit Secsn", "quantity":2},{"product":"Sandwich Ayari", "quantity":2}],
               [{"product":"'Plat Escalope'", "quantity":1}],
               [{"product":"Pizza The Hawaiian", "quantity":1},{"product":"Salade Maison", "quantity":1}],
               [{"product":"Pizza Garden Specieal", "quantity":2}],
               [{"product": "sandwich Ayari", "quantity":3}],
               [{"product": "Pizza Chees", "quantity":2}],
               [{"product":"Pizza Chees", "quantity":2},{"product":"Pizza The Hawaiian'", "quantity":2}],
               [{"product":"Plat Tunisien", "quantity":3}],
               [{"product":"Sandwich Baguette XL", "quantity":2},{"product":"SAndwich Philly Steak", "quantity":3}],
               [{"product":"Pancake Fruit Secsn", "quantity":3}],
               [{"product":"Sandwich Zinger", "quantity":4}],
               [{"product":"sandwich Scalope", "quantity":3}],
               [{"product":"Lablebi", "quantity":3}],
               [{"product":"Sandwich Zinger", "quantity":3},{"product":"SAndwich Philly Steak", "quantity":4}]
        ]
customer = [i for i in range(1,1000)]

"""Liste de platss"""
SandwichAyari = []
sandwichScalope = []
Lablebi = []
SandwichZinger = []
SandwichTuinisien = []
AndwichPhillySteak = []
SaladeMaison = []
plattunisien = []
platSclop = []
SandwichBaguetteXL = []
PancakeFruitSecsn = []
PizzaChees = []
PizzaGardenSpecieal = []
PizzaTheHawaiian = [] 

for i in range(1,1000):
    SandwichAyari.append(random.randint(0,1))
for i in range(1,1000):
    sandwichScalope.append(random.randint(0,1))
for i in range(1,1000):
    Lablebi.append(random.randint(0,1))
for i in range(1,1000):
    SandwichZinger.append(random.randint(0,1))
for i in range(1,1000):
    SandwichTuinisien.append(random.randint(0,1))
for i in range(1,1000):
    AndwichPhillySteak.append(random.randint(0,1))
for i in range(1,1000):
    SaladeMaison.append(random.randint(0,1))

for i in range(1,1000):
    plattunisien.append(random.randint(0,1))

for i in range(1,1000):
    PizzaTheHawaiian.append(random.randint(0,1))

for i in range(1,1000):
    PizzaGardenSpecieal.append(random.randint(0,1))

for i in range(1,1000):
    PizzaChees.append(random.randint(0,1))

for i in range(1,1000):
    PancakeFruitSecsn.append(random.randint(0,1))

for i in range(1,1000):
    SandwichBaguetteXL.append(random.randint(0,1))

for i in range(1,1000):
    platSclop.append(random.randint(0,1))

""""""
dataProduct= pd.DataFrame(
        {
                'customer':customer,
                'platScalop': platSclop,
                'Sandwich Baguette XL':SandwichBaguetteXL,
                'Pancake Fruit Secsn':SandwichBaguetteXL,
                'Pizza Chees': PizzaChees,
                'Pizza Garden Specieal':PizzaGardenSpecieal,
                'Pizza The Hawaiian': PizzaTheHawaiian,
                'Plat Tunisien': plattunisien,
                'SAlade Maison': SaladeMaison,
                'SAndwich Philly Steak': AndwichPhillySteak,
                'Sandwich Tuinisien' : SandwichTuinisien,
                'Sandwich Zinger' : SandwichZinger,
                'Lablebi':Lablebi,
                'sandwich Scalope' :sandwichScalope,
                'Sandwic hAyari' : SandwichAyari
                })
#    
    
dataProduct.to_csv(r'filecsv/dataProduct.csv', index= False)


liste =[]
dataPlats = pd.read_csv('filecsv/dataPlats.csv')


dataPlats = pd.read_csv('filecsv/dataPlats.csv')
dataPlatstest = dataPlats.drop(['customer'], axis = 1)

#def getcosune(x,y):
#    cos = 0
#    cos = sum(x*y) / math.sqrt(sum(x*x)) * math.sqrt(sum(y*y))
#    return cos
dataplat_ibs = pd.DataFrame(index = dataPlatstest.columns, columns=dataPlatstest.columns)

for i in range(0,len(dataplat_ibs.columns)):
    for j in range(0,len(dataplat_ibs.columns)):
        dataplat_ibs.ix[i,j] = 1 - cosine(dataPlatstest.ix[:,i], dataPlatstest.ix[:,j])

##Maintenant on peut chercher le voisin de chaque produit on tri d'ordre decroissant chaque colone et on choisi le meillaurs 10 elements
data_neighbours = pd.DataFrame(columns=range(1,7))
data_neighbours[1] = dataPlatstest.columns

    #data_neighbours[1] ==> Numero de produit
for i in range(0,len(dataplat_ibs.columns)):
    data_neighbours.ix[i,2:6] = dataplat_ibs.ix[0:,i].sort_values(ascending=True)[:5].index
 

for i in data_neighbours : 
    data_neighbours[i] = data_neighbours[i].astype(np.int64)

for i  in data_neighbours.head(1).ix[:0,2:6]:
    liste.append(i)

df = pd.DataFrame(columns=['idProduct'])    
df['idProduct']= data_neighbours.head(1)
    




 liste =  data_neighbours.head(2).ix[:0,2]

d = {'idproduit': data_neighbours.head(2).ix[:0,2]}

dfx = pd.DataFrame(data=d)
data_neighbours[5]
print(data_neighbours.loc[0])

recomandeddata = data_neighbours.head(6).ix[:6,1:4]
dftesMatrix = data_neighbours.to_numpy()


data_neighbours = data_neighbours.rename(columns = {2:'idProduit'})
#data_neighbours.loc[(data_neighbours[1]<2),[data_neighbours[2],data_neighbours[3]]]

df1 = df.head(2)

for i in df1 :
    for j in i :
        print(j)
        
data_neighbours[data_neighbours[1]<2]

data_neighbours.loc[1]
data_neighbours[1] = 'product'


liste = []
row = 3
for  row in data_neighbours.iterrows():
    for value in row :
        liste.append(value)


data_neighbours.head(1)



a = input('Entrz votre plats : ')
print(recomandeddata.loc[a,:])    

returnlistproduct(4)

from flask import Flask
from flask_cors import CORS
app  = Flask(__name__)

cors = CORS(app,ressources = {r'/items':{"origins": "*"}})


@app.route('/items')
def ItemBasedColl():
    dataPlats = pd.read_csv('filecsv/dataPlats.csv')
    dataPlatstest = dataPlats.drop(['customer'], axis = 1)
    
    dataplat_ibs = pd.DataFrame(index = dataPlatstest.columns, columns=dataPlatstest.columns)

    for i in range(0,len(dataplat_ibs.columns)):
        for j in range(0,len(dataplat_ibs.columns)):
            dataplat_ibs.ix[i,j] = 1 - cosine(dataPlatstest.ix[:,i], dataPlatstest.ix[:,j])
    data_neighbours = pd.DataFrame(index= dataplat_ibs.columns , columns=range(1,6))

    for i in range(0,len(dataplat_ibs.columns)):
        data_neighbours.ix[i,:5] = dataplat_ibs.ix[0:,i].sort_values(ascending=True)[:5].index
        d = data_neighbours.head(1).ix[:6,2:3]
    return d
#    return data_neighbours.head(1).ix[:6,2:4]

if __name__ == "__main__":
	app.run()    
    

df1 = ItemBasedColl()



"""Item Based compile with sucess"""














"""User Based"""
# Helper function to get similarity scores

def getScore(history, similarities):
    return sum(history*similarities)/sum(similarities)



data_sims = pd.DataFrame(index=dataPlats.index,columns=dataPlats.columns)
data_sims.ix[:,:1] = dataPlats.ix[:,:1]

for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        user = data_sims.index[i]
        product = data_sims.columns[j]
 
        if dataPlats.ix[i][j] == 1:
            data_sims.ix[i][j] = 0
        else:
            product_top_names = data_neighbours.ix[product][1:10]
            product_top_sims = dataplat_ibs.ix[product].sort_values(ascending=True)[1:10]
            user_purchases = dataPlatstest.ix[user,product_top_names]
 
            data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)


#We can now produc a matrix of User Based recommendations as follows:

data_recomanded = pd.DataFrame(index=data_sims.index, columns = ['user','1','2','3','4','5','6'])
data_recomanded.ix[0:,0] = data_sims.ix[:,0]
# Instead of top song scores, we want to see names
for i in range(0,len(data_sims.index)):
    data_recomanded.ix[i,1:] = data_sims.ix[i,:].sort_values(ascending=True).ix[1:7,].index.transpose()


print (data_recomanded.ix[:10,:4])

"""User Based COmpile with success"""



def userBasedColl():
    dataPlats = pd.read_csv('filecsv/dataPlats.csv')
    dataPlatstest = dataPlats.drop(['customer'], axis = 1)
    
    data_sims = pd.DataFrame(index=dataPlats.index,columns=dataPlats.columns)
    data_sims.ix[:,:1] = dataPlats.ix[:,:1]
    
    dataplat_ibs = pd.DataFrame(index = dataPlatstest.columns, columns=dataPlatstest.columns)
    data_neighbours = pd.DataFrame(index= dataplat_ibs.columns , columns=range(1,6))


    for i in range(0,len(data_sims.index)):
        for j in range(1,len(data_sims.columns)):
            user = data_sims.index[i]
            product = data_sims.columns[j]
            if dataPlats.ix[i][j] == 1:
                 data_sims.ix[i][j] = 0
            else:
                product_top_names = data_neighbours.ix[product][1:10]
                product_top_sims = dataplat_ibs.ix[product].sort_values(ascending=True)[1:10]
                user_purchases = dataPlatstest.ix[user,product_top_names]
                 
                data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)
    
    data_recomanded = pd.DataFrame(index=data_sims.index, columns = ['user','1','2','3','4','5','6'])
    data_recomanded.ix[0:,0] = data_sims.ix[:,0]
    for i in range(0,len(data_sims.index)):
        data_recomanded.ix[i,1:] = data_sims.ix[i,:].sort_values(ascending=True).ix[1:7,].index.transpose()
        
        
    return (data_recomanded.ix[:10,:4])

                 
userBasedColl()




userBasedColl()



"""Content Based filtering"""
a = input('Entrez votre plats : ')
patern = re.compile(a)
f = pd.read_csv('filecsv/dataProduct.csv')
for i in dataPlatstest.columns:
    if patern.match(i):
        print(i)
        
for i in f.columns:
  if a in i : 
      print('Vous povez aimez aussi ',i)
    
for i in dataPlats:
    print(i)






from flask.ext.api import FlaskAPI


app.route('/content')
def contentFiltring(x):
    listp = []
    dataPlats = pd.read_csv('filecsv/dataProduct.csv')
    for i in dataPlats.columns:
        if x in i :
            listp.append(i)
    return listp

b = contentFiltring(a)
print(b)
#for i in dataPlats.index:
#    for j in dataPlats.columns :
#        if dataPlats.ix[i][j]==1:
#            dataPlats.ix[i][j]=0

#from sklearn.neighbors import NearestNeighbors
#
#model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
#
#model_knn.fit(dataPlats)










"""Essay with KMEANS"""



K = range(1,15)
KM = [KMeans(n_clusters=k).fit(dataPlats) for k in K]

centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(dataPlats, cent, 'euclidean') for cent in centroids]

cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/dataPlats.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(dataPlats)**2)/dataPlats.shape[0]
bss = tss-wcss



kIdx = 10-1
# elbow curve

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()






kmeans = KMeans(n_clusters=100, init='k-means++', n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(dataPlats)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='green')


    



#quantity.append(np.random.randint(1,3,(1,10000))
##RandomQuantity
for x in range(1,100):
    quantitylist.append(random.randint(1,5))
   
#Random lis to product
for i in range(1,100):
    productRandom.append(random.choice(producttest))

##C reate Data
dataTraining = pd.DataFrame(
        {'customer':customer,
         'product':productRandom,
#         'quantity':quantitylist
})

#dataTraining.to_csv(r'filecsv/datatrainingcsv')

"""amelioration kmeans"""

lenthdata = len(dataPlats.index) -1
cluster  = y_kmeans[lenthdata]
data = []
list_data=[]
x=0
i=0
while (x <= lenthdata):
    i = i+1
    if(y_kmeans[x]== cluster):
        data.insert(i,x)
    x= x+1
datacluster = dataPlats.copy()
datacluster.drop(datacluster.index, inplace=True)
a= 0 

for i in data:
    a=a+1
    datacluster.loc[dataPlats.index[a]] = dataPlats.loc[i]


listOption=[]

for index, row in datacluster.iterrows():
    q=0
    for varlue in row:
        listOption.append(row.index[q])
        q=q+1
""""""



X = dataPlats.copy()     
        
y = dataPlats['customer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)


df_training = X_train.pivot(index='customer', columns='product', values='quantity')
df_training = df_training.fillna(0)

"""Weighted Average Approach"""
df_training_dummy = df_training.copy().fillna(0)

#Cosine Similarity 

Similarity_matrix = cosine_similarity(df_training_dummy,df_training_dummy)
similarity_matrix_df = pd.DataFrame(Similarity_matrix,
                                    index=df_training.index,
                                    columns=df_training.index)

def calculate_simularity(id_plats,id_user):
    cosine_scores=0
    rating_score=0
    if id_plats in df_training:
        cosine_scores = similarity_matrix_df[id_user]  #similarity of id_user with every other user
        #print(cosine_scores)
        rating_score= df_training[id_plats]
        #won't consider users who havent rated id_movie so drop similarity scores and ratings corresponsing to np.na
        index_notrated = rating_score[rating_score.isnull()].index
        rating_score = rating_score.dropna()
        cosine_scores = cosine_scores.drop(index_notrated)
        #calculating rating by weighted mean of ratings and cosine scores of the users who have rated the movie
        rating_plats = np.dot(rating_score,cosine_scores)/cosine_scores.sum()
    else:
        return 2.5
        return rating_plats
        

print(df_training.index[0])
print(df_training.columns[0])
calculate_simularity(df_training.columns[0],df_training.index[0])
"""Method 1 Cosine Semularity"""
"""KNN Methode"""
reader = Reader()
data = Dataset.load_from_df(dataTraining, reader)

knn = KNNBasic()
cross_validate(knn, data, measures=['RMSE', 'mae'], cv = 3)




svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'mae'], cv = 3)

trainset = data.build_full_trainset()

svd.predict(dataTraining['customer'][0], dataTraining['product'][1])


NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(dataPlats)




       
#print(random.choice(product))
#
#
#for i in range(10000):
#    pro.append('ok')
#
##productRandom = random.shuffle(product)
#
#literal_eval(product)
#print(np.random.randint(1,4,(1,100))) ## Quantite
#
##qua= np.random.randint(1,4,(1,10))
##qua = [np.random.randint(1,4,(1,10))]
#qua = [i  for i in range(1,10)]

X = dataPlats.copy()

y= dataPlats.index

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 

from sklearn.neighbors import KNeighborsClassifier
    
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred)


#dataTraining['customer']= customer
#dataTraining['product']= literal_eval(product)
#
#
#
#
#for i in range():
#    df.append(secrets.choice(product))



"""KMEANS """
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, random_state=0)
y_predict = kmeans.fit_predict(dataPlats)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red')


#mu = 0.5
#sigma = 0.1
#np.random.seed(0)
#
#X = np.random.normal(mu, sigma, (395,1))
#Y = np.random.normal((mu * 2), (sigma * 3), (395,1))
#
#plt.scatter(X,Y, color='g')


#df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

"""User Based """
dataPlatstest = dataPlats.drop('customer',1)
  
        

dataPlats.loc[(dataOLats.customer = 1), ['1','2']]

