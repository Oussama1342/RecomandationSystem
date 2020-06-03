#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:44:19 2020

@author: oussama
"""

import numpy as np 
import pandas as pd

import re
from csv import writer
from csv import reader
import csv
import matplotlib.pyplot as plt
import seaborn as sbrn
import statsmodels as stat
from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import SGDRegressor 

#import seaborn as sns



"""
default_text = 'Gender'
text1 = 'age'

with open('filecsv/data3.csv', 'r') as read_obj, open('filecsv/outputfile.csv', 'w', newline='') as write_obj :
    
    csv_reader = reader(read_obj)
    csv_writer = writer(write_obj)
    for row in csv_reader:
        row.append(default_text)
        csv_writer.writerow(row)
        
"""
##Import Data
Dataset =pd.read_csv("filecsv/data3last.csv")
Dataset['orderItems']=Dataset['orderItems'].replace({'product':''}, regex=True)
Dataset['orderItems']= Dataset['orderItems'].map(lambda x: re.sub(r'\W+', '', x))


Dataset["age"] = Dataset["age"].fillna(-0.2)
bins = [18, 27, 45, np.inf]
labels = [ 'Student', 'employee', 'retraite']
Dataset['status'] = pd.cut(Dataset["age"], bins, labels = labels)

Dataset['createdAt'] = pd.to_datetime(Dataset['createdAt']).dt.strftime('%Y-%m-%d')

Dataset.insert(0, 'Ones', 1)  
Dataset = Dataset.drop(['customer'], axis = 1)
Dataset = Dataset.drop(['resto'], axis = 1)
Dataset = Dataset.drop(['total'], axis = 1)
Dataset = Dataset.drop(['createdAt'], axis = 1)
#Dataset = Dataset.drop(['age'], axis = 1)



#X = Dataset.iloc[[ ], [0,5,6]]

X = Dataset.iloc[:, [1,2]].values
Y = Dataset.iloc[:,0].values
x1=Dataset["age"]
x2=Dataset["gender"]
plt.scatter(x2,Y)

plt.hist(x1[:,0], bins=20)
plt.hist(Y[:,0], bins=20)
Dataset.head()

X=Dataset[["gender","status"]].values
Y=Dataset["orderItems"].values



fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(X[1], X[3], Y[0], c='r', marker='^')

 
ax.set_xlabel('Student')
ax.set_ylabel('Gender')
ax.set_zlabel('plats')
plt.show()

##########################333

X = np.hstack((X**2, X))
theta = np.random.randn(3,1)





##Convert to Matrix
"""X = np.matrix(X.values)
Y = np.matrix(Y.values)"""

# Visualisation de donnees
"""
sns.barplot(x="gender",y="age",data= Dataset)

sns.barplot(x="gender",y="orderItems",data= Dataset)"""


# Data cleaning






##Drop name of custommer




### convert Age Group to numeric value####


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labEncr_X = LabelEncoder()
X[:,0] = labEncr_X.fit_transform(X[:,0])
#X[:,1] = labEncr_X.fit_transform(X[:,1])

onehotEncr = OneHotEncoder(categorical_features=[0])
#onehotEncr = OneHotEncoder(categorical_features=[1])

X = onehotEncr.fit_transform(X).toarray()

labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

plt.scatter(X[:,2], Y, marker='o')


"""New"""

np.random.seed(0) 
poly_features = PolynomialFeatures(degree=2, include_bias=False) 
X = poly_features.fit_transform(X) 
model = SGDRegressor(max_iter=10000, eta0=0.001) 

model.fit(X,Y) 
print('Coeff R2 =', model.score(X, Y)) 

plt.scatter(X[:,4], Y, marker='o')
plt.scatter(X[:,0], model.predict(X), c='red', marker='+') 
























""""""
X.shape
Y.shape
Y = Y.reshape(X.shape[0],1)

plt.scatter(X[:,0],
            X[:,1],
            c= Y)

plt.hist(X[:,0], bins=20)
plt.hist2d(X[:,0], X[:,3])
plt.xlabel('longueur sepal')
plt.ylabel('largeur sepal')
plt.colorbar()


""""""




##Implement Lineaire Regression


from sklearn.linear_model import LinearRegression

modeleReg=LinearRegression()


modeleReg.fit(X,Y)

print(modeleReg.intercept_)

print(modeleReg.coef_)   ###Theta 1

print(modeleReg.predict(X))

modeleReg.score(X,Y)
RMSE=np.sqrt(((Y-modeleReg.predict(X))**2).sum()/len(Y))










































"""   Optimisation de regression Lineaire  """
 
z2 = np.matrix(modeleReg.coef_)
X2 = np.matrix(X)
Y2 = np.matrix(Y)
theta = np.matrix(np.array(modeleReg.coef_))

alpha = 0.1
iters = 1000

def model(X, theta):
    return X.dot(theta)   #"""ax+b"""



plt.scatter(X,Y)
plt.plot(X,model(X,theta), c='r')


#"""Creation COst function""""

def costfunction(X,y,theta):
    return 1/(2*len(y)) * np.sum((model(X, theta)-y)**2)
    

costfunction(X,Y,theta)

#"""Gradien Descendent""""


def grad(X,y,theta):
    m=len(y)
    return  1/m * X.T.dot(model(X,theta)-y)


def gradien_descent(X, y ,theta ,learning_rate, n_iteration):
    
    cost_history = np.zeros(n_iteration)
    
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X,y,theta)
        cost_history[i] = costfunction(X, y , theta)
        
    return theta,cost_history
g2, cost2 = gradien_descent(X, Y, theta, alpha, iters)


f = g2[0, 0] + (g2[0, 1] * X2)

































""" Classification Data using DEcision Tree"""

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


"""Vsislisation Decision Tree"""
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = Dataset.shape[1],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

""""""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')







##Applique les algorithmes ##############



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,  test_size = 0.2, random_state = 0)

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

y_pred = gaussian.predict(X_test)

acc_gaussian = round(accuracy_score(y_pred, X_test) * 100, 2)




#####################33

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, Y_test) * 100, 2)
print(acc_logreg)


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, Y_test) * 100, 2)
print(acc_svc)


###################################

# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(accuracy_score(y_pred, Y_test) * 100, 2)
print(acc_linear_svc)


































from sklearn.linear_model import LinearRegression






"""
regresseur = LinearRegression()
regresseur.fit(X_train,Y_train)

y_prediction = regresseur.predict(X_test)



mplt.scatter(X_train,Y_train, color = 'red')
mplt.plot(X_train, regresseur.predict(X_train), color='green')
mplt.title(' Rendment note sur munite de revision')
mplt.xlabel('minute passe a revizer ' )
mplt.ylabel('note en pourcentage')
mplt.show()"""













































#age_mapping = {'Student': 1, 'employee': 2, 'Adult': 3, 'Senior': 4}
#
#Dataset['AgeGroup'] = Dataset['AgeGroup'].map(age_mapping)


###Convert gender fieled to numeric value

#gender_mapping = {'H':1, 'F':0}
#
#Dataset['gender'] = Dataset['gender'].map(gender_mapping)



#############AAppliquer les differsnts Algorithmes #####################













































































































































#Dataset['createdAt'] = Dataset['createdAt'].dt.strftime('%m/%d/%Y')
#
#Dataset['createdAt']  = pd.to_datetime(Dataset['createdAt'] .dt.strftime('%Y-%m'))


#X = Dataset.iloc[:,-3:-2]
#
#
#lentghDataSet = len(X.index)
#
#
#bad_chars = [';', ':', '!', "*"] 
#
#Dataset['orderItems']=Dataset['orderItems'].map(lambda x: x.lstrip('{').rstrip('}'))
#Dataset['orderItems']=Dataset['orderItems'].replace({':':''}, regex=True)
#
#Dataset['orderItems']=Dataset['orderItems'].isalnum()
#
#
#
#i=0
#while i<4:
#    Dataset['orderItems']=Dataset['orderItems'].str.slice_replace(0, 1)
#    i=i+1
#
#Dataset['orderItems']=Dataset['orderItems'].astype(str).str[:-1]










#
#for val in Dataset:
#    #X[val] = X[val].translate(None, ''.join(bad_chars)) 
#
#    Dataset['orderItems']=Dataset['orderItems'].replace({':':''}, regex=True)
#    #X[val] = X[val].replace({'[':''}, regex=True)
#    #X[val] = X[val].replace({'{':''}, regex=True)
#    Dataset['orderItems']=Dataset['orderItems'].replace({'}':''}, regex=True)
#
#
#
#
#for i in X : 
#    X[i] = X[i].replace(i, '')
#
#X['orderItems']= X['orderItems'].str.replace(r'\^[a-zA-Z]a+','')
#
#X = X.replace(r'\D+', '', regex=True)
#Dataset = Dataset.replace(r'[<%]', '', regex=True)
#
#
#
#for val in X:
#    re.sub(r'\W', ' ', str(val))
#    val=val+1
#   
#
#re.sub('[^A-Za-z0-9]+', '', X['orderItems'])
#'HelloPeopleWhitespace7331'     
    



#X.to_csv(r'xtest.csv', index = False)

#X1 = pd.read_csv("xtest.csv")


#def preprocess_text(document):
#    
#    document = re.sub(r'\W', ' ', str(document))
#    return preprocessed_text
#
#
#plats = X.to_frame()
#from nltk.stem import WordNetLemmatizer
#
#q=0
#for i in X :
#    re.sub(r'\W', ' ', str(i))
#    q=q+1
    


    
    