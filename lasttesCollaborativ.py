
# --- Import Libraries --- #
import pandas as pd
from scipy.spatial.distance import cosine


data  = pd.read_csv('filecsv/datatest.csv')


 #Drop any column named "user"
data_germany = data.drop('user', 1)



# Create a placeholder dataframe listing item vs. item
#This DataFrame To store the simularite

data_ibs = pd.DataFrame(index=data_germany.columns,columns=data_germany.columns)

for i in range(0,len(data_ibs.columns)) :
    # Loop through the columns for each column
    for j in range(0,len(data_ibs.columns)) :
        data_ibs.ix[i,j] = 1 -cosine(data_germany.ix[:,1],data_germany.ix[:,j])
        
        

data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,11))


for i in range(0,len(data_ibs.columns)):
    data_neighbours.ix[i,:10] = data_ibs.ix[0:,i].order(ascending=False)[:10].index
    
    
#----And Item Based Recommandtion ----# 
    
    
    
# --- Start User Based Recommendations --- #
def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)

data_sims = pd.DataFrame(index=data.index,columns=data.columns)
data_sims.ix[:,:1] = data.ix[:,:1]
    
for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        user = data_sims.index[i]
        product = data_sims.columns[j]
 
        if data.ix[i][j] == 1:
            data_sims.ix[i][j] = 0
        else:
            product_top_names = data_neighbours.ix[product][1:10]
            product_top_sims = data_ibs.ix[product].order(ascending=False)[1:10]
            user_purchases = data_germany.ix[user,product_top_names]
 
            data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)

