# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:02:46 2019

@author: Rituraj Saha
"""


# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset=pd.read_csv("googleplaystore.csv")
newdataset=dataset

# Fitting the Data
newdataset=newdataset.drop(columns="Genres") #removineg genres
newdataset=newdataset.drop(columns="Type") #removing type
newdataset=newdataset.drop(columns="Last Updated")  #removing last update
newdataset=newdataset.drop(columns="Android Ver") #removing android version


newdataset["Rating"].fillna(newdataset["Rating"].mean(), inplace=True) #filling nan with mean
newdataset.dropna(inplace=True) #deleting nan cols


newdataset['Installs'] = newdataset["Installs"].str.replace(",","") #extracting data
newdataset['Installs'] = newdataset["Installs"].str.replace("+","") #extracting data
newdataset['Size'] = newdataset["Size"].str.replace("M","") #extracting data
newdataset['Size'] = newdataset["Size"].str.replace("k","") #extracting data
newdataset['Size'] = newdataset["Size"].str.replace("Varies with device","0") #extracting data
newdataset['Price'] = newdataset["Price"].str.replace("$","") #extracting data

# Categorizing the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
newdataset["Category"]=le.fit_transform(newdataset["Category"])
newdataset["Content Rating"]=le.fit_transform(newdataset["Content Rating"])

x=newdataset.iloc[:,[1,3,4,5,6,7]].values
y=newdataset.iloc[:,2].values




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)


# Fitting Model to the Training set(RFR)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=5000,random_state=0)
regressor.fit(x_train, y_train)



# Predicting the Test set results
y_pred = regressor.predict(x_test)


# Analyzing the outut
from sklearn.metrics import mean_squared_error,r2_score 
print("mean_squared_error: ",mean_squared_error(y_test,y_pred)) 
print("r2_score: ",r2_score(y_test, y_pred)) 
print("score: ",regressor.score(x_test,y_test))







x=newdataset.iloc[:,[1,2,3,4,6,7]].values
y=newdataset.iloc[:,5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)


# Fitting Model to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Analyzing the outut
from sklearn.metrics import mean_squared_error,r2_score 
print("mean_squared_error: ",mean_squared_error(y_test,y_pred)) 
print("r2_score: ",r2_score(y_test, y_pred)) 







x=newdataset.iloc[:,[1,5]].values
# Fitting Model to the Training set(hierarchical clustering)
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(x_train)

# Making the dendogram
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("dendogram")
plt.show()








x=newdataset.iloc[:,[1,5]].values

# elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,21):
        km=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=2)
        km.fit(x)
        wcss.append(km.inertia_)

plt.plot(range(1,21),wcss)
plt.title("elbow")
plt.xlabel("clusters")
plt.ylabel("wcss")
plt.show()
        

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x= sc_x.fit_transform(x)


# Fitting Model to the Training set(kmeans)
km=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=2)
ykm=km.fit_predict(x)

# visualizing the clusters
plt.scatter(x[ykm== 0,0],x[ykm== 0,1],s=50,c="red",label="")
plt.scatter(x[ykm== 1,0],x[ykm== 1,1],s=50,c="blue",label="")
plt.scatter(x[ykm== 2,0],x[ykm== 2,1],s=50,c="yellow",label="")
#plt.scatter(x[ykm== 3,0],x[ykm== 3,1],s=50,c="green",label="")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=70,c="black",label="")
plt.xlabel("Category")
plt.ylabel("No. of Installs")
plt.legend()
plt.show()