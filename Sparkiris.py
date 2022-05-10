import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans as km
from sklearn import datasets

#to load the iris dataset
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.tail(5)) #to print the last 5 rows of dataset
print("\n")
x=df.iloc[:,[0,1,2,3]].values

#to plot the elbow graph
inertia_values=[]
for i in range(1,11):
  model=km(n_clusters=i, init='k-means++', max_iter=300, random_state=10)
  model.fit_predict(x)
  inertia_values.append(model.inertia_)
  
#plotting the results in a line graph
plt.plot(range(1,11), inertia_values, color="green")
plt.xlabel("No. of clusters")
plt.ylabel("SSE/Inertia")
plt.title("Elbow Graph")
plt.grid()
plt.show()
print("\n")

"""Since there's not much of change after 2 in the elbow graph we assume 3 clusters for K-means clustering in iris dataset"""
#creating a cluster
K=km(init='random',n_clusters=3,max_iter=300,random_state=10)
K.fit(x)
clusters=K.cluster_centers_
iden_clusters=K.fit_predict(x)

#plotting the graph to depict K-means clustering
plt.scatter(x[iden_clusters==0,0],x[iden_clusters==0,1],s=100,c='pink',label='Iris-setosa')
plt.scatter(x[iden_clusters==1,0],x[iden_clusters==1,1],s=100,c='purple',label='Iris-versicolor')
plt.scatter(x[iden_clusters==2,0],x[iden_clusters==2,1],s=100,c='blue',label='Iris-virginica')
plt.scatter(clusters[:, 0], clusters[:, 1], s=100, c='yellow',label='Centroids')
plt.legend()
plt.grid()
plt.show()
