import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset=pd.read_csv(r'Machine Learning Models\csv\student_clustering.csv')

# number of row and col
print(dataset.shape)

# First five rows
print(dataset.head())

plt.scatter(dataset['cgpa'],dataset['iq'])
plt.show()

list=[]

for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit_predict(dataset)
    list.append(km.inertia_)

print(list)
plt.plot(range(1,11),list)
plt.show()

X = dataset.iloc[:,:].values
km = KMeans(n_clusters=4)
#cluster assign
Ymeans = km.fit_predict(X)

plt.scatter(X[Ymeans == 0,0], X[Ymeans == 0,1], color='blue')
plt.scatter(X[Ymeans == 1,0], X[Ymeans == 1,1], color='red')
plt.scatter(X[Ymeans == 2,0], X[Ymeans == 2,1], color='green')
plt.scatter(X[Ymeans == 3,0], X[Ymeans == 3,1], color='yellow')

plt.show()