import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

dataset=pd.read_csv(r'Machine Learning Models\csv\shopping_data.csv')

# print(dataset.head())

#data of salary and spending
data = dataset.iloc[:,3:5].values
# print(data)

plt.figure(figsize=(10,5))
plt.title('Dendogram')
dend=shc.dendrogram(shc.linkage(data,method='ward'))
# plot denodogram
plt.show()

cluster= AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
# lables formed for the data
labels=cluster.fit_predict(data)

plt.figure(figsize=(10,7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()