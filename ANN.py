import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df=pd.read_csv(r'Machine Learning Models\csv\Admission_Predict.csv')
# df.info()

df.drop(columns=['Serial No.'],inplace=True)
X=df[['GRE Score','TOEFL Score','CGPA']]
y=df.iloc[:,-1]

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=24)

print(X_test)
scaler= MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)

model = Sequential()
model.add(Dense(3, activation='relu', input_dim=3))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
# model.summary()

model.compile(loss='mean_squared_error', optimizer='adam') 
history=model.fit(X_test_scaled,y_train, epochs=100, validation_split=0.2)
print(history)

y_pred= model.predict(X_test_scaled)
print(r2_score(y_test,y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()