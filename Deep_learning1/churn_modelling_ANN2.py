
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('Churn_Modelling.csv')
df.head()

df.info()

df.describe()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

df['Geography'] = lb.fit_transform(df['Geography'])
df['Gender'] = lb.fit_transform(df['Gender'])

df

df.info()

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

df.shape

y = df.pop('Exited')
X = df

X.shape

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout


classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 10))


classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))


classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])


model_history=classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


classifier.summary()
#total params are total number of weights and biases


Y_pred = classifier.predict(X_test)
Y_pred

Y_pred = (Y_pred > 0.5)

Y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
print(cm)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_pred,y_test)

print(score)
