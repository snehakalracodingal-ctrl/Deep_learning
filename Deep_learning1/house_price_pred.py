import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# load dataset
df = pd.read_csv("housing_data.csv")
df.head()


y = df.pop('AboveMedianPrice')
x = df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


model = Sequential()
model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model_history=model.fit(X_train, y_train, batch_size = 10, epochs = 100)


model.summary()
Y_pred = model.predict(X_test)
Y_pred

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, Y_pred)

mae

"""### Model Accuracy = 1-0.30 = 0.70"""
