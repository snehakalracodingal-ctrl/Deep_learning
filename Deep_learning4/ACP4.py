from random import Random

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


df = pd.read_csv('MARUTI.NS.csv')
user_y = input("What you want to predict?\n1. Open\n2. High\n3. Low\n4. Close\n5. RSI(Buy/Sell)\nInput: ")
column_map = {"1": "Open", "2": "High", "3": "Low", "4": "Close", "5": "RSI"}
user_y = column_map[user_y]
user_interval = int(input("Number of interval in which you want data to predict: "))
y = df.loc[user_interval:,user_y]
y.index=np.arange(0, len(y))
x = df.drop(['Date', user_y], axis=1).iloc[:-user_interval, :]

train_accuracy = []
test_accuracy = []
train_rsme = []
test_rsme = []
kfolds = []
kfold_accuracy = []

print(x)

def accuracy_score(y_train, y_pred_train, y_test, y_pred_test):
  print("Accuracy in Training: ", r2_score(y_train, y_pred_train)*100)
  train_accuracy.append(r2_score(y_train, y_pred_train)*100)
  print("Accuracy in Testing: ", r2_score(y_test, y_pred_test)*100)
  test_accuracy.append(r2_score(y_test, y_pred_test)*100)

def rsme(y_train, y_pred_train, y_test, y_pred_test):
  rsme_test = mean_squared_error(y_test, y_pred_test) ** 0.5
  rsme_train = mean_squared_error(y_train, y_pred_train) ** 0.5
  train_rsme.append(rsme_train)
  test_rsme.append(rsme_test)
  print("RSME Train: ", rsme_train)
  print("RSME Test: ", rsme_test)

def k_fold(algo, kx, ky, cv):
  kf = KFold(n_splits=cv, random_state=None)
  kfolds.append(cv)
  scores = []
  model = algo
  for train_index, test_index in kf.split(kx):
    X_train, X_test, Y_train, Y_test = kx.iloc[train_index], kx.iloc[test_index], ky[train_index], ky[test_index]
    model.fit(X_train, np.ravel(Y_train))
    Y_pred_train = model.predict(X_train)
    scores.append(r2_score(Y_train, Y_pred_train))
  print("Accuracy: {}".format(np.mean(scores)*100))
  kfold_accuracy.append(np.mean(scores)*100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
knn_model = KNeighborsRegressor(n_neighbors=7)
knn_model.fit(x_train, y_train)
y_pred_train_knn = knn_model.predict(x_train)
y_pred_test_knn = knn_model.predict(x_test)
accuracy_score(y_train, y_pred_train_knn, y_test, y_pred_test_knn)

rsme(y_train, y_pred_train_knn, y_test, y_pred_test_knn)

k_fold(KNeighborsRegressor(), x, y, 10)


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=10)
rf_model.fit(x_train, y_train)
y_pred_train_rf = rf_model.predict(x_train)
y_pred_test_rf = rf_model.predict(x_test)
accuracy_score(y_train, y_pred_train_rf, y_test, y_pred_test_rf)

rsme(y_train, y_pred_train_rf, y_test, y_pred_test_rf)

k_fold(RandomForestRegressor(), x, y.astype('int'), 10)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler().fit_transform(x)
y = np.array(y).reshape(-1, 1)
sc_y = StandardScaler().fit_transform(y)
x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(sc_x, sc_y, test_size = 0.3, random_state = 0)

from sklearn.svm import SVR
svr_model = SVR(kernel='rbf')
svr_model.fit(x_train_sc, y_train_sc[:,0])
y_pred_train_svr = svr_model.predict(x_train_sc)
y_pred_test_svr = svr_model.predict(x_test_sc)
accuracy_score(y_train_sc, y_pred_train_svr, y_test_sc, y_pred_test_svr)

rsme(y_train_sc, y_pred_train_svr, y_test_sc, y_pred_test_svr)

kf = KFold(n_splits=5, random_state=None)
kfolds.append(5)
scores = []
model = SVR(kernel='rbf')
for train_index, test_index in kf.split(sc_x):
    X_train, X_test, Y_train, Y_test = sc_x[train_index], sc_x[test_index], sc_y[train_index], sc_y[test_index]
    model.fit(X_train, np.ravel(Y_train))
    Y_pred_train = model.predict(X_train)
    scores.append(r2_score(Y_train, Y_pred_train))
print("Accuracy: {}".format(np.mean(scores)*100))
kfold_accuracy.append(np.mean(scores)*100)



from sklearn.linear_model import LogisticRegression
lgr_model = LogisticRegression(solver='liblinear')
lgr_model.fit(x_train, y_train.astype('int'))
y_pred_train_lgr = lgr_model.predict(x_train)
y_pred_test_lgr = lgr_model.predict(x_test)
accuracy_score(y_train.astype('int'), y_pred_train_lgr, y_test.astype('int'), y_pred_test_lgr)

rsme(y_train.astype('int'), y_pred_train_lgr, y_test.astype('int'), y_pred_test_lgr)

k_fold(LogisticRegression(solver='liblinear'), x, y.astype('int'), 5)



from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_train_lr = lr_model.predict(x_train)
y_pred_test_lr = lr_model.predict(x_test)
accuracy_score(y_train, y_pred_train_lr, y_test, y_pred_test_lr)

rsme(y_train, y_pred_train_lr, y_test, y_pred_test_lr)

k_fold(LinearRegression(), x, y, 10)



df_algo = pd.DataFrame({'Algorithm': ['KNN', 'Random Forest', 'SVR', 'Logistic', 'Multiple Linear'], 'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy, 'Train RMSE': train_rsme, 'Test RMSE': test_rsme, 'KFold': kfolds, 'Kfold Accuracy': kfold_accuracy})
df_algo.set_index('Algorithm', inplace=True)
print(df_algo)

df_algo[['Train Accuracy', 'Test Accuracy', 'Train RMSE', 'Test RMSE', 'Kfold Accuracy']].plot(subplots=False,figsize=(12,5))

df_model_best = df_algo.loc[((df_algo['Test Accuracy']>75) & (df_algo['Train Accuracy']>80)), ['Train Accuracy', 'Test Accuracy', 'Kfold Accuracy']]
print(df_model_best)

interval = int(input("Number of interval in which you want data to predict: "))
df_new = pd.DataFrame(columns=df.columns)
for i in range(0, len(df), interval):
  df_new = pd.concat([df_new, df.iloc[[i], :]], ignore_index=True)
print(df_new)



y_predict=rf_model.predict([df.drop(['Date', user_y], axis=1).iloc[-1, :].to_numpy()])
print(y_predict)