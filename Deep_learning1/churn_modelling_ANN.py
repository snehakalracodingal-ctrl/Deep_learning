
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

"""#### **3.2. Feature Selection**"""

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

df.shape

"""#### **3.3. Train Test Split**"""

y = df.pop('Exited')
X = df

X.shape

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""#### **3.4. Feature Scaling**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train