# -*- coding: utf-8 -*-
"""
@author: Roshan Joe Vincent
"""

#Importing Libraries
import pandas as pd

#Loading Data from CSV file
dataset = pd.read_csv('Weather.csv')

#Check if table has missing values
pd.isnull(dataset).any(1).nonzero()[0]

#Drop rows that have missing values
dataset.drop(pd.isnull(dataset).any(1).nonzero()[0], inplace = True)

#Breaking down Independent and Dependent variables
X = dataset.iloc[: , 1:4].values #upperbound is omitted
y = dataset.iloc[:, 4].values.astype(int)
y_temp = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_test[0,0]=0.22;
X_test[0,0]=35;
X_test[0,0]=45;

# Perform min max Feature Scaling
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range=(0,1))
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)


#Fitting the Classifier
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting rain/no
y_pred = classifier.predict(X_test)

if y_pred[0]==1:
    print("Our system predicted today will be rain");
else:
    print("Our system predicted today will not be rain");
#Analysing our model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)