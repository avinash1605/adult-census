# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:26:46 2018

@author: 1628065
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('adult_data.csv')
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,14].values

#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_X = LabelEncoder()
X[:,1]=labenc_X.fit_transform(X[:,1])
X[:,3]=labenc_X.fit_transform(X[:,3])
X[:,5]=labenc_X.fit_transform(X[:,5])
X[:,6]=labenc_X.fit_transform(X[:,6])
X[:,7]=labenc_X.fit_transform(X[:,7])
X[:,8]=labenc_X.fit_transform(X[:,8])
X[:,9]=labenc_X.fit_transform(X[:,9])
X[:,13]=labenc_X.fit_transform(X[:,13])
labenc_Y = LabelEncoder()
Y=labenc_X.fit_transform(Y)
#removing missing values
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X)
X=imp.transform(X)

#Splitting data sets into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=0)#preferably random state value is used as 0
#SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#applying the GAUSSIAN BAYES
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() #WE CALCULATE / DETERMINE THE PROBABILITY OF TWO SO NO RANDOM STATE IS REQD
classifier.fit(X_train,Y_train)

#predict result
y_pred = classifier.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

from sklearn.metrics import accuracy_score
percentage_accuracy= accuracy_score(Y_test,y_pred)*100

