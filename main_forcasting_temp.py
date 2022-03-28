# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:40:28 2022

@author: Featherine
"""

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as md

import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('features.csv')
df = df.fillna(0)

# df = df[0:48]

df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')


# Get average per hour
df['hour'] = df['DateTime'].dt.hour
# df[['1006', 'Temp']] = df[['1006', 'Temp']].groupby(df['hour']).transform('mean')

df = df.drop(['DateTime'], axis=1)


y = np.array(df['1006'])
X = np.array(df['Temp']).reshape((-1,1))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,                                     
                                                    train_size = 0.7, 
                                                    test_size = 0.3, 
                                                    random_state = 10)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


degrees = [1, 2, 3, 6, 10, 15, 20]



y_train_pred = np.zeros((len(X_train), len(degrees)))
y_test_pred = np.zeros((len(X_test), len(degrees)))

for i, degree in enumerate(degrees):
    
    # make pipeline: create features, then feed them to linear_reg model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    # predict on test and train data
    # store the predictions of each degree in the corresponding column
    y_train_pred[:, i] = model.predict(X_train)
    y_test_pred[:, i] = model.predict(X_test)
    
    
    
plt.figure(figsize=(16, 8))

X_train = scaler.inverse_transform(X_train)
X_test = scaler.inverse_transform(X_test)

y_max = np.max(y_train)*1.1
y_min = np.min(y_train)*0.9

# train data
plt.subplot(121)
plt.scatter(X_train, y_train)
# plt.yscale('log')
plt.title("Training data")
for i, degree in enumerate(degrees):
        
    dummy = np.concatenate((X_train, y_train_pred[:, i].reshape((-1,1))), axis=1)
    dummy = pd.DataFrame(dummy)
    dummy = dummy.drop_duplicates(keep='last')
    dummy = dummy.sort_values(by=[0])
    plt.plot(dummy[0], dummy[1], label=str(degree))
    plt.legend(loc='upper left')
    plt.ylim([y_min, y_max])
    
# test data
plt.subplot(122)
plt.scatter(X_test, y_test)
# plt.yscale('log')
plt.title("Testing data")
for i, degree in enumerate(degrees): 
    
    dummy = np.concatenate((X_test, y_test_pred[:, i].reshape((-1,1))), axis=1)
    dummy = pd.DataFrame(dummy)
    dummy = dummy.drop_duplicates(keep='last')
    dummy = dummy.sort_values(by=[0])
    plt.plot(dummy[0], dummy[1], label=str(degree))
    plt.legend(loc='upper left')
    plt.ylim([y_min, y_max])
    
plt.savefig('Forcasting_Time.png')
    
# compare r2 for train and test sets (for all polynomial fits)
print("R-squared values: \n")

for i, degree in enumerate(degrees):
    train_r2 = round(sklearn.metrics.r2_score(y_train, y_train_pred[:, i]), 2)
    test_r2 = round(sklearn.metrics.r2_score(y_test, y_test_pred[:, i]), 2)
    print("Polynomial degree {0}: train score={1}, test score={2}".format(degree, 
                                                                         train_r2, 
                                                                         test_r2))
    
input_hour = 5
input_hour = np.array(input_hour).reshape((-1,1))
input_hour = scaler.transform(input_hour)
predict = model.predict(input_hour)
print('The predicted number of cars is: ' + str(predict))

