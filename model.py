#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:22:13 2021

@author: arun_singh-ggn
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

import pickle

salary_data = pd.read_csv("salary_predict_dataset.csv")

map_exp = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
           'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15}

salary_data['experience'].replace(map_exp, inplace=True)

salary_data['experience'].fillna(0, inplace=True)

salary_data['test_score'].fillna(salary_data['test_score'].mean(), inplace=True)

salary_data['interview_score'].fillna(salary_data['interview_score'].mean(), inplace=True)

features_data = salary_data.iloc[:, 0:3]
X = features_data.to_numpy()

y = salary_data.iloc[:, -1]


model = LinearRegression()

model.fit(X, y)

#print('score', model.score(X, y))

#print('predict', model.predict([[0,2,1]]))

pickle.dump(model, open("model.pkl", "wb"))

