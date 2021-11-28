#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 19:16:46 2021

@author: arun_singh-ggn
"""

import flask
import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

@app.route('/')
@app.route('/index')

def index():
    return flask.render_template('index.html')


def predictSalary(input):
    features = np.array(input).reshape(1,3)
    model = pickle.load(open("model.pkl", "rb"))
    result = model.predict(features)
    return result

@app.route('/result', methods=['POST'])

def result():
    if request.method == 'POST':
        input = request.form.values()
        input = list(map(float, input))
        result = predictSalary(input)
        
        return render_template("result.html", prediction="Estimated salary - "+str(result))
    
    
    
if __name__ == '__main__':
    app.run()    