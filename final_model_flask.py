# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import joblib

app = Flask(__name__,template_folder='tmp')
model=joblib.load('cat_boostmodel.pkl')


@app.route('/',methods=['GET'])
def home():
    return render_template('test.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    x_columns=['Name','Owner','Month_of_Purchase','Year_of_Purchase','Kilometers',
               'Rating','Fuel_Type','Transmission','RTO','Insurance_Type','Insurance']

    values=[[x for x in request.form.values()]]
    
    data=pd.DataFrame(values,columns=x_columns)
    dataset=data[['Name','Rating','Kilometers','Month_of_Purchase','Year_of_Purchase',
                  'Owner','Fuel_Type','Transmission','RTO','Insurance','Insurance_Type']]
    prediction=model.predict(dataset)[0]
    print(prediction)
    
    return render_template('test.html',prediction_text='predicted price is {}'.format(np.round(prediction)))
 
if __name__=='__main__':
    app.run(debug=True)