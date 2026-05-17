from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_object

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    
    else :
        data = CustomData(
            gender=request.form['gender'],
            race_ethnicity=request.form['race_ethnicity'],
            parental_level_of_education=request.form['parental_level_of_education'],
            lunch=request.form['lunch'],
            test_preparation_course=request.form['test_preparation_course'],
            writing_score=float(request.form['writing_score']),
            reading_score=float(request.form['reading_score'])
        )
        
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=result[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')