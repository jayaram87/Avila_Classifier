from flask import Flask, request, render_template, redirect, url_for, flash
from flask_cors import cross_origin
import os, sys, pickle
from src.exception import CustomException
from src.pipeline.pipeline import Pipeline
from src.config.configuration import Configuration
from datetime import datetime
import pandas as pd
import numpy as np

ARTIFACT_DIR = os.path.join('avila_classifier', 'artifact')

app = Flask(__name__)
app.secret_key = "project"

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/train')
def train():
    message = ''
    pipeline = Pipeline(Configuration(current_time_stamp=f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'))
    if not Pipeline.experiment.status:
        message = 'Training started'
        pipeline.run()
    else:
        message = 'Training in progress'

    if os.path.exists(pipeline.experiment.file_path):
        experiment_file = pd.read_csv(pipeline.experiment.file_path)
        experiment_file = experiment_file[-5:]
    else:
        experiment_file = pd.DataFrame()
    
    data = {
        'message': message,
        'training_model_details': experiment_file.to_html(classes='table table-sm col-6', index=False)
    }

    return render_template('train.html', context=data)

@cross_origin
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        intercolumnar_distance = float(request.form['intercolumnar_distance'])
        upper_margin = float(request.form['upper_margin'])
        lower_margin = float(request.form['lower_margin'])
        exploitation = float(request.form['exploitation'])
        row_number = float(request.form['row_number'])
        modular_ratio = float(request.form['modular_ratio'])
        interlinear_spacing = float(request.form['interlinear_spacing'])
        weight = float(request.form['weight'])
        peak_number = float(request.form['peak_number'])
        ratio = float(request.form['ratio'])

        if not os.path.exists(os.path.join(ARTIFACT_DIR, 'experiment', 'experiment.csv')):
            flash(f'Model hasnt been trained yet, please train the model first')
            return redirect(url_for('home'))
        
        experiment_df = pd.read_csv(os.path.join(ARTIFACT_DIR, 'experiment', 'experiment.csv'))
        last_trained_df = experiment_df[experiment_df['model_accepted'] == 'Best model generated'].iloc[-1]
        
        if len(last_trained_df) == 0:
            return redirect(url_for('home'))
        
        with open(last_trained_df['model_file_path'], 'rb') as file:
            model = pickle.load(file)
        with open(last_trained_df['le_file_path'], 'rb') as file:
            le = pickle.load(file)
        values = np.array([intercolumnar_distance, upper_margin, lower_margin, exploitation, row_number, modular_ratio, interlinear_spacing, weight, peak_number, ratio]).reshape(1,-1)
        prediction = le.inverse_transform(model.predict(values))
        
        return render_template('result.html', context=prediction)
    
    return render_template('predict.html')


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        raise CustomException(e, sys) from e