from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Input csv file in it
def process_data(file_path):
    df = pd.read_csv(file_path)
    
    # Preprocessing datasets
    X = df.drop('target', axis=1) 
    y = df['target']  
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    return accuracy, f1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            accuracy, f1 = process_data(file_path)
            
            
            return jsonify({
                'accuracy': accuracy,
                'f1': f1,
                'status': 'Success'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        heart_rate = int(request.form['heart_rate'])
        blood_pressure = int(request.form['blood_pressure'])

        # Example prediction logic (simplified for demonstration)
        if heart_rate < 60 or heart_rate > 100:
            heart_status = "Abnormal"
        else:
            heart_status = "Normal"

        if blood_pressure < 90 or blood_pressure > 140:
            bp_status = "Abnormal"
        else:
            bp_status = "Normal"

        # Combine evaluation results
        overall_health = "Healthy" if heart_status == "Normal" and bp_status == "Normal" else "Needs Attention"

        # Return results
        return jsonify({
            'heart_status': heart_status,
            'bp_status': bp_status,
            'overall_health': overall_health
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
