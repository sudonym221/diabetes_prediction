from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    inputs = [float(request.form[feature]) for feature in features]
    prediction = model.predict([inputs])[0]
    return render_template('result.html', result='Diabetic' if prediction else 'Not Diabetic')

if __name__ == '__main__':
    app.run(debug=True)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
