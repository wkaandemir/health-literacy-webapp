# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = 'ModelK-Nearest_Neighbors.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    gender = float(request.form['gender'])
    classs = float(request.form['classs'])
    mother = float(request.form['mother'])
    father = float(request.form['father'])
    live = float(request.form['live'])
    news = float(request.form['news'])
    health = float(request.form['health'])
    economy = float(request.form['economy'])

    input_data = np.array([age, gender, classs, mother, father, live, news, health, economy]).reshape(1, -1)

    prediction = model.predict(input_data)

    if prediction == 1:
        return render_template('result.html', prediction_text="Your Health Literacy is High")
    else:
        return render_template('result.html', prediction_text="Your Health Literacy is Low")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
