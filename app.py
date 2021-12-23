import numpy as np
from flask import Flask, render_template, request
import pickle

model = pickle.load(open("knn_model.pkl", "rb"))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['Age'])
        sex = request.form.get('Sex')
        cp = request.form.get('ChestPainType')
        trestbps = int(request.form['RestingBP'])
        chol = int(request.form['Cholesterol'])
        fbs = request.form.get('FastingBS')
        restecg = request.form.get('RestingECG')
        maxheart = int(request.form['MaxHR'])
        exang = request.form.get('ExerciseAngina')
        oldpeak = int(request.form['Oldpeak'])
        slope = request.form.get('ST_Slope')
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, maxheart, exang, oldpeak, slope]])
        my_prediction = model.predict(data)
        if my_prediction > 0.5:
            my_prediction = 1
        else:
            my_prediction = 0
        return render_template('result.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
