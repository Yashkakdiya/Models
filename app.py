from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Pclass = float(request.form['Pclass'])
        Sex = float(request.form['Sex'])  # 0 or 1
        Age = float(request.form['Age'])
        SibSp = float(request.form['SibSp'])
        Parch = float(request.form['Parch'])
        Fare = float(request.form['Fare'])

        input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare]])
        prediction = model.predict(input_data)[0]

        result = "🎉 Survived!" if prediction == 1 else "💀 Did not survive."

        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction="⚠️ Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
