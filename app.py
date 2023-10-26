import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

#Load the model
model = pickle.load(open('models/reg_model.pkl','rb'))
scl = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    df = pd.DataFrame(data, index= [0])
    scaled_data = scl.transform(df)
    print(scaled_data)
    pred = model.predict(scaled_data)
    print(pred[0])
    return jsonify(pred[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_data = scl.transform(np.array(data).reshape(1,-1))
    print(scaled_data)
    pred = model.predict(scaled_data)
    return render_template("home.html", prediction_text = f"The predicted House Price is {round(pred[0],2)}" )

if __name__ == "__main__":
    app.run(debug=True)
