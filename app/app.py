import flask
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import flask_monitoringdashboard as dashboard

with open(f'model/model_flask.pickle', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__, template_folder='templates')

dashboard.bind(app)

@app.route('/',methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = ['C_MNTH', 'C_WDAY', 
    'C_HOUR', 'C_WTHR', 'C_RALN', 'C_TRAF', 'V_TYPE', 'P_SEX','PP_CAR']).astype(object)      
    prediction = model.predict_proba(data_unseen)[:,1]
    text = str('Dado un accidente, probabilidad de que haya algún fallecido:')

    return render_template('index.html',result=prediction)


if __name__ == '__main__':
    app.run('0.0.0.0', port=80)
