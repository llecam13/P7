# -*- coding: utf-8 -*-


import flask
import json
import requests
#import streamlit as st

from tools.preprocess import cleaning
import pandas as pd
import xgboost
import pickle


#construction API Flask: api de prediction interrogé
#par le dashboard streamlit qui affichera les résultats
app = flask.Flask(__name__)
app.config["DEBUG"] = True


#page d'accueil du site
@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1> Bienvenu sur l'api Flask: projet 7 scoring credit! </h1>"


# endpoint: prédiction octroie de credit (oui, non pour un client)

#Load Dataframe
path_df = 'restricted_dataset'
df = pd.read_csv(path_df)
df = df.drop(['Unnamed: 0'], axis = 1)

#Load model
path_model = 'model_final.pickle.dat'
model = pickle.load(open(path_model,'rb'))

#@app.route('/api/credit/<ID>', methods=['POST'])

@app.route('/<int:ID>')
def predict_credit(ID):     
    
    #retourne pour un client si crédit est accordé
    #ID = int(request.args.get('ID'))
    data = cleaning(df)
    
    data_for_prediction = data[data['ID'] == ID].iloc[:,:-2]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    
    prediction = model.predict(data_for_prediction_array)
    proba = model.predict_proba(data_for_prediction_array)
    
    dictionnaire = {
        'individual_data' : data_for_prediction.to_json(),
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dictionnaire)

    return flask.jsonify(dictionnaire)

#si ce script est le script lancer:
if __name__ == "__main__":
    # url local: http://localhost:5000/ ou http://127.0.0.1:5000
    app.run()