# -*- coding: utf-8 -*-

#Import
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

import sklearn
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pathlib import Path
import urllib
from urllib.request import urlopen
import json
import requests


from tools.preprocess import cleaning
from tools.dashboard_functions import *



def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


#functions and load
@st.cache #mise en cache de la fonction pour exécution unique
def load_data(PATH):
    data=pd.read_csv(PATH)
    return data

path_df = 'Dashboard/app/saved_model_data/restricted_dataset'
df = pd.read_csv(path_df)
df = df.drop(["Unnamed: 0"], axis = 1)

#Load variable descriptions:
path_desc = 'Dashboard/app/saved_model_data/desc_features.csv'
desc = load_data(path_desc)


liste_id = df['ID'].tolist()
data = cleaning(df)

#dashboard display with navbar:
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options = ["Home",
                   "Comprendre nos clients",
                   "Comprendre le modèle",
                   "Prédire et expliquer"],
        icons = ["house", "book", "bar-chart", "bullseye"],
        menu_icon = "cast",
        styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#FF6F61", "font-size": "25px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#6B5B95"},
    })
    
    #choix variable:
    st.write('Description des variables :')
    dict_desc = dict(zip(desc.Feature, desc.description))
    
    option = st.selectbox(
     'Veuillez indiquez la variable à expliquer :',
     dict_desc.keys())

    st.write(dict_desc[option])
    
    st.write("liste des id clients disponibles pour la version déployé de l'app (4000 clients sélectionné au hasard) :")
    st.write(liste_id)
    
    
    
if selected == "Home":

    #présentation du dashboard:
        st.title("Implémentez un modéle de scoring de credit")
        st.subheader(" Bienvenue sur le Dashboard ")

if selected == "Comprendre nos clients":
    st.title(f"Analyse exploratoire des données clients:")
    

    st.write("--------------------------------------------------")
    st.title("Observer les données non-filtrées :")    
    
    informations_data(df)
    visualisation_distribution_target(df)
    
    
    
    agree = st.checkbox('Observer quelques graphiques disponibles ?')
    if agree:
        visualisation_univar(df)
    
    st.write("--------------------------------------------------")
    st.title("Observer les données filtrées :")
        
    Gender = list(df['CODE_GENDER'].unique())
    
    marit_status = list(df['NAME_FAMILY_STATUS_Married'].unique())
    
    amount = df["AMT_CREDIT"]

        
    st.markdown("Filtres disponibles pour les données :")
        
    gender_choice = st.multiselect("sexe : F = " + str(1) + " ; M = " + str(0) , Gender, Gender)
        
        
    marital_choice = st.multiselect("Married = " + str(1) + " ; not Married = " + str(0) ,
                                                marit_status, marit_status)
        
        
    amount_credit = st.slider('AMT_CREDIT',
                                          float(amount.min()), float(amount.max()),
                                          (float(amount.min()), float(amount.max())),
                                          1000.0)
    
 
    #creates masks from the sidebar selection widgets
    mask_gender = df['CODE_GENDER'].isin(gender_choice)
    mask_marital = df['NAME_FAMILY_STATUS_Married'].isin(marital_choice)
    
        #get the parties with a number of members in the range of nb_mbrs
    mask_amount_credit = df['AMT_CREDIT'].between(amount_credit[0], amount_credit[1])
    #mask_amount_days_emp = df['DAYS_EMPLOYED'].between( amount_days_emp[0], amount_days_emp[1])
    
    #mask_amount_days_phone = df['DAYS_LAST_PHONE_CHANGE'].between(amount_days_phone[0], amount_days_phone[1])
    
    #mask_amount_pay_rate = df['PAYMENT_RATE'].between(pay_rate_choice[0], pay_rate_choice[1])

    
    # & mask_amount_days_emp & mask_amount_days_phone & mask_amount_pay_rate
    df_filtered = df[mask_gender & mask_marital & mask_amount_credit]

    
    st.write('----------------------------------------')
    st.write("tableau de données filtrées : ")
    #observation du dataframe:
        
    st.write('Description des variables :')
    #dict_df = dict(zip(df.columns, df.columns))
    
    Colonnes = list(df.columns)
    
    Col_choice = st.multiselect('variables à observer dans les données filtrées (pas plus de quelques unes) :',
                                Colonnes, default=("ID","TARGET"))
    
    st.dataframe(df_filtered[Col_choice])
    
    informations_data(df_filtered)
    visualisation_distribution_target(df_filtered)
    
    agree_2 = st.checkbox('Observer les changements ?')
    if agree_2:
        visualisation_univar(df_filtered)
    
                    
if selected == "Comprendre le modèle":
    st.title(f"Comprendre le modèle de score-crédit:")
    st.markdown(f"Informations sur le modèle choisie:")
    
    #st.write('Caractéristiques globales du modèle:')
    
    #nombre de client pour sub_sample:
    nb_input = st.text_input('Combien de clients voulez-vous tirer au sort pour observation ?', )
    
    if nb_input.isdigit():                  
        with st.spinner('Chargement des caractéristiques globales du modèle...'):
            interpretation_global(int(nb_input))
        st.success('Done!')
    else:
        st.write("Veuillez saisir un nombre (raisonnable).")
    
if selected == "Prédire et expliquer":
    st.title(f"Prédire et expliquer le risque de défaut d'un client:")
    st.markdown("Analyse des résultats de prédiction d'offre de crédit:")


    #choix du client:
    id_input = st.text_input('identifiant client:', )
    
    if id_input == '':
        st.write('Veuillez saisir un identifiant.')

    elif (int(id_input) not in liste_id):
        st.write('Veuillez vérifier si l\'identifiant saisie est correct.')
        st.write('Si oui, veuillez vérifier que les informations clients ont bien été renseigné. Pour rappel les champs à renseigner sont:')

        st.write(df.columns)
    
    #identifiant correct:
    elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API

    #Appel de l'API : 
        API_url = "https://llctest.herokuapp.com/" + str(int(id_input))
    
        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(API_url)

            API_data = json.loads(json_url.read())
            classe_predite = API_data['prediction']
            if classe_predite == 1:
                etat = 'client à risque'
                proba = 1-API_data['proba'] 
            else:
                etat = 'client peu risqué'
                proba = 1-API_data['proba'] 

            #affichage de la prédiction
            prediction = API_data['proba']
            classe = df[df['ID']==int(id_input)]['TARGET'].values[0]
            
            if np.isnan(classe):
                classe_reelle = "pas de données réelles pour ce client (client test)"
            else :
                classe_reelle = str(classe).replace('0.0', 'pas de défaut sur le crédit octroyé').replace('1.0', 'défaut sur le crédit octroyé')
                
            chaine_prediction = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut '
            chaine_realite = 'classe réelle : ' + str(classe_reelle)
            
        st.markdown(chaine_prediction)
        st.markdown(chaine_realite)
        
        #st.write('Caractéristiques locales pour le client considéré:')

        with st.spinner('Chargement des détails de la prédiction...'):
            interpretation_client(id_input)
        
   
        st.success('Done!')
