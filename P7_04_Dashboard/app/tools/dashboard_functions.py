# -*- coding: utf-8 -*-


import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#metrics
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import fbeta_score, precision_score, recall_score
from tools.preprocess import cleaning
from tools.analysis_functions import *
from io import BytesIO
import joblib
import shap
shap.initjs()

path_df = 'Dashboard/app/saved_model_data/restricted_dataset'
df = pd.read_csv(path_df)
df = df.drop(["Unnamed: 0"], axis = 1)

#Load model
path_model = 'Dashboard/app/saved_model_data/model_final_recompiled.joblib.dat'
model = joblib.load(open(path_model,'rb'))
#data = cleaning(df)

#Load explainer:
file_path = 'Dashboard/app/saved_model_data/restricted_shap_values'
with open(file_path, 'rb') as f:
     shap_vals = pickle.load(f)

file_path = 'Dashboard/app/saved_model_data/restricted_model_explainer'
#model_explainer = pickle.load(open(file_path,'rb'))
with open(file_path, 'rb') as f:     
     model_explainer = joblib.load(f)
     
file_path = 'Dashboard/app/saved_model_data/restricted_expected_values'
with open(file_path, 'rb') as f:     
     exp_vals = pickle.load(f)
   
     
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()] 
pred_data = df.drop(['TARGET', 'ID'], axis = 1)
true_y = train_df["TARGET"]
labels = train_df["ID"]

predictions = model.predict(train_df.drop(["TARGET", "ID"], axis = 1))
probas_predictions = model.predict_proba(train_df.drop(["TARGET", "ID"], axis = 1))[:, 1]




def credit_score(y_true, y_pred):
    '''
    fonction de calcul du score métier pour le projet.

    Parameters
    ----------
    y_true : list de int, numpy array
        vrai valeurs de la prédiction pour la target y.
    y_pred : list de int, numpy array
        valeurs prédites par le modèle pour la target y.

    Returns
    -------
    cs : int
        valeur de la métrique métier à minimiser.

    '''

    # false positive
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    # false negative
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # true positive
    tp = ((y_pred == 1) & (y_true == 1)).sum()

    # true negative
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    cs = 10*fn + fp   
    return cs

def credit_metric(x, y):
    return 10*x+y

def st_shap(plot, height=None):
    '''
    fonction d'affichage pour les force plot dans le dashboard streamlit
    '''
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
def informations_data(datafr):
    '''
    Parameters
    ----------
    datafr : panda dataframe
        sur lequel on veut des infos.

    Print
    -------
    shape et head
    '''
    st.header("Description de nos données :")
    
    st.write('-'*100)
    st.write("nombre total de client : " + str(datafr.shape[0]) )
    st.write("nombre total de variables : " + str(datafr.shape[1]))
    st.write('-'*100)
    st.write('les premiers clients dans nos données :')
    st.dataframe(datafr.head())
    

def visualisation_distribution_target(datafr):
    '''Parameters
    ----------
    datafr : panda dataframe
        sur lequel on veut des infos.

    Returns
    -------
    pie chart plotly de la target y.

    '''
    
    target_distribution = datafr.TARGET.value_counts(dropna = False)
    labels = ['Pas de défaut', 'Défaut', 'Info non disponible']

    fig = go.Figure(data = [
        go.Pie(values = target_distribution, labels = labels, textinfo = 'label+percent+value')], 
         layout = go.Layout(title = 'Visualiser le déséquilibre des classes'))
    
    st.header("Camembert : Distribution de la Target")
    st.plotly_chart(fig)


def visualisation_univar(datafr):
    '''Parameters
    ----------
    datafr : panda dataframe
        sur lequel on veut des infos.

    Returns
    -------
    quelques graphs d'analyse univariée des colonnes du df.

    '''
    
    df_cat = datafr[["TARGET", "CODE_GENDER", "NAME_FAMILY_STATUS_Married"]].astype('category')
    df_num = datafr[["AMT_CREDIT", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE",
                     "PAYMENT_RATE"]]
    
    st.markdown('Nos trois variables catégorielles: défaut, sexe et statut marital :')
    for i in df_cat.columns :
        fig2 = plt.figure(figsize = (5,4))
        sns.countplot(x = i, data = df_cat)
        #st.pyplot(fig2)
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")
        st.image(buf2)
        
    
    st.markdown('Les distributions des données filtrées par variables sélectonnées :')
    for j in df_num.columns :
        #fig = plt.figure(figsize = (5,4))
        #sns.displot(df_num[j], color = 'black', rug = True)
        #st.pyplot(fig)
        #buf = BytesIO()
        #fig.savefig(buf, format="png")
        #st.image(buf)
        
        fig3 = plt.figure(figsize = (5,4))
        sns.boxplot(df_num[j])
        #st.pyplot(fig3)
        buf3 = BytesIO()
        fig3.savefig(buf3, format="png")
        st.image(buf3)
    


def interpretation_global(sample_nb):
    '''
    

    Parameters
    ----------
    sample_nb : int
        nombre d'individus à tirer aléatoirement pour visualisation shap.
        

    Returns
    -------
    Shap summary plot, decision plot, infos sur le modèle en format streamlit
    '''
                        
    
    auc_train_model = roc_auc_score(true_y, probas_predictions)
    
    st.write('-----------------------------------------------------')
    st.write('AUC pour toutes les données disponibles : '+str(auc_train_model))
    
    # ROC curve
    fpr_train_gbt, tpr_train_gbt, _ = roc_curve(true_y, probas_predictions)
    
    
    st.write('-----------------------------------------------------')
    st.write('Courbe ROC pour toutes les données disponibles')
    
    fig = plt.figure()
    plt.plot(fpr_train_gbt, tpr_train_gbt, color='blue', label='AUC_globale = %0.2f' % auc_train_model)
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), color='red')
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('Credit Default- Gradient Boosting')
    #plt.savefig('roc_curve.png', dpi=300)
    plt.show()
    st.pyplot(fig)
    
    st.write('---------------------------------------------------')

    st.write('Matrice de confusion simple pour les données disponibles : ' )

    tn, fp, fn, tp = confusion_matrix(true_y, predictions).ravel()
    st.write('positif : client fait défaut')
    st.write('négatif : client ne fait pas défaut')
    st.write("vrai negatif : ", tn, ", faux positif : ", fp, ", faux negatif : ", fn, ", vrai positif : ", tp)
    st.write("--------------------------------------------------")


    st.write('score crédit global des clients actuels : ', credit_metric(fn,fp))

    fbs_gbt = fbeta_score(true_y, predictions, beta=2)

    st.write('score f2 qui privilégie la détection de faux négatifs : ', fbs_gbt)

    auc_gbt = roc_auc_score(true_y, predictions)

    st.write('aire sous la courbe ROC (bon si proche de 1) : ', auc_gbt)

    st.write('Proportion de prédiction correcte parmis tout ce que le modèle prédit comme "mauvais clients" (vrais positifs / tous prédits positifs) :' , precision_score(true_y, predictions))

    st.write('Probabilité de détecter un vrai défaut (vrais positifs détectés / tous vrais positifs) :', recall_score(true_y, predictions))

    st.write('Proportion de prédiction correcte parmis tout ce que le modèle prédit comme "bon clients" (vrais négatifs détéctés / tous vrais négatifs) :' , tn/(tn+fp))
    
    
    st.write('-------------------------------------------------------')
    st.write("Explication globale du modèle avec SHAP:")
    
   
    #plot 1
    st.write("Classement et résumé global de l'importance des features pour le modèle d'après leurs influences respectives dans l'octroie de crédit des clients:")
    fig1 = plt.figure()
    sum_plot = shap.summary_plot(shap_vals, pred_data)
    st.pyplot(fig1)
    
    #plot 2
    st.write("Visualisation sous forme de bar plot:")
    fig2 = plt.figure()
    shap.summary_plot(shap_vals, pred_data, plot_type="bar")
    st.pyplot(fig2)
    
    
    #plot 3
    st.write("Description du processus de décision pour le sous-ensemble aléatoire de clients:")
    sub_sample = pred_data.sample(n = sample_nb)
    shap_values_sub = model_explainer.shap_values(sub_sample)
    fig4 = plt.figure()
    dec_plot_sample = shap.decision_plot(exp_vals.tolist(),
                               shap_values_sub,
                               features=pred_data,
                               highlight = [1]
                               )
    st.pyplot(fig4)
    
    
    #plot 4
    st.write("Influences respectives des features pour la décision d'octroyer le crédit aux clients du sous ensemble :")
    
    for i,j in enumerate(shap_values_sub):
        st.write('-------------------------------------------------------------------------')
        st.write('client aléatoire ' + str(i+1))
        fig = plt.figure()
        B_plot = shap.bar_plot(j, pred_data)
        st.pyplot(fig)
    
    
    
    
def interpretation_client(id_input):
    '''
    Fonction qui interpréte le score d'un client en utilisant SHAP'
    Parameters
    -------
    id_input : int
    numéro identification du client pour interprétation de sa prédiction par le modèle
    
    Returns
    -------
    série de graphiques et d'infos clients (SHAP)'

    '''
    
    data = df[df['ID']==int(id_input)]
    
    st.write('--------------------------------------------')
    st.write('caractéristiques du client sélectionné :')
    
    id_target_data = data[['ID','TARGET']]
    st.dataframe(id_target_data)
    
    individual_data = data.drop(['TARGET', 'ID'], axis = 1)
    st.dataframe(individual_data)
    
    shap_values = model_explainer.shap_values(individual_data)
    
    #choix variable:
    st.write('Valeurs des variables pour le client :')
    feature_values = dict(individual_data)
    
    option = st.selectbox(
    'Veuillez indiquez la variable à chercher : ',
    feature_values.keys())
    
    st.write(feature_values[option])
    
    st.write ('----------------------------------------------')
    st.write('Graphiques explicatifs de la prédiction pour le client (variables ayant le plus significativement contribué à la décision):')
    st.write ('----------------------------------------------')
    
    st.write('Contribution des variables principales à la prédiction pour ce client:')
    
    fig2 = plt.figure()
    dec_plot = shap.decision_plot(exp_vals.tolist(),
                               shap_values,
                               features=pred_data
                               )
    st.pyplot(fig2)
    
    st.write ('----------------------------------------------')
    st.write('Contribution des variables les plus imprtantes dans le classement du client :')
    fig = plt.figure()
    # Insert first SHAP plot here
    F_plot = shap.force_plot(model_explainer.expected_value,
                             shap_values,
                             individual_data)
    st_shap(F_plot, 150)
    #st.pyplot(fig)
    
    st.write ('----------------------------------------------')
    st.write("Top 7 des variables les plus imprtantes pour ce client (et sens de l'influence)")
    
    fig3 = plt.figure()
    B_plot = shap.bar_plot(shap_values[0], pred_data)
    st.pyplot(fig3)
    
    
    
    
    
    

    