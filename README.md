# P7

Le projet consiste à construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique et 
un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle
et d’améliorer la connaissance client des chargés de relation client.

Le repo contient : - le code du dashboard déployé
                   - le code de l'API déployé
                   - les notebook, la note méthodologique et la présentation du traitement des données et de la construction du modèle machine learning de classification
                   - requirements.txt et README.md
                   
données kaggle : https://www.kaggle.com/competitions/home-credit-default-risk/data                   
      
Pour rester dans les limites de taille (utilisation gratuite de streamlit cloud et heroku), on a tiré au hasard 4000 clients 
et retiré quelques fonctionnalités du dashboard (comparaison avec les autres clients par exemple)

URL dashboard streamlit déployé (streamlit share) : https://llecam13-dashboard-dashboardappdashboard-withnavbar-fouu2s.streamlitapp.com

URL API flask déployé (heroku) : https://llctest.herokuapp.com

Outils techniques : - Python (pandas, numpy, matplotlib, scikit-learn, ...)
                    - Modèle XGBoost avec comparaison avec DummyClassifier, RandomForest, LightGBM et Logistic Regression
                    - Interprétation des résultats avec shap
                    - Flask et streamlit
                    - Déploiement heroku et streamlit cloud
                    - Format des données : csv, pandas dataframe, JSON
