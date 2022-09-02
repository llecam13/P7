# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler

import os
import warnings
warnings.filterwarnings('ignore')

def cleaning(data):
    #sert à assurer que les ID et Targets
    #sont les deux dernières colonnes
    ID = data['ID']
    target = data['TARGET']

    df = data.drop(['TARGET', 'ID'], axis = 1)
    
    df['ID'] = ID
    df['TARGET'] = target
    
    return df

#if __name__ == "__main__":
 