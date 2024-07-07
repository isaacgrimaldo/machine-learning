# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Pre pantilla

#Como importar las librerias
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.impute import SimpleImputer

# importacion de datos
data = pd.read_csv('data.csv');
X = data.iloc[:,:-1].values; # matriz
y = data.iloc[:, 3].values   # vetores

# tratamiento de datos
imputer = SimpleImputer(
    missing_values = np.nan,
    strategy = 'mean',
    fill_value=None, 
    verbose=0, 
    copy=True
    );
imputer = imputer.fit(X[:, 1:3]);
X = [:, 1:3] = imputer.transform(X[:, 1:3]);
