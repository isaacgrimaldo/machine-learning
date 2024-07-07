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
from sklearn.impute import SimpleImputer;
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;
from sklearn.compose import ColumnTransformer;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler

# importacion de datos
data = pd.read_csv('data.csv');
X = data.iloc[:,:-1].values; # matriz
y = data.iloc[:, 3].values   # vetores


# tratamiento de datos
imputer = SimpleImputer(
    missing_values = np.nan,
    strategy = 'mean',
    fill_value=None,  
    );
imputer = imputer.fit(X[:, 1:3]);
X[:, 1:3] = imputer.transform(X[:, 1:3]);

# datos categoricos para los paises
labelencoder_X = LabelEncoder();
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]);

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                          # Leave the rest of the columns untouched
);
X = np.array(ct.fit_transform(X), dtype=np.float);

# tarea crear categorizacion de quienes compraron el producto
labelencoder_y = LabelEncoder();
y = labelencoder_y.fit_transform(y);

#Divir la data en conjuntos de entrenamiento y conjunto de testing
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size= 0.2, random_state = 0);


# escalador de datos
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train);
X_test  = sc_X.transform(X_test); 






