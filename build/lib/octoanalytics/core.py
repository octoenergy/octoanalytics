"""
This module implements the main functionality of octoanalytics.

Author: Jean Bertin
"""

__author__ = "Jean Bertin"
__email__ = "jean.bertin@octopusenergy.fr"
__status__ = "planning"

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import holidays

def eval_forecast(data, country_code='FR'):
    # ... (prétraitement des données)

    # Définir les variables explicatives et la cible
    features = ['hour', 'dayofweek', 'week', 'month', 'year', 'is_holiday']
    target = 'consumption'

    # Diviser les données en ensembles d'entraînement, de validation et de test
    train_data, temp_data = train_test_split(data, test_size=0.4, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    X_test, y_test = test_data[features], test_data[target]

    # Entraîner le modèle avec la validation croisée pour le nombre optimal d'estimateurs
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    # Prédictions et évaluation
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    print(f"Validation MAPE: {val_mape:.2f}%")
    print(f"Test MAPE: {test_mape:.2f}%")

    return model, y_test_pred, y_test, test_mape, y_val_pred, val_mape



