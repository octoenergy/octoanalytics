"""
This module implements the main functionality of octorisk.

Author: Jean Bertin
"""

__author__ = "Jean Bertin"
__email__ = "jean.bertin@octopusenergy.fr"
__status__ = "planning"

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import holidays

def eval_forecast(data, country_code='FR'):
    """
    Forecast hourly electricity consumption using XGBoost and time-based calendar features.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe with at least two columns:
        - 'date' (datetime)
        - 'consumption' (float, the target variable)
    country_code : str
        ISO country code for holiday detection (e.g. 'FR' for France).

    Returns
    -------
    model : trained XGBRegressor model
    y_test_pred : predictions on the test set
    y_test_actual : actual values of the test set
    test_mape : MAPE on the test set
    y_val_pred : predictions on the validation set
    val_mape : MAPE on the validation set
    """

    # Ensure proper datetime format
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # Feature engineering: calendar-based features
    data['hour'] = data['date'].dt.hour
    data['dayofweek'] = data['date'].dt.dayofweek
    data['week'] = data['date'].dt.isocalendar().week.astype(int)
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Holiday feature (binary flag)
    holiday_dates = holidays.country_holidays(country_code)
    data['is_holiday'] = data['date'].dt.date.astype('datetime64').isin(holiday_dates).astype(int)

    # Define features and target
    features = ['hour', 'dayofweek', 'week', 'month', 'year', 'is_holiday']
    target = 'consumption'

    # Split the data: 60% train, 20% validation, 20% test
    train_data, temp_data = train_test_split(data, test_size=0.4, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    X_test, y_test = test_data[features], test_data[target]

    # Train the model using the validation set for early stopping
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Predictions and evaluation
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    print(f"Validation MAPE: {val_mape:.2f}%")
    print(f"Test MAPE: {test_mape:.2f}%")

    return model, y_test_pred, y_test, test_mape, y_val_pred, val_mape
