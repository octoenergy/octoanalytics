<p align="center">
  <img src="images/logo_octoanalytics.png" alt="octoanalytics logo" width="200"/>
</p>

<h1 align="center">octoanalytics</h1>


[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**octoanalytics** is an Python package by **Octopus Energy** that provides tools for quantitative analysis and risk calculation on energy data. It helps to analyze time series energy consumption data, extract relevant features, and predict future consumption using machine learning models.

## Key Features

- **Time-based Feature Engineering**: Extract hourly, daily, and yearly features, as well as detect holidays using a calendar.
- **Forecasting Model**: Utilizes **XGBoost** regression models to predict hourly energy consumption.
- **Model Evaluation**: Computes **MAPE** (Mean Absolute Percentage Error) on the validation and test datasets.

## Installation

To install `octoanalytics`, you can use pip:

```bash
pip install octoanalytics
```

### Requirements

- Python 3.7 or higher
- `pandas`
- `numpy`
- `xgboost`
- `sklearn`
- `holidays`

These dependencies will be automatically installed when you install `octoanalytics`.

## Usage

### 1. Importing the package

To use `octoanalytics`, import the `eval_forecast` module as shown below:

```python
from octoanalytics import eval_forecast
```

### 2. Input Data Format

The data required for the function must be a DataFrame with the following columns:
- **'date'**: A column containing date-time values in `datetime` format.
- **'consumption'**: A column containing energy consumption values (the target variable).

Example of how the input data should look:

```python
import pandas as pd

data = pd.DataFrame({
    'date': ['2025-01-01 00:00', '2025-01-01 01:00', '2025-01-01 02:00', ...],
    'consumption': [120.5, 115.3, 113.7, ...]
})

data['date'] = pd.to_datetime(data['date'])
```

### 3. Main Function: `eval_forecast`

The `eval_forecast` function trains a machine learning model to forecast energy consumption using **XGBoost**. Here's how to use it:

```python
model, y_test_pred, y_test, test_mape, y_val_pred, val_mape = eval_forecast(data, country_code='FR')
```

#### Parameters

- `data` (pd.DataFrame): A DataFrame containing the columns `date` and `consumption`.
- `country_code` (str): The ISO code for the country to detect holidays (default is `'FR'` for France).

#### Return Values

- `model`: The trained XGBoost model.
- `y_test_pred`: The model's predictions on the test set.
- `y_test`: The actual values of the test set.
- `test_mape`: The **Mean Absolute Percentage Error (MAPE)** of the model on the test set.
- `y_val_pred`: The model's predictions on the validation set.
- `val_mape`: The **MAPE** of the model on the validation set.

### 4. Example Usage

```python
import pandas as pd
from octoanalytics import eval_forecast

# Example data (replace with your actual dataset)
data = pd.DataFrame({
    'date': ['2025-01-01 00:00', '2025-01-01 01:00', '2025-01-01 02:00'],
    'consumption': [120.5, 115.3, 113.7]
})
data['date'] = pd.to_datetime(data['date'])

# Run the forecast function
model, y_test_pred, y_test, test_mape, y_val_pred, val_mape = eval_forecast(data)

# Print the results
print(f"Validation MAPE: {val_mape:.2f}%")
print(f"Test MAPE: {test_mape:.2f}%")
```

## Detailed Description of `eval_forecast`

The `eval_forecast` function is used to train a forecasting model for energy consumption using the **XGBoost** algorithm. Here's how it works:

1. **Data Preprocessing**: The function extracts time-based features such as hour, day of the week, month, year, and week of the year. It also adds a binary feature indicating whether a given date is a holiday in the specified country.
   
2. **Data Splitting**: The data is split into three sets:
    - **Training set**: 60% of the data.
    - **Validation set**: 20% of the data.
    - **Test set**: 20% of the data.
   
3. **Training the XGBoost Model**: The model is trained on the training set, with early stopping based on validation data to prevent overfitting.

4. **Model Evaluation**: The **MAPE** (Mean Absolute Percentage Error) is computed on both the validation and test sets.

### XGBoost Model Parameters

- **n_estimators**: The number of boosting rounds (default is 100).
- **learning_rate**: The learning rate for adjusting tree weights (default is 0.1).
- **max_depth**: The maximum depth of the decision trees (default is 5).

These parameters can be adjusted by modifying the call to the `XGBRegressor` model in the `eval_forecast` function.

## Model Evaluation

The **MAPE** (Mean Absolute Percentage Error) is calculated on both the validation and test sets. It is expressed as a percentage and provides an indication of how well the model is performing. A lower MAPE value indicates better model performance.

## Developer

- **Author**: Jean Bertin
- **Email**: [jean.bertin@octopusenergy.fr](mailto:jean.bertin@octopusenergy.fr)
- **Status**: In development (planning)

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Contributions

Contributions are welcome! If you would like to suggest a feature or report a bug, please open an **issue** or submit a **pull request**.
