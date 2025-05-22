

<p align="center">
  <img src="images/logo_octoanalytics.png" alt="octoanalytics logo" width="200"/>
</p>

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**octoanalytics** is a Python package by **Octopus Energy** providing tools for quantitative analysis and risk calculation on energy data. It facilitates analyzing energy consumption time series, incorporating temperature data, forecasting consumption, retrieving market prices, and computing risk premiums.

---

## Key Features

- **Smoothed Temperature Retrieval**: Fetches hourly smoothed temperature data for major French cities and computes a national average.
- **Energy Consumption Forecasting**: Gradient Boosting model based on time features and temperature.
- **Interactive Plotting**: Visualize forecasts vs actual consumption with MAPE annotation.
- **Spot and Forward Price Data**: Functions to query EPEX spot prices and EEX forward prices from Databricks.
- **Risk Premium Calculation**: Computes risk premiums from forward price curves and forecast errors.
- **Data Preprocessing**: Automatic feature engineering, imputation, and scaling.

---

## Installation

To install `octoanalytics`, use:

```bash
pip install octoanalytics
```

Dependencies such as `pandas`, `numpy`, `scikit-learn`, `holidays`, `plotly`, `tqdm`, and `tentaclio` will be installed automatically.

---

## Usage

### Importing the package

```python
from octoanalytics import eval_forecast, plot_forecast, calculate_mape, get_temp_smoothed_fr, get_spot_price_fr, get_forward_price_fr, get_pfc_fr, calculate_prem_risk_vol
```

### Data format for forecasting

The input data should be a DataFrame with at least:

- A datetime column (default named `'datetime'`)
- A consumption column (default named `'consumption'`)

Example:

```python
import pandas as pd

data = pd.DataFrame({
    'datetime': ['2025-01-01 00:00', '2025-01-01 01:00', '2025-01-01 02:00'],
    'consumption': [120.5, 115.3, 113.7]
})

data['datetime'] = pd.to_datetime(data['datetime'])
```

### Forecasting energy consumption

Use the `eval_forecast` function to train and predict consumption:

```python
forecast_df = eval_forecast(data)
print(forecast_df.head())
```

This returns the test set with a `'forecast'` column containing predicted values.

### Plotting forecasts

Visualize actual vs predicted consumption with:

```python
plot_forecast(data)
```

### Calculate MAPE (Mean Absolute Percentage Error)

```python
mape_value = calculate_mape(data)
print(f"MAPE: {mape_value:.2f}%")
```

### Retrieve smoothed temperature data for France

```python
temp_df = get_temp_smoothed_fr('2025-01-01', '2025-01-31')
print(temp_df.head())
```

### Retrieve spot prices for French electricity market

Requires a Databricks token:

```python
spot_prices = get_spot_price_fr(token='your_token_here', start_date='2025-01-01', end_date='2025-01-31')
print(spot_prices.head())
```

### Retrieve forward prices and Price Forward Curve (PFC)

```python
forward_prices = get_forward_price_fr(token='your_token_here', cal_year=2026)
pfc = get_pfc_fr(token='your_token_here', cal_year=2026)
```

### Calculate premium risk volatility

```python
premium = calculate_prem_risk_vol(token='your_token_here', input_df=data, datetime_col='datetime', target_col='consumption', plot_chart=True, quantile=50)
print(f"Risk premium at 50th percentile: {premium}")
```

---

## Function Descriptions

### `eval_forecast(df, datetime_col='datetime', target_col='consumption')`

Trains a Gradient Boosting model using time features and smoothed temperature data to forecast energy consumption. Splits data into train/test sets and returns test data with forecasts.

---

### `plot_forecast(df, datetime_col='datetime', target_col='consumption')`

Plots interactive time series comparing actual consumption with forecasts, showing MAPE on the plot.

---

### `calculate_mape(df, datetime_col='datetime', target_col='consumption')`

Returns the MAPE between actual and predicted consumption using the forecasting model.

---

### `get_temp_smoothed_fr(start_date, end_date)`

Fetches hourly smoothed temperatures averaged over multiple major French cities.

---

### `get_spot_price_fr(token, start_date, end_date)`

Retrieves hourly spot prices for the French electricity market (EPEX) from Databricks.

---

### `get_forward_price_fr(token, cal_year)`

Fetches annual forward prices for French electricity (EEX) for a specified calendar year.

---

### `get_pfc_fr(token, cal_year)`

Retrieves and resamples hourly Price Forward Curve data for French electricity (EEX) for a specified calendar year.

---

### `calculate_prem_risk_vol(token, input_df, datetime_col, target_col, plot_chart=False, quantile=50)`

Calculates a risk premium based on forecast errors and forward price curves. Returns the premium value for the requested quantile and optionally plots the distribution.

---

## Author

- Jean Bertin  
- Email: [jean.bertin@octopusenergy.fr](mailto:jean.bertin@octopusenergy.fr)  
- Status: In development (planning)

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

## Contributions

Contributions are welcome! Please open issues or pull requests on GitHub for suggestions or bug reports.
