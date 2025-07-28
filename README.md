

<p align="center">
  <img src="images/logo_octoanalytics.png" alt="octoanalytics logo" width="200"/>
</p>

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# octoanalytics 📊⚡

**Energy consumption forecasting & risk premium analysis for the French electricity market**

---

## Description


`octoanalytics` is a Python toolkit for:

- Retrieving and smoothing **weather data** via Open-Meteo API.
- Training and evaluating **load forecasting models** (MW) using Random Forest.
- Generating **interactive visualizations** comparing actual vs. forecasted values.
- Accessing **spot and forward price data** (annual, monthly, PFC) from **Databricks SQL**.
- Computing **volume and shape risk premiums**, key for energy portfolio management.

---

## Installation

```bash
pip install octoanalytics
```

> ⚠️ You need a valid Databricks access token to retrieve market data.

---

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `plotly`
- `matplotlib`
- `tqdm`
- `requests`
- `tentaclio`
- `yaspin`
- `holidays`
- `dotenv`
- `databricks-sql-connector`

---

## Main Features

### 🔁 Weather data retrieval and smoothing

```python
from octoanalytics import get_temp_smoothed_fr

temp_df = get_temp_smoothed_fr(start_date="2024-01-01", end_date="2024-12-31")
```

---

### ⚡ Load forecasting

```python
from octoanalytics import eval_forecast

forecast_df = eval_forecast(df=load_df(), temp_df=temp_df, cal_year=2024)
```

---

### 💰 Risk premium calculation

#### Volume Risk

```python
from octoanalytics import calculate_prem_risk_vol

premium = calculate_prem_risk_vol(forecast_df, spot_df, forward_df)
```

#### Shape Risk

```python
from octoanalytics import calculate_prem_risk_shape

shape_risk = calculate_prem_risk_shape(forecast_df, pfc_df, spot_df)
```

---

### 🔌 Databricks SQL connections

```python
from octoanalytics import get_spot_price_fr, get_forward_price_fr_annual

spot_df = get_spot_price_fr(token=DB_TOKEN, start_date="2024-01-01", end_date="2024-12-31")
forward_df = get_forward_price_fr_annual(token=DB_TOKEN, cal_year=2025)
```

---

## Package Structure

```
octoanalytics/
│
├── __init__.py
├── core.py              # Main logic
├── ...
```

---

## Author

**Jean Bertin**  
📧 jean.bertin@octopusenergy.fr

---

## License

MIT – free to use, modify, and distribute.

---

## Roadmap

- [ ] Add XGBoost model
- [ ] Load anomaly detection
- [ ] Flask REST API deployment
- [ ] Automatic PDF report generation

---

## Full Demo

To be included in `examples/forecast_demo.ipynb`.
