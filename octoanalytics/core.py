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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import holidays
import matplotlib.pyplot as plt
import holidays
import requests
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import tentaclio as tio
import os
from dotenv import load_dotenv





def get_temp_smoothed_fr(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves smoothed hourly average temperatures across several major French cities.

    :param start_date: Start date (format 'YYYY-MM-DD')
    :param end_date: End date (format 'YYYY-MM-DD')
    :return: DataFrame with columns ['datetime', 'temperature']
    """
    # Selected cities to smooth data at a national scale
    cities = {
        "Paris": (48.85, 2.35),
        "Lyon": (45.76, 4.84),
        "Marseille": (43.30, 5.37),
        "Lille": (50.63, 3.07),
        "Toulouse": (43.60, 1.44),
        "Strasbourg": (48.58, 7.75),
        "Nantes": (47.22, -1.55),
        "Bordeaux": (44.84, -0.58)
    }

    city_dfs = []

    for city, (lat, lon) in tqdm(cities.items(), desc="Fetching city data"):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "Europe/Paris"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame({
                'datetime': data['hourly']['time'],
                city: data['hourly']['temperature_2m']
            })
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            city_dfs.append(df)
        except Exception as e:
            print(f"Error with {city}: {e}")

    # Merge all city data and compute the mean temperature
    df_all = pd.concat(city_dfs, axis=1)
    df_all['temperature'] = df_all.mean(axis=1)

    # Return only datetime and the averaged temperature
    return df_all[['temperature']].reset_index()


def eval_forecast(df, datetime_col='datetime', target_col='consumption'):
    # 1. Standardize date formats and clean NaN values
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=[datetime_col, target_col])

    # 2. Sort the data and split into two halves
    df = df.sort_values(datetime_col)
    midpoint = len(df) // 2
    train_df = df.iloc[:midpoint]
    test_df = df.iloc[midpoint:]

    # 3. Retrieve smoothed temperatures
    full_start = df[datetime_col].min().strftime('%Y-%m-%d')
    full_end = df[datetime_col].max().strftime('%Y-%m-%d')

    temp_df = get_temp_smoothed_fr(full_start, full_end)
    temp_df = temp_df.rename(columns={'datetime': datetime_col})
    temp_df[datetime_col] = pd.to_datetime(temp_df[datetime_col], errors='coerce')

    # 4. Merge temperature data with train/test sets
    train_df = pd.merge(train_df, temp_df, on=datetime_col, how='left')
    test_df = pd.merge(test_df, temp_df, on=datetime_col, how='left')

    # 5. Feature engineering
    def add_time_features(df):
        df['hour'] = df[datetime_col].dt.hour
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['week'] = df[datetime_col].dt.isocalendar().week.astype(int)
        df['month'] = df[datetime_col].dt.month
        df['year'] = df[datetime_col].dt.year
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        fr_holidays = holidays.country_holidays('FR')
        df['is_holiday'] = df[datetime_col].dt.date.apply(lambda d: 1 if d in fr_holidays else 0)
        return df

    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)

    # 6. Define X and y
    features = ['hour', 'dayofweek', 'week', 'month', 'year', 'is_weekend', 'is_holiday', 'temperature']
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]  # useful for later evaluation

    # 7. Imputation and normalization
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 8. Train the model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 9. Make predictions
    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    test_df['forecast'] = y_pred

    return test_df


def plot_forecast(df, datetime_col='datetime', target_col='consumption'):
    # 1. Call eval_forecast
    forecast_df = eval_forecast(df, datetime_col=datetime_col, target_col=target_col)

    # 2. Calculate MAPE
    y_true = forecast_df[target_col].values
    y_pred = forecast_df['forecast'].values
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 3. Create the interactive plot
    fig = go.Figure()

    # Actual values series
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col],
        y=forecast_df[target_col],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Forecast series
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col],
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # 4. Layout with black background
    fig.update_layout(
        title='Forecast vs Actual Consumption',
        xaxis_title='Date',
        yaxis=dict(title='Consumption', color='white', gridcolor='gray'),
        xaxis=dict(color='white', gridcolor='gray'),
        legend=dict(x=0.01, y=0.99, font=dict(color='white')),
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(t=120)  # increase top margin
    )

    # 5. Add MAPE at the top center
    fig.add_annotation(
        text=f"MAPE: {mape:.2f}%",
        xref="paper", yref="paper",
        x=0.5, y=1,  # slightly above the title
        showarrow=False,
        font=dict(size=14, color="white"),
        align="center"
    )

    # 6. Show the plot
    fig.show()


def calculate_mape(df, datetime_col='datetime', target_col='consumption'):
    # 1. Compute forecasts using eval_forecast
    forecast_df = eval_forecast(df, datetime_col=datetime_col, target_col=target_col)

    # 2. Extract actual and predicted values
    y_true = forecast_df[target_col]
    y_pred = forecast_df['forecast']

    # 3. Calculate MAPE (as a percentage)
    mape_value = mean_absolute_percentage_error(y_true, y_pred) * 100

    return mape_value


def get_spot_price_fr(token: str, start_date: str, end_date: str):
    """
    Retrieves electricity spot prices in France from Databricks (EPEX spot).

    :param token: Databricks personal access token, as a string.
    :param start_date: Start date in 'YYYY-MM-DD' format.
    :param end_date: End date in 'YYYY-MM-DD' format.
    :return: DataFrame with columns ['delivery_from', 'price_eur_per_mwh'].
    """
    databricks_url = f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"

    with tqdm(total=1, desc="Loading spot prices from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        with tio.db(databricks_url) as client:
            query = f"""
                SELECT delivery_from, price_eur_per_mwh
                FROM consumer.inter_energymarkets_epex_hh_spot_prices
                WHERE price_date >= '{start_date}' AND price_date <= '{end_date}'
                ORDER BY delivery_from;
            """
            spot_df = client.get_df(query)
            pbar.update(1)

    # Cleaning / typing
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'], utc=True).dt.tz_localize(None)
    spot_df['price_eur_per_mwh'] = spot_df['price_eur_per_mwh'].astype(float)

    return spot_df

def get_forward_price_fr(token: str, cal_year: int) -> pd.DataFrame:
    """
    Retrieves annual forward electricity prices in France for a given year from Databricks (EEX).

    :param token: Databricks personal access token.
    :param cal_year: Calendar year for delivery (e.g., 2026).
    :return: DataFrame with columns ['trading_date', 'forward_price', 'cal_year'].
    """
    databricks_url = (
        f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )

    with tqdm(total=1, desc="Loading forward prices from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        with tio.db(databricks_url) as client:
            query = f"""
                SELECT setllement_price, trading_date
                FROM consumer.stg_eex_power_future_results_fr 
                WHERE long_name = 'EEX French Power Base Year Future' 
                AND delivery_start >= '{cal_year}-01-01'
                AND delivery_end <= '{cal_year}-12-31'
                ORDER BY trading_date
            """
            forward_df = client.get_df(query)
            pbar.update(1)

    # Cleaning / typing
    forward_df.rename(columns={'setllement_price': 'forward_price'}, inplace=True)
    forward_df['trading_date'] = pd.to_datetime(forward_df['trading_date'], utc=True)
    forward_df['forward_price'] = forward_df['forward_price'].astype(float)
    forward_df['cal_year'] = cal_year

    return forward_df

def get_pfc_fr(token: str, cal_year: int) -> pd.DataFrame:
    """
    Retrieves the Price Forward Curve (PFC) of electricity in France for a given calendar year,
    from Databricks (EEX).

    :param token: Databricks personal access token.
    :param cal_year: Calendar year for delivery (e.g., 2026).
    :return: Hourly-indexed DataFrame with columns ['pfc_forward_price', 'cal_year'].
    """
    databricks_url = (
        f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )

    with tqdm(total=1, desc="Loading PFC curve from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        with tio.db(databricks_url) as client:
            query = f"""
                SELECT 
                    delivery_from AS time_utc,
                    forward_price AS pfc_forward_price
                FROM consumer.stg_octo_curves
                WHERE mode = 'EOD_EEX' 
                  AND asset = 'FRPX'
                  AND price_date = (
                    SELECT MAX(price_date) 
                    FROM consumer.stg_octo_curves 
                    WHERE mode = 'EOD_EEX' AND asset = 'FRPX'
                  )
                  AND delivery_from >= '{cal_year}-01-01'
                  AND delivery_from < '{cal_year + 1}-01-01'
                ORDER BY delivery_from
            """
            pfc = client.get_df(query)
            pbar.update(1)

    pfc.rename(columns={'setllement_price': 'forward_price'}, inplace=True)

    # Type conversion
    pfc['time_utc'] = pd.to_datetime(pfc['time_utc'], utc=True)
    pfc['pfc_forward_price'] = pfc['pfc_forward_price'].astype(float)

    # Resample to hourly frequency and compute mean
    pfc = pfc.set_index('time_utc').resample('H').mean()

    # Add cal_year column for reference
    pfc['cal_year'] = cal_year

    return pfc

def calculate_prem_risk_vol(token: str, input_df: pd.DataFrame, datetime_col: str, target_col: str, plot_chart: bool = False, quantile: int = 50) -> float:
    """
    Calculates a risk premium based on multiple forward prices,
    and returns the value corresponding to the specified quantile.

    :param token: Databricks token.
    :param input_df: DataFrame containing consumption data.
    :param datetime_col: Name of the datetime column in input_df.
    :param target_col: Name of the actual consumption column in input_df.
    :param plot_chart: If True, displays the premium distribution chart.
    :param quantile: Quantile to return (between 1 and 100).
    :return: Risk premium corresponding to the requested quantile.
    """
    # 1. Forecast evaluation
    forecast_df = eval_forecast(input_df, datetime_col=datetime_col, target_col=target_col)
    forecast_df[datetime_col] = pd.to_datetime(forecast_df[datetime_col])

    # 2. Determine the dominant year
    year_counts = forecast_df[datetime_col].dt.year.value_counts()
    if year_counts.empty:
        raise ValueError("No valid data in eval_forecast.")
    major_year = year_counts.idxmax()
    print(f"Dominant year: {major_year} with {year_counts.max()} occurrences")

    # 3. Retrieve spot prices for the covered period
    start_date = forecast_df[datetime_col].min().strftime('%Y-%m-%d')
    end_date = forecast_df[datetime_col].max().strftime('%Y-%m-%d')
    spot_df = get_spot_price_fr(token, start_date, end_date)
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'])

    # 4. Retrieve forward prices for the dominant year
    forward_df = get_forward_price_fr(token, major_year)
    if forward_df.empty:
        raise ValueError(f"No forward prices found for year {major_year}")
    forward_prices = forward_df['forward_price'].tolist()

    # 5. Prepare dataframe for merging
    forecast_df = forecast_df.rename(columns={datetime_col: 'datetime', target_col: 'consommation_realisee'})
    forecast_df = forecast_df[['datetime', 'consommation_realisee', 'forecast']]
    merged_df = pd.merge(forecast_df, spot_df, left_on='datetime', right_on='delivery_from', how='inner')
    if merged_df.empty:
        raise ValueError("No match between consumption and spot prices")

    # 6. Compute annual consumption once
    merged_df['diff_conso'] = merged_df['consommation_realisee'] - merged_df['forecast']
    conso_totale_MWh = merged_df['consommation_realisee'].sum()
    if conso_totale_MWh == 0:
        raise ValueError("Annual consumption is zero, division not possible")

    # 7. Calculate premiums for each forward price
    premiums = []
    for fwd_price in forward_prices:
        merged_df['diff_price'] = merged_df['price_eur_per_mwh'] - fwd_price
        merged_df['produit'] = merged_df['diff_conso'] * merged_df['diff_price']
        premium = abs(merged_df['produit'].sum()) / conso_totale_MWh
        premiums.append(premium)

    # 8. Optional: display chart
    if plot_chart:
        premiums_sorted = sorted(premiums)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=premiums_sorted,
            x=list(range(1, len(premiums_sorted)+1)),
            mode='lines+markers',
            name='Premiums',
            line=dict(color='cyan')
        ))
        fig.update_layout(
            title="Risk premium distribution (volume)",
            xaxis_title="Index (sorted)",
            yaxis_title="Premium",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            hovermode='closest'
        )
        fig.show()

    # 9. Return the requested quantile
    if not (1 <= quantile <= 100):
        raise ValueError("Quantile must be an integer between 1 and 100.")
    quantile_value = np.percentile(premiums, quantile)
    return quantile_value



