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
from dateutil.relativedelta import relativedelta
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional
from databricks import sql


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

def eval_forecast(df, temp_df, cal_year, datetime_col='timestamp', target_col='MW'):
    """
    Génère un forecast pour l’année civile cal_year si les données d'entrée couvrent bien cette période.
    """
    import holidays

    np.random.seed(42)
    df = df.copy()

    # Nettoyage de base
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[datetime_col, target_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # 🔍 Vérification de la couverture complète sur l’année cal_year
    expected_start = pd.Timestamp(f"{cal_year}-01-01 00:00:00")
    expected_end = pd.Timestamp(f"{cal_year}-12-31 23:59:59")

    if not ((df[datetime_col] <= expected_start).any() and (df[datetime_col] >= expected_end).any()):
        raise ValueError(
            f"Les données ne couvrent pas toute l’année civile {cal_year}.\n"
            f"Période attendue : {expected_start.date()} à {expected_end.date()}\n"
            f"Données disponibles de {df[datetime_col].min().date()} à {df[datetime_col].max().date()}"
        )

    # ✅ Les données couvrent bien l’année -> split test/train
    test_start = expected_start
    test_end = expected_end

    # Nettoyage température (déjà lissée)
    temp_df[datetime_col] = pd.to_datetime(temp_df['datetime'])
    temp_df = temp_df.drop(columns=['datetime'])

    # Merge température
    df = pd.merge(df, temp_df, on=datetime_col, how='left')
    df['temperature'] = df['temperature'].ffill().bfill()

    # Features
    def add_features(df):
        df['hour'] = df[datetime_col].dt.hour
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['month'] = df[datetime_col].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['minute'] = df[datetime_col].dt.minute
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['dayofyear'] = df[datetime_col].dt.dayofyear
        df['week'] = df[datetime_col].dt.isocalendar().week.astype(int)
        df['quarter'] = df[datetime_col].dt.quarter
        df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
        df['year'] = df[datetime_col].dt.year
        fr_holidays = holidays.country_holidays('FR')
        df['is_holiday'] = df[datetime_col].dt.date.astype(str).isin(fr_holidays).astype(int)
        df['heating_on'] = (df['temperature'] < 15).astype(int)
        df['cooling_on'] = (df['temperature'] > 25).astype(int)
        df['temp_below_10'] = np.maximum(0, 10 - df['temperature'])
        df['temp_above_30'] = np.maximum(0, df['temperature'] - 30)
        df['temp_diff_15'] = df['temperature'] - 15
        return df

    df = add_features(df)
    df = df.dropna().reset_index(drop=True)

    # Split train/test
    train_df = df[(df[datetime_col] < test_start) | (df[datetime_col] > test_end)].copy()
    test_df = df[(df[datetime_col] >= test_start) & (df[datetime_col] <= test_end)].copy()

    if len(train_df) < 1000:
        raise ValueError("Pas assez de données pour entraîner le modèle.")

    # Modélisation
    features = [col for col in train_df.columns if col not in [datetime_col, target_col]]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[features], train_df[target_col])

    # Prédiction
    test_df['forecast'] = model.predict(test_df[features])

    return test_df[[datetime_col, target_col, 'forecast']]





    df = add_features(df)

    # Ajouter les lags au DataFrame global pour cohérence
    df['lag_1'] = df[target_col].shift(1)
    df['lag_48'] = df[target_col].shift(48)
    df['lag_336'] = df[target_col].shift(336)

    df = df.dropna().reset_index(drop=True)

    # Réappliquer split après l'ajout des lags
    train_df = df[(df[datetime_col] < test_start) | (df[datetime_col] > test_end)].copy()
    test_df = df[(df[datetime_col] >= test_start) & (df[datetime_col] <= test_end)].copy()

    # Sélection des features
    features = [col for col in train_df.columns if col not in [datetime_col, target_col]]

    # Modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[features], train_df[target_col])

    # Prédictions
    test_df['forecast'] = model.predict(test_df[features])

    return test_df[[datetime_col, target_col, 'forecast']]

def plot_forecast(df, temp_df, datetime_col='timestamp', target_col='MW', save_path=None):
    # 1. Call eval_forecast (avec température en paramètre)
    forecast_df = eval_forecast(df, temp_df=temp_df, datetime_col=datetime_col, target_col=target_col)

    # 2. Calculate MAPE
    y_true = forecast_df[target_col].values
    y_pred = forecast_df['forecast'].values
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # 3. Create the interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col], y=forecast_df[target_col],
        mode='lines', name='Valeurs réelles', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col], y=forecast_df['forecast'],
        mode='lines', name='Prévision', line=dict(color='red', dash='dash')
    ))

    # 4. Layout
    fig.update_layout(
        title=f'Prévision vs Réel — MAPE: {mape:.2f}%',
        xaxis_title='Date',
        yaxis_title=target_col,
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(x=0.01, y=0.99, font=dict(color='white')),
        xaxis=dict(color='white', gridcolor='gray'),
        yaxis=dict(color='white', gridcolor='gray'),
        margin=dict(t=100)
    )

    # 5. Save or show
    if save_path:
        fig.write_html(save_path)
        print(f"Graph saved as interactive HTML at: {save_path}")
    else:
        fig.show()

    return fig

def get_spot_price_fr(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves electricity spot prices in France from Databricks (EPEX spot).

    :param token: Databricks personal access token
    :param start_date: Format 'YYYY-MM-DD'
    :param end_date: Format 'YYYY-MM-DD'
    :return: DataFrame with ['delivery_from', 'price_eur_per_mwh']
    """
    with tqdm(total=1, desc="Loading spot prices from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        connection = sql.connect(
            server_hostname="octoenergy-oefr-prod.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/ddb864eabbe6b908",
            access_token=token
        )

        cursor = connection.cursor()

        query = f"""
            SELECT delivery_from, price_eur_per_mwh
            FROM consumer.inter_energymarkets_epex_hh_spot_prices
            WHERE source_identifier = 'epex'
              AND price_date >= '{start_date}'
              AND price_date <= '{end_date}'
            ORDER BY delivery_from
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        spot_df = pd.DataFrame(rows, columns=columns)

        cursor.close()
        connection.close()
        pbar.update(1)

    # Cleaning
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'], utc=True).dt.tz_localize(None)
    spot_df['price_eur_per_mwh'] = spot_df['price_eur_per_mwh'].astype(float)

    return spot_df

def get_forward_price_fr_annual(token: str, cal_year: int) -> pd.DataFrame:
    """
    Retrieves annual forward electricity prices in France for a given year from Databricks (EEX).

    :param token: Databricks personal access token.
    :param cal_year: Calendar year for delivery (e.g., 2026).
    :return: DataFrame with columns ['trading_date', 'forward_price', 'cal_year'].
    """
    from databricks import sql
    from tqdm import tqdm

    with tqdm(total=1, desc="Loading forward prices from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        connection = sql.connect(
            server_hostname="octoenergy-oefr-prod.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/ddb864eabbe6b908",
            access_token=token
        )

        cursor = connection.cursor()

        query = f"""
            SELECT setllement_price AS forward_price, trading_date
            FROM consumer.stg_eex_power_future_results_fr 
            WHERE long_name = 'EEX French Power Base Year Future' 
              AND delivery_start >= '{cal_year}-01-01'
              AND delivery_end <= '{cal_year}-12-31'
            ORDER BY trading_date
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        forward_df = pd.DataFrame(rows, columns=columns)

        cursor.close()
        connection.close()
        pbar.update(1)

    # Cleaning / typing
    forward_df['trading_date'] = pd.to_datetime(forward_df['trading_date'], utc=True)
    forward_df['forward_price'] = forward_df['forward_price'].astype(float)
    forward_df['cal_year'] = cal_year

    return forward_df

def get_forward_price_fr_months(token: str, cal_year_month: str) -> pd.DataFrame:
    """
    Retrieves monthly forward electricity prices in France from Databricks (EEX).

    :param token: Databricks personal access token.
    :param cal_year_month: Month of delivery in 'YYYY-MM' format (e.g., '2025-03').
    :return: DataFrame with ['trading_date', 'forward_price', 'cal_year'].
    """
    # Début et fin du mois
    start_date = datetime.strptime(cal_year_month, "%Y-%m")
    end_date = start_date + relativedelta(months=1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    databricks_url = (
        f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )

    with tqdm(total=1, desc="Loading forward prices from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        with tio.db(databricks_url) as client:
            query = f"""
                SELECT setllement_price, trading_date, delivery_start, delivery_end
                FROM consumer.stg_eex_power_future_results_fr 
                WHERE long_name = 'EEX French Power Base Month Future' 
                AND delivery_start >= '{start_str}'
                AND delivery_start < '{end_str}'
                AND setllement_price IS NOT NULL
                ORDER BY trading_date
            """
            forward_df = client.get_df(query)
            pbar.update(1)

    # Nettoyage
    forward_df.rename(columns={'setllement_price': 'forward_price'}, inplace=True)
    forward_df['trading_date'] = pd.to_datetime(forward_df['trading_date'], utc=True)
    forward_df['forward_price'] = forward_df['forward_price'].astype(float)
    forward_df['cal_year'] = cal_year_month

    # Suppression des doublons
    forward_df = forward_df.drop_duplicates()

    return forward_df

def get_pfc_fr(token: str, price_date: int, delivery_year: int) -> pd.DataFrame:
    connection = sql.connect(
        server_hostname="octoenergy-oefr-prod.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/ddb864eabbe6b908",
        access_token=token
    )
    
    cursor = connection.cursor()
    query = f"""
        SELECT delivery_from,
               price_date,
               forward_price
        FROM consumer.stg_octo_curves
        WHERE mode = 'EOD'
          AND asset = 'FRPX'
          AND year(delivery_from) = '{delivery_year}'
          AND year(price_date) = '{price_date}'
        ORDER BY price_date
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=colnames)
    
    cursor.close()
    connection.close()
    return df

def calculate_prem_risk_vol(forecast_df: pd.DataFrame, spot_df: pd.DataFrame, forward_df: pd.DataFrame, plot_chart: bool = False, quantile: int = 50, variability_factor: float = 1.1, save_path: Optional[str] = None) -> float:
    """
    Calcule la prime de risque (volume) à partir d’un DataFrame de prévision déjà préparé.
    """

    forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime']).dt.tz_localize(None)

    # 1. Année la plus récente
    latest_date = forecast_df['datetime'].max()
    latest_year = latest_date.year
    print(f"Using year from latest date: {latest_year} (latest forecast: {latest_date.strftime('%Y-%m-%d')})")

    # 2. Spot data
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from']).dt.tz_localize(None)

    # 3. Forward check
    if forward_df.empty:
        raise ValueError(f"No forward prices provided.")
    forward_prices = forward_df['forward_price'].tolist()

    # 4. Merge données
    merged_df = pd.merge(forecast_df, spot_df, left_on='datetime', right_on='delivery_from', how='inner')
    if merged_df.empty:
        raise ValueError("No data available to merge spot and forecast")

    # 5. Calcul consommation et prime
    merged_df['diff_conso'] = (merged_df['consommation_realisee'] - merged_df['forecast']) * variability_factor
    conso_totale_MWh = merged_df['consommation_realisee'].sum()
    if conso_totale_MWh == 0:
        raise ValueError("Annual consumption is zero, division not possible")

    premiums = []
    for fwd_price in forward_prices:
        merged_df['diff_price'] = merged_df['price_eur_per_mwh'] - fwd_price
        merged_df['produit'] = merged_df['diff_conso'] * merged_df['diff_price']
        premium = abs(merged_df['produit'].sum()) / conso_totale_MWh
        premiums.append(premium)

    # 6. Affichage / export
    if plot_chart or save_path:
        premiums_sorted = sorted(premiums)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=premiums_sorted,
            x=list(range(1, len(premiums_sorted) + 1)),
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
        if save_path:
            fig.write_html(save_path)
            print(f"Graphique interactif enregistré : {save_path}")
        if plot_chart:
            fig.show()

    # 7. Quantile
    if not (1 <= quantile <= 100):
        raise ValueError("Quantile must be an integer between 1 and 100.")
    quantile_value = np.percentile(premiums, quantile)
    print(f"Quantile {quantile} = {quantile_value:.4f}")
    return quantile_value


def calculate_prem_risk_shape(forecast_df: pd.DataFrame, pfc_df: pd.DataFrame, spot_df: pd.DataFrame, quantile: int = 70, plot_chart: bool = False) -> float:

    df_conso_prev = forecast_df.copy()
    df_conso_prev = df_conso_prev.rename(columns={'datetime': 'delivery_from'})
    df_conso_prev['delivery_from'] = pd.to_datetime(df_conso_prev['delivery_from']).dt.tz_localize('UTC')
    df_conso_prev['forecast'] = df_conso_prev['forecast'] / 1_000_000  # MWh -> GWh

    # PFC
    pfc = pfc_df.copy()
    pfc['delivery_from'] = pd.to_datetime(pfc['delivery_from']).dt.tz_localize('UTC')

    # Merge PFC + forecast
    df = pd.merge(pfc, df_conso_prev[['delivery_from', 'forecast']], on='delivery_from', how='left').dropna()
    df['value'] = df['forward_price'] * df['forecast']
    df['delivery_month'] = df['delivery_from'].dt.to_period('M')
    df['price_date'] = pfc['price_date']  # si pas déjà là

    gb_month = df.groupby(['price_date', 'delivery_month']).agg(
        bl_volume_month=('forecast', 'mean'),
        bl_value_month=('value', 'sum'),
        forward_price_sum_month=('forward_price', 'sum')
    )
    gb_month['bl_value_month'] = gb_month['bl_value_month'] / gb_month['forward_price_sum_month']
    gb_month.reset_index(inplace=True)

    # Spot
    spot = spot_df.copy()
    spot = spot.rename(columns={'price_eur_per_mwh': 'spot_price'})
    spot['delivery_from'] = pd.to_datetime(spot['delivery_from']).dt.tz_localize('UTC')

    df = df.merge(spot[['delivery_from', 'spot_price']], on='delivery_from', how='left').dropna()
    df = df.merge(gb_month, on=['price_date', 'delivery_month'], how='left').dropna()

    # Résiduels
    df['residual_volume'] = df['forecast'] - df['bl_value_month']
    df['residual_value'] = df['residual_volume'] * df['spot_price']

    agg = df.groupby(['price_date']).agg(
        residual_value_month=('residual_value', 'sum'),
        conso_month=('forecast', 'sum')
    )
    agg['shape_cost'] = agg['residual_value_month'] / agg['conso_month']
    agg['abs_shape_cost'] = agg['shape_cost'].abs()

    if plot_chart:
        sorted_vals = agg['abs_shape_cost'].sort_values().reset_index(drop=True)
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_vals, color='white')
        plt.title('Shape Risk (Sorted)', color='white')
        plt.xlabel('Simulation', color='white')
        plt.ylabel('€/MWh', color='white')
        plt.grid(True, color='gray')
        plt.show()

    quantile_value = np.percentile(agg['abs_shape_cost'], quantile)
    print(f"Quantile {quantile} = {quantile_value:.4f} €/MWh")

    return quantile_value







