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





def get_temperature_lissee_france(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Récupère les températures horaires moyennes 'lissées' sur plusieurs grandes villes françaises.
    
    :param start_date: Date de début (format 'YYYY-MM-DD')
    :param end_date: Date de fin (format 'YYYY-MM-DD')
    :return: DataFrame avec colonnes ['datetime', 'temperature']
    """
    # Villes choisies pour lisser à l’échelle nationale
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

    for city, (lat, lon) in tqdm(cities.items(), desc="Récupération des villes"):
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
            print(f"Erreur avec {city}: {e}")

    # Fusion et moyenne
    df_all = pd.concat(city_dfs, axis=1)
    df_all['temperature'] = df_all.mean(axis=1)

    # Retourner seulement datetime + temperature
    return df_all[['temperature']].reset_index()


def eval_forecast(df, datetime_col='datetime', target_col='consumption'):
    # 1. Harmoniser les formats de dates et nettoyer les NaN
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=[datetime_col, target_col])
    
    # 2. Trier les données et découper en deux
    df = df.sort_values(datetime_col)
    midpoint = len(df) // 2
    train_df = df.iloc[:midpoint]
    test_df = df.iloc[midpoint:]

    # 3. Récupérer les températures lissées
    full_start = df[datetime_col].min().strftime('%Y-%m-%d')
    full_end = df[datetime_col].max().strftime('%Y-%m-%d')

    temp_df = get_temperature_lissee_france(full_start, full_end)
    temp_df = temp_df.rename(columns={'datetime': datetime_col})
    temp_df[datetime_col] = pd.to_datetime(temp_df[datetime_col], errors='coerce')

    # 4. Fusionner température avec train/test
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

    # 6. Définir X et y
    features = ['hour', 'dayofweek', 'week', 'month', 'year', 'is_weekend', 'is_holiday', 'temperature']
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]  # utile pour évaluer plus tard

    # 7. Imputation et normalisation
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 8. Entraîner le modèle
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 9. Prédictions
    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    test_df['forecast'] = y_pred

    return test_df


def plot_forecast(df, datetime_col='datetime', target_col='consumption'):
    # 1. Appeler eval_forecast
    forecast_df = eval_forecast(df, datetime_col=datetime_col, target_col=target_col)

    # 2. Calcul de la MAPE
    y_true = forecast_df[target_col].values
    y_pred = forecast_df['forecast'].values
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 3. Créer le graphique interactif
    fig = go.Figure()

    # Série réelle
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col],
        y=forecast_df[target_col],
        mode='lines',
        name='Réalisé',
        line=dict(color='blue')
    ))

    # Série prévisionnelle
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col],
        y=forecast_df['forecast'],
        mode='lines',
        name='Prévision',
        line=dict(color='red', dash='dash')
    ))

    # 4. Mise en page avec fond noir
    fig.update_layout(
        title='Prévision de la consommation vs Réalité',
        xaxis_title='Date',
        yaxis=dict(title='Consommation', color='white', gridcolor='gray'),
        xaxis=dict(color='white', gridcolor='gray'),
        legend=dict(x=0.01, y=0.99, font=dict(color='white')),
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(t=120)  # augmenter l’espace en haut
    )

    # 5. Ajouter la MAPE en haut centré
    fig.add_annotation(
        text=f"MAPE : {mape:.2f}%",
        xref="paper", yref="paper",
        x=0.5, y=1,  # légèrement au-dessus du titre
        showarrow=False,
        font=dict(size=14, color="white"),
        align="center"
    )

    # 6. Afficher
    fig.show()


def calculate_mape(df, datetime_col='datetime', target_col='consumption'):
    # 1. Calculer les prévisions avec eval_forecast
    forecast_df = eval_forecast(df, datetime_col=datetime_col, target_col=target_col)
    
    # 2. Extraire les valeurs réelles et prédites
    y_true = forecast_df[target_col]
    y_pred = forecast_df['forecast']
    
    # 3. Calculer la MAPE (en %)
    mape_value = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return mape_value


def get_spot_price_fr(token: str, start_date: str, end_date: str) :
    
    """
    Récupère les prix spot de l'électricité en France depuis Databricks (EPEX spot).

    :param token: Token personnel Databricks, attendu sous forme de chaîne de caractères (string).
    :param start_date: Date de début au format 'YYYY-MM-DD'.
    :param end_date: Date de fin au format 'YYYY-MM-DD'.
    :return: DataFrame avec les colonnes ['delivery_from', 'price_eur_per_mwh'].
    """
    databricks_url = f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"

    with tqdm(total=1, desc="Chargement des prix spot depuis Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
        with tio.db(databricks_url) as client:
            query = f"""
                SELECT delivery_from, price_eur_per_mwh
                FROM consumer.inter_energymarkets_epex_hh_spot_prices
                WHERE price_date >= '{start_date}' AND price_date <= '{end_date}'
                ORDER BY delivery_from;
            """
            spot_df = client.get_df(query)
            pbar.update(1)

    # Nettoyage / typage
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'], utc=True).dt.tz_localize(None)
    spot_df['price_eur_per_mwh'] = spot_df['price_eur_per_mwh'].astype(float)

    return spot_df


def get_forward_price_fr(token: str, cal_year: int) -> pd.DataFrame:
    """
    Récupère les prix forward annuels de l'électricité en France pour une année donnée depuis Databricks (EEX).

    :param token: Token personnel Databricks.
    :param cal_year: Année calendaire de livraison souhaitée (ex: 2026).
    :return: DataFrame avec les colonnes ['trading_date', 'setllement_price', 'cal_year'].
    """
    databricks_url = (
        f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )

    with tqdm(total=1, desc="Chargement des prix forward depuis Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
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

    # Nettoyage / typage
    forward_df.rename(columns={'setllement_price': 'forward_price'}, inplace=True)
    forward_df['trading_date'] = pd.to_datetime(forward_df['trading_date'], utc=True)
    forward_df['forward_price'] = forward_df['forward_price'].astype(float)
    forward_df['cal_year'] = cal_year

    return forward_df


def get_pfc_fr(token: str, cal_year: int) -> pd.DataFrame:
    """
    Récupère la courbe Price Forward Curve (PFC) de l'électricité en France pour une année calendaire donnée,
    depuis Databricks (EEX).

    :param token: Token personnel Databricks.
    :param cal_year: Année calendaire de livraison souhaitée (ex: 2026).
    :return: DataFrame indexé en datetime hourly avec colonnes ['forward_price', 'cal_year'].
    """
    databricks_url = (
        f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )

    with tqdm(total=1, desc="Chargement de la courbe PFC depuis Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
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

    # Conversion des types
    pfc['time_utc'] = pd.to_datetime(pfc['time_utc'], utc=True)
    pfc['pfc_forward_price'] = pfc['pfc_forward_price'].astype(float)

    # Resample à fréquence horaire et calcul de la moyenne
    pfc = pfc.set_index('time_utc').resample('H').mean()

    # Ajout de la colonne cal_year pour référence
    pfc['cal_year'] = cal_year

    return pfc


def calculate_prem_risk_vol(token: str, input_df: pd.DataFrame, datetime_col: str, target_col: str, plot_chart: bool = False, quantile: int = 50 ) -> float:
    """
    Calcule un premium de risque basé sur plusieurs prix forward,
    et retourne la valeur correspondant au quantile spécifié.

    :param token: Token Databricks.
    :param input_df: DataFrame contenant les données de consommation.
    :param datetime_col: Nom de la colonne datetime dans input_df.
    :param target_col: Nom de la colonne de consommation réalisée dans input_df.
    :param plot_chart: Si True, affiche la distribution des premiums.
    :param quantile: Quantile à retourner (entre 1 et 100).
    :return: Premium de risque correspondant au quantile demandé.
    """
    # 1. Évaluation de la prévision
    forecast_df = eval_forecast(input_df, datetime_col=datetime_col, target_col=target_col)
    forecast_df[datetime_col] = pd.to_datetime(forecast_df[datetime_col])
    
    # 2. Déterminer l'année majoritaire
    year_counts = forecast_df[datetime_col].dt.year.value_counts()
    if year_counts.empty:
        raise ValueError("Pas de données valides dans eval_forecast.")
    major_year = year_counts.idxmax()
    print(f"Année majoritaire : {major_year} avec {year_counts.max()} occurrences")

    # 3. Récupération des prix spot pour la période couverte
    start_date = forecast_df[datetime_col].min().strftime('%Y-%m-%d')
    end_date = forecast_df[datetime_col].max().strftime('%Y-%m-%d')
    spot_df = get_spot_price_fr(token, start_date, end_date)
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'])

    # 4. Récupération des prix forward pour l'année majoritaire
    forward_df = get_forward_price_fr(token, major_year)
    if forward_df.empty:
        raise ValueError(f"Aucun prix forward trouvé pour l'année {major_year}")
    forward_prices = forward_df['forward_price'].tolist()

    # 5. Préparer le dataframe pour la jointure
    forecast_df = forecast_df.rename(columns={datetime_col: 'datetime', target_col: 'consommation_realisee'})
    forecast_df = forecast_df[['datetime', 'consommation_realisee', 'forecast']]
    merged_df = pd.merge(forecast_df, spot_df, left_on='datetime', right_on='delivery_from', how='inner')
    if merged_df.empty:
        raise ValueError("Aucune correspondance entre la conso et les prix spot")

    # 6. Calcul une seule fois la consommation annuelle
    merged_df['diff_conso'] = merged_df['consommation_realisee'] - merged_df['forecast']
    conso_totale_MWh = merged_df['consommation_realisee'].sum()
    if conso_totale_MWh == 0:
        raise ValueError("Consommation annuelle nulle, division impossible")

    # 7. Calcul des premiums pour chaque prix forward
    premiums = []
    for fwd_price in forward_prices:
        merged_df['diff_price'] = merged_df['price_eur_per_mwh'] - fwd_price
        merged_df['produit'] = merged_df['diff_conso'] * merged_df['diff_price']
        premium = abs(merged_df['produit'].sum()) / conso_totale_MWh
        premiums.append(premium)

    # 8. Optionnel : afficher le graphique
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
            title="Distribution des premiums de risque volume",
            xaxis_title="Index (classé)",
            yaxis_title="Premium",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            hovermode='closest'
        )
        fig.show()

    # 9. Retourner le quantile demandé
    if not (1 <= quantile <= 100):
        raise ValueError("Le quantile doit être un entier entre 1 et 100.")
    quantile_value = np.percentile(premiums, quantile)
    return quantile_value



