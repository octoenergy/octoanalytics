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
from yaspin import yaspin
import pandas as pd
from databricks import sql


def get_temp_smoothed_fr(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Récupère les températures moyennes horaires lissées sur plusieurs grandes villes françaises.

    Paramètres :
    -----------
    start_date : str
        Date de début (format 'YYYY-MM-DD').
    end_date : str
        Date de fin (format 'YYYY-MM-DD').

    Retour :
    -------
    pd.DataFrame
        DataFrame avec les colonnes ['datetime', 'temperature'] représentant la température moyenne lissée.
    """

    # 1. Définition des grandes villes françaises pour lisser les données à l'échelle nationale
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

    city_dfs = []  # 2. Liste pour stocker les DataFrames de chaque ville

    # 3. Boucle sur chaque ville pour récupérer les données météo horaires via l'API Open-Meteo
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
            # 4. Requête GET à l'API météo
            response = requests.get(url, params=params)
            response.raise_for_status()  # Vérifie que la requête est OK
            data = response.json()

            # 5. Construction du DataFrame pour la ville courante
            df = pd.DataFrame({
                'datetime': data['hourly']['time'],
                city: data['hourly']['temperature_2m']
            })
            df['datetime'] = pd.to_datetime(df['datetime'])  # Conversion en datetime
            df.set_index('datetime', inplace=True)  # Mise en index sur la date/heure

            city_dfs.append(df)  # 6. Ajout du DataFrame ville à la liste

        except Exception as e:
            print(f"Error with {city}: {e}")  # 7. Gestion simple des erreurs

    # 8. Fusion de tous les DataFrames de villes selon l'index datetime (concaténation horizontale)
    df_all = pd.concat(city_dfs, axis=1)

    # 9. Calcul de la moyenne horaire sur toutes les villes (lissage national)
    df_all['temperature'] = df_all.mean(axis=1)

    # 10. Retourne uniquement la colonne datetime et la température moyenne lissée (réinitialisation de l'index)
    return df_all[['temperature']].reset_index()

def eval_forecast(df, temp_df, cal_year, datetime_col='timestamp', target_col='MW'):
    """
    Génère un forecast pour l’année civile cal_year si les données d'entrée couvrent bien cette période.

    Paramètres :
    -----------
    df : pd.DataFrame
        Données historiques avec colonnes incluant datetime_col et target_col.
    temp_df : pd.DataFrame
        Données de température lissée avec colonnes ['datetime', 'temperature'].
    cal_year : int
        Année civile sur laquelle faire le forecast.
    datetime_col : str, optionnel (par défaut 'timestamp')
        Nom de la colonne datetime dans df.
    target_col : str, optionnel (par défaut 'MW')
        Nom de la colonne cible (valeur à prédire) dans df.

    Retour :
    -------
    pd.DataFrame
        DataFrame contenant datetime_col, target_col et la colonne 'forecast' correspondant à la prévision.
    """


    np.random.seed(42)  # 1. Fixe la graine aléatoire pour reproductibilité
    df = df.copy()

    # 2. Nettoyage basique et uniformisation du datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[datetime_col, target_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # 3. Vérification que les données couvrent bien toute l'année cal_year
    expected_start = pd.Timestamp(f"{cal_year}-01-01 00:00:00")
    expected_end = pd.Timestamp(f"{cal_year}-12-31 23:59:59")

    if not ((df[datetime_col] <= expected_start).any() and (df[datetime_col] >= expected_end).any()):
        raise ValueError(
            f"Les données ne couvrent pas toute l’année civile {cal_year}.\n"
            f"Période attendue : {expected_start.date()} à {expected_end.date()}\n"
            f"Données disponibles de {df[datetime_col].min().date()} à {df[datetime_col].max().date()}"
        )

    # 4. Définition de la période de test sur l'année cal_year
    test_start = expected_start
    test_end = expected_end

    # 5. Nettoyage et préparation des données température
    temp_df[datetime_col] = pd.to_datetime(temp_df['datetime'])
    temp_df = temp_df.drop(columns=['datetime'])

    # 6. Fusion des températures avec les données principales
    df = pd.merge(df, temp_df, on=datetime_col, how='left')
    df['temperature'] = df['temperature'].ffill().bfill()  # Remplissage des valeurs manquantes

    # 7. Fonction interne pour ajouter des variables dérivées (features)
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
        
        # 7a. Ajout d'indicateurs de jours fériés français
        fr_holidays = holidays.country_holidays('FR')
        df['is_holiday'] = df[datetime_col].dt.date.astype(str).isin(fr_holidays).astype(int)

        # 7b. Indicateurs de chauffage/climatisation
        df['heating_on'] = (df['temperature'] < 15).astype(int)
        df['cooling_on'] = (df['temperature'] > 25).astype(int)

        # 7c. Variables température transformées (différences seuil)
        df['temp_below_10'] = np.maximum(0, 10 - df['temperature'])
        df['temp_above_30'] = np.maximum(0, df['temperature'] - 30)
        df['temp_diff_15'] = df['temperature'] - 15
        return df

    df = add_features(df)
    df = df.dropna().reset_index(drop=True)  # 8. Suppression des lignes avec valeurs manquantes

    # 9. Split en données d'entraînement et test (test = année cal_year)
    train_df = df[(df[datetime_col] < test_start) | (df[datetime_col] > test_end)].copy()
    test_df = df[(df[datetime_col] >= test_start) & (df[datetime_col] <= test_end)].copy()

    if len(train_df) < 1000:
        raise ValueError("Pas assez de données pour entraîner le modèle.")

    # 10. Entraînement du modèle Random Forest sur les features dérivées
    features = [col for col in train_df.columns if col not in [datetime_col, target_col]]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[features], train_df[target_col])

    # 11. Prédiction sur la période test
    test_df['forecast'] = model.predict(test_df[features])

    # 12. Retour des résultats : datetime, valeur réelle, et prévision
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
    """
    Génère un graphique interactif comparant les valeurs réelles et la prévision issue de eval_forecast,
    avec calcul du MAPE (Mean Absolute Percentage Error).

    Paramètres :
    -----------
    df : pd.DataFrame
        Données historiques contenant au moins datetime_col et target_col.
    temp_df : pd.DataFrame
        Données de température pour passer à eval_forecast.
    datetime_col : str, optionnel (par défaut 'timestamp')
        Nom de la colonne datetime dans df.
    target_col : str, optionnel (par défaut 'MW')
        Nom de la colonne cible dans df.
    save_path : str ou None, optionnel
        Chemin de sauvegarde du graphique au format HTML. Si None, le graphique s'affiche directement.

    Retour :
    -------
    fig : plotly.graph_objs._figure.Figure
        Figure Plotly générée.
    """


    # 1. Appel à eval_forecast pour obtenir les prévisions
    forecast_df = eval_forecast(df, temp_df=temp_df, datetime_col=datetime_col, target_col=target_col)

    # 2. Calcul du MAPE (Mean Absolute Percentage Error) en ignorant les valeurs nulles
    y_true = forecast_df[target_col].values
    y_pred = forecast_df['forecast'].values
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # 3. Création du graphique interactif avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col], y=forecast_df[target_col],
        mode='lines', name='Valeurs réelles', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col], y=forecast_df['forecast'],
        mode='lines', name='Prévision', line=dict(color='red', dash='dash')
    ))

    # 4. Mise en forme du graphique (layout) avec thème sombre et légende
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

    # 5. Enregistrement du graphique au format HTML ou affichage direct
    if save_path:
        fig.write_html(save_path)
        print(f"Graph saved as interactive HTML at: {save_path}")
    else:
        fig.show()

    return fig

def get_spot_price_fr(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Récupère les prix spot de l’électricité en France depuis Databricks (marché EPEX spot).

    Paramètres :
    -----------
    token : str
        Token personnel d’accès à Databricks.
    start_date : str
        Date de début au format 'YYYY-MM-DD'.
    end_date : str
        Date de fin au format 'YYYY-MM-DD'.

    Retour :
    -------
    pd.DataFrame
        DataFrame avec les colonnes ['delivery_from', 'price_eur_per_mwh'] contenant
        les dates/heures de livraison et les prix spot correspondants.
    """

    # 1. Initialisation du spinner de chargement avec yaspin
    with yaspin(text="Chargement des prix spot depuis Databricks...", color="cyan") as spinner:

        # 2. Connexion à Databricks via token personnel
        connection = sql.connect(
            server_hostname="octoenergy-oefr-prod.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/ddb864eabbe6b908",
            access_token=token
        )

        cursor = connection.cursor()

        # 3. Construction et exécution de la requête SQL pour récupérer les prix spot entre start_date et end_date
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

        # 4. Fermeture du curseur et de la connexion
        cursor.close()
        connection.close()

        # 5. Indication de succès via le spinner (affiche une coche verte)
        spinner.ok("✅")

    # 6. Conversion des résultats en DataFrame pandas
    spot_df = pd.DataFrame(rows, columns=columns)

    # 7. Nettoyage et typage des colonnes
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'], utc=True).dt.tz_localize(None)
    spot_df['price_eur_per_mwh'] = spot_df['price_eur_per_mwh'].astype(float)

    # 8. Retour du DataFrame final avec les prix spot
    return spot_df

def get_forward_price_fr_annual(token: str, cal_year: int) -> pd.DataFrame:
    """
    Récupère les prix forward annuels d’électricité en France pour une année donnée depuis Databricks (EEX).

    Paramètres :
    -----------
    token : str
        Token personnel d’accès à Databricks.
    cal_year : int
        Année civile de livraison souhaitée (ex. 2026).

    Retour :
    -------
    pd.DataFrame
        DataFrame contenant les colonnes ['trading_date', 'forward_price', 'cal_year'] avec les
        dates de trading, les prix forward correspondants et l’année civile associée.
    """

    # 1. Initialisation du spinner de chargement avec yaspin
    with yaspin(text="Chargement des prix forward depuis Databricks...", color="cyan") as spinner:

        # 2. Connexion à Databricks via token personnel
        connection = sql.connect(
            server_hostname="octoenergy-oefr-prod.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/ddb864eabbe6b908",
            access_token=token
        )

        cursor = connection.cursor()

        # 3. Construction et exécution de la requête SQL pour récupérer les prix forward de l’année cal_year
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

        # 4. Fermeture du curseur et de la connexion
        cursor.close()
        connection.close()

        # 5. Indication de succès via le spinner (affiche une coche verte)
        spinner.ok("✅")

    # 6. Conversion des résultats en DataFrame pandas
    forward_df = pd.DataFrame(rows, columns=columns)

    # 7. Nettoyage et typage des colonnes
    forward_df['trading_date'] = pd.to_datetime(forward_df['trading_date'], utc=True)
    forward_df['forward_price'] = forward_df['forward_price'].astype(float)
    forward_df['cal_year'] = cal_year

    # 8. Retour du DataFrame final avec les prix forward annuels
    return forward_df

def get_forward_price_fr_months(token: str, cal_year_month: str) -> pd.DataFrame:
    """
    Récupère les prix forward mensuels d’électricité en France depuis Databricks (EEX).

    Paramètres :
    -----------
    token : str
        Token personnel d’accès à Databricks.
    cal_year_month : str
        Mois de livraison au format 'YYYY-MM' (exemple : '2025-03').

    Retour :
    -------
    pd.DataFrame
        DataFrame avec les colonnes ['trading_date', 'forward_price', 'cal_year'] 
        contenant les dates de trading, les prix forward mensuels et le mois associé.
    """

    # 1. Calcul des bornes temporelles pour le mois donné
    start_date = datetime.strptime(cal_year_month, "%Y-%m")
    end_date = start_date + relativedelta(months=1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # 2. Construction de l'URL de connexion Databricks avec le token
    databricks_url = (
        f"databricks+thrift://{token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )

    # 3. Initialisation du spinner yaspin pour indiquer le chargement
    with yaspin(text="Chargement des prix forward mensuels depuis Databricks...", color="cyan") as spinner:
        
        # 4. Connexion et récupération des données via tio.db
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

        # 5. Indication de succès via le spinner (affiche une coche verte)
        spinner.ok("✅")

    # 6. Nettoyage des données et renommage des colonnes
    forward_df.rename(columns={'setllement_price': 'forward_price'}, inplace=True)
    forward_df['trading_date'] = pd.to_datetime(forward_df['trading_date'], utc=True)
    forward_df['forward_price'] = forward_df['forward_price'].astype(float)
    forward_df['cal_year'] = cal_year_month

    # 7. Suppression des doublons éventuels
    forward_df = forward_df.drop_duplicates()

    # 8. Retour du DataFrame final avec les prix forward mensuels
    return forward_df

def get_pfc_fr(token: str, price_date: int, delivery_year: int) -> pd.DataFrame:
    """
    Récupère les courbes de prix Price Forward Curve (« PFC ») pour la France depuis Databricks.

    Paramètres :
    -----------
    token : str
        Token personnel d’accès à Databricks.
    price_date : int
        Année de la date de prix (exemple : 2024).
    delivery_year : int
        Année de livraison des prix forward (exemple : 2025).

    Retour :
    -------
    pd.DataFrame
        DataFrame contenant les colonnes ['delivery_from', 'price_date', 'forward_price'] 
        correspondant aux dates de livraison, date de prix et prix forward associés.
    """

    # 1. Initialisation du spinner yaspin pour indiquer le chargement
    with yaspin(text="Chargement des courbes de prix forward depuis Databricks...", color="cyan") as spinner:

        # 2. Connexion à la base Databricks avec token personnel
        connection = sql.connect(
            server_hostname="octoenergy-oefr-prod.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/ddb864eabbe6b908",
            access_token=token
        )

        cursor = connection.cursor()

        # 3. Requête SQL pour récupérer les données filtrées sur mode, asset, année livraison et année date prix
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

        # 4. Exécution de la requête et récupération des résultats
        cursor.execute(query)
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]

        # 5. Fermeture des connexions
        cursor.close()
        connection.close()

        # 6. Indication de succès via le spinner (affiche une coche verte)
        spinner.ok("✅")

    # 7. Conversion en DataFrame pandas avec noms des colonnes
    df = pd.DataFrame(rows, columns=colnames)

    # 8. Retour du DataFrame final avec les courbes de prix forward
    return df

def calculate_prem_risk_vol(forecast_df: pd.DataFrame,spot_df: pd.DataFrame,forward_df: pd.DataFrame,quantile: int = 70,plot_chart: bool = False,variability_factor: float = 1.1,save_path: Optional[str] = None) -> float:
    """
    Calcule la prime de risque volume à partir des prévisions de consommation, des prix spot et 
    d’un ensemble de prix forward. Cette prime mesure l’impact de l’erreur de prévision sur la valeur 
    économique, en supposant un écart entre consommation réelle et prévision.

    Paramètres :
    -----------
    forecast_df : pd.DataFrame
        Données de consommation et prévisions, contenant :
            - une colonne 'timestamp' (datetime)
            - une colonne 'forecast' (prévisions de consommation en MW)
            - une colonne 'MW' (consommation réalisée en MW)
    spot_df : pd.DataFrame
        Données de prix spot avec les colonnes ['delivery_from', 'price_eur_per_mwh'].
    forward_df : pd.DataFrame
        Liste des prix forward (calendaires ou autres), avec au minimum la colonne ['forward_price'].
    quantile : int, par défaut 70
        Le quantile à extraire (entre 1 et 100) de la distribution des primes calculées.
    plot_chart : bool, par défaut False
        Si True, affiche un graphique interactif de la distribution des primes de risque volume.
    variability_factor : float, par défaut 1.1
        Facteur multiplicatif appliqué à l’erreur de prévision pour simuler une incertitude plus élevée.
    save_path : str, optionnel
        Si défini, sauvegarde le graphique au format HTML à ce chemin.

    Retour :
    -------
    float
        La valeur du quantile demandé (en €/MWh), représentant la prime de risque volume.
    """

    # 1. Conversion des colonnes temporelles en datetime sans timezone
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp']).dt.tz_localize(None)
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from']).dt.tz_localize(None)

    # 2. Année de référence basée sur la dernière date de prévision
    latest_date = forecast_df['timestamp'].max()
    latest_year = latest_date.year
    print(f"Using year from latest date: {latest_year} (latest forecast: {latest_date.strftime('%Y-%m-%d')})")

    # 3. Vérification des prix forward
    if forward_df.empty:
        raise ValueError("No forward prices provided.")
    forward_prices = forward_df['forward_price'].tolist()

    # 4. Jointure forecast + spot
    merged_df = pd.merge(
        forecast_df,
        spot_df,
        left_on='timestamp',
        right_on='delivery_from',
        how='inner'
    )
    if merged_df.empty:
        raise ValueError("No data available to merge spot and forecast.")

    # 5. Simulation de l’erreur de prévision (écart entre réel et prévu)
    merged_df['diff_conso'] = (merged_df['MW'] - merged_df['forecast']) * variability_factor
    conso_totale_MWh = merged_df['MW'].sum()
    if conso_totale_MWh == 0:
        raise ValueError("Annual consumption is zero, division not possible.")

    # 6. Calcul de la prime de risque pour chaque prix forward
    premiums = []
    for fwd_price in forward_prices:
        merged_df['diff_price'] = merged_df['price_eur_per_mwh'] - fwd_price
        merged_df['produit'] = merged_df['diff_conso'] * merged_df['diff_price']
        premium = abs(merged_df['produit'].sum()) / conso_totale_MWh
        premiums.append(premium)

    # 7. Visualisation de la distribution (optionnel)
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
            yaxis_title="Premium (€/MWh)",
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

    # 8. Extraction du quantile demandé
    if not (1 <= quantile <= 100):
        raise ValueError("Quantile must be an integer between 1 and 100.")
    quantile_value = np.percentile(premiums, quantile)
    print(f"Quantile {quantile} risque volume = {quantile_value:.4f} €/MWh")
    return float(quantile_value)

def calculate_prem_risk_shape(forecast_df: pd.DataFrame,pfc_df: pd.DataFrame,spot_df: pd.DataFrame,quantile: int = 70,plot_chart: bool = False,save_path: Optional[str] = None) -> float:
    """
    Calcule la prime de risque de shape à partir d'une prévision de consommation, des prix forward (PFC)
    et des prix spot. Le résultat représente une mesure du risque pris lorsqu'on achète un produit
    à profil plat et qu'on revend au profil réel sur le marché spot.

    Paramètres :
    -----------
    forecast_df : pd.DataFrame
        Données de prévision de consommation, avec :
            - une colonne 'timestamp' (datetime)
            - une colonne 'forecast' (prévisions de consommation en MW)
    pfc_df : pd.DataFrame
        Données de prix forward (PFC) avec les colonnes :
            ['delivery_from', 'forward_price', 'price_date'].
    spot_df : pd.DataFrame
        Données de prix spot avec les colonnes :
            ['delivery_from', 'price_eur_per_mwh'].
    quantile : int, par défaut 70
        Le quantile à extraire de la distribution des coûts shape (en valeur absolue).
    plot_chart : bool, par défaut False
        Si True, affiche un graphique interactif (Plotly) des valeurs triées de prime de shape.
    save_path : str, optionnel
        Si défini, sauvegarde le graphique interactif au format HTML à ce chemin.

    Retour :
    -------
    float
        La valeur du quantile demandé (en €/MWh), mesurant la prime de risque de shape.
    """

    # 1. Prétraitement de la prévision de consommation
    df_conso_prev = forecast_df.copy()
    df_conso_prev = df_conso_prev.rename(columns={'timestamp': 'delivery_from'})
    df_conso_prev['delivery_from'] = pd.to_datetime(df_conso_prev['delivery_from'], utc=True)
    df_conso_prev['forecast'] = df_conso_prev['forecast'] / 1_000_000  # Conversion MW -> GWh

    # 2. Prétraitement des données PFC
    pfc = pfc_df.copy()
    pfc['delivery_from'] = pd.to_datetime(pfc['delivery_from'], utc=True)

    # 3. Fusion PFC + prévisions conso (jour)
    df = pd.merge(pfc, df_conso_prev[['delivery_from', 'forecast']], on='delivery_from', how='left').dropna()
    df['value'] = df['forward_price'] * df['forecast']
    df['delivery_month'] = pd.to_datetime(df['delivery_from'].dt.tz_localize(None)).dt.to_period('M')
    df['price_date'] = pfc['price_date']

    # 4. Agrégation mensuelle pour simuler un profil plat
    gb_month = df.groupby(['price_date', 'delivery_month']).agg(
        bl_volume_month=('forecast', 'mean'),
        bl_value_month=('value', 'sum'),
        forward_price_sum_month=('forward_price', 'sum')
    )
    gb_month['bl_value_month'] = gb_month['bl_value_month'] / gb_month['forward_price_sum_month']
    gb_month.reset_index(inplace=True)

    # 5. Prétraitement des données spot
    spot = spot_df.copy()
    spot = spot.rename(columns={'price_eur_per_mwh': 'spot_price'})
    spot['delivery_from'] = pd.to_datetime(spot['delivery_from'], utc=True)

    # 6. Fusion conso + PFC + spot
    df = df.merge(spot[['delivery_from', 'spot_price']], on='delivery_from', how='left').dropna()
    df = df.merge(gb_month, on=['price_date', 'delivery_month'], how='left').dropna()

    # 7. Calcul des volumes résiduels entre profil réel et plat
    df['residual_volume'] = df['forecast'] - df['bl_value_month']
    df['residual_value'] = df['residual_volume'] * df['spot_price']

    # 8. Agrégation mensuelle des coûts shape
    agg = df.groupby(['price_date']).agg(
        residual_value_month=('residual_value', 'sum'),
        conso_month=('forecast', 'sum')
    )
    agg['shape_cost'] = agg['residual_value_month'] / agg['conso_month']
    agg['abs_shape_cost'] = agg['shape_cost'].abs()

    # 9. Affichage graphique (optionnel)
    if plot_chart or save_path:
        sorted_vals = agg['abs_shape_cost'].sort_values().reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sorted_vals,
            x=list(range(1, len(sorted_vals) + 1)),
            mode='lines+markers',
            name='Shape Risk',
            line=dict(color='cyan')
        ))
        fig.update_layout(
            title="Shape Risk Distribution",
            xaxis_title="Index (sorted)",
            yaxis_title="€/MWh",
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

    # 10. Extraction du quantile souhaité
    quantile_value = np.percentile(agg['abs_shape_cost'], quantile)
    print(f"Quantile {quantile} risque shape = {quantile_value:.4f} €/MWh")
    return float(quantile_value)




