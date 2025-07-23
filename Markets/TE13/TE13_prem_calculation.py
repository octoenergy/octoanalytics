from octoanalytics import get_temp_smoothed_fr
from octoanalytics import eval_forecast
from octoanalytics import plot_forecast
from octoanalytics import get_spot_price_fr
from octoanalytics import get_forward_price_fr_annual
from octoanalytics import get_pfc_fr
from octoanalytics import calculate_prem_risk_vol
from octoanalytics import calculate_prem_risk_shape

from pathlib import Path
import pandas as pd

# Charging token with .env file
load_dotenv()
my_token = os.environ["DATABRICKS_TOKEN"]


databricks = (
        f"databricks+thrift://{my_token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )


INPUT_FOLDER  = Path("Markets/TE13/input_data/")   # dossier des .csv
OUTPUT_FOLDER = Path("Markets/TE13/premium_risk_charts/")     # sous-dossier d’export
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)    # crée s'il n’existe pas



# 1. Année civile complète pour les prix spot et forward (entrée utilisateur)
cal_year = 2024

# 2. Charger les prix spot , forward  et la PFC sur l’année civile complète
spot_df = get_spot_price_fr(my_token, start_date = f"{cal_year}-01-01", end_date = f"{cal_year}-12-31")
forward_df = get_forward_price_fr_annual(my_token, cal_year=cal_year)
pfc_df = get_pfc_fr(token=my_token, price_date= cal_year-1 , delivery_year=cal_year)


# 3. Load input data
input_data = pd.read_csv(INPUT_FOLDER / 'C2_ENT3_HCE_cdc_historique.csv')
input_data['timestamp'] = pd.to_datetime(input_data['timestamp'], utc=True)
print(input_data['timestamp'].dt.tz)  # sera None si naïf


# 4. Charger les températures lissées France sur la période des deux années d'historique glissant
temp_start_date = input_data['timestamp'].min().strftime('%Y-%m-%d')
temp_end_date = input_data['timestamp'].max().strftime('%Y-%m-%d')
temp_df = get_temp_smoothed_fr(temp_start_date, temp_end_date)

# 5. Génerer le forecast sur l'année civile complète
forecast_df = eval_forecast(input_data, temp_df=temp_df, cal_year=cal_year, datetime_col='timestamp', target_col='MW')
forecast_df = forecast_df.rename(columns={'timestamp': 'datetime', 'MW': 'consommation_realisee'})

# 6. Calculer la prime de risque
calculate_prem_risk_vol(forecast_df=forecast_df,spot_df=spot_df,forward_df=forward_df,plot_chart=True,quantile=70)


# 7. Calculer la prime de shape
calculate_prem_risk_shape(forecast_df=forecast_df,pfc_df=pfc_df,spot_df=spot_df,quantile=70,plot_chart=True)











#output.to_csv('Markets/TE13/output_data/Prev_C2_ENT3_HCE.csv', index=False, sep=';', encoding='utf-8')






# Calcul des primes de risque pour chaque fichier CSV dans le dossier


for file_path in INPUT_FOLDER.iterdir():
    if not file_path.is_file() or file_path.suffix.lower() != ".csv":
        continue                                   # ignore dossiers & non-CSV

    print(file_path.name)

    # Nom « propre » pour les exports
    base_name = (file_path.stem.split("_cdc_historique")[0]
                 or file_path.stem)

    # Lecture CSV (latin1 si nécessaire)
    df = pd.read_csv(file_path, encoding="latin1")

    # --------- 1) Forecast ----------------------------------
    forecast_html = OUTPUT_FOLDER / f"{base_name}_forecast.html"
    plot_forecast(
        df,
        datetime_col="timestamp",
        target_col="MW",
        save_path=forecast_html
    )

    # --------- 2) Prime de risque ---------------------------
    premium_html = OUTPUT_FOLDER / f"{base_name}_premium_dist.html"
    calculate_prem_risk_vol(
        token=my_token,
        input_df=df,
        datetime_col="timestamp",
        target_col="MW",
        plot_chart=False,
        save_path=premium_html
    )

    print(f"✔️  exports créés dans {OUTPUT_FOLDER.name} :")
    print(f"   - {forecast_html.name}")
    print(f"   - {premium_html.name}")




# ------------------------------------------------------------------
# PARAMÈTRES
# ------------------------------------------------------------------
QUANTILES     = [70, 80, 90]                     # colonnes à produire
TOKEN         = my_token                         # ton token Databricks
DATETIME_COL  = "timestamp"
TARGET_COL    = "MW"

# ------------------------------------------------------------------
# UTILITAIRE LECTURE ROBUSTE CSV (encodage incertain)
# ------------------------------------------------------------------
def read_csv_robust(fp):
    """Essaie UTF-8, latin-1, cp1252 ; lève une erreur sinon."""
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Impossible de lire {fp} avec encodages connus.")

# ------------------------------------------------------------------
# BOUCLE SUR LES FICHIERS
# ------------------------------------------------------------------
rows = []

for file_path in INPUT_FOLDER.iterdir():
    if not file_path.is_file() or file_path.suffix.lower() != ".csv":
        continue  # ignore sous-dossiers ou autres extensions

    base_name = file_path.stem.split("_cdc_historique")[0] or file_path.stem
    df = read_csv_robust(file_path)

    result_line = {"file": base_name}

    for q in QUANTILES:
        premium = calculate_prem_risk_vol(
            token=TOKEN,
            input_df=df,
            datetime_col=DATETIME_COL,
            target_col=TARGET_COL,
            quantile=q,
            plot_chart=False,
            save_path=None      # pas de graphique
        )
        result_line[f"q{q}"] = premium

    rows.append(result_line)

# ------------------------------------------------------------------
# TABLEAU FINAL
# ------------------------------------------------------------------
summary_df = pd.DataFrame(rows).set_index("file").sort_index()
print(summary_df)           # affiche dans la console/notebook

# (Facultatif) — Export
summary_df.to_csv(INPUT_FOLDER / "premium_summary.csv")          # CSV




# -------------------- Calcul premium de risque volume


import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import tentaclio as tio
from scipy.stats import norm
from numpy import percentile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


# Fabrication des données de forecast
df_conso_prev = eval_forecast(input_data, datetime_col='timestamp', target_col='MW')

#rename des colonnes
df_conso_prev.rename(columns={'timestamp': 'delivery_from','MW': 'consumption_2024', 'forecast': 'prev_2024'}, inplace=True)

#conversion de la colonne delivery_from en datetime
df_conso_prev['delivery_from'] = pd.to_datetime(df_conso_prev['delivery_from'])

#consersion au pas de temps demi horaire en faisant la moyenne
#df_conso_prev = df_conso_prev.resample('30T', on='delivery_from').sum().reset_index()

#selection des colonnes
df_conso_prev = df_conso_prev[['delivery_from','prev_2024']]

df_conso_prev['prev_2024'] = df_conso_prev['prev_2024'] / 1000000 # Conversion de MWh en Wh

#PFC du UK avec Prix FR
def get_PFC(price_date, delivery_from ):
    with tio.db(databricks) as client:
        query = f"""

        SELECT delivery_from
       ,price_date
       ,forward_price
        FROM consumer.stg_octo_curves 
        WHERE mode='EOD' AND asset = 'FRPX' 
        and year(delivery_from)= '{delivery_from}'
        and year(price_date) = '{price_date}' and year(price_date) <2025
        ORDER BY price_date

        """
    
        df_PFC = client.get_df(query)  

        return df_PFC  
    
pfc = get_PFC(2023, 2024 )
#pfc2 = get_forward_price_fr_annual(my_token, 2024)


# 1) convertir / localiser en UTC
pfc['delivery_from'] = (pd.to_datetime(pfc['delivery_from']).dt.tz_convert('UTC'))
df_conso_prev['delivery_from'] = (pd.to_datetime(df_conso_prev['delivery_from']).dt.tz_localize('UTC', ambiguous='infer'))

df = pfc.merge(df_conso_prev,how='left',left_on = 'delivery_from',right_on = 'delivery_from').dropna()

df['value'] = df['forward_price'] * df['prev_2024']
df['delivery_year'] = df['delivery_from'].dt.to_period('y')
df['delivery_month'] = df['delivery_from'].dt.to_period('M')



#BLOCS MONTH
gb_month = df.groupby(['price_date','delivery_month']).agg(bl_volume_month=('prev_2024', 'mean')
                                       ,bl_value_month=('value', 'sum')
                                       ,forward_price_sum_month=('forward_price','sum'))
gb_month['bl_value_month'] = gb_month['bl_value_month'] / gb_month['forward_price_sum_month']


gb_month.reset_index(inplace=True)


#PFC du UK avec Prix FRANCE1
def get_spot_price(delivery_year):
    with tio.db(databricks) as client:
        query = f"""

        select delivery_from,price_eur_per_mwh
        from consumer.inter_energymarkets_epex_hh_spot_prices
        where source_identifier = 'epex'
        and year(delivery_from) = {delivery_year}
        order by delivery_from

        """
        df_spot_price = client.get_df(query)  

        return df_spot_price 
spot_price = get_spot_price(2024)
#merge df et spot price
df = df.merge(spot_price,how='left',left_on = 'delivery_from',right_on = 'delivery_from').dropna()

#rename des colonnes
df.rename(columns={'price_eur_per_mwh': 'spot_price'}, inplace=True)
#merge gb_month et df on price_date et delivery_month
df = df.merge(gb_month,how='left',left_on = ['price_date','delivery_month'],right_on = ['price_date','delivery_month']).dropna()
df['residual_volume'] =  df['prev_2024'] - df['bl_value_month']
df['residual_value'] = df['residual_volume'] * df['spot_price']
#par price_date faire BLOC_CAL_BL = valorisation/(Prix_BL*nb_ligne)

redudual_value_MWh = df.groupby(['price_date']).agg(residual_value_month=('residual_value', 'sum')
                                           ,conso_month=('prev_2024', 'sum'))
redudual_value_MWh['shape_cost'] = redudual_value_MWh['residual_value_month']/redudual_value_MWh['conso_month']
redudual_value_MWh['abs_shape_cost'] = abs(redudual_value_MWh['shape_cost'])
#tracé du graphique

fig = px.bar(
    redudual_value_MWh.reset_index(),
    x='price_date',
    y='shape_cost',
    title="Exposure Value by Price Date (€/MWh)",
    height=500
)

fig.show()
# Trier les valeurs de 'residual_value_year' (et non le DataFrame entier)
sorted_values = redudual_value_MWh['abs_shape_cost'].sort_values().reset_index(drop=True)

# Tracer le graphique
plt.figure(figsize=(12, 6))
plt.plot(sorted_values)
plt.title('shape risk')
plt.xlabel('Simulation')
plt.ylabel('€/MWh')
plt.grid(True)
plt.show()

#centile 0,8 de simulation
centile = np.percentile(redudual_value_MWh['abs_shape_cost'], 80)
print(" le premium à appliquer est de : ", centile, '€/MWh')
