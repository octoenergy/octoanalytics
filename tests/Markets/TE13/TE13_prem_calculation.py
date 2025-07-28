from pathlib import Path
import pandas as pd

# Charging token with .env file
load_dotenv()
my_token = os.environ["DATABRICKS_TOKEN"]


databricks = (
        f"databricks+thrift://{my_token}@octoenergy-oefr-prod.cloud.databricks.com"
        "?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"
    )


INPUT_FOLDER  = Path("tests/Markets/TE13/input_data/")   # dossier des .csv
OUTPUT_FOLDER = Path("tests/Markets/TE13/premium_risk_charts/")     # sous-dossier d’export
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


# 4. Charger les températures lissées France sur la période des deux années d'historique glissant
temp_start_date = input_data['timestamp'].min().strftime('%Y-%m-%d')
temp_end_date = input_data['timestamp'].max().strftime('%Y-%m-%d')
temp_df = get_temp_smoothed_fr(temp_start_date, temp_end_date)
temp_df['timestamp'] = temp_df['timestamp'].dt.tz_convert('UTC') if temp_df['timestamp'].dt.tz else temp_df['timestamp'].dt.tz_localize('UTC')
temp_df['datetime'] = temp_df['datetime'].dt.tz_convert('UTC') if temp_df['datetime'].dt.tz else temp_df['datetime'].dt.tz_localize('UTC')


# 5. Génerer le forecast sur l'année civile complète
#forecast_df = eval_forecast(input_data, temp_df=temp_df, cal_year=cal_year, datetime_col='timestamp', target_col='MW')
forecast_df = eval_forecast(input_data, temp_df, cal_year=2024, plot_chart=True)


# 6. Calculer la prime de risque
calculate_prem_risk_vol(forecast_df=forecast_df,spot_df=spot_df,forward_df=forward_df,quantile=70, plot_chart=False)

# 7. Calculer la prime de shape
calculate_prem_risk_shape(forecast_df=forecast_df,pfc_df=pfc_df,spot_df=spot_df,quantile=70,plot_chart=False)


