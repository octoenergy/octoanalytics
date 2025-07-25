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





# Iteration pour calculer les primes de risque pour chaque fichier CSV dans le dossier input_data
# ------------------------------------------------------------------

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


