from octoanalytics import get_temp_smoothed_fr
from octoanalytics import eval_forecast
from octoanalytics import plot_forecast
from octoanalytics import calculate_mape
from octoanalytics import get_spot_price_fr
from octoanalytics import get_forward_price_fr
from octoanalytics import get_pfc_fr
from octoanalytics import calculate_prem_risk_vol


# Load input data
input_df_c2 = pd.read_csv('tests/data/C2_cdc_historique_vdp_lot1.csv')
input_df_c4 = pd.read_csv('tests/data/C4_cdc_historique_vdp_lot1.csv')

input_df_c4 = pd.read_csv('tests/data/C4_cdc_historique_WAAT.csv')




# Charging token with .env file
load_dotenv()
my_token = os.environ["DATABRICKS_TOKEN"]

# Testing the functions

output = eval_forecast(input_df_c4, datetime_col='timestamp', target_col='MW',keep_explanatory_variables="Yes")
output.to_csv('tests/data/output_C4_cdc_historique_WAAT.csv', index=False)

plot_forecast(input_df_c2, datetime_col='timestamp', target_col='MW')

calculate_mape(input_df_c4, datetime_col='timestamp', target_col='MW')

output_spot = get_spot_price_fr(my_token, start_date = "2024-06-12", end_date = "2024-09-12")

output_forward = get_forward_price_fr(my_token, cal_year = 2025)

output_pfc = get_pfc_fr(my_token, cal_year = 2028) 

calculate_prem_risk_vol(my_token, input_df_c4, datetime_col='timestamp', target_col='MW', plot_chart=True, quantile = 50, variability_factor = 1.1)







databricks_url = f"databricks+thrift://{my_token}@octoenergy-oefr-prod.cloud.databricks.com?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"

with tqdm(total=1, desc="Loading spot prices from Databricks", bar_format='{l_bar}{bar} [elapsed: {elapsed}]') as pbar:
    with tio.db(databricks_url) as client:
        query = f"""
                SELECT delivery_from, price_eur_per_mwh
                FROM consumer.inter_energymarkets_epex_hh_spot_prices
                WHERE price_date >= '2024-06-12' AND price_date <= '2024-08-12'
                ORDER BY delivery_from;
        """
        spot_df = client.get_df(query)
        pbar.update(1)




databricks = f"databricks+thrift://{my_token}@octoenergy-oefr-prod.cloud.databricks.com?HTTPPath=/sql/1.0/warehouses/ddb864eabbe6b908"

# Read from DATABRICKS and rename 'utc' to 'time_utc'
with tio.db(databricks) as client:

        query = f"""
                    SELECT 
                        delivery_from AS time_utc
                        , forward_price 
                    FROM consumer.stg_octo_curves 
                    WHERE mode='EOD_EEX' AND asset = 'FRPX' AND price_date = (SELECT MAX(price_date) FROM consumer.stg_octo_curves WHERE mode='EOD_EEX' AND asset = 'FRPX') 
                    ORDER BY delivery_from
                """

        pfc = client.get_df(query)
pfc.assign(time_utc=lambda df: pd.to_datetime(df['time_utc'], utc=True), forward_price=lambda df: df['forward_price'].astype(float))