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

output = eval_forecast(input_df_c4, datetime_col='timestamp', target_col='MW',keep_explanatory_variables="No")
output.to_csv('tests/data/output_C4_cdc_historique_WAAT.csv', index=False, sep=';', encoding='utf-8')

plot_forecast(input_df_c2, datetime_col='timestamp', target_col='MW')

calculate_mape(input_df_c4, datetime_col='timestamp', target_col='MW')

output_spot = get_spot_price_fr(my_token, start_date = "2024-06-12", end_date = "2024-09-12")

output_forward = get_forward_price_fr(my_token, cal_year = 2025)

output_pfc = get_pfc_fr(my_token, cal_year = 2028) 

calculate_prem_risk_vol(my_token, input_df_c4, datetime_col='timestamp', target_col='MW', plot_chart=True, quantile = 50, variability_factor = 1.1)


