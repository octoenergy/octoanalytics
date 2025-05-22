from octoanalytics import get_temperature_lissee_france
from octoanalytics import eval_forecast
from octoanalytics import plot_forecast
from octoanalytics import calculate_mape
from octoanalytics import get_spot_price_fr
from octoanalytics import get_forward_price_fr
from octoanalytics import get_pfc_fr
from octoanalytics import calculate_prem_risk_vol


# Load input data
input_df_c2 = pd.read_csv('tests/data/cdc_c2_sdet_lot4.csv', index_col=0)
input_df_c4 = pd.read_csv('tests/data/cdc_c4_sdet_lot4.csv', index_col=0)


# Charging token with .env file
load_dotenv()
my_token = os.environ["DATABRICKS_TOKEN"]

# Testing the functions

eval_forecast(input_df_c2, datetime_col='interval_start', target_col='interval_value_W')

plot_forecast(input_df_c2, datetime_col='interval_start', target_col='interval_value_W')

calculate_mape(input_df_c2, datetime_col='interval_start', target_col='interval_value_W')

output_spot = get_spot_price_fr(my_token, start_date = "2024-06-12", end_date = "2024-09-12")

output_forward = get_forward_price_fr(my_token, cal_year = 2026)

output_pfc = get_pfc_fr(my_token, cal_year = 2028) 

calculate_prem_risk_vol(my_token, input_df_c4, datetime_col='interval_start', target_col='interval_value_W', plot_chart=True, quantile = 40)


