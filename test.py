# test_spot_price_script.py
from octoanalytics import get_temperature_lissee_france




# Charger tes données
train_df = pd.read_csv('data_2023.txt')
test_df = pd.read_csv('data_2024.txt')

# Appeler la fonction pour obtenir le DataFrame avec les prévisions
output = eval_forecast(train_df, test_df, datetime_col='AXPO_SELL', target_col='MW')
output.to_csv('forecast_2024.csv', index=False)  # Enregistrer le DataFrame dans un fichier CSV


# Calculer le MAPE
y_test = output['MW']
y_pred = output['forecast']
temperature = output['temperature']
dates = output['AXPO_SELL']
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# Créer le graphique interactif avec Plotly
fig = go.Figure()

# Courbe réelle
fig.add_trace(go.Scatter(
    x=dates, y=y_test, mode='lines', name='Consommation Réelle', line=dict(color='blue')
))

# Courbe prévision
fig.add_trace(go.Scatter(
    x=dates, y=y_pred, mode='lines', name='Prévision', line=dict(color='red')
))

# Courbe température (axe secondaire)
fig.add_trace(go.Scatter(
    x=dates, y=temperature, mode='lines', name='Température (°C)', line=dict(color='green'), yaxis='y2'
))

# Mettre à jour les axes
fig.update_layout(
    title=f'Réel vs Prévision vs Température\nMAPE = {mape:.2f}%',
    xaxis_title='Date',
    yaxis=dict(title='MW'),
    yaxis2=dict(
        title='Température (°C)',
        overlaying='y',
        side='right'
    ),
    xaxis_tickangle=45
)

fig.show()


# Requete from table stg_eex_power_future_results_fr

#SELECT setllement_price,traded_volume, trading_date,delivery_start,delivery_end FROM stg_eex_power_future_results_fr
#WHERE delivery_start = "2024-01-01" AND delivery_end = "2024-12-31" AND product = "F7BY"
#ORDER BY trading_date DESC











