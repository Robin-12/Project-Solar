import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt  


# Load datasets
forecast = pd.read_csv('forecast.csv')
sunset = pd.read_csv('sunset.csv')

# Convert to datetime
forecast['timestamp'] = pd.to_datetime(forecast['timestamp'])
sunset['datum'] = pd.to_datetime(sunset['datum'])
sunset['Opkomst_datum'] = pd.to_datetime(sunset['Opkomst'])
sunset['Op ware middag_datum'] = pd.to_datetime(sunset['Op ware middag'])
sunset['Ondergang_datum'] = pd.to_datetime(sunset['Ondergang'])

# Merge datasets
forecast['date'] = forecast['timestamp'].dt.date
sunset['date'] = sunset['datum'].dt.date
df = pd.merge(forecast, sunset, on='date', how='left')
df.drop(columns=['datum', 'date'], inplace=True)
tijdstippen = df['timestamp'].dt.hour
tijdstippen_array = np.array(tijdstippen)

# Load model
model = joblib.load('zonnepaneel.pkl')

# Predict
predictions = model.predict(df)

# Print prediction
for tijd, voorspelling in zip(tijdstippen, predictions):
    print(f'Voorspelling voor {tijd}u: {voorspelling:.2f} kWh')

# Plot
plt.plot(df[tijdstippen], predictions)                



