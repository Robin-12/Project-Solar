import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for calculating additional features
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate the night column
        X['night'] = ((X['timestamp'] < X['Opkomst_datum']) | (X['timestamp'] > X['Ondergang_datum'])).astype(int)
        
        # Calculate hours_from_true_noon
        X['hours_from_true_noon'] = (X['timestamp'] - X['Op ware middag_datum']).dt.total_seconds() / 3600
        
        # Calculate ratio_hours_from_true_noon
        X['ratio_hours_from_true_noon'] = X['hours_from_true_noon'] / X['Uren zonlicht']
        
        return X

# Load datasets
forecast = pd.read_csv('forecast.csv')
sunset = pd.read_csv('sunset.csv')

# Convert to datetime
forecast['timestamp'] = pd.to_datetime(forecast['timestamp'], utc=True)
sunset['datum'] = pd.to_datetime(sunset['datum'])
sunset['Opkomst_datum'] = pd.to_datetime(sunset['Opkomst'])
sunset['Op ware middag_datum'] = pd.to_datetime(sunset['Op ware middag'])
sunset['Ondergang_datum'] = pd.to_datetime(sunset['Ondergang'])

# Merge datasets
forecast['date'] = forecast['timestamp'].dt.date
sunset['date'] = sunset['datum'].dt.date
merged = pd.merge(forecast, sunset, on='date', how='left')

# Define the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('feature_engineering', FeatureEngineering()),
    ('scaler', StandardScaler())
])

# Select the relevant columns for transformation
columns_to_transform = [
    'temp', 'pressure', 'cloudiness', 'humidity_relative', 'Uren zonlicht', 
    'night', 'hours_from_true_noon', 'ratio_hours_from_true_noon'
]

# Separate features and target if needed
X = merged[['timestamp', 'temp', 'pressure', 'cloudiness', 'humidity_relative', 'Opkomst_datum', 'Op ware middag_datum', 'Ondergang_datum', 'Uren zonlicht']]
X_transformed = pipeline.fit_transform(X)

# Convert the result back to a DataFrame
result_columns = columns_to_transform
result_df = pd.DataFrame(X_transformed, columns=result_columns)

print(result_df.head())
