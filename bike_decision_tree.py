# Part 1: Download and Load Dataset
import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Constants
URL = "www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset"
DATASET = "london_merged.csv"
DATEN_ORDNER = Path("datasets")

# Function: Download and Extract Dataset
def load_data(url: str, ziel_ordner: Path):
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    if "datasets" in path_parts:
        index = path_parts.index("datasets") + 1
        quelle = '/'.join(path_parts[index:])
        zip_name = f"{path_parts[-1]}.zip"
        zip_file = ziel_ordner / zip_name

        if not zip_file.is_file():
            ziel_ordner.mkdir(parents=True, exist_ok=True)
            os.system(f"kaggle datasets download -d {quelle} -p {ziel_ordner}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(path=ziel_ordner)
    else:
        print("Invalid URL format")

# Function: Load CSV with Pandas
def pd_openCSVFile(csv_name: str, ordner: Path, encoding='utf-8'):
    csv_file = ordner / csv_name
    return pd.read_csv(csv_file, encoding=encoding)

# Load Dataset
load_data(url=URL, ziel_ordner=DATEN_ORDNER)
df = pd_openCSVFile(csv_name=DATASET, ordner=DATEN_ORDNER)

# Part 2: Preprocessing and Feature Engineering
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour_of_day'] = df['timestamp'].dt.hour

if 'day_of_week' in df.columns:
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

if 'month' in df.columns:
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df = df.drop(columns=['timestamp', 't2', 'day_of_week', 'month'], errors='ignore')

if 't1' in df.columns and 'wind_speed' in df.columns:
    df['temp_wind_interaction'] = df['t1'] * df['wind_speed']

# Prepare Data
X = df.drop('cnt', axis=1)
y = df['cnt']

numeric_features = ['t1', 'hum', 'wind_speed', 'hour_of_day',
                    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'temp_wind_interaction']
numeric_features = [feature for feature in numeric_features if feature in X.columns]

categorical_features = ['weather_code', 'is_holiday', 'is_weekend', 'season']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best Model: Decision Tree with Optimal Parameters
best_params = {'max_depth': 10}  # From results

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(**best_params))
])

# Train and Evaluate
dt_pipeline.fit(X_train, y_train)
dt_y_pred = dt_pipeline.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)

print(f"Decision Tree - MSE: {dt_mse}, R²: {dt_r2}")

