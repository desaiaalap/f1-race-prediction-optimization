# main.py

from ingestion.bahrain_data_fetcher import fetch_multiple_years
from preprocessing.feature_builder import build_driver_features
from modeling.model_runner import train_and_evaluate

# Step 1: Fetch all race + qualifying data in memory
df_laps, df_qual = fetch_multiple_years(range(2018, 2024),
                                        save_laps=True,
                                        save_telemetry=True,
                                        save_weather=True,
                                        save_qualifying=True)

# Step 2: Build features and train model
df_features = build_driver_features(df_laps, df_qual)
model = train_and_evaluate(df_features)
