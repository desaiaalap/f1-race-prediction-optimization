import joblib
import pandas as pd
from ingestion.bahrain_data_fetcher import fetch_multiple_years
from preprocessing.feature_builder import build_driver_features

# Load the trained model
model = joblib.load('trained_model.pkl')
print("✅ Loaded trained model")

# Fetch the 2024 data (only)
df_laps_2024, df_qual_2024 = fetch_multiple_years([2024],
                                                  save_laps=True,
                                                  save_telemetry=True,
                                                  save_weather=True,
                                                  save_qualifying=True)

# Step 2: Feature engineering for 2024 data
df_features_2024 = build_driver_features(df_laps_2024, df_qual_2024)

# Step 3: Predict on 2024 data using the loaded model
predictions_2024 = model.predict(df_features_2024)
# Get the predicted probabilities (not just binary prediction)
pred_probs = model.predict_proba(df_features_2024)[:, 1]  # Get probability for Top 10 (class 1)

# Create a DataFrame for predictions
df_features_2024['predicted_prob'] = pred_probs

# Rank drivers based on the predicted probabilities (descending)
df_features_2024['PredictedPosition'] = df_features_2024['predicted_prob'].rank(ascending=False, method='min')

# Sort by predicted position to see the order
df_sorted_2024 = df_features_2024.sort_values(by='PredictedPosition', ascending=True)

# Print the predicted standings with their positions
print("🚦 Predicted Standings for 2024 Bahrain GP:")
print(df_sorted_2024[['DriverNumber', 'PredictedPosition']].reset_index(drop=True))
