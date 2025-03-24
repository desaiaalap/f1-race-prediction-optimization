import os
import requests
import pandas as pd
from datetime import datetime

# Create output directory
RAW_DATA_PATH = os.path.join("data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# Example: Get data for 2023 Bahrain GP (race_id = 1098)
BASE_URL = "https://api.openf1.org/v1"
RACE_ID = 1098  # You can loop through race IDs later

def fetch_openf1_data(endpoint: str, race_id: int):
    url = f"{BASE_URL}/{endpoint}?race_id={race_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        print(f"✅ Fetched {len(df)} records from {endpoint}")
        return df
    except Exception as e:
        print(f"❌ Error fetching {endpoint}: {e}")
        return pd.DataFrame()

def save_df(df: pd.DataFrame, filename: str):
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved: {filepath}")

if __name__ == "__main__":
    print(f"🚦 Starting OpenF1 data download for race_id={RACE_ID}")

    # 1. Lap-by-lap timing data
    laps_df = fetch_openf1_data("lap_times", RACE_ID)
    if not laps_df.empty:
        save_df(laps_df, f"lap_times_race_{RACE_ID}.csv")

    # 2. Pit stop data
    pit_df = fetch_openf1_data("pit", RACE_ID)
    if not pit_df.empty:
        save_df(pit_df, f"pit_data_race_{RACE_ID}.csv")

    print("✅ Data download complete.")
