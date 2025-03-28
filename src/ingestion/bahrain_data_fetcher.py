# src/ingestion/bahrain_data_fetcher.py

import fastf1
import pandas as pd
import os

CACHE_PATH = 'data/cache'
RAW_DATA_PATH = 'data/raw'
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(RAW_DATA_PATH, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_PATH)

def save_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(RAW_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"💾 Saved: {path}")

def download_bahrain_race(year: int,
                          save_laps: bool = True,
                          save_telemetry: bool = True,
                          save_weather: bool = True,
                          save_qualifying: bool = True):
    print(f"\n🚦 Downloading Bahrain GP {year} data...")

    try:
        schedule = fastf1.get_event_schedule(year)
        bahrain = schedule[schedule['EventName'].str.contains("Bahrain", case=False)]

        if bahrain.empty:
            print(f"⚠️ No Bahrain GP found for {year}")
            return None, None
        
        gp_round = int(bahrain.iloc[0]['RoundNumber'])
        session = fastf1.get_session(year, gp_round, 'R')
        session.load()

        # 1. Save lap data
        laps_df, qual_df = None, None
        if save_laps:
            laps_df = session.laps.copy()
            laps_df['year'] = year
            save_csv(laps_df, f"{year}_Bahrain_laps.csv")

        # 2. Save telemetry
        if save_telemetry:
            try:
                fastest = session.laps.pick_fastest()
                telemetry = fastest.get_telemetry()
                save_csv(telemetry, f"{year}_Bahrain_fastestlap_telemetry.csv")
            except Exception:
                print(f"⚠️ Telemetry missing for fastest lap in {year}")

        # 3. Save weather data
        if save_weather:
            weather = session.weather_data
            if weather is not None:
                save_csv(weather, f"{year}_Bahrain_weather.csv")

        # 4. Save qualifying data
        if save_qualifying:
            try:
                qual_df = session.results[['DriverNumber', 'GridPosition']].copy()
                qual_df['DriverNumber'] = qual_df['DriverNumber'].astype(str)
                qual_df['year'] = year
                save_csv(qual_df, f"{year}_Bahrain_qualifying.csv")
            except Exception:
                print(f"⚠️ Qualifying results missing for {year}")

        print(f"✅ Finished {year}")
        return laps_df, qual_df

    except Exception as e:
        print(f"❌ Failed {year}: {e}")
        return None, None

def fetch_multiple_years(years, **kwargs):
    all_laps = []
    all_qual = []
    for year in years:
        laps_df, qual_df = download_bahrain_race(year, **kwargs)
        if laps_df is not None:
            all_laps.append(laps_df)
        if qual_df is not None:
            all_qual.append(qual_df)
    return pd.concat(all_laps, ignore_index=True), pd.concat(all_qual, ignore_index=True)
