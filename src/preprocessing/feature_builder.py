# src/preprocessing/feature_builder.py

import pandas as pd

def build_driver_features(laps_df: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    laps_df = laps_df[laps_df['LapTime'].notna()].copy()

    # Ensure LapTime is timedelta type
    if not pd.api.types.is_timedelta64_dtype(laps_df['LapTime']):
        laps_df['LapTime'] = pd.to_timedelta(laps_df['LapTime'], errors='coerce')
    laps_df['lap_time_sec'] = laps_df['LapTime'].dt.total_seconds()

    laps_df['is_pit'] = laps_df['PitInTime'].notna()

    # Final position from last lap
    final_laps = (
        laps_df.groupby(['year', 'DriverNumber'])['LapNumber']
        .max()
        .reset_index()
    )
    final_pos = laps_df.merge(final_laps, on=['year', 'DriverNumber', 'LapNumber'])[
        ['year', 'DriverNumber', 'Position']
    ].rename(columns={"Position": "FinalPosition"})

    # Now we merge the final positions with the main DataFrame (laps_df)
    laps_df = laps_df.merge(final_pos, on=['year', 'DriverNumber'], how='left')

    # Feature engineering (driver-level aggregation)
    summary = laps_df.groupby(['year', 'DriverNumber']).agg({
        'lap_time_sec': 'mean',
        'is_pit': 'sum',
        'Stint': 'nunique',
        'Compound': lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN',
        'FinalPosition': 'mean',  # Average position over all laps
    }).reset_index()

    summary.rename(columns={
        'lap_time_sec': 'avg_lap_time',
        'is_pit': 'total_pit_stops',
        'Stint': 'num_stints',
        'Compound': 'main_compound',
        'FinalPosition': 'avg_finish_position',  # Renamed the final position column for clarity
    }, inplace=True)

    # Merge with final positions
    summary = summary.merge(final_pos, on=['year', 'DriverNumber'])

    # Additional feature engineering:
    # 1. Driver's position at lap 10, lap 20, lap 30, etc.
    for lap in [5, 10, 15, 20, 25, 30, 40, 50]:
        lap_position = laps_df[laps_df['LapNumber'] == lap][['year', 'DriverNumber', 'Position']]
        lap_position.rename(columns={'Position': f'position_at_lap_{lap}'}, inplace=True)
        summary = summary.merge(lap_position, on=['year', 'DriverNumber'], how='left')

    # 2. Race distance completed (total laps completed)
    total_laps = laps_df.groupby(['year', 'DriverNumber'])['LapNumber'].max().reset_index()
    total_laps.rename(columns={'LapNumber': 'total_laps'}, inplace=True)
    summary = summary.merge(total_laps, on=['year', 'DriverNumber'], how='left')

    # 3. Average finishing position from previous races
    driver_avg_position = laps_df.groupby('DriverNumber')['FinalPosition'].mean().reset_index()
    driver_avg_position.rename(columns={'FinalPosition': 'avg_finish_position'}, inplace=True)
    summary = summary.merge(driver_avg_position, on='DriverNumber', how='left')

    # Merge qualifying grid position
    qualifying_df['DriverNumber'] = qualifying_df['DriverNumber'].astype(str)
    summary['DriverNumber'] = summary['DriverNumber'].astype(str)
    summary = summary.merge(qualifying_df, on=['year', 'DriverNumber'], how='left')

    # 4. Lap time differences within a stint (to capture tire wear)
    laps_df['lap_time_diff'] = laps_df.groupby(['year', 'DriverNumber', 'Stint'])['lap_time_sec'].diff().fillna(0)

    return summary
