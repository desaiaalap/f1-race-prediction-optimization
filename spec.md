# Formula 1: Race Prediction & Strategy Optimization

## Goal
Predict finishing positions and recommend pit-stop strategies that beat teams’ real choices.

## Data
- Source  : FastF1 API (`fastf1.ergast`)
- Seasons : **2022, 2023, 2024**
- Saved at `data/raw/{season}_{race}.parquet`

## Pipeline
1. Ingestion & caching
2. Pre-processing (tire-wear %, lap deltas, weather buckets)
3. Modeling  
   - XGBoost classifier  
   - TensorFlow sequence net on lap series
4. Evaluation (macro-F1, top-10 accuracy)
5. Strategy RL agent (keras-rl) to choose pit lap & compound

## Success criteria
- ≥ 85 % macro-F1 on 2023 test races
- RL agent beats real strategy in ≥ 60 % of 2024 races
- End-to-end CLI run ≤ 10 min on an M2 laptop

## Tools
Poetry, FastF1, XGBoost, TensorFlow, Optuna, TensorBoard
