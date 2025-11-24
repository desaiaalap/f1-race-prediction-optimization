# 🏎️ Formula 1: Race Prediction

This project predicts Formula 1 race outcomes using machine learning. Built as a personal deep dive into motorsports analytics, the goal was to explore how data impacts race predictions.

## 📌 What It Does

- **Race Outcome Prediction**
  Trains an XGBoost model on FastF1 telemetry data to predict whether a driver finishes in the top 10.
  Uses features like tire compound, lap times, pit stop behavior, stint count, and qualifying position.

- **Driver Performance Feature Engineering**
  Extracts 60+ features: average lap time, tire changes, pit stop counts, lap-wise position tracking, and stint analysis.

- **Testable and Modular**
  Includes scripts for ingestion (`FastF1`), feature building, model training, and prediction on unseen data.

---

## 🧠 Tech Stack

- Python, Pandas, NumPy
- FastF1 (telemetry ingestion)
- XGBoost (binary classification)
- scikit-learn (preprocessing and evaluation)

---

## 🔍 Folder Structure

```
f1-race-prediction-optimization/
├── data/
│ ├── raw/ # Race CSVs (laps, weather, qualifying)
│ └── cache/ # FastF1 cache
├── src/
│ ├── ingestion/ # Fetching race data
│ ├── preprocessing/ # Feature engineering
│ └── modeling/ # Model training + prediction
├── main.py # End-to-end pipeline
├── test.py # Predict on new race (2024)
└── README.md
```


---

## ✅ Current Capabilities

- Predicts top-10 finishers for Bahrain GP 2024 using 2018–2023 race data
- Modular codebase for easy extension to other GPs or seasons
- Achieves strong prediction accuracy with XGBoost classifier

---

## 🚧 Future Improvements

- Add precision/recall/F1 metrics for better model evaluation
- Plot telemetry trendlines and position changes throughout races
- Expand to other circuits and multi-season analysis
- Add feature importance visualization

---

## 📬 Contact

Built by [Aalap Desai](https://github.com/desaiaalap) as part of a motorsports data science portfolio.  
Feel free to explore or fork the repo.

---

