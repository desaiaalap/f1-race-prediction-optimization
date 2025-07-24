# 🏎️ Formula 1: Race Prediction & Strategy Optimization

This project predicts Formula 1 race outcomes and simulates race strategies using machine learning and reinforcement learning. Built as a personal deep dive into motorsports analytics, the goal was to explore how data impacts decision-making on the track.

## 📌 What It Does

- **Race Outcome Prediction**  
  Trains an XGBoost model on FastF1 telemetry data to predict whether a driver finishes in the top 10.  
  Uses features like tire compound, lap times, pit stop behavior, stint count, and qualifying position.

- **Q-Learning Based Strategy Simulator**  
  Simulates full-length races lap-by-lap to find optimal pit stop strategies based on tire wear, stint length, and lap pace evolution.

- **Driver Performance Feature Engineering**  
  Extracts 60+ features: average lap time, tire changes, pit stop counts, lap-wise position tracking, and stint analysis.

- **Testable and Modular**  
  Includes scripts for ingestion (`FastF1`), feature building, model training, prediction on unseen data, and race strategy simulation.

---

## 🧠 Tech Stack

- Python, Pandas, NumPy
- FastF1 (telemetry ingestion)
- XGBoost (binary classification)
- TensorFlow (used in early experiments)
- Q-Learning (custom reinforcement learning simulation)

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
│ ├── modeling/ # Model training + prediction
│ └── simulation/ # Q-learning race strategy simulator
├── main.py # End-to-end pipeline
├── test.py # Predict on new race (2024)
├── dashboards/ (optional) # Future: visualizations
└── README.md
```


---

## ✅ Current Capabilities

- Predicts top-10 finishers for Bahrain GP 2024 using 2018–2023 race data
- Simulates race strategies for a selected driver using Q-learning
- Modular codebase for easy extension to other GPs or seasons

---

## 🚧 Future Improvements

- Add precision/recall/F1 metrics for better model evaluation
- Plot race strategy heatmaps and telemetry trendlines
- Expand to other circuits and multi-driver strategy optimization

---

## 📬 Contact

Built by [Aalap Desai](https://github.com/desaiaalap) as part of a motorsports data science portfolio.  
Feel free to explore or fork the repo.

---

