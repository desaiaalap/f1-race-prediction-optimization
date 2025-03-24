# 🏎️ Formula 1: Race Outcome Prediction & Strategy Optimization

This is an end-to-end machine learning and simulation project that predicts Formula 1 race outcomes and optimizes pit strategies using real-time telemetry and timing data from the OpenF1 API — built completely using free and local tools.

## 🎯 Project Goals

- **Predict Race Outcomes**: Estimate finishing positions using lap and pit data.
- **Optimize Race Strategies**: Apply reinforcement learning to determine optimal pit stop windows and tire strategies.
- **Simulate Race Scenarios**: Deploy an interactive race simulation dashboard to replicate the role of an F1 race strategist.

---

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| 📊 **Race Data Ingestion** | Pulls lap-by-lap, pit stop, and telemetry data from the OpenF1 API. |
| 🛠️ **Feature Engineering** | Extracts strategy-driven features like tire stint performance, pit deltas, and lap consistency. |
| 🤖 **Machine Learning Models** | Uses classification and regression models to predict positions and win probabilities. |
| 🎮 **Strategy Simulator** | Reinforcement learning agent chooses pit strategies based on current race states. |
| 💻 **Local Deployment** | Fully local simulation via Streamlit or FastAPI + Uvicorn — no paid cloud platforms required. |
| 📈 **Experiment Tracking** | MLflow used locally to track model runs and strategy experiments. |

---

## 🧰 Tech Stack

- **Languages**: Python 3.8+
- **Data & ML Tools**: OpenF1 API, Pandas, NumPy, Scikit-learn, TensorFlow, MLflow (local)
- **Visualization**: Matplotlib, Seaborn
- **Dashboard**: Streamlit or Dash (local)
- **Deployment**: FastAPI (local with Uvicorn)
- **Logging**: Built-in Python logging, CSV-based logs (no external services)

---

## 📦 Folder Structure

