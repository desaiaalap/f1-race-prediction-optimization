# src/modeling/model_runner.py

import pandas as pd
from xgboost import XGBClassifier  # Switch to classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

def train_and_evaluate(df: pd.DataFrame):
    df = df.copy()

    # Ensure proper types
    df['DriverNumber'] = df['DriverNumber'].astype(str)
    df['main_compound'] = df['main_compound'].astype(str)

    # Define features and target
    X = df.drop(columns=['FinalPosition'])
    y = (df['FinalPosition'] <= 10).astype(int)  # Classification target: Top 10 = 1, others = 0

    # Split train/test (use 2023 for test)
    X_train = X[X['year'] < 2023]
    y_train = y[X['year'] < 2023]
    X_test = X[X['year'] == 2023]
    y_test = y[X['year'] == 2023]

    cat_cols = ['DriverNumber', 'main_compound']
    num_cols = ['avg_lap_time', 'total_pit_stops', 'num_stints', 'position_at_lap_10', 'position_at_lap_20', 'position_at_lap_30', 'total_laps']

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ]), num_cols)
    ])

    # Set up the XGBoost model with hyperparameter tuning
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(random_state=42, objective='binary:logistic'))
    ])

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__n_estimators': [100, 200],
        'clf__subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate the model
    preds = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"📊 Accuracy on 2023 Test Set: {accuracy:.2f}")
    
    joblib.dump(best_model, 'trained_model.pkl')
    print("✅ Model saved as 'trained_model.pkl'")
    
    return best_model



    
