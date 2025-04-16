import pandas as pd
from sklearn.model_selection import GridSearchCV
from models_hyperparams_grid import *
import joblib
import os

df = pd.read_csv("./input/all_columns/df_train_collated_reduced.csv")
X_train = df.drop(columns="demand")
y_train = df["demand"]

best_models = {}
os.makedirs("models", exist_ok=True)

for model_class, model_name in models_to_test.items():
    print(f"\nTraining model: {model_name}")
    model = model_class()
    param_grid = possible_hyperparams_per_model.get(model_name, {})

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        model_filename = f"models/{model_name}.pkl"
        joblib.dump(grid_search.best_estimator_, model_filename)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    else:
        print(f"No hyperparameters defined for {model_name}. Skipping grid search.")