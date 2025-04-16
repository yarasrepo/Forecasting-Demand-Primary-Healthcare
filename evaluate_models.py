import os
import pandas as pd
import joblib

from shallow_models import plot_roc_auc_curve, plot_precision_recall_curve, get_results, get_regression_results

models_path = "./models"
metrics_path_root = "./metrics"
scaler_txt = "MinMax"

df = pd.read_csv("./input/all_columns/df_test_collated_reduced.csv")
X_test = df.drop(columns = "demand")
y_test = df["demand"]

for model_file in os.listdir(models_path):
    if not model_file.endswith(".pkl"):
        continue

    model_name = model_file.replace(".pkl", "")
    model = joblib.load(os.path.join(models_path, model_file))

    print(f"\n=== Evaluating {model_name.upper()} ===")
    y_pred = model.predict(X_test)

    # f2, gmean, roc_auc, acc, prec, rec, f1 = get_results(y_test, y_pred)
    # results_dict = {
    #     "F2": f2, "G-Mean": gmean, "ROC-AUC": roc_auc,
    #     "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
    # }

    mae, rmse, r2 = get_regression_results(y_test, y_pred)
    results_dict = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    save_dir = os.path.join(metrics_path_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    results_df = pd.DataFrame([results_dict])
    results_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = model.predict(X_test)

        # Plot ROC AUC and PR curves
    plot_roc_auc_curve(y_test, y_proba, fig_name="roc_auc.png", save_dir=save_dir)
    plot_precision_recall_curve(y_test, y_proba, save_dir=save_dir, fig_name="precision_recall")