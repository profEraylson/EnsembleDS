import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return 100 * np.mean(diff)

def mase(y_true, y_pred, seasonality=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    if n <= seasonality:
        return np.nan
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    naive_forecast = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    mae_naive = np.mean(naive_forecast)
    return mae_forecast / mae_naive if mae_naive != 0 else np.nan

def evaluate_models(model_preds, model_names, y_true, seasonality=1):
    results = []

    for preds, name in zip(model_preds, model_names):
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))  # Alterado aqui
        smape_value = smape(y_true, preds)
        mase_value = mase(y_true, preds, seasonality)
        
        results.append({
            "Modelo": name,
            "MAE": mae,
            "RMSE": rmse,
            "SMAPE": smape_value,
            "MASE": mase_value
        })

    return pd.DataFrame(results)



