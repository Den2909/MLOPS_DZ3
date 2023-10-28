import pandas as pd
import os
import pickle
import mlflow
import mlflow.sklearn
from sktime.forecasting.naive import NaiveForecaster

# Инициализация MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Укажите адрес сервера MLflow
mlflow.set_experiment("train_model")
mlflow.start_run()
# Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path = os.path.join(datasets_dir, 'data_train.csv')
y_train = pd.read_csv(data_csv_path, parse_dates=['Date'], index_col="Date")

SEASON = 52

# Создаем и обучаем прогнозировщик
forecaster = NaiveForecaster(strategy="mean", sp=SEASON)
forecaster.fit(y_train)

# Сохраняем обученную модель
model_output_path = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/models/model_forecaster.pkl')
with open(model_output_path, "wb") as model_file:
    pickle.dump(forecaster, model_file)

# Логируем модель в MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(forecaster, "forecast_model")
    mlflow.log_params({"seasonality": SEASON})

# Завершение MLflow run
mlflow.end_run()