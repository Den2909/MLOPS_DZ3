import pandas as pd
import os
import pickle
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
import mlflow
import mlflow.sklearn

# Инициализация MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_model")
mlflow.start_run()

# Загрузка данных
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path_test = os.path.join(datasets_dir, 'data_test.csv')
y_test = pd.read_csv(data_csv_path_test, parse_dates=['Date'], index_col="Date")

fh = ForecastingHorizon(y_test.index, is_relative=False)

f_input = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/models/model_forecaster.pkl')
with open(f_input, "rb") as fd:
    forecaster = pickle.load(fd)

y_pred = forecaster.predict(fh)  

# Вычисление sMAPE
smape_value = MeanAbsolutePercentageError(symmetric=True)(y_pred, y_test)

# Логирование метрик и параметров
mlflow.log_params({
    "seasonality": forecaster.sp,
})
mlflow.log_metric("sMAPE", smape_value)

print(f'Симметричная средняя абсолютная процентная ошибка (sMAPE) = {smape_value:.3f}')

# Завершение MLflow run
mlflow.end_run()