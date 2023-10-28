import pandas as pd
import os
import pickle
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.naive import NaiveForecaster  # Импорт класса NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import r2_score
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.performance_metrics.forecasting import MeanSquaredError
smape = MeanAbsolutePercentageError(symmetric = True)

# Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path_test = os.path.join(datasets_dir, 'data_test.csv')
y_test = pd.read_csv(data_csv_path_test, parse_dates=['Date'], index_col="Date")

fh = ForecastingHorizon(y_test.index, is_relative=False)

f_input = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/models/model_forecaster')
with open(f_input, "rb") as fd:
    forecaster = pickle.load(fd)

y_pred = forecaster.predict(fh)  
print(f'Симметричная средняя абсолютная процентная ошибка (sMAPE) = {smape(y_pred, y_test):.3f}')