import pandas as pd
import os
from sktime.forecasting.model_selection import temporal_train_test_split


import pandas as pd
import os
import pickle  # Добавлен импорт модуля pickle
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster






import pandas as pd
import os
import pickle  # Добавлен импорт модуля pickle
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    MultiplexForecaster,
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import  MeanAbsolutePercentageError, MeanSquaredError
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.tbats import TBATS
smape = MeanAbsolutePercentageError(symmetric = True)
rmse  = MeanSquaredError(square_root=True)

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier







#Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path = os.path.join(datasets_dir, 'data_proc.csv')
y = pd.read_csv(data_csv_path, parse_dates=['Date'], index_col="Date")

# Разбиваем данные
TEST_SIZE = int(0.45*y.size)

y_train, y_test = temporal_train_test_split(y, test_size=TEST_SIZE)

print(f'Соотношение данных: Train: {y_train.shape[0]}, Test: {y_test.shape[0]}')

# Записываем train в файл по указанному пути
data_csv_path_train = os.path.join(datasets_dir, 'data_train.csv')
y_train.to_csv(data_csv_path_train)

print("Датасет train сохранен по пути:", data_csv_path_train)

# Записываем test в файл по указанному пути
data_csv_path_test = os.path.join(datasets_dir, 'data_test.csv')
y_test.to_csv(data_csv_path_test)

print("Датасет test сохранен по пути:", data_csv_path_test)
print(y_test)
print(y_train)

SEASON = 52
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = NaiveForecaster(strategy="mean", sp=SEASON)
forecaster.fit(y_train)

y_pred = forecaster.predict(fh)





print(f'sMAPE = {smape(y_pred.values, y_test.values):.3f}')