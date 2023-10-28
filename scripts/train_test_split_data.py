import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sktime.forecasting.model_selection import temporal_train_test_split

# Инициализация MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_test_split")
mlflow.start_run()

# Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path = os.path.join(datasets_dir, 'data_proc.csv')
y = pd.read_csv(data_csv_path, parse_dates=['Date'], index_col="Date")

# Разбиваем данные
TEST_SIZE = int(0.45 * y.shape[0])
y_train, y_test = temporal_train_test_split(y, test_size=TEST_SIZE)

# Записываем train и test данные в файлы
data_csv_path_train = os.path.join(datasets_dir, 'data_train.csv')
data_csv_path_test = os.path.join(datasets_dir, 'data_test.csv')
y_train.to_csv(data_csv_path_train)
y_test.to_csv(data_csv_path_test)

# Логирование параметров и метрик в MLflow
mlflow.log_params({
    "test_size": TEST_SIZE
})
mlflow.log_metric("train_data_rows", len(y_train))
mlflow.log_metric("test_data_rows", len(y_test))

print(f'Соотношение данных: Train: {y_train.shape[0]}, Test: {y_test.shape[0]}')
print("Датасет train сохранен по пути:", data_csv_path_train)
print("Датасет test сохранен по пути:", data_csv_path_test)

# Завершение MLflow run
mlflow.end_run()