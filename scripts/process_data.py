import pandas as pd
import os
import mlflow
import mlflow.sklearn


# Инициализация MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("process_data")
mlflow.start_run()

# Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path = os.path.join(datasets_dir, 'data.csv')
df = pd.read_csv(data_csv_path, parse_dates=['Date'], index_col="Date")

df.index = pd.to_datetime(df.index)
df.fillna(0, inplace=True)

# Для упрощения анализа и без значительных потерь значимости удалим быструю составляющую при помощи перегруппировки данных.
y = df.Consumption.asfreq('7d')

# Записываем обработанные данные в файл по указанному пути
data_csv_path_proc = os.path.join(datasets_dir, 'data_proc.csv')
y.to_csv(data_csv_path_proc)

# Записываем данные в MLflow
mlflow.log_params({
    "resampling_frequency": '7d'
})

mlflow.log_metric("data_rows", len(y))


print("Датасет успешно обработан и сохранен по пути:", data_csv_path_proc)

# Завершение MLflow run
mlflow.end_run()