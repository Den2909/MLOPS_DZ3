import pandas as pd
import os
from sktime.forecasting.model_selection import temporal_train_test_split



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
