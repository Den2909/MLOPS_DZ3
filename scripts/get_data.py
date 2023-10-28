import pandas as pd
import os

# Указываем URL для загрузки датасета
url = 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
# Указываем путь для сохранения файла
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path = os.path.join(datasets_dir, 'data.csv')

# Загружаем датасет
df_all = pd.read_csv(url, index_col='utc_timestamp', parse_dates=True, low_memory=False)

# Записываем данные в файл по указанному пути
df_all.to_csv(data_csv_path, sep=',', index=True)

print("Датасет успешно загружен и сохранен по пути:", data_csv_path)