import pandas as pd
import os

# Указываем URL для загрузки датасета
url = 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
# Указываем путь для сохранения файла
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ3/MLOPS_DZ3/datasets')
data_csv_path = os.path.join(datasets_dir, 'data.csv')

# Загружаем датасет
df_all = pd.read_csv(url, index_col='utc_timestamp', parse_dates=True, low_memory=False)

#выделим только данные относящиеся к целевой стране. В качестве примера рассмотрим германию. Возьмём период с 2015 по 2019 годы.
def extract_country(df_all, country_code, year_min=None, year_max=None):
    """Extract data for a single country"""

    # List of columns to extract
    columns = [col for col in df_all.columns if col.startswith(country_code)]

    # Extract columns and remove country codes from column labels
    columns_map = {col : col[3:] for col in columns}
    df_out = df_all[columns].rename(columns=columns_map)

    # Exclude years outside of specified range, if any
    if year_min is not None:
        df_out = df_out[df_out.index.year >= year_min]
    if year_max is not None:
        df_out = df_out[df_out.index.year <= year_max]

    return df_out
df_hrly = extract_country(df_all, country_code='DE', year_min=2015, year_max=2019)

#выделим необходимые, кроме того приведем колонки к более интерпретируемому виду
def transform_dataframe(df, cols_map):
    # Rename columns for convenience
    df = df[list(cols_map.keys())].rename(columns=cols_map)
    # Convert from MW to GW
    df = df / 1000
    df = df.resample('D').sum(min_count=24)
    df = df.rename_axis('Date')
    df.index = df.index.strftime('%Y-%m-%d')
    return df

cols_map = {'load_actual_entsoe_transparency' : 'Consumption',
            'wind_generation_actual' : 'Wind',
            'solar_generation_actual' : 'Solar'}
df_daily = transform_dataframe(df_hrly, cols_map)

# Compute wind + solar generation
df_daily['Wind+Solar'] = df_daily[['Wind', 'Solar']].sum(axis=1, skipna=False)




# Записываем данные в файл по указанному пути
df_daily.to_csv(data_csv_path, sep=',', index=True)

print("Датасет успешно загружен и сохранен по пути:", data_csv_path)