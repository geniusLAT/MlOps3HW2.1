import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_and_save_processed_dataframe(input_file_path: str = "./data/cars_raw.csv", output_file_path: str = "./data/prep_data.csv"):
    # Загрузка данных
    df = pd.read_csv(input_file_path)

    # Проверяем наличие колонки 'Price'
    if 'Price' not in df.columns:
        raise ValueError("Столбец 'Price' не найден в данных.")

    # Удаление строк с пропущенными значениями
    df.dropna(inplace=True)

    # Разделяем данные на признаки и целевую переменную
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Определяем числовые и категориальные признаки
    numeric_features = []
    categorical_features = []

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            numeric_features.append(column)
        elif pd.api.types.is_object_dtype(X[column]) or isinstance(X[column].dtype, pd.CategoricalDtype):
            categorical_features.append(column)

    # Создаем пайплайн для предобработки данных
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Заполнение пропусков средним значением
        ('scaler', StandardScaler())                   # Стандартизация числовых признаков
    ])

    # # Объединяем трансформеры в один ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # Применяем предобработку к данным
    features_normalized = preprocessor.fit_transform(X)

    # Преобразование нормализованных признаков в DataFrame
    processed_data = pd.DataFrame(
        features_normalized, columns=numeric_features)

    # Добавление целевого признака 'Price'
    processed_data['Price'] = y.reset_index(drop=True)

    # Сохранение обработанных данных в файл
    processed_data.to_csv(output_file_path, index=False)