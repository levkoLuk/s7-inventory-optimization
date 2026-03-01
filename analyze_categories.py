import pandas as pd
import numpy as np
import os

# Параметры для классификации
XYZ_THRESHOLDS = {'X': 0.1, 'Y': 0.25}
ABC_THRESHOLDS = {'A': 80, 'B': 95}
OUTPUT_DIR = "output/"


def analyze_categories(filepath="final_orders_train.csv"):
    """Загружаем данные и делаем ABC/XYZ анализ"""
    try:
        df = pd.read_csv(filepath)
        print(f"Загружено {len(df)} записей")
    except FileNotFoundError:
        print(f"Ошибка: файл '{filepath}' не найден")
        return None, None
    except Exception as e:
        print(f"Ошибка при чтении: {e}")
        return None, None

    # Приводим даты к нормальному виду
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')

    # Считаем время доставки
    df['lead_time_days'] = (df['delivery_date'] - df['order_date']).dt.days

    # Заполняем пропуски в количестве
    df['valid_delivered_qty'] = df['valid_delivered_qty'].fillna(0)
    df = df[df['valid_delivered_qty'] >= 0]

    # Считаем цену за единицу
    df['unit_price'] = df.apply(
        lambda row: row['amount'] / row['valid_delivered_qty']
        if row['valid_delivered_qty'] > 0 else 0,
        axis=1
    )

    # Убираем выбросы по цене
    q1, q3 = df['unit_price'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr

    outliers_count = len(df[(df['unit_price'] < lower_bound) | (df['unit_price'] > upper_bound)])
    df = df[(df['unit_price'] >= lower_bound) & (df['unit_price'] <= upper_bound)]

    if outliers_count > 0:
        print(f"Отфильтровано {outliers_count} выбросов")

    # Группируем по категориям
    category_summary = df.groupby('product_category').agg(
        total_amount=('amount', 'sum'),
        total_qty=('valid_delivered_qty', 'sum'),
        order_count=('order_id', 'nunique'),
        avg_lead_time_days=('lead_time_days', 'mean'),
        min_lead_time_days=('lead_time_days', 'min'),
        max_lead_time_days=('lead_time_days', 'max'),
        avg_qty_per_order=('valid_delivered_qty', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0),
        std_qty_per_order=('valid_delivered_qty', lambda x: x[x > 0].std() if len(x[x > 0]) > 1 else 0),
        num_unique_products=('product_id', 'nunique')
    ).reset_index()

    # Считаем долю в общей стоимости
    category_summary['percentage_of_total_amount'] = (
                                                             category_summary['total_amount'] / category_summary[
                                                         'total_amount'].sum()
                                                     ) * 100

    # Сортируем для ABC анализа
    category_summary = category_summary.sort_values(
        'percentage_of_total_amount',
        ascending=False
    ).reset_index(drop=True)

    category_summary['cumulative_percentage'] = category_summary['percentage_of_total_amount'].cumsum()

    # Присваиваем ABC классы
    def assign_abc(percentage):
        if percentage <= ABC_THRESHOLDS['A']:
            return 'A'
        elif percentage <= ABC_THRESHOLDS['B']:
            return 'B'
        else:
            return 'C'

    category_summary['abc_class'] = category_summary['cumulative_percentage'].apply(assign_abc)

    # XYZ анализ по временным рядам
    df['order_month'] = pd.to_datetime(df['order_date']).dt.to_period('M')
    monthly_demand = df.groupby(['product_category', 'order_month'])['valid_delivered_qty'].sum().reset_index()

    xyz_results = []
    for cat in df['product_category'].unique():
        cat_demand = monthly_demand[monthly_demand['product_category'] == cat]['valid_delivered_qty']

        if len(cat_demand) > 1 and cat_demand.mean() > 0:
            cv = cat_demand.std() / cat_demand.mean()
        else:
            cv = 0

        xyz_results.append({
            'product_category': cat,
            'cv_time_series': round(cv, 4)
        })

    xyz_df = pd.DataFrame(xyz_results)

    def assign_xyz(volatility_coef):
        if volatility_coef < XYZ_THRESHOLDS['X']:
            return 'X'
        elif volatility_coef < XYZ_THRESHOLDS['Y']:
            return 'Y'
        else:
            return 'Z'

    xyz_df['xyz_class'] = xyz_df['cv_time_series'].apply(assign_xyz)

    category_summary = category_summary.merge(xyz_df, on='product_category', how='left')
    category_summary['demand_volatility_coef'] = category_summary['cv_time_series']

    # Детализация по продуктам
    product_demand = df.groupby(['product_category', 'product_id']).agg(
        total_qty=('valid_delivered_qty', 'sum'),
        total_amount=('amount', 'sum'),
        mean_qty=('valid_delivered_qty', 'mean'),
        std_qty=('valid_delivered_qty', 'std'),
        order_count=('order_id', 'nunique')
    ).reset_index()

    product_demand['demand_volatility_coef'] = product_demand.apply(
        lambda row: row['std_qty'] / row['mean_qty'] if row['mean_qty'] > 0 else 0,
        axis=1
    )
    product_demand['demand_volatility_coef'] = product_demand['demand_volatility_coef'].fillna(0)

    # Округляем числа
    numeric_cols = ['total_amount', 'total_qty', 'avg_lead_time_days',
                    'demand_volatility_coef', 'cv_time_series']
    for col in numeric_cols:
        if col in category_summary.columns:
            category_summary[col] = category_summary[col].round(2)

    # Создаем папку и сохраняем результаты
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Сохраняем с правильными параметрами для Excel
    category_summary.to_csv(
        f'{OUTPUT_DIR}category_summary.csv',
        index=False,
        sep=';',
        encoding='utf-8-sig',
        decimal=','
    )
    product_demand.to_csv(
        f'{OUTPUT_DIR}product_demand.csv',
        index=False,
        sep=';',
        encoding='utf-8-sig',
        decimal=','
    )

    print(f"Найдено {len(category_summary)} категорий")
    print(f"Найдено {len(product_demand)} продуктов")
    print(f"Результаты в папке {OUTPUT_DIR}")

    return category_summary, product_demand


if __name__ == "__main__":
    print("Запуск ABC/XYZ анализа...")
    category_summary, product_demand = analyze_categories()

    if category_summary is not None:
        print("\nРезультаты по категориям:")
        print(category_summary[[
            'product_category', 'abc_class', 'xyz_class',
            'total_amount', 'total_qty', 'demand_volatility_coef'
        ]].to_string(index=False))

"""
import pandas as pd
import numpy as np

def analyze_categories(filepath="final_orders_train.csv"):
    """
    Загружает данные о заказах, очищает их и проводит первоначальный анализ категорий.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return "Ошибка: файл 'final_orders_train.csv' не найден."
    except Exception as e:
        return f"Ошибка при чтении файла: {e}"

    # Преобразование дат
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])

    # Расчет времени доставки (lead time)
    df['lead_time_days'] = (df['delivery_date'] - df['order_date']).dt.days

    # Обработка пропущенных значений в valid_delivered_qty, если они есть
    # Предполагаем, что если valid_delivered_qty отсутствует, то товар не был доставлен или равен 0
    df['valid_delivered_qty'] = df['valid_delivered_qty'].fillna(0)

    # Вычисление стоимости за единицу, если amount и valid_delivered_qty > 0
    # Избегаем деления на ноль
    df['unit_price'] = df.apply(lambda row: row['amount'] / row['valid_delivered_qty'] if row['valid_delivered_qty'] > 0 else 0, axis=1)

    # Агрегация данных по категориям
    category_summary = df.groupby('product_category').agg(
        total_amount=('amount', 'sum'),
        total_qty=('valid_delivered_qty', 'sum'),
        order_count=('order_id', 'nunique'),
        avg_lead_time_days=('lead_time_days', 'mean'),
        min_lead_time_days=('lead_time_days', 'min'),
        max_lead_time_days=('lead_time_days', 'max'),
        avg_qty_per_order=('valid_delivered_qty', lambda x: x[x > 0].mean()), # Среднее количество для доставленных
        std_qty_per_order=('valid_delivered_qty', lambda x: x[x > 0].std()), # Стандартное отклонение для доставленных
        num_unique_products=('product_id', 'nunique')
    ).reset_index()

    # Добавляем в summary волатильность спроса (коэффициент вариации)
    category_summary['demand_volatility_coef'] = category_summary['std_qty_per_order'] / category_summary['avg_qty_per_order']
    
    # Агрегация данных по продуктам для дополнительного анализа (для XYZ анализа)
    product_demand = df.groupby(['product_category', 'product_id']).agg(
        total_qty=('valid_delivered_qty', 'sum'),
        mean_qty=('valid_delivered_qty', 'mean'),
        std_qty=('valid_delivered_qty', 'std'),
        total_amount = ('amount', 'sum')
    ).reset_index()

    # Для XYZ анализа нужен коэффициент вариации по каждому продукту
    product_demand['demand_volatility_coef'] = product_demand['std_qty'] / product_demand['mean_qty']
    # Заменим NaN на 0 для продуктов с постоянным спросом или единичной закупкой
    product_demand['demand_volatility_coef'] = product_demand['demand_volatility_coef'].fillna(0)

    # ABC анализ на уровне product_category (по общей стоимости)
    category_summary = category_summary.sort_values(by='total_amount', ascending=False)
    category_summary['cumulative_amount'] = category_summary['total_amount'].cumsum()
    category_summary['percentage_of_total_amount'] = (category_summary['cumulative_amount'] / category_summary['total_amount'].sum()) * 100

    def assign_abc(percentage):
        if percentage <= 80:
            return 'A'
        elif percentage <= 95:
            return 'B'
        else:
            return 'C'

    category_summary['abc_class'] = category_summary['percentage_of_total_amount'].apply(assign_abc)

    # XYZ анализ на уровне product_category (по средней волатильности спроса агрегированной на уровне product_category)
    def assign_xyz(volatility_coef):
        if volatility_coef < 0.1: # Менее 10% - стабильный спрос
            return 'X'
        elif volatility_coef < 0.25: # От 10% до 25% - колеблющийся спрос
            return 'Y'
        else: # Более 25% - нерегулярный спрос
            return 'Z'
    
    category_summary['xyz_class'] = category_summary['demand_volatility_coef'].apply(assign_xyz)


    return category_summary, product_demand

if __name__ == "__main__":
    category_summary, product_demand = analyze_categories()
    print("Category Summary:")
    print(category_summary.to_json(orient='records', indent=2))
    print("\nProduct Demand (first 5 rows):")
    print(product_demand.head().to_json(orient='records', indent=2))
"""
