
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

