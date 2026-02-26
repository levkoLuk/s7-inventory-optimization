import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

# Загружаем функцию анализа категорий
from analyze_categories import analyze_categories

class InventoryOptimizer:
    def __init__(self, category_data, product_demand_data, df_full_orders):
        self.category_data = category_data
        self.product_demand_data = product_demand_data
        self.df_full_orders = df_full_orders

        # Предположения для моделирования (могут быть настроены)
        self.holding_cost_rate = 0.20  # 20% от стоимости запаса в год
        self.ordering_cost = 100       # Стоимость размещения одного заказа
        # self.stockout_cost_multiplier = 2 # будет использоваться, если будет явная задача минимизации дефицита
        self.service_level_a = 0.99    # Уровень сервиса для A-категорий (99%)
        self.service_level_b = 0.95    # Уровень сервиса для B-категорий (95%)
        self.service_level_c = 0.90    # Уровень сервиса для C-категорий (90%)

    def get_service_level(self, abc_class):
        if abc_class == 'A':
            return self.service_level_a
        elif abc_class == 'B':
            return self.service_level_b
        else:
            return self.service_level_c

    def calculate_eoq(self, annual_demand, ordering_cost, unit_cost, holding_cost_rate):
        """ Расчет экономического размера заказа (EOQ) """
        if holding_cost_rate == 0 or unit_cost == 0 or annual_demand == 0:
            # Если нет стоимости хранения, или товар бесплатный, или спрос нулевой, EOQ теряет смысл
            return 0  # Возвращаем 0, если невозможно рассчитать или не имеет смысла
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / (unit_cost * holding_cost_rate))
        return eoq

    def calculate_safety_stock(self, mean_daily_demand, std_daily_demand, lead_time_days, service_level):
        """ Расчет страхового запаса """
        if service_level >= 1.0 or service_level <= 0.0:  # Проверка на корректность service_level
            return 0
        z_score = norm.ppf(service_level)
        
        # Проверка на NaN, inf и отрицательные значения в std_daily_demand и mean_daily_demand
        if np.isnan(std_daily_demand) or np.isinf(std_daily_demand) or std_daily_demand < 0:
            std_daily_demand = 0
        if np.isnan(mean_daily_demand) or np.isinf(mean_daily_demand) or mean_daily_demand < 0:
            mean_daily_demand = 0

        # Усредняем lead_time_days, если это Series
        if isinstance(lead_time_days, pd.Series):
            lead_time_days_val = lead_time_days.mean()
        else:
            lead_time_days_val = lead_time_days
        
        # Проверка на NaN, inf и отрицательные значения в lead_time_days_val
        if np.isnan(lead_time_days_val) or np.isinf(lead_time_days_val) or lead_time_days_val < 0:
            lead_time_days_val = 0

        # Если спрос нулевой или очень низкий и нет волатильности, страховой запас не нужен
        if mean_daily_demand == 0 and std_daily_demand == 0:
            return 0

        # Консервативный подход к расчету SS: учитываем неопределенность спроса во время LT
        # Здесь используем стандартное отклонение ЗАКАЗОВ, а не дневного спроса. 
        # Это может быть неточно, если std_daily_demand сильно отличается от std_qty_per_order.
        # Для более точного расчета std_daily_demand требуется больше гранулярности в данных. 
        # Пока используем std_qty_per_order как прокси для волатильности.

        # Если std_daily_demand был рассчитан как std_qty_per_order / sqrt(365), то это уже масштабированное значение
        # Если 'std_daily_demand' здесь - это 'std_qty_per_order' из summary, то нужно масштабировать по LT
        # Предположим, что std_daily_demand уже представляет вариабельность на период в день
        effective_std_dev = std_daily_demand * np.sqrt(max(1, lead_time_days_val))
        safety_stock = z_score * effective_std_dev

        return max(0, safety_stock) # Страховой запас не может быть отрицательным


    def optimize_category_inventory(self, category_id):
        cat_info = self.category_data[self.category_data['product_category'] == category_id].iloc[0]
        abc_class = cat_info['abc_class']
        xyz_class = cat_info['xyz_class']
        
        annual_demand_qty = cat_info['total_qty']
        avg_lead_time_days = cat_info['avg_lead_time_days']
        
        # Получаем данные по продуктам в этой категории для более точного расчета Unit Cost
        products_in_category = self.product_demand_data[self.product_demand_data['product_category'] == category_id]
        
        # Средняя стоимость единицы для категории (взвешенная по объему, если возможно)
        if cat_info['total_qty'] > 0:
            avg_unit_cost = cat_info['total_amount'] / cat_info['total_qty']
        else:
            avg_unit_cost = 0

        service_level = self.get_service_level(abc_class)

        # Прогнозирование среднего дневного спроса
        mean_daily_demand = annual_demand_qty / 365.0
        
        # Оценка стандартного отклонения дневного спроса
        # Используем std_qty_per_order как индикатор волатильности спроса
        # Если в категории очень мало заказов, std_qty_per_order может быть NaN или 0
        std_qty_per_order = cat_info.get('std_qty_per_order', 0.0)
        if pd.isna(std_qty_per_order):
            std_qty_per_order = 0.0

        # Для Z-класса спроса, где данные очень разрежены, std_qty_per_order может быть плохой метрикой
        # В таких случаях лучше использовать более высокий коэффициент вариации или другой метод оценки.
        # Здесь я буду использовать std_qty_per_order как прокси для волатильности спроса за средний заказ.
        # Но для daily спроса, это не совсем корректно, нужно делить на sqrt(числа дней), что уже сделано в calculate_safety_stock.
        # Дополнительная корректировка для std_daily_demand для Z-класса, чтобы увеличить SS.
        if xyz_class == 'Z':
            # Если спрос нерегулярный, увеличиваем воспринимаемую дисперсию
            std_daily_demand = std_qty_per_order / np.sqrt(365.0) # Масштабируем до дневного
            std_daily_demand *= 1.5  # Дополнительный множитель для Z-класса
        else:
            std_daily_demand = std_qty_per_order / np.sqrt(365.0) # Масштабируем до дневного

        # Расчет страхового запаса
        safety_stock = self.calculate_safety_stock(mean_daily_demand, std_daily_demand, avg_lead_time_days, service_level)

        # Расчет размера заказа (EOQ)
        order_quantity = self.calculate_eoq(annual_demand_qty, self.ordering_cost, avg_unit_cost, self.holding_cost_rate)

        # Корректировка order_quantity для категорий с низким спросом или высокой волатильностью
        if order_quantity == 0 or np.isinf(order_quantity) or order_quantity > annual_demand_qty: # Если EOQ не имеет смысла или слишком большой
             # Для Z-классов с редким спросом, возможно, просто 1 заказ на год или квартал.
             # Для C-X категории (стабильный низкий спрос) EOQ может быть верен.
             if xyz_class == 'X':
                 # Если EOQ маленький и стабильный, оставляем его
                 pass # EOQ уже рассчитан
             else:
                 # Для Z-класса, особенно A-Z, B-Z, C-Z, EOQ может быть не лучшей метрикой.
                 # Может быть лучше заказывать покрытие на период Lead Time + буфер. 
                 # Или просто заказывать достаточно, чтобы держать 1-2 кратный Lead Time спрос + SS.
                 # Давайте сделаем заказ равным среднему спросу за LT + 2 * SS для консервативности
                 order_quantity = (mean_daily_demand * avg_lead_time_days * 1.5) + safety_stock # 1.5x спрос на LT + SS 
                 if order_quantity == 0: # Если все еще 0 (например, при нулевом спросе)
                     order_quantity = 1 if annual_demand_qty > 0 else 0 # Заказать хотя бы 1, если есть спрос
        
        order_quantity = round(order_quantity, 0)
        safety_stock = round(safety_stock, 0)

        # Расчет точки перезаказа (Reorder Point)
        reorder_point = (mean_daily_demand * avg_lead_time_days) + safety_stock
        reorder_point = round(reorder_point, 0)

        return {
            'product_category': category_id,
            'abc_class': abc_class,
            'xyz_class': xyz_class,
            'annual_demand_qty': annual_demand_qty,
            'avg_unit_cost': avg_unit_cost,
            'avg_lead_time_days': avg_lead_time_days,
            'service_level': service_level,
            'mean_daily_demand': mean_daily_demand,
            'std_daily_demand_used_for_ss': std_daily_demand,
            'safety_stock': safety_stock,
            'order_quantity': order_quantity,
            'reorder_point': reorder_point
        }

    def run_optimization(self):
        results = []
        for category_id in self.category_data['product_category'].unique():
            results.append(self.optimize_category_inventory(category_id))

        # Дополнительная проверка: если category_data содержит менее 5 уникальных категорий,
        # то мы должны убедиться, что все 5 категорий изначальные были проанализированы.
        # Но в данном случае, category_data уже содержит 5 категорий из analyze_categories.
        return pd.DataFrame(results)

# Ниже будет пример использования и визуализации результатов
if __name__ == "__main__":
    output_dir = "output/"
    category_summary, product_demand = analyze_categories()
    
    # Для df_full_orders нужно прочесть файл заново и подготовить его
    df_full_orders = pd.read_csv("final_orders_train.csv")
    df_full_orders['order_date'] = pd.to_datetime(df_full_orders['order_date'])
    df_full_orders['delivery_date'] = pd.to_datetime(df_full_orders['delivery_date'])
    df_full_orders['lead_time_days'] = (df_full_orders['delivery_date'] - df_full_orders['order_date']).dt.days
    df_full_orders['valid_delivered_qty'] = df_full_orders['valid_delivered_qty'].fillna(0)

    optimizer = InventoryOptimizer(category_summary, product_demand, df_full_orders)
    optimization_results = optimizer.run_optimization()

    print("Результаты оптимизации запасов по категориям:")
    print(optimization_results.to_json(orient='records', indent=2))

    # Визуализация прогнозов закупок (пример)
    # Поскольку мы не делаем детальное прогнозирование на год,
    # мы покажем только общие рекомендации по закупкам.
    # В реальной системе это были бы графики с планируемыми заказами во времени.

    # Графики для EOQ и Safety Stock
    plt.figure(figsize=(14, 7))
    sns.barplot(x='product_category', y='order_quantity', data=optimization_results, hue='abc_class', palette='viridis', dodge=False)
    plt.title('Рекомендованный размер заказа (EOQ / Приблизительный) по категориям')
    plt.xlabel('Категория продукта')
    plt.ylabel('Рекомендованный объем заказа')
    plt.legend(title='ABC Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('recommended_order_quantity.png')
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.barplot(x='product_category', y='safety_stock', data=optimization_results, hue='abc_class', palette='plasma', dodge=False)
    plt.title('Рекомендованный страховой запас по категориям')
    plt.xlabel('Категория продукта')
    plt.ylabel('Страховой запас')
    plt.legend(title='ABC Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('recommended_safety_stock.png')
    plt.close()

    # График для Reorder Point
    plt.figure(figsize=(14, 7))
    sns.barplot(x='product_category', y='reorder_point', data=optimization_results, hue='abc_class', palette='magma', dodge=False)
    plt.title('Рекомендованная точка перезаказа по категориям')
    plt.xlabel('Категория продукта')
    plt.ylabel('Точка перезаказа')
    plt.legend(title='ABC Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('recommended_reorder_point.png')
    plt.close()

    print("\nСгенерированы графики: recommended_order_quantity.png, recommended_safety_stock.png, recommended_reorder_point.png")
