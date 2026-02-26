import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Добавим импорт os

from analyze_categories import analyze_categories
from optimize_inventory import InventoryOptimizer

class EconomicImpactEvaluator:
    def __init__(self, optimization_results, holding_cost_rate, ordering_cost, category_data, product_demand_data, df_full_orders):
        self.optimization_results = optimization_results
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        self.category_data = category_data
        self.product_demand_data = product_demand_data
        self.df_full_orders = df_full_orders
        
        # Предполагаемые стандартные издержки для сравнения "как было"
        # Например, если запасы не оптимизированы, заказывают покрытие на 6 месяцев спроса,
        # и нет страхового запаса, что приводит к дефициту.
        self.initial_coverage_days = 180 # Покрытие на 180 дней спроса в "как было"
        self.initial_stockout_cost_multiplier = 5.0 # В 5 раз выше за единицу, чем в оптимизированном сценарии, из-за частых дефицитов
        self.optimized_stockout_cost_multiplier = 2.0 # В 2 раза выше за единицу в оптимизированном сценарии
        

    def calculate_current_costs(self, cat_info):
        # Расчет затрат "как было"
        annual_demand_qty = cat_info['annual_demand_qty']
        avg_unit_cost = cat_info['avg_unit_cost']
        mean_daily_demand = cat_info['mean_daily_demand']

        # 1. Затраты на хранение (предположим, поддерживался запас на N дней спроса)
        initial_stock_level = mean_daily_demand * self.initial_coverage_days
        initial_holding_cost = initial_stock_level * avg_unit_cost * self.holding_cost_rate

        # 2. Затраты на размещение заказов (предположим, заказывали 2 раза в год)
        initial_num_orders = annual_demand_qty / initial_stock_level if initial_stock_level > 0 else 2 # Если мало товара, то все равно 2 заказа
        initial_ordering_cost = initial_num_orders * self.ordering_cost

        # 3. Затраты на дефицит (сложно оценить без симуляции, поэтому используем множитель)
        # Предположим, что без оптимизации дефицит случался в 10% случаев от годового спроса
        initial_stockout_qty_percent = 0.10
        initial_stockout_cost = (annual_demand_qty * initial_stockout_qty_percent) * (avg_unit_cost * self.initial_stockout_cost_multiplier)

        return {
            'initial_holding_cost': initial_holding_cost,
            'initial_ordering_cost': initial_ordering_cost,
            'initial_stockout_cost': initial_stockout_cost,
            'initial_total_cost': initial_holding_cost + initial_ordering_cost + initial_stockout_cost
        }

    def calculate_optimized_costs(self, cat_info):
        # Расчет затрат "как будет"
        annual_demand_qty = cat_info['annual_demand_qty']
        avg_unit_cost = cat_info['avg_unit_cost']
        order_quantity = cat_info['order_quantity']
        safety_stock = cat_info['safety_stock']
        service_level = cat_info['service_level']

        # 1. Затраты на хранение (средний запас = SS + Q/2)
        optimized_avg_inventory = safety_stock + (order_quantity / 2)
        optimized_holding_cost = optimized_avg_inventory * avg_unit_cost * self.holding_cost_rate

        # 2. Затраты на размещение заказов
        optimized_num_orders = annual_demand_qty / order_quantity if order_quantity > 0 else 0
        optimized_ordering_cost = optimized_num_orders * self.ordering_cost

        # 3. Затраты на дефицит (зависят от уровня сервиса)
        # Допустим, 1 - service_level - это процент дефицита от ANNUAL_DEMAND_QTY
        optimized_stockout_qty_percent = (1 - service_level) 
        optimized_stockout_cost = (annual_demand_qty * optimized_stockout_qty_percent) * (avg_unit_cost * self.optimized_stockout_cost_multiplier)

        return {
            'optimized_holding_cost': optimized_holding_cost,
            'optimized_ordering_cost': optimized_ordering_cost,
            'optimized_stockout_cost': optimized_stockout_cost,
            'optimized_total_cost': optimized_holding_cost + optimized_ordering_cost + optimized_stockout_cost
        }

    def evaluate_economic_effect(self):
        impact_results = []
        for index, row in self.optimization_results.iterrows():
            initial_costs = self.calculate_current_costs(row)
            optimized_costs = self.calculate_optimized_costs(row)

            total_initial_cost = initial_costs['initial_total_cost']
            total_optimized_cost = optimized_costs['optimized_total_cost']
            
            economic_effect = total_initial_cost - total_optimized_cost
            # Проверим на NaN или Inf перед добавлением
            if np.isinf(economic_effect) or np.isnan(economic_effect):
                economic_effect = 0.0 # В случае некорректных расчетов, эффект равен 0

            impact_results.append({
                'product_category': row['product_category'],
                'abc_class': row['abc_class'],
                'xyz_class': row['xyz_class'],
                **initial_costs,
                **optimized_costs,
                'economic_effect': economic_effect
            })
        return pd.DataFrame(impact_results)

    def analyze_risks_and_scenarios(self, original_opt_results):
        # Ключевые риски и их митигация (качественный анализ)
        risks_and_mitigation = {
            "Категория 0 (A-Z) и 4 (A-Z)": {
                "Риски": [
                    "Высокая стоимость дефицита из-за критичности компонента и высокой цены",
                    "Заморозка капитала из-за высокой стоимости и длительного Lead Time",
                    "Неточный прогноз спроса для Z-класса"
                ],
                "Митигация": [
                    "Тесное сотрудничество с поставщиками (долгосрочные контракты, обмен данными)",
                    "Регулярный пересмотр прогнозов и уровней запасов",
                    "Альтернативные источники снабжения (если применимо)"
                ]
            },
            "Категория 2 (B-Z)": {
                "Риски": [
                    "Дефицит из-за самого длительного Lead Time и нерегулярного спроса",
                    "Устаревание запасов (особенно при переизбытке)"
                ],
                "Митигация": [
                    "Активный поиск и развитие альтернативных каналов поставок",
                    "Улучшение качества прогнозирования, основанного на событиях"
                ]
            },
            "Категория 3 (C-Z)": {
                "Риски": [
                    "Переизбыток запасов из-за больших объемов и высокой волатильности (C-класс не прощает таких затрат на хранение)",
                    "Сложности с управлением большим физическим объемом запасов"
                ],
                "Митигация": [
                    "Детальный анализ затрат на хранение против затрат на дефицит для этой категории",
                    "Поиск возможностей для JIT-поставок (Just-in-Time) или консолидации заказов"
                ]
            },
            "Категория 1 (C-X)": {
                "Риски": [
                    "Недооценка редкого, но критичного спроса (даже если стабильный, но очень редкий)"
                ],
                "Митигация": [
                    "Поддержание минимального, но достаточного страхового запаса для критичных C-X компонентов",
                    "Автоматизация, чтобы избежать ручных ошибок"
                ]
            }
        }

        # Сценарный анализ
        # Сценарий 1: Увеличение срока поставки на 20% для всех категорий
        scenario1_results = []
        for index, row in original_opt_results.iterrows():
            temp_row = row.copy()
            temp_row['avg_lead_time_days'] *= 1.2
            
            # Пересчитаем SS для нового Lead Time
            original_optimizer = InventoryOptimizer(self.category_data, self.product_demand_data, self.df_full_orders) # Используем старый optimizer
            new_safety_stock = original_optimizer.calculate_safety_stock(
                row['mean_daily_demand'],
                row['std_daily_demand_used_for_ss'],
                temp_row['avg_lead_time_days'],
                row['service_level']
            )

            # Если новый SS значительно больше старого, это означает, что оригинальный SS
            # будет недостаточным. Тут мы _оцениваем_ потенциальные доп. затраты или потери
            potential_additional_stockout_cost_from_scenario = 0
            if new_safety_stock > row['safety_stock'] and row['avg_unit_cost'] > 0:
                # Условная оценка потерь от дефицита, если бы мы не увеличили SS
                # Предположим, дефицит возникает на разнице (new_SS_needed - original_SS)
                potential_additional_stockout_cost_from_scenario = (new_safety_stock - row['safety_stock']) * row['avg_unit_cost'] * self.optimized_stockout_cost_multiplier
                
            scenario1_results.append({
                'product_category': row['product_category'],
                'abc_class': row['abc_class'],
                'xyz_class': row['xyz_class'],
                'scenario': 'Увеличение LT на 20%',
                'original_SS': row['safety_stock'],
                'new_SS_needed': round(new_safety_stock,2),
                'potential_additional_stockout_cost': round(potential_additional_stockout_cost_from_scenario, 2)
            })

        # Сценарий 2: Снижение спроса на 10% для всех категорий
        scenario2_results = []
        for index, row in original_opt_results.iterrows():
            temp_row = row.copy()
            temp_row['annual_demand_qty'] = row['annual_demand_qty'] * 0.9
            temp_row['mean_daily_demand'] = row['mean_daily_demand'] * 0.9

            # Пересчитаем SS и Order_Quantity для нового спроса
            original_optimizer = InventoryOptimizer(self.category_data, self.product_demand_data, self.df_full_orders) 
            
            # Пересчитаем SS
            new_safety_stock = original_optimizer.calculate_safety_stock(
                temp_row['mean_daily_demand'],
                row['std_daily_demand_used_for_ss'], # Предполагаем, что std_daily_demand не меняется пропорционально
                row['avg_lead_time_days'],
                row['service_level']
            )

            # Пересчитаем Order Quantity
            new_order_quantity = original_optimizer.calculate_eoq(
                temp_row['annual_demand_qty'],
                self.ordering_cost,
                row['avg_unit_cost'],
                self.holding_cost_rate
            )


            # Экономический эффект: снижение затрат на хранение и заказы
            # Если новый Order Quantity или Safety Stock отрицательный или NaN, то обнуляем его.
            new_safety_stock_for_calc = max(0, new_safety_stock) if not np.isnan(new_safety_stock) else 0
            new_order_quantity_for_calc = max(0, new_order_quantity) if not np.isnan(new_order_quantity) else 0

            original_avg_inventory = row['safety_stock'] + (row['order_quantity']/2)
            new_avg_inventory = new_safety_stock_for_calc + (new_order_quantity_for_calc/2)

            potential_decrease_holding_cost = (original_avg_inventory - new_avg_inventory) * row['avg_unit_cost'] * self.holding_cost_rate
            
            original_num_orders = row['annual_demand_qty'] / row['order_quantity'] if row['order_quantity'] > 0 else 0
            new_num_orders = temp_row['annual_demand_qty'] / new_order_quantity_for_calc if new_order_quantity_for_calc > 0 else 0

            potential_decrease_ordering_cost = (original_num_orders - new_num_orders) * self.ordering_cost
            
            potential_total_savings = potential_decrease_holding_cost + potential_decrease_ordering_cost

            scenario2_results.append({
                'product_category': row['product_category'],
                'abc_class': row['abc_class'],
                'xyz_class': row['xyz_class'],
                'scenario': 'Снижение спроса на 10%',
                'original_total_cost': round(self.calculate_optimized_costs(row)['optimized_total_cost'], 2),
                'new_SS_needed': round(new_safety_stock,2),
                'new_order_quantity': round(new_order_quantity,2),
                'potential_total_savings': round(potential_total_savings, 2)
            })

        return {
            "risks_and_mitigation": risks_and_mitigation,
            "scenario_analysis_lead_time_increase": pd.DataFrame(scenario1_results),
            "scenario_analysis_demand_decrease": pd.DataFrame(scenario2_results)
        }


if __name__ == "__main__":
    output_dir = "output/"
    category_summary, product_demand = analyze_categories()
    
    df_full_orders = pd.read_csv("final_orders_train.csv")
    df_full_orders['order_date'] = pd.to_datetime(df_full_orders['order_date'])
    df_full_orders['delivery_date'] = pd.to_datetime(df_full_orders['delivery_date'])
    df_full_orders['lead_time_days'] = (df_full_orders['delivery_date'] - df_full_orders['order_date']).dt.days
    df_full_orders['valid_delivered_qty'] = df_full_orders['valid_delivered_qty'].fillna(0)

    optimizer = InventoryOptimizer(category_summary, product_demand, df_full_orders)
    optimization_results = optimizer.run_optimization()

    evaluator = EconomicImpactEvaluator(optimization_results, optimizer.holding_cost_rate, optimizer.ordering_cost, category_summary, product_demand, df_full_orders)
    economic_impact_df = evaluator.evaluate_economic_effect()

    print("\nЭкономический эффект предложенных стратегий (годовой):")
    print(economic_impact_df.to_json(orient='records', indent=2))

    total_economic_effect = economic_impact_df['economic_effect'].sum()
    print(f"\nОбщий годовой экономический эффект: {total_economic_effect:,.2f}")

    # Проводим сценарный анализ
    risks_and_scenarios = evaluator.analyze_risks_and_scenarios(optimization_results)

    print("\nАнализ рисков и предложения по митигации:")
    for category, detail in risks_and_scenarios["risks_and_mitigation"].items():
        print(f"\n--- {category} ---")
        print("  Риски:")
        for risk in detail["Риски"]:
            print(f"    - {risk}")
        print("  Митигация:")
        for mitigation in detail["Митигация"]:
            print(f"    - {mitigation}")

    print("\nСценарный анализ: Увеличение срока поставки на 20%")
    print(risks_and_scenarios["scenario_analysis_lead_time_increase"].to_json(orient='records', indent=2))

    print("\nСценарный анализ: Снижение спроса на 10%")
    print(risks_and_scenarios["scenario_analysis_demand_decrease"].to_json(orient='records', indent=2))

    # Визуализация экономического эффекта
    plt.figure(figsize=(14, 7))
    sns.barplot(x='product_category', y='economic_effect', hue='abc_class', data=economic_impact_df, palette='viridis', dodge=False)
    plt.title('Годовой экономический эффект по категориям')
    plt.xlabel('Категория продукта')
    plt.ylabel('Экономический эффект (руб.)')
    plt.legend(title='ABC Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'economic_effect_by_category.png')) # Используем os.path.join
    plt.close()

    print("\nСгенерирован график: economic_effect_by_category.png")

    # ----- Генерация плана закупок в CSV -----
    # Собираем данные для CSV
    if not isinstance(optimization_results, pd.DataFrame) or optimization_results.empty:
        print("Ошибка: optimization_results не является DataFrame или пуст. Невозможно сгенерировать procurement_plan.csv")
    else:
        procurement_plan_df = optimization_results[[
            'product_category', 
            'abc_class', 
            'xyz_class', 
            'annual_demand_qty', 
            'avg_unit_cost',
            'avg_lead_time_days',
            'service_level',
            'mean_daily_demand',
            'safety_stock', 
            'order_quantity', 
            'reorder_point'
        ]].copy()

        # Добавляем рекомендации по стратегии
        procurement_plan_df['recommended_strategy'] = procurement_plan_df.apply(lambda row: 
            "Агрессивное управление с фокусом на глубокое прогнозирование и сокращение LT" if row['abc_class'] == 'A' and row['xyz_class'] == 'Z' else
            "Фокус на сокращение LT и снижении дефицита, повышенный SS" if row['abc_class'] == 'B' and row['xyz_class'] == 'Z' else
            "Эффективное управление объемными, но дешевыми компонентами" if row['product_category'] == 3 and row['abc_class'] == 'C' and row['xyz_class'] == 'Z' else
            "Упрощенное управление, фокус на доступности при минимальных затратах" if row['abc_class'] == 'C' and row['xyz_class'] == 'X' else
            "Общая рекомендация: оптимизация с учетом особенностей ABC/XYZ", axis=1
        )

        try:
            procurement_plan_df.to_csv(os.path.join(output_dir, 'procurement_plan.csv'), index=False, float_format='%.2f') # Используем os.path.join
            print("\nПлан закупок сохранен в " + os.path.join(output_dir, 'procurement_plan.csv')) # Обновляем сообщение
        except Exception as e:
            print(f"Ошибка при сохранении procurement_plan.csv: {e}")
