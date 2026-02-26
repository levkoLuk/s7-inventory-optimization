import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загружаем функцию анализа категорий
from analyze_categories import analyze_categories

def plot_category_data(category_summary, product_demand, df_full):
    output_dir = "output/"

    # 1. ABC-анализ: Кривая Лоренца для общей стоимости по категориям
    plt.figure(figsize=(10, 6))
    plt.plot(category_summary['percentage_of_total_amount'], category_summary['cumulative_amount'] / category_summary['total_amount'].sum(), marker='o')
    plt.title('ABC Анализ по категориям: Кривая Лоренца (Общая Стоимость)')
    plt.xlabel('Процент категорий (накопительным итогом)')
    plt.ylabel('Процент общей стоимости (накопительным итогом)')
    plt.grid(True)
    plt.axvline(x=0.8, color='r', linestyle='--', label='80%')
    plt.axvline(x=0.95, color='g', linestyle='--', label='95%')
    plt.legend()
    plt.savefig(f'{output_dir}abc_analysis_category.png')
    plt.close()

    # 2. XYZ-анализ: Гистограмма коэффициента вариации спроса для каждой категории.
    plt.figure(figsize=(10, 6))
    sns.barplot(x='product_category', y='demand_volatility_coef', data=category_summary)
    plt.title('XYZ Анализ по категориям: Коэффициент Вариации Спроса')
    plt.xlabel('Категория продукта')
    plt.ylabel('Коэффициент вариации спроса')
    plt.savefig(f'{output_dir}xyz_analysis_category.png')
    plt.close()

    # 3. Распределение сроков поставки по категориям
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='product_category', y='lead_time_days', data=df_full)
    plt.title('Распределение сроков поставки по категориям')
    plt.xlabel('Категория продукта')
    plt.ylabel('Срок поставки (дни)')
    plt.savefig(f'{output_dir}lead_time_distribution_category.png')
    plt.close()

    # 4. Суммарное количество и стоимость по категориям
    fig, ax1 = plt.subplots(figsize=(12, 7))

    sns.barplot(x='product_category', y='total_qty', data=category_summary, ax=ax1, color='skyblue', alpha=0.6)
    ax1.set_xlabel('Категория продукта')
    ax1.set_ylabel('Общее количество (шт.)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    sns.lineplot(x='product_category', y='total_amount', data=category_summary, ax=ax2, color='red', marker='o')
    ax2.set_ylabel('Общая стоимость', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Общее количество и стоимость по категориям')
    plt.savefig(f'{output_dir}total_qty_amount_category.png')
    plt.close()

    # 5. Пример временного ряда для категории с наибольшим объемом покупок.
    # Найдем категорию с наибольшим total_qty
    top_category = category_summary.loc[category_summary['total_qty'].idxmax(), 'product_category']
    top_category_data = df_full[df_full['product_category'] == top_category].set_index('order_date').resample('ME')['valid_delivered_qty'].sum()

    plt.figure(figsize=(14, 7))
    top_category_data.plot()
    plt.title(f'Временной ряд потребления для категории {top_category}')
    plt.xlabel('Дата заказа')
    plt.ylabel('Суммарное количество доставлено')
    plt.grid(True)
    plt.savefig(f'{output_dir}time_series_category_{top_category}.png')
    plt.close()

    print(f"Сгенерированы и сохранены следующие графики в директорию '{output_dir}': \n"
          f"abc_analysis_category.png, \n"
          f"xyz_analysis_category.png, \n"
          f"lead_time_distribution_category.png, \n"
          f"total_qty_amount_category.png, \n"
          f"time_series_category_{top_category}.png")

if __name__ == "__main__":
    output_dir = "output/"
    category_summary, product_demand = analyze_categories()
    
    # Чтобы построить boxplot для lead_time_days, нужна исходная DataFrame
    df_full = pd.read_csv("final_orders_train.csv")
    df_full['order_date'] = pd.to_datetime(df_full['order_date'])
    df_full['delivery_date'] = pd.to_datetime(df_full['delivery_date'])
    df_full['lead_time_days'] = (df_full['delivery_date'] - df_full['order_date']).dt.days

    plot_category_data(category_summary, product_demand, df_full)
