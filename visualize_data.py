import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
from analyze_categories import analyze_categories

OUTPUT_DIR = "output/"
DPI = 300


def plot_category_data(category_summary, product_demand, df_full, output_dir=OUTPUT_DIR):
    """Строим все графики"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Генерация графиков в {output_dir}")

    # 1. ABC анализ
    try:
        plt.figure(figsize=(12, 7))
        category_sorted = category_summary.sort_values(
            'percentage_of_total_amount',
            ascending=False
        ).reset_index(drop=True)

        plt.plot(
            category_sorted['product_category'].astype(str),
            category_sorted['cumulative_percentage'],
            marker='o',
            linewidth=2.5,
            markersize=10,
            color='#2c3e50',
            markerfacecolor='#3498db',
            markeredgecolor='white',
            markeredgewidth=2
        )

        plt.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=1.5,
                    label='80% (A-класс)', alpha=0.8)
        plt.axhline(y=95, color='#f39c12', linestyle='--', linewidth=1.5,
                    label='95% (B-класс)', alpha=0.8)

        plt.title('ABC Анализ')
        plt.xlabel('Категория')
        plt.ylabel('Накопленная доля (%)')
        plt.legend()
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filepath = os.path.join(output_dir, 'abc_analysis_category.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Сохранен: abc_analysis_category.png")
    except Exception as e:
        print(f"Ошибка ABC: {e}")

    # 2. XYZ анализ
    try:
        plt.figure(figsize=(12, 7))
        bars = sns.barplot(
            x='product_category',
            y='demand_volatility_coef',
            data=category_summary,
            palette='coolwarm',
            edgecolor='black',
            linewidth=0.5
        )

        for container in bars.containers:
            bars.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

        plt.axhline(y=0.1, color='#27ae60', linestyle='--', linewidth=1.5,
                    label='X (< 0.1)', alpha=0.7)
        plt.axhline(y=0.25, color='#f39c12', linestyle='--', linewidth=1.5,
                    label='Y (0.1-0.25)', alpha=0.7)

        plt.title('XYZ Анализ')
        plt.xlabel('Категория')
        plt.ylabel('Коэффициент вариации')
        plt.legend()
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        filepath = os.path.join(output_dir, 'xyz_analysis_category.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Сохранен: xyz_analysis_category.png")
    except Exception as e:
        print(f"Ошибка XYZ: {e}")

    # 3. Lead Time
    try:
        plt.figure(figsize=(12, 7))
        df_filtered = df_full[
            (df_full['lead_time_days'] >= 0) &
            (df_full['lead_time_days'] <= df_full['lead_time_days'].quantile(0.99))
            ].copy()

        sns.boxplot(
            x='product_category',
            y='lead_time_days',
            data=df_filtered,
            palette='viridis',
            showfliers=True,
            flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3}
        )

        plt.title('Распределение сроков поставки')
        plt.xlabel('Категория')
        plt.ylabel('Дней')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        filepath = os.path.join(output_dir, 'lead_time_distribution_category.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Сохранен: lead_time_distribution_category.png")
    except Exception as e:
        print(f"Ошибка Lead Time: {e}")

    # 4. Количество и стоимость
    try:
        fig, ax1 = plt.subplots(figsize=(12, 7))

        color_qty = 'skyblue'
        bars = ax1.bar(
            category_summary['product_category'].astype(str),
            category_summary['total_qty'],
            color=color_qty,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            label='Количество'
        )
        ax1.set_xlabel('Категория')
        ax1.set_ylabel('Количество (шт.)', color=color_qty)
        ax1.tick_params(axis='y', labelcolor=color_qty)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        ax2 = ax1.twinx()
        color_amount = '#e74c3c'
        line = ax2.plot(
            category_summary['product_category'].astype(str),
            category_summary['total_amount'],
            color=color_amount,
            marker='o',
            linewidth=2.5,
            markersize=8,
            markerfacecolor='white',
            markeredgecolor=color_amount,
            markeredgewidth=2,
            label='Стоимость'
        )
        ax2.set_ylabel('Стоимость (руб.)', color=color_amount)
        ax2.tick_params(axis='y', labelcolor=color_amount)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title('Количество и стоимость')
        plt.tight_layout()

        filepath = os.path.join(output_dir, 'total_qty_amount_category.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Сохранен: total_qty_amount_category.png")
    except Exception as e:
        print(f"Ошибка dual-axis: {e}")

    # 5. Временной ряд
    try:
        if category_summary.empty:
            print("category_summary пуст")
        else:
            top_category = category_summary.loc[
                category_summary['total_qty'].idxmax(),
                'product_category'
            ]

            df_filtered = df_full[df_full['product_category'] == top_category].copy()

            if df_filtered.empty:
                print(f"Нет данных для категории {top_category}")
            else:
                df_filtered = df_filtered.set_index('order_date')

                # Используем 'M' вместо 'ME' для совместимости
                try:
                    monthly_data = df_filtered.resample('M')['valid_delivered_qty'].sum()
                except Exception:
                    monthly_data = df_filtered.resample('MS')['valid_delivered_qty'].sum()

                if monthly_data.empty or monthly_data.sum() == 0:
                    print(f"Нет данных для временного ряда")
                else:
                    plt.figure(figsize=(16, 8))

                    plt.plot(
                        monthly_data.index,
                        monthly_data.values,
                        marker='o',
                        linewidth=2,
                        markersize=5,
                        color='#3498db',
                        markerfacecolor='white',
                        markeredgecolor='#3498db'
                    )

                    if len(monthly_data) >= 3:
                        rolling_avg = monthly_data.rolling(window=3, min_periods=1).mean()
                        plt.plot(
                            rolling_avg.index,
                            rolling_avg.values,
                            linewidth=2,
                            color='#e74c3c',
                            linestyle='--',
                            label='Скользящее среднее (3 мес.)',
                            alpha=0.8
                        )

                    plt.title(f'Потребление: Категория {top_category}')
                    plt.xlabel('Дата')
                    plt.ylabel('Количество')
                    plt.legend()
                    plt.grid(True, alpha=0.3, linestyle='--')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    filepath = os.path.join(output_dir, f'time_series_category_{top_category}.png')
                    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
                    plt.close()
                    print(f"Сохранен: time_series_category_{top_category}.png")
    except Exception as e:
        print(f"Ошибка временного ряда: {e}")

    print(f"\nГрафики в: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    print("Запуск визуализации...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    category_summary, product_demand = analyze_categories()

    if category_summary is None:
        print("Не удалось загрузить данные")
        exit(1)

    df_full = pd.read_csv("final_orders_train.csv")
    df_full['order_date'] = pd.to_datetime(df_full['order_date'], errors='coerce')
    df_full['delivery_date'] = pd.to_datetime(df_full['delivery_date'], errors='coerce')
    df_full['lead_time_days'] = (
            df_full['delivery_date'] - df_full['order_date']
    ).dt.days
    df_full['valid_delivered_qty'] = df_full['valid_delivered_qty'].fillna(0)

    print(f"Загружено {len(df_full)} записей")

    plot_category_data(category_summary, product_demand, df_full, OUTPUT_DIR)

    print("Визуализация завершена!")

"""
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
"""
