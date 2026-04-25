import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Настройки
sizes = [100, 1000]  # Размеры выборок
distributions = {
    "Нормальное (0, 1)": lambda n: np.random.normal(0, 1, n),
    "Равномерное (0, 10)": lambda n: np.random.uniform(0, 10, n),
    "Экспоненциальное (scale=1)": lambda n: np.random.exponential(1, n)
}


def analyze_sample(data, name, size):
    # Расчет статистики
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_res = stats.mode(data, keepdims=True)
    mode_val = mode_res.mode[0]

    sample_range = np.ptp(data)
    var_biased = np.var(data)  # Смещенная
    var_unbiased = np.var(data, ddof=1)  # Несмещенная

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    print(f"--- {name} (N={size}) ---")
    print(f"Среднее: {mean_val:.3f}, Медиана: {median_val:.3f}, Мода: {mode_val:.3f}")
    print(f"Дисперсия (смещ/несмещ): {var_biased:.3f} / {var_unbiased:.3f}")
    print(f"Размах: {sample_range:.3f}, IQR: {iqr:.3f}\n")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Гистограмма и Полигон частот
    counts, bins, _ = axes[0].hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Гистограмма')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    axes[0].plot(bin_centers, counts, color='red', marker='o', label='Полигон частот')

    # Линии статистики
    axes[0].axvline(mean_val, color='green', linestyle='--', label=f'Среднее: {mean_val:.2f}')
    axes[0].axvline(median_val, color='orange', linestyle='-', label=f'Медиана: {median_val:.2f}')
    axes[0].legend()
    axes[0].set_title(f"Гистограмма и Статистика ({name})")

    # 2. Эмпирическая функция распределения (ECDF)
    x_sorted = np.sort(data)
    y_values = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    axes[1].step(x_sorted, y_values, where='post', color='purple')

    # Квартили на ECDF
    for q, label in zip([q1, median_val, q3], ['Q1', 'Q2/Med', 'Q3']):
        axes[1].axvline(q, color='gray', linestyle=':', alpha=0.6)
        axes[1].text(q, 0.1, label, rotation=90)

    axes[1].set_title("Эмпирическая функция распределения (ECDF)")
    plt.tight_layout()
    plt.show()


# Запуск анализа
for name, dist_func in distributions.items():
    for size in sizes:
        analyze_sample(dist_func(size), name, size)
