import numpy as np
from scipy import stats

# 1. Исходные данные
data = np.array([
    -1.00, 0.92, 0.06, -0.21, 0.21, 0.44, -0.14, -0.67, 0.44,
    0.65, -0.33, 0.19, -0.12, -0.77, 0.15, 0.67, -0.99, 0.59,
    0.28, 0.24, 0.13, -0.37, 0.14, -0.09, 0.79, -0.64, 0.30,
    -0.83, -0.17, -1.00, 0.10, -0.21, -0.23, -0.92, -0.57, 0.27,
    1.00, 0.48, -0.97, -0.42, -0.46, -0.81, -0.07, -0.59, 1.00,
    -0.95, 0.61, -0.29, -1.00, -0.03, 0.39, -0.85, 0.45, 0.29,
    0.78, 0.17, 0.87, -0.96, 0.21, -0.48, -0.29, 0.07, -0.36,
    0.08, -1.00, 0.98, 0.85, 0.32, -0.24, 0.42, -1.00, 0.24,
    0.88, -0.74, -0.28, 0.36, 0.46, 0.64, 0.90, 0.01, -0.24,
    0.36, 0.01, -0.45, -0.22, -0.29, -0.77, 0.40, -1.00, 0.15,
    -0.16, -0.27, -0.27, -0.10, 0.69, 0.40, -0.08, -0.81, 0.17
])

n = len(data)
k = 7

# --- ШАГ 1: Группировка данных ---
min_val, max_val = data.min(), data.max()
h = (max_val - min_val) / k
bins = np.linspace(min_val, max_val, k + 1)

# Частоты (ni) и относительные частоты (wi)
frequencies, _ = np.histogram(data, bins=bins)
relative_freqs = frequencies / n

# Середины интервалов (xi)
midpoints = bins[:-1] + h / 2

# --- ШАГ 2: Точечные оценки ---
# Выборочное среднее: x_cp = sum(xi * wi)
s_mean = np.sum(midpoints * relative_freqs)

# Выборочная дисперсия: D = sum(wi * (xi - x_cp)^2)
s_var = np.sum(relative_freqs * (midpoints - s_mean) ** 2)

# Среднеквадратическое отклонение: sigma = sqrt(D)
sigma = np.sqrt(s_var)

print(f"--- ТОЧЕЧНЫЕ ОЦЕНКИ ---")
print(f"Выборочное среднее: {s_mean:.4f}")
print(f"Выборочная дисперсия: {s_var:.4f}")
print(f"Среднеквадратическое отклонение: {sigma:.4f}")
print("-" * 30)


# --- ШАГ 3: Интервальные оценки ---
def calculate_interval(gamma):
    # Уровень значимости
    alpha = 1 - gamma
    # Находим t_gamma (критическое значение Стьюдента для n-1 степеней свободы)
    t_gamma = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # Точность оценки delta = (t_gamma * sigma_b) / sqrt(n)
    delta = (t_gamma * sigma) / np.sqrt(n)

    lower = s_mean - delta
    upper = s_mean + delta

    return t_gamma, delta, lower, upper


# Расчет для двух вероятностей
for g in [0.95, 0.99]:
    tg, d, low, up = calculate_interval(g)
    print(f"Для gamma = {g}:")
    print(f"  t_gamma = {tg:.3f}")
    print(f"  Точность (delta) = {d:.4f}")
    print(f"  Доверительный интервал: ({low:.4f} < M < {up:.4f})")
    print()