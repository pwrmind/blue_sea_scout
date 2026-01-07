import torch
import pandas as pd
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
import matplotlib.pyplot as plt

# 1. Глобальные настройки CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# 2. Мощный движок расчета экономики (Векторизованный)
class MarketEngine:
    def __init__(self, data_df):
        # Переводим всё в тензоры сразу при инициализации
        self.prices = torch.tensor(data_df['price'].values, device=device, dtype=dtype)
        self.purchase = torch.tensor(data_df['purchase_price'].values, device=device, dtype=dtype)
        self.weights = torch.tensor(data_df['weight'].values, device=device, dtype=dtype)
        self.reviews = torch.tensor(data_df['reviews'].values, device=device, dtype=dtype)
        self.revenue = torch.tensor(data_df['revenue'].values, device=device, dtype=dtype)
        
    def get_unit_economy(self, current_prices, logistic_cost=60.0, marketplace_fee=0.15):
        """Мгновенный расчет прибыли на CUDA"""
        tax = 0.06
        net_revenue = current_prices * (1 - marketplace_fee - tax)
        logistics = self.weights * logistic_cost
        profit = net_revenue - self.purchase - logistics
        roi = (profit / self.purchase) * 100
        return profit, roi

# 3. Байесовская оптимизация и поиск ниш (BoTorch)
def find_optimal_niche_points(engine):
    # Подготовка признаков: нормализуем цену и отзывы
    train_x = torch.stack([engine.prices, engine.reviews], dim=-1)
    train_x = (train_x - train_x.mean(dim=0)) / train_x.std(dim=0)
    
    # Целевая метрика: Profitability Index (сочетание прибыли и легкости захода)
    # Мы ищем где Revenue высокое, а Reviews низкие
    profit_index = (engine.revenue / (engine.reviews + 1)).log().unsqueeze(-1)
    profit_index = (profit_index - profit_index.mean()) / profit_index.std()

    # Обучаем Гауссовский процесс на GPU
    gp = SingleTaskGP(train_x, profit_index).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)

    # Функция захвата (Acquisition Function): ищем "золотую середину" между известным и неизвестным
    # UCB помогает найти точки с высоким потенциалом (Exploration)
    UCB = UpperConfidenceBound(gp, beta=0.1)
    
    return gp, UCB

# --- ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ ---

# Генерируем данные о 5000 товарах (симуляция парсинга маркетплейса)
n_items = 5000
df_market = pd.DataFrame({
    'price': np.random.uniform(500, 10000, n_items),
    'purchase_price': np.random.uniform(200, 4000, n_items),
    'weight': np.random.uniform(0.1, 5.0, n_items),
    'reviews': np.random.lognormal(2, 1, n_items),
    'revenue': np.random.uniform(5000, 1000000, n_items)
})

# Запуск движка
engine = MarketEngine(df_market)
profit, roi = engine.get_unit_economy(engine.prices)

print(f"Средний ROI по рынку: {roi.mean().item():.2f}%")

# Запуск AI-поиска ниш
print("Обучение Байесовской модели поиска аномалий...")
gp_model, ucb_func = find_optimal_niche_points(engine)

# Проверка: рассчитаем потенциал для сетки цен
test_prices = torch.linspace(500, 10000, 100, device=device).unsqueeze(-1)
test_reviews = torch.full((100, 1) , 10.0, device=device) # Что если у нас всего 10 отзывов?
test_x = torch.cat([test_prices, test_reviews], dim=-1)
# Нормализация для теста
test_x_norm = (test_x - test_x.mean(dim=0)) / (test_x.std(dim=0) + 1e-6)

with torch.no_grad():
    posterior = gp_model.posterior(test_x_norm)
    mean_potential = posterior.mean
    lower, upper = posterior.mvn.confidence_region()

# Визуализация "Карты возможностей"
plt.figure(figsize=(10, 6))
plt.fill_between(test_prices.cpu().flatten(), lower.cpu(), upper.cpu(), alpha=0.2, label='Неопределенность (Риск)')
plt.plot(test_prices.cpu().numpy(), mean_potential.cpu().numpy(), label='Потенциал прибыли', color='blue')
plt.title("Анализ ниши: Потенциал прибыли vs Цена (при 10 отзывах)")
plt.xlabel("Цена товара")
plt.ylabel("Индекс привлекательности")
plt.legend()
plt.grid(True)
plt.show()
