import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# 1. Настройки среды 2026
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64 

class MarketEngine:
    def __init__(self, df):
        self.df = df
        self.prices = torch.tensor(df['price'].values, device=device, dtype=dtype)
        self.purchase = torch.tensor(df['purchase_price'].values, device=device, dtype=dtype)
        self.weights = torch.tensor(df['weight'].values, device=device, dtype=dtype)
        self.reviews = torch.tensor(df['reviews'].values, device=device, dtype=dtype)
        self.revenue = torch.tensor(df['revenue'].values, device=device, dtype=dtype)
        
    def calculate_roi_vectorized(self, test_prices):
        """Расчет ROI с учетом распределенных затрат"""
        avg_purchase = self.purchase.mean()
        avg_weight = self.weights.mean()
        # Динамические параметры
        tax, fee, logis = 0.06, 0.15, 60.0
        profits = (test_prices * (1 - fee - tax)) - avg_purchase - (avg_weight * logis)
        return (profits / avg_purchase) * 100

def train_model(engine):
    raw_x = torch.stack([engine.prices, engine.reviews], dim=-1)
    x_min, x_max = raw_x.min(dim=0).values, raw_x.max(dim=0).values
    train_x = (raw_x - x_min) / (x_max - x_min)
    
    # Целевая переменная: Эффективность выручки на единицу социального доказательства
    y = (engine.revenue / (engine.reviews + 1)).log().unsqueeze(-1)
    train_y = (y - y.mean()) / y.std()

    gp = SingleTaskGP(train_x, train_y).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    return gp, x_min, x_max

def generate_analytics(engine, gp_model, x_min, x_max):
    # --- ВИЗУАЛИЗАЦИЯ: HEATMAP ---
    res = 40 # разрешение сетки
    p_grid = torch.linspace(500, 12000, res, device=device, dtype=dtype)
    r_grid = torch.linspace(1, 200, res, device=device, dtype=dtype)
    grid_p, grid_r = torch.meshgrid(p_grid, r_grid, indexing='ij')
    
    flat_x = torch.stack([grid_p.flatten(), grid_r.flatten()], dim=-1)
    scaled_x = (flat_x - x_min) / (x_max - x_min)
    
    with torch.no_grad():
        posterior = gp_model.posterior(scaled_x)
        scores = posterior.mean.reshape(res, res).cpu().numpy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(scores.T, cmap='magma', 
                xticklabels=[f'{int(p)}' for p in p_grid[::5]], 
                yticklabels=[f'{int(r)}' for r in r_grid[::5]])
    plt.title('2D КАРТА ПОТЕНЦИАЛА: ЦЕНА vs ОТЗЫВЫ (2026)')
    plt.xlabel('Цена (руб)')
    plt.ylabel('Количество отзывов')
    plt.gca().invert_yaxis()
    plt.show()

    # --- ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТЧЕТА ---
    test_prices = torch.linspace(500, 12000, 100, device=device, dtype=dtype)
    # Симулируем выход новичка (ровно 10 отзывов)
    test_reviews = torch.full_like(test_prices, 10.0)
    
    raw_test_x = torch.stack([test_prices, test_reviews], dim=-1)
    scaled_test_x = (raw_test_x - x_min) / (x_max - x_min)
    
    with torch.no_grad():
        post = gp_model.posterior(scaled_test_x)
        score_mean = post.mean.flatten()
        uncertainty = torch.sqrt(post.variance.flatten())
        roi = engine.calculate_roi_vectorized(test_prices)

    # Формируем таблицу
    results = pd.DataFrame({
        'Price': test_prices.cpu().numpy(),
        'ROI_%': roi.cpu().numpy(),
        'AI_Score': score_mean.cpu().numpy(),
        'Risk': uncertainty.cpu().numpy()
    })

    # Улучшенный рейтинг: Вес AI (70%) + Вес ROI (30%) за вычетом Риска
    norm_score = (results['AI_Score'] - results['AI_Score'].min()) / (results['AI_Score'].max() - results['AI_Score'].min())
    norm_roi = (results['ROI_%'] - results['ROI_%'].min()) / (results['ROI_%'].max() - results['ROI_%'].min())
    results['Final_Rating'] = (0.7 * norm_score + 0.3 * norm_roi) / (1 + results['Risk'])

    # Фильтр: Только прибыльные и только лучшие по рейтингу
    return results[results['ROI_%'] > 20].sort_values('Final_Rating', ascending=False).head(10)

if __name__ == "__main__":
    # Создание данных
    np.random.seed(42)
    df_market = pd.DataFrame({
        'price': np.random.uniform(1000, 12000, 1000),
        'purchase_price': np.random.uniform(500, 3500, 1000),
        'weight': np.random.uniform(0.3, 3.5, 1000),
        'reviews': np.random.lognormal(3, 1, 1000),
        'revenue': np.random.uniform(50000, 1500000, 1000)
    })

    print("--- 1. ИСХОДНЫЕ ДАННЫЕ РЫНКА ---")
    print(df_market.head())

    engine = MarketEngine(df_market)
    print("\n--- 2. ОБУЧЕНИЕ НЕЙРОННОЙ МОДЕЛИ (CUDA)... ---")
    gp, x_min, x_max = train_model(engine)
    
    print("\n--- 3. АНАЛИЗ И ВИЗУАЛИЗАЦИЯ... ---")
    final_decisions = generate_analytics(engine, gp, x_min, x_max)

    print("\n" + "█"*20 + " РЕЗУЛЬТАТЫ: ЛУЧШИЕ ТОЧКИ ВХОДА " + "█"*20)
    print(final_decisions[['Price', 'ROI_%', 'Final_Rating']].to_string(
        index=False, formatters={'Price': '{:,.0f} р.'.format, 'ROI_%': '{:,.1f}%'.format}
    ))
    print("█"*72)
