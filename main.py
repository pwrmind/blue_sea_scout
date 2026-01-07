import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Настройки CUDA и точности
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
        
    def calculate_roi(self, test_prices):
        avg_purchase = self.purchase.mean()
        avg_weight = self.weights.mean()
        tax, fee, logis = 0.06, 0.15, 60.0
        profits = (test_prices * (1 - fee - tax)) - avg_purchase - (avg_weight * logis)
        return (profits / avg_purchase) * 100

def train_model(engine):
    raw_x = torch.stack([engine.prices, engine.reviews], dim=-1)
    x_min, x_max = raw_x.min(dim=0).values, raw_x.max(dim=0).values
    train_x = (raw_x - x_min) / (x_max - x_min)
    
    y = (engine.revenue / (engine.reviews + 1)).log().unsqueeze(-1)
    train_y = (y - y.mean()) / y.std()

    gp = SingleTaskGP(train_x, train_y).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    return gp, x_min, x_max

def visualize_and_report(engine, gp_model, x_min, x_max):
    # 1. Генерируем тестовую сетку для визуализации
    test_prices = torch.linspace(500, 15000, 100, device=device, dtype=dtype).unsqueeze(-1)
    test_reviews = torch.full((100, 1), 10.0, device=device, dtype=dtype) # Для 10 отзывов
    scaled_x = (torch.cat([test_prices, test_reviews], dim=-1) - x_min) / (x_max - x_min)
    
    with torch.no_grad():
        posterior = gp_model.posterior(scaled_x)
        mean = posterior.mean.flatten()
        lower, upper = posterior.mvn.confidence_region()
        roi = engine.calculate_roi(test_prices.flatten())

    # 2. ПОСТРОЕНИЕ ГРАФИКА
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Ось потенциала (AI Score)
    ax1.set_xlabel('Цена товара (руб)')
    ax1.set_ylabel('AI Потенциал Ниши', color='cyan')
    ax1.plot(test_prices.cpu(), mean.cpu(), color='cyan', label='Прогноз потенциала', linewidth=2)
    ax1.fill_between(test_prices.cpu().flatten(), lower.cpu(), upper.cpu(), color='cyan', alpha=0.1, label='Зона риска')
    ax1.tick_params(axis='y', labelcolor='cyan')

    # Вторая ось для ROI
    ax2 = ax1.twinx()
    ax2.set_ylabel('ROI %', color='lime')
    ax2.plot(test_prices.cpu(), roi.cpu(), color='lime', linestyle='--', label='Юнит-экономика (ROI)')
    ax2.axhline(20, color='red', alpha=0.5, linestyle=':', label='Порог 20% ROI')
    ax2.tick_params(axis='y', labelcolor='lime')

    plt.title('КАРТА ВОЗМОЖНОСТЕЙ РЫНКА (2026)', fontsize=15)
    fig.tight_layout()
    plt.show()

    # 3. ФИНАЛЬНАЯ ТАБЛИЦА ПРИНЯТИЯ РЕШЕНИЙ
    mask = (roi > 20.0) & (mean > mean.mean())
    decisions = pd.DataFrame({
        'Цена_Входа': test_prices.flatten()[mask].cpu().numpy(),
        'ROI_%': roi[mask].cpu().numpy(),
        'AI_Score': mean[mask].cpu().numpy(),
        'Risk': (upper - lower)[mask].cpu().numpy()
    })
    decisions['Rating'] = decisions['AI_Score'] / (1 + decisions['Risk'])
    
    return decisions.sort_values('Rating', ascending=False).head(5)

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Имитация входящих данных
    np.random.seed(42)
    data = pd.DataFrame({
        'price': np.random.uniform(1000, 12000, 1000),
        'purchase_price': np.random.uniform(500, 3000, 1000),
        'weight': np.random.uniform(0.5, 3.0, 1000),
        'reviews': np.random.lognormal(3, 1, 1000),
        'revenue': np.random.uniform(50000, 2000000, 1000)
    })

    print("--- 1. СЫРЫЕ ДАННЫЕ (ОБЗОР) ---")
    print(data.head())
    print(f"\nВсего SKU в базе: {len(data)}")

    engine = MarketEngine(data)
    
    print("\n--- 2. ОБУЧЕНИЕ МОДЕЛИ НА CUDA... ---")
    gp_model, x_min, x_max = train_model(engine)
    
    print("\n--- 3. ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИИ И ОТЧЕТА ---")
    final_decision = visualize_and_report(engine, gp_model, x_min, x_max)

    print("\n" + "█"*15 + " ФИНАЛЬНЫЙ ACTION PLAN " + "█"*15)
    if not final_decision.empty:
        print(final_decision[['Цена_Входа', 'ROI_%', 'Rating']].to_string(
            index=False, formatters={'Цена_Входа': '{:,.0f} р.'.format, 'ROI_%': '{:,.1f}%'.format}
        ))
    else:
        print("Нерентабельно: поднимите маржу или ищите другие ниши.")
