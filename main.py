import torch
import pandas as pd
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt

# 1. Настройки CUDA и точности
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64 

# 2. Движок расчета экономики
class MarketEngine:
    def __init__(self, data_df):
        self.prices = torch.tensor(data_df['price'].values, device=device, dtype=dtype)
        self.purchase = torch.tensor(data_df['purchase_price'].values, device=device, dtype=dtype)
        self.weights = torch.tensor(data_df['weight'].values, device=device, dtype=dtype)
        self.reviews = torch.tensor(data_df['reviews'].values, device=device, dtype=dtype)
        self.revenue = torch.tensor(data_df['revenue'].values, device=device, dtype=dtype)
        
    def get_unit_economy(self, current_prices, logistic_cost=60.0, marketplace_fee=0.15):
        tax = 0.06
        net_revenue = current_prices * (1 - marketplace_fee - tax)
        logistics = self.weights * logistic_cost
        profit = net_revenue - self.purchase - logistics
        roi = (profit / self.purchase) * 100
        return profit, roi

# 3. Обучение модели
def train_bayesian_model(engine):
    # Приводим X к Unit Cube (0, 1) для стабильности BoTorch
    raw_x = torch.stack([engine.prices, engine.reviews], dim=-1)
    x_min = raw_x.min(dim=0).values
    x_max = raw_x.max(dim=0).values
    train_x = (raw_x - x_min) / (x_max - x_min)
    
    # Готовим Y: Индекс потенциала (Revenue / Reviews)
    profit_index = (engine.revenue / (engine.reviews + 1)).log().unsqueeze(-1)
    train_y = (profit_index - profit_index.mean()) / profit_index.std()

    # Создание и обучение GP модели
    gp = SingleTaskGP(train_x, train_y).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    
    return gp, x_min, x_max

# 4. Генерация отчета
def get_niche_report(engine, gp_model, x_min, x_max, n_candidates=10):
    # Сетка гипотетических цен при малом числе отзывов (5 шт)
    test_prices = torch.linspace(500, 10000, 50, device=device, dtype=dtype).unsqueeze(-1)
    test_reviews = torch.full((50, 1), 5.0, device=device, dtype=dtype)
    raw_x = torch.cat([test_prices, test_reviews], dim=-1)
    
    # Масштабирование
    scaled_x = (raw_x - x_min) / (x_max - x_min)
    
    with torch.no_grad():
        posterior = gp_model.posterior(scaled_x)
        scores = posterior.mean.cpu().numpy().flatten()
        variance = posterior.variance.cpu().numpy().flatten()
        uncertainty = np.sqrt(variance)

    report = pd.DataFrame({
        'Recommended_Price': test_prices.cpu().numpy().flatten(),
        'Potential_Score': scores,
        'Risk': uncertainty
    })

    # Расчет ROI для отчета
    avg_purchase = engine.purchase.mean().item()
    avg_weight = engine.weights.mean().item()
    tax, fee, logis = 0.06, 0.15, 60.0
    
    prices = report['Recommended_Price'].values
    profits = (prices * (1 - fee - tax)) - avg_purchase - (avg_weight * logis)
    report['ROI_Forecast'] = (profits / avg_purchase) * 100
    report['Score_Index'] = report['Potential_Score'] / (1 + report['Risk'])
    
    return report.sort_values(by='Score_Index', ascending=False).head(n_candidates)

# --- ГЛАВНЫЙ БЛОК ЗАПУСКА ---
if __name__ == "__main__":
    print(f"Запуск анализа на устройстве: {device}")
    
    # 1. Генерируем тестовые данные (имитация 1000 товаров)
    n_items = 1000
    df_market = pd.DataFrame({
        'price': np.random.uniform(500, 10000, n_items),
        'purchase_price': np.random.uniform(200, 4000, n_items),
        'weight': np.random.uniform(0.1, 5.0, n_items),
        'reviews': np.random.lognormal(2, 1, n_items),
        'revenue': np.random.uniform(5000, 1000000, n_items)
    })

    # 2. Инициализируем движок и данные
    engine = MarketEngine(df_market)
    
    # 3. Обучаем AI модель
    print("Обучение Байесовской модели... (это может занять 10-30 сек)")
    gp_model, x_min, x_max = train_bayesian_model(engine)
    
    # 4. Формируем финальную таблицу
    final_report = get_niche_report(engine, gp_model, x_min, x_max)

    # 5. Красивый вывод
    print("\n" + "="*70)
    print(" СВОДНАЯ ТАБЛИЦА ЛУЧШИХ ТОЧЕК ВХОДА (ПРОГНОЗ 2026)")
    print("="*70)
    
    # Форматирование для печати
    output = final_report.copy()
    output['Recommended_Price'] = output['Recommended_Price'].map('{:,.0f} руб.'.format)
    output['ROI_Forecast'] = output['ROI_Forecast'].map('{:,.1f}%'.format)
    
    print(output[['Recommended_Price', 'Potential_Score', 'Risk', 'ROI_Forecast', 'Score_Index']].to_string(index=False))
    print("="*70)
    print("ИНСТРУКЦИЯ: Ищите максимальный Score_Index при положительном ROI.")
