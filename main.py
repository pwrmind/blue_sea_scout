import torch
import pandas as pd
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Настройки 2026
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64 

class MarketEngine:
    def __init__(self, data_df):
        self.prices = torch.tensor(data_df['price'].values, device=device, dtype=dtype)
        self.purchase = torch.tensor(data_df['purchase_price'].values, device=device, dtype=dtype)
        self.weights = torch.tensor(data_df['weight'].values, device=device, dtype=dtype)
        self.reviews = torch.tensor(data_df['reviews'].values, device=device, dtype=dtype)
        self.revenue = torch.tensor(data_df['revenue'].values, device=device, dtype=dtype)
        
    def calculate_roi_for_prices(self, test_prices):
        """Расчет ROI для произвольного набора цен"""
        avg_purchase = self.purchase.mean()
        avg_weight = self.weights.mean()
        tax, fee, logis = 0.06, 0.15, 60.0
        
        profits = (test_prices * (1 - fee - tax)) - avg_purchase - (avg_weight * logis)
        roi = (profits / avg_purchase) * 100
        return roi

def train_bayesian_model(engine):
    raw_x = torch.stack([engine.prices, engine.reviews], dim=-1)
    x_min, x_max = raw_x.min(dim=0).values, raw_x.max(dim=0).values
    train_x = (raw_x - x_min) / (x_max - x_min)
    
    # Целевая метрика: Логарифм выручки на 1 отзыв (эффективность ниши)
    profit_index = (engine.revenue / (engine.reviews + 1)).log().unsqueeze(-1)
    train_y = (profit_index - profit_index.mean()) / profit_index.std()

    gp = SingleTaskGP(train_x, train_y).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    return gp, x_min, x_max

def get_final_decision(engine, gp_model, x_min, x_max):
    """Инструмент сам принимает решение и фильтрует данные"""
    # 1. Генерируем широкую сетку цен
    test_prices = torch.linspace(500, 15000, 100, device=device, dtype=dtype).unsqueeze(-1)
    test_reviews = torch.full((100, 1), 5.0, device=device, dtype=dtype)
    raw_x = torch.cat([test_prices, test_reviews], dim=-1)
    
    # 2. Получаем предсказания AI
    scaled_x = (raw_x - x_min) / (x_max - x_min)
    with torch.no_grad():
        posterior = gp_model.posterior(scaled_x)
        score = posterior.mean.flatten()
        uncertainty = torch.sqrt(posterior.variance.flatten())

    # 3. Рассчитываем ROI для всей сетки
    roi = engine.calculate_roi_for_prices(test_prices.flatten())

    # 4. АВТОМАТИЧЕСКАЯ ФИЛЬТРАЦИЯ (Constraint)
    # Оставляем только где ROI > 20% и Score выше среднего
    mask = (roi > 20.0) & (score > score.mean())
    
    final_table = pd.DataFrame({
        'Price': test_prices.flatten()[mask].cpu().numpy(),
        'AI_Score': score[mask].cpu().numpy(),
        'Risk': uncertainty[mask].cpu().numpy(),
        'ROI': roi[mask].cpu().numpy()
    })

    # Считаем финальный рейтинг привлекательности
    final_table['Final_Rating'] = final_table['AI_Score'] / (1 + final_table['Risk'])
    
    # Возвращаем только ТОП-5 реально выгодных стратегий
    return final_table.sort_values(by='Final_Rating', ascending=False).head(5)

if __name__ == "__main__":
    # Имитация данных
    df_market = pd.DataFrame({
        'price': np.random.uniform(500, 10000, 1000),
        'purchase_price': np.random.uniform(300, 2000, 1000),
        'weight': np.random.uniform(0.5, 2.0, 1000),
        'reviews': np.random.lognormal(2, 1, 1000),
        'revenue': np.random.uniform(10000, 500000, 1000)
    })

    engine = MarketEngine(df_market)
    gp_model, x_min, x_max = train_bayesian_model(engine)
    
    # Получаем готовое решение
    decisions = get_final_decision(engine, gp_model, x_min, x_max)

    print("\n" + "!"*25 + " РЕКОМЕНДАЦИИ К ЗАКУПКЕ " + "!"*25)
    if decisions.empty:
        print("ВНИМАНИЕ: При текущих затратах прибыльных ниш не обнаружено.")
    else:
        print(decisions[['Price', 'ROI', 'Final_Rating']].to_string(
            index=False, 
            formatters={'Price': '{:,.0f} руб.'.format, 'ROI': '{:,.1f}%'.format}
        ))
    print("!"*74)
