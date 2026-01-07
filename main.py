"""
Продвинутая система анализа рыночных возможностей с машинным обучением
Версия 2026.2 (исправленная)
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Машинное обучение
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Оптимизация
import optuna

# Визуализация
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow не установлен. Логирование экспериментов отключено.")

# ==================== КОНФИГУРАЦИЯ ====================

class MarketConfig:
    """Конфигурация системы анализа рынка"""
    
    def __init__(self):
        # Параметры данных
        self.min_price = 500
        self.max_price = 15000
        self.min_reviews = 1
        self.max_reviews = 500
        
        # Экономические параметры
        self.base_fee = 0.15  # Базовая комиссия маркетплейса
        self.base_tax = 0.06  # НДС
        self.base_logistics = 60.0  # руб/кг
        self.target_roi = 0.20  # Минимальный ROI
        self.risk_free_rate = 0.08  # Безрисковая ставка
        
        # Параметры модели
        self.ai_weight = 0.6
        self.roi_weight = 0.3
        self.risk_weight = 0.1
        self.normalization_method = 'z-score'  # 'minmax' или 'z-score'
        
        # Параметры обучения
        self.train_test_split = 0.8
        self.n_folds = 5
        self.random_seed = 42
        
        # Визуализация
        self.heatmap_resolution = 30  # Уменьшено для скорости
        self.color_map = 'viridis'
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path='market_config.json'):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path='market_config.json'):
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# ==================== ОБРАБОТКА ДАННЫХ ====================

class DataProcessor:
    """Класс для обработки и обогащения данных"""
    
    def __init__(self, config):
        self.config = config
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def create_features(self, df):
        """Создание фичей для модели"""
        
        # Базовые фичи
        df = df.copy()
        df['price_norm'] = (df['price'] - df['price'].mean()) / df['price'].std()
        df['log_reviews'] = np.log1p(df['reviews'])
        df['log_revenue'] = np.log1p(df['revenue'])
        
        # Интерактивные фичи
        df['price_to_weight'] = df['price'] / (df['weight'] + 1e-5)
        df['reviews_per_revenue'] = df['reviews'] / (df['revenue'] + 1e-5)
        
        # Статистические фичи (скользящие окна)
        if len(df) > 100:
            df['price_ma_7'] = df['price'].rolling(7, min_periods=1).mean()
            df['price_std_7'] = df['price'].rolling(7, min_periods=1).std()
        
        # Целевая переменная: нормализованная эффективность
        df['target'] = (df['revenue'] / (df['reviews'] + 10))  # +10 для устойчивости
        df['log_target'] = np.log1p(df['target'])
        
        return df
    
    def prepare_tensors(self, df, device='cpu'):
        """Подготовка тензоров для обучения"""
        
        features = ['price_norm', 'log_reviews', 'price_to_weight']
        features = [f for f in features if f in df.columns]
        
        X = df[features].values
        y = df['log_target'].values.reshape(-1, 1)
        
        # Нормализация
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Конвертация в тензоры
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64, device=device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float64, device=device)
        
        return X_tensor, y_tensor, features
    
    def validate_data(self, df):
        """Валидация входных данных"""
        required_columns = ['price', 'purchase_price', 'weight', 'reviews', 'revenue']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
        
        # Проверка на аномалии
        stats = {
            'n_rows': len(df),
            'price_range': (df['price'].min(), df['price'].max()),
            'reviews_range': (df['reviews'].min(), df['reviews'].max()),
            'missing_values': df.isnull().sum().sum(),
            'zero_revenue': (df['revenue'] <= 0).sum()
        }
        
        return stats


# ==================== МОДЕЛЬ ЭКОНОМИКИ ====================

class EconomicModel:
    """Модель экономических расчетов"""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_detailed_margin(self, price, purchase_price, weight, 
                                 category_fee=None, region_tax=None):
        """Детальный расчет маржинальности"""
        
        # Динамические параметры
        fee = category_fee if category_fee else self.config.base_fee
        tax = region_tax if region_tax else self.config.base_tax
        logistics = self.config.base_logistics * weight
        
        # Расчет выручки и затрат
        revenue = price * (1 - fee - tax)
        costs = purchase_price + logistics
        
        # Расчет показателей
        profit = revenue - costs
        margin = (profit / revenue) * 100 if revenue > 0 else 0
        roi = (profit / purchase_price) * 100 if purchase_price > 0 else 0
        
        return {
            'price': float(price),
            'revenue': float(revenue),
            'costs': float(costs),
            'profit': float(profit),
            'margin_%': float(margin),
            'roi_%': float(roi),
            'fee_%': fee * 100,
            'tax_%': tax * 100,
            'logistics_cost': float(logistics)
        }
    
    def calculate_batch_roi(self, prices, purchase_prices, weights):
        """Векторизованный расчет ROI для батча"""
        fees = torch.full_like(prices, self.config.base_fee)
        taxes = torch.full_like(prices, self.config.base_tax)
        logistics = weights * self.config.base_logistics
        
        revenue = prices * (1 - fees - taxes)
        costs = purchase_prices + logistics
        profits = revenue - costs
        
        roi = (profits / purchase_prices) * 100
        roi = torch.nan_to_num(roi, nan=0.0, posinf=0.0, neginf=0.0)
        
        return roi
    
    def calculate_risk_metrics(self, returns_series):
        """Расчет метрик риска"""
        if len(returns_series) < 2:
            return {}
        
        returns = np.array(returns_series)
        
        metrics = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'sharpe_ratio': float((np.mean(returns) - self.config.risk_free_rate) / np.std(returns) 
                                  if np.std(returns) > 0 else 0),
            'var_95': float(np.percentile(returns, 5)),  # Value at Risk 95%
            'max_drawdown': float(self._calculate_max_drawdown(returns)),
            'skewness': float(self._calculate_skewness(returns)),
            'kurtosis': float(self._calculate_kurtosis(returns))
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns):
        """Расчет максимальной просадки"""
        if len(returns) == 0:
            return 0
        cumulative = np.cumprod(1 + returns/100)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) * 100 if len(drawdown) > 0 else 0
    
    def _calculate_skewness(self, returns):
        """Асимметрия распределения"""
        try:
            from scipy.stats import skew
            return skew(returns) if len(returns) > 2 else 0
        except ImportError:
            return 0
    
    def _calculate_kurtosis(self, returns):
        """Эксцесс распределения"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(returns) if len(returns) > 3 else 0
        except ImportError:
            return 0


# ==================== МОДЕЛЬ МАШИННОГО ОБУЧЕНИЯ ====================

class MarketPredictor:
    """Модель прогнозирования рыночного потенциала"""
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        self.dtype = torch.float64
        self.models = []
        self.train_losses = []
        
    def train_gp_model(self, X_train, y_train, X_val=None, y_val=None, 
                      kernel_type='rbf', optimize_hyperparams=True):
        """Обучение гауссовского процесса"""
        
        # Выбор ядра
        if kernel_type == 'rbf':
            base_kernel = RBFKernel()
        elif kernel_type == 'matern':
            base_kernel = MaternKernel(nu=2.5)
        else:
            base_kernel = RBFKernel()
        
        kernel = ScaleKernel(base_kernel)
        
        # Создание модели
        model = SingleTaskGP(
            train_X=X_train,
            train_Y=y_train,
            covar_module=kernel
        ).to(self.device)
        
        # Обучение
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Оптимизация гиперпараметров
        if optimize_hyperparams:
            self._optimize_hyperparameters(model, mll, X_val, y_val)
        else:
            fit_gpytorch_mll(mll)
        
        self.models.append(model)
        return model
    
    def _optimize_hyperparameters(self, model, mll, X_val=None, y_val=None):
        """Оптимизация гиперпараметров с помощью Optuna"""
        
        def objective(trial):
            # Предлагаем гиперпараметры
            lengthscale = trial.suggest_float('lengthscale', 0.1, 10.0)
            noise = trial.suggest_float('noise', 1e-4, 0.1)
            
            # Устанавливаем гиперпараметры
            model.covar_module.base_kernel.lengthscale = lengthscale
            model.likelihood.noise = noise
            
            # Обучаем
            fit_gpytorch_mll(mll)
            
            # Оцениваем
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    posterior = model.posterior(X_val)
                    pred_mean = posterior.mean
                    loss = torch.mean((pred_mean - y_val) ** 2)
                return loss.item()
            else:
                return mll(model(model.train_inputs[0]), model.train_targets).item()
        
        # Оптимизация
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=5, show_progress_bar=False)  # Уменьшено до 5 испытаний для скорости
            
            # Применяем лучшие параметры
            best_params = study.best_params
            model.covar_module.base_kernel.lengthscale = best_params['lengthscale']
            model.likelihood.noise = best_params['noise']
        except:
            pass  # Если оптимизация не удалась, используем параметры по умолчанию
        
        # Финальное обучение
        fit_gpytorch_mll(mll)
    
    def predict(self, X_test, model_index=-1):
        """Прогнозирование с помощью обученной модели"""
        model = self.models[model_index] if self.models else None
        if model is None:
            raise ValueError("Модель не обучена")
        
        with torch.no_grad():
            posterior = model.posterior(X_test)
            mean = posterior.mean
            variance = posterior.variance
            lower, upper = posterior.mvn.confidence_region()
        
        return {
            'mean': mean.cpu().numpy(),
            'variance': variance.cpu().numpy(),
            'lower': lower.cpu().numpy(),
            'upper': upper.cpu().numpy(),
            'uncertainty': torch.sqrt(variance).cpu().numpy()
        }
    
    def cross_validate(self, X, y, n_splits=3):  # Уменьшено до 3 фолдов для скорости
        """Кросс-валидация временных рядов"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"  Обучение фолда {fold+1}/{n_splits}...")
            model = self.train_gp_model(
                X_train, y_train, X_val, y_val,
                optimize_hyperparams=True
            )
            
            # Прогноз на валидации
            predictions = self.predict(X_val)
            mse = np.mean((predictions['mean'] - y_val.cpu().numpy()) ** 2)
            scores.append(mse)
            
            print(f"  Fold {fold+1}: MSE = {mse:.4f}")
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'all_scores': scores
        }


# ==================== ВИЗУАЛИЗАЦИЯ И АНАЛИЗ ====================

class MarketVisualizer:
    """Класс для визуализации результатов анализа"""
    
    def __init__(self, config):
        self.config = config
        plt.style.use('dark_background')
        self.colors = {
            'primary': '#00FFFF',
            'secondary': '#FF00FF',
            'success': '#00FF00',
            'warning': '#FFFF00',
            'danger': '#FF0000',
            'background': '#0A0A0A'
        }
    
    def create_dashboard(self, analysis_results, save_path=None):
        """Создание интерактивного дашборда"""
        try:
            # Основные графики
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Карта потенциала рынка',
                    'Распределение ROI по ценам',
                    'Топ точек входа',
                    'Анализ риска'
                ),
                specs=[
                    [{"type": "heatmap"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. Heatmap потенциала
            if 'heatmap_data' in analysis_results:
                hm_data = analysis_results['heatmap_data']
                fig.add_trace(
                    go.Heatmap(
                        z=hm_data['scores'],
                        x=hm_data['prices'],
                        y=hm_data['reviews'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    row=1, col=1
                )
            
            # 2. График ROI
            if 'roi_curve' in analysis_results:
                roi_data = analysis_results['roi_curve']
                fig.add_trace(
                    go.Scatter(
                        x=roi_data['prices'],
                        y=roi_data['roi_values'],
                        mode='lines',
                        name='ROI',
                        line=dict(color=self.colors['success'], width=2)
                    ),
                    row=1, col=2
                )
                
                # Линия порога
                fig.add_hline(
                    y=self.config.target_roi * 100,
                    line_dash="dash",
                    line_color=self.colors['warning'],
                    row=1, col=2
                )
            
            # 3. Топ точки входа
            if 'top_entries' in analysis_results and len(analysis_results['top_entries']['prices']) > 0:
                top_data = analysis_results['top_entries']
                fig.add_trace(
                    go.Bar(
                        x=top_data['prices'],
                        y=top_data['ratings'],
                        text=top_data['roi_values'],
                        textposition='auto',
                        marker_color=self.colors['primary'],
                        name='Рейтинг'
                    ),
                    row=2, col=1
                )
            
            # 4. Распределение риска
            if 'risk_distribution' in analysis_results:
                risk_data = analysis_results['risk_distribution']
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(risk_data['values'])),
                        y=risk_data['values'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=risk_data['values'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Уровень риска'
                    ),
                    row=2, col=2
                )
            
            # Обновление layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text="Панель управления рыночным анализом",
                showlegend=True,
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color='white')
            )
            
            if save_path:
                try:
                    fig.write_html(save_path)
                except:
                    print(f"Не удалось сохранить дашборд в {save_path}")
            
            return fig
        except Exception as e:
            print(f"Ошибка при создании дашборда: {e}")
            return None
    
    def create_simple_report(self, analysis_results, save_path='market_report.png'):
        """Создание простого статического отчета"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # 1. Heatmap
            if 'heatmap_data' in analysis_results:
                hm_data = analysis_results['heatmap_data']
                im = axes[0].imshow(hm_data['scores'], cmap='viridis', aspect='auto',
                                extent=[hm_data['prices'][0], hm_data['prices'][-1],
                                        hm_data['reviews'][0], hm_data['reviews'][-1]])
                axes[0].set_title('Карта рыночного потенциала')
                axes[0].set_xlabel('Цена (руб)')
                axes[0].set_ylabel('Отзывы')
                plt.colorbar(im, ax=axes[0])
            
            # 2. ROI кривая
            if 'roi_curve' in analysis_results:
                roi_data = analysis_results['roi_curve']
                axes[1].plot(roi_data['prices'], roi_data['roi_values'], 
                            color='lime', linewidth=2)
                axes[1].axhline(y=self.config.target_roi * 100, color='red', 
                            linestyle='--', alpha=0.5)
                axes[1].fill_between(roi_data['prices'], 0, roi_data['roi_values'],
                                    where=(roi_data['roi_values'] > self.config.target_roi * 100),
                                    color='green', alpha=0.3)
                axes[1].set_title('ROI по ценам')
                axes[1].set_xlabel('Цена (руб)')
                axes[1].set_ylabel('ROI %')
            
            # 3. Топ точки входа (если есть)
            if 'top_entries' in analysis_results and len(analysis_results['top_entries']['prices']) > 0:
                top_data = analysis_results['top_entries']
                bars = axes[2].bar(range(len(top_data['prices'])), top_data['ratings'],
                                color='cyan', alpha=0.7)
                axes[2].set_title('Топ точек входа')
                axes[2].set_xlabel('Точки входа')
                axes[2].set_ylabel('Рейтинг')
                axes[2].set_xticks(range(len(top_data['prices'])))
                
                # Создаем подписи для осей X
                x_labels = [f"{p:.0f}" for p in top_data['prices']]
                axes[2].set_xticklabels(x_labels, rotation=45, ha='right')
                
                # Добавляем значения ROI на бары
                for i, (bar, roi) in enumerate(zip(bars, top_data['roi_values'])):
                    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f'{roi:.1f}%', ha='center', va='bottom', fontsize=8)
            else:
                axes[2].text(0.5, 0.5, 'Нет прибыльных точек входа', 
                            ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('Топ точек входа')
            
            # 4. Распределение риска
            if 'risk_distribution' in analysis_results:
                risk_data = analysis_results['risk_distribution']
                axes[3].hist(risk_data['values'], bins=20, color='magenta', 
                            alpha=0.7, edgecolor='white')
                if 'var_95' in risk_data:
                    axes[3].axvline(x=risk_data['var_95'], color='red', 
                                linestyle='--', label='VaR 95%')
                axes[3].set_title('Распределение риска')
                axes[3].set_xlabel('Уровень риска')
                axes[3].set_ylabel('Частота')
                axes[3].legend()
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, facecolor=self.colors['background'])
            plt.show()
            
            return fig
        except Exception as e:
            print(f"Ошибка при создании отчета: {e}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Ошибка визуализации: {str(e)}', 
                    ha='center', va='center', fontsize=12)
            plt.show()
            return None


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

class MarketAnalysisPipeline:
    """Основной пайплайн анализа рынка"""
    
    def __init__(self, config=None):
        self.config = config or MarketConfig()
        self.data_processor = DataProcessor(self.config)
        self.economic_model = EconomicModel(self.config)
        self.predictor = MarketPredictor(self.config)
        self.visualizer = MarketVisualizer(self.config)
        
        # Логирование
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(f"MarketAnalysis_{self.experiment_id}")
                mlflow.start_run(run_name=f"run_{self.experiment_id}")
                mlflow.log_params(self.config.to_dict())
            except:
                print("Не удалось инициализировать MLflow")
    
    def run_analysis(self, data):
        """Запуск полного анализа"""
        
        print("=" * 80)
        print("ЗАПУСК РЫНОЧНОГО АНАЛИЗА")
        print("=" * 80)
        
        try:
            # 1. Валидация данных
            print("\n1. ВАЛИДАЦИЯ ДАННЫХ...")
            stats = self.data_processor.validate_data(data)
            print(f"   • Записей: {stats['n_rows']}")
            print(f"   • Диапазон цен: {stats['price_range'][0]:.0f} - {stats['price_range'][1]:.0f} руб")
            print(f"   • Пропуски: {stats['missing_values']}")
            
            # 2. Подготовка данных
            print("\n2. ПОДГОТОВКА ДАННЫХ...")
            enriched_data = self.data_processor.create_features(data)
            X, y, feature_names = self.data_processor.prepare_tensors(
                enriched_data, device=self.predictor.device
            )
            print(f"   • Фичи: {feature_names}")
            print(f"   • Размерность: {X.shape}")
            
            # 3. Обучение модели
            print("\n3. ОБУЧЕНИЕ МОДЕЛИ...")
            cv_results = self.predictor.cross_validate(X, y, n_splits=min(3, self.config.n_folds))
            print(f"   • Средняя MSE: {cv_results['mean_score']:.4f}")
            print(f"   • Стандартное отклонение: {cv_results['std_score']:.4f}")
            
            # 4. Прогнозирование
            print("\n4. ПРОГНОЗИРОВАНИЕ...")
            
            # Создание тестовой сетки
            price_grid = torch.linspace(
                self.config.min_price, 
                self.config.max_price, 
                self.config.heatmap_resolution,
                device=self.predictor.device,
                dtype=self.predictor.dtype
            )
            
            review_grid = torch.linspace(
                self.config.min_reviews,
                self.config.max_reviews,
                self.config.heatmap_resolution,
                device=self.predictor.device,
                dtype=self.predictor.dtype
            )
            
            # Прогноз для heatmap
            grid_p, grid_r = torch.meshgrid(price_grid, review_grid, indexing='ij')
            
            # Подготовка фичей для предсказания
            price_mean = data['price'].mean()
            price_std = data['price'].std()
            weight_mean = enriched_data['weight'].mean()
            
            # Создаем тензор с фичами для каждого сочетания цены и отзывов
            price_flat = grid_p.flatten()
            review_flat = grid_r.flatten()
            
            # Создаем фичи в правильном порядке
            feature1 = (price_flat - price_mean) / price_std
            feature2 = torch.log1p(review_flat)
            feature3 = price_flat / weight_mean
            
            flat_x = torch.stack([feature1, feature2, feature3], dim=-1)
            
            predictions = self.predictor.predict(flat_x)
            
            # 5. Экономический анализ
            print("\n5. ЭКОНОМИЧЕСКИЙ АНАЛИЗ...")
            
            # ROI для разных цен
            avg_purchase = torch.tensor(data['purchase_price'].mean(), 
                                    device=self.predictor.device)
            avg_weight = torch.tensor(data['weight'].mean(),
                                    device=self.predictor.device)
            
            roi_values = self.economic_model.calculate_batch_roi(
                price_grid,
                torch.full_like(price_grid, avg_purchase),
                torch.full_like(price_grid, avg_weight)
            )
            
            # 6. Генерация рекомендаций
            print("\n6. ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ...")
            
            # Для новичка (10 отзывов)
            newbie_reviews = torch.full_like(price_grid, 10.0)
            
            # Подготовка фичей для новичка
            feature1_newbie = (price_grid - price_mean) / price_std
            feature2_newbie = torch.log1p(newbie_reviews)
            feature3_newbie = price_grid / weight_mean
            
            newbie_features = torch.stack([feature1_newbie, feature2_newbie, feature3_newbie], dim=-1)
            
            newbie_predictions = self.predictor.predict(newbie_features)
            
            # Расчет рейтинга
            ai_scores = newbie_predictions['mean'].flatten()
            roi_scores = roi_values.cpu().numpy()
            risk_scores = newbie_predictions['uncertainty'].flatten()
            
            # Нормализация
            ai_norm = (ai_scores - ai_scores.min()) / (ai_scores.max() - ai_scores.min() + 1e-8)
            roi_norm = (roi_scores - roi_scores.min()) / (roi_scores.max() - roi_scores.min() + 1e-8)
            risk_norm = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)
            
            # Комбинированный рейтинг
            ratings = (
                self.config.ai_weight * ai_norm +
                self.config.roi_weight * roi_norm -
                self.config.risk_weight * risk_norm
            )
            
            # Фильтрация и сортировка - ИСПРАВЛЕНИЕ ЗДЕСЬ
            profitable_mask = roi_scores > (self.config.target_roi * 100)
            profitable_indices = np.where(profitable_mask)[0]
            
            if len(profitable_indices) > 0:
                profitable_ratings = ratings[profitable_indices]
                
                # Получаем топ-N индексов (максимум 10)
                n_top = min(10, len(profitable_indices))
                top_indices_in_profitable = np.argsort(profitable_ratings)[-n_top:][::-1]
                top_indices = profitable_indices[top_indices_in_profitable]
                
                # Преобразуем в списки для безопасного доступа
                price_list = price_grid.cpu().numpy()
                roi_list = roi_scores
                ai_list = ai_scores
                risk_list = risk_scores
                ratings_list = ratings
                
                top_entries = pd.DataFrame({
                    'Цена_входа': [price_list[i] for i in top_indices],
                    'ROI_%': [roi_list[i] for i in top_indices],
                    'AI_Score': [ai_list[i] for i in top_indices],
                    'Риск': [risk_list[i] for i in top_indices],
                    'Рейтинг': [ratings_list[i] for i in top_indices]
                })
            else:
                top_entries = pd.DataFrame()
            
            # 7. Подготовка результатов для визуализации
            print("\n7. ПОДГОТОВКА ОТЧЕТА...")
            
            analysis_results = {
                'heatmap_data': {
                    'scores': predictions['mean'].reshape(
                        self.config.heatmap_resolution, 
                        self.config.heatmap_resolution
                    ).T,
                    'prices': price_grid.cpu().numpy(),
                    'reviews': review_grid.cpu().numpy()
                },
                'roi_curve': {
                    'prices': price_grid.cpu().numpy(),
                    'roi_values': roi_scores
                }
            }
            
            # Добавляем топ записи, если они есть
            if not top_entries.empty:
                analysis_results['top_entries'] = {
                    'prices': top_entries['Цена_входа'].values,
                    'ratings': top_entries['Рейтинг'].values,
                    'roi_values': top_entries['ROI_%'].values
                }
            
            analysis_results['risk_distribution'] = {
                'values': risk_scores,
                'var_95': np.percentile(risk_scores, 5) if len(risk_scores) > 0 else 0
            }
            
            analysis_results['metrics'] = {
                'avg_roi': float(np.mean(roi_scores[profitable_mask]) if profitable_mask.any() else 0),
                'max_roi': float(np.max(roi_scores) if len(roi_scores) > 0 else 0),
                'avg_risk': float(np.mean(risk_scores) if len(risk_scores) > 0 else 0),
                'success_rate': float(np.mean(profitable_mask) * 100 if len(profitable_mask) > 0 else 0)
            }
            
            # 8. Визуализация
            print("\n8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
            
            # Статический отчет
            self.visualizer.create_simple_report(
                analysis_results, 
                save_path=f'market_report_{self.experiment_id}.png'
            )
            
            # Интерактивный дашборд
            dashboard = self.visualizer.create_dashboard(
                analysis_results,
                save_path=f'dashboard_{self.experiment_id}.html'
            )
            
            # 9. Логирование результатов
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics(analysis_results['metrics'])
                    mlflow.log_artifact(f'market_report_{self.experiment_id}.png')
                    mlflow.log_artifact(f'dashboard_{self.experiment_id}.html')
                    mlflow.end_run()
                except:
                    print("Не удалось сохранить результаты в MLflow")
            
            print("\n" + "=" * 80)
            print("АНАЛИЗ ЗАВЕРШЕН")
            print("=" * 80)
            
            return {
                'config': self.config,
                'top_entries': top_entries,
                'analysis_results': analysis_results,
                'predictor': self.predictor,
                'dashboard': dashboard
            }
            
        except Exception as e:
            print(f"\n❌ ОШИБКА ВО ВРЕМЯ АНАЛИЗА: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== ТЕСТОВЫЙ ЗАПУСК ====================

def generate_sample_data(n_samples=500):  # Уменьшено для скорости
    """Генерация реалистичных тестовых данных"""
    
    np.random.seed(42)
    
    # Базовые параметры
    base_prices = np.random.uniform(800, 12000, n_samples)
    
    # Эмулируем категории товаров
    categories = np.random.choice(['Электроника', 'Одежда', 'Косметика', 
                                'Книги', 'Спорт'], n_samples)
    
    # Закупочные цены зависят от категории
    purchase_prices = np.where(
        categories == 'Электроника', base_prices * 0.6,
        np.where(categories == 'Одежда', base_prices * 0.4,
                base_prices * 0.3)
    ) + np.random.normal(0, 100, n_samples)
    
    # Вес товаров
    weights = np.where(
        categories == 'Электроника', np.random.uniform(0.5, 3.0, n_samples),
        np.where(categories == 'Книги', np.random.uniform(0.3, 1.0, n_samples),
                np.random.uniform(0.2, 2.0, n_samples))
    )
    
    # Отзывы зависят от цены и категории
    reviews_base = np.random.lognormal(3, 1, n_samples)
    reviews = reviews_base * (1 + 0.0001 * base_prices)  # Дороже = больше отзывов
    
    # Выручка зависит от цены и отзывов
    revenue_base = base_prices * np.random.uniform(10, 100, n_samples)
    revenue = revenue_base * (1 + 0.5 * np.log1p(reviews) / 10)
    
    # Добавляем сезонность
    months = np.random.randint(1, 13, n_samples)
    revenue *= 1 + 0.2 * np.sin(2 * np.pi * (months - 1) / 12)
    
    # Собираем DataFrame
    df = pd.DataFrame({
        'price': np.maximum(base_prices, 100),
        'purchase_price': np.maximum(purchase_prices, 50),
        'weight': np.maximum(weights, 0.1),
        'reviews': np.maximum(reviews, 1),
        'revenue': np.maximum(revenue, 1000),
        'category': categories,
        'month': months
    })
    
    return df


if __name__ == "__main__":
    
    # Генерация тестовых данных
    print("Генерация тестовых данных...")
    sample_data = generate_sample_data(500)  # Уменьшил объем данных для быстрого теста
    
    # Создание и настройка конфигурации
    config = MarketConfig()
    config.target_roi = 0.15  # Снизил порог ROI для большего количества результатов
    config.ai_weight = 0.5
    config.roi_weight = 0.4
    config.risk_weight = 0.1
    config.heatmap_resolution = 30  # Уменьшил разрешение для быстрой визуализации
    
    # Запуск пайплайна
    pipeline = MarketAnalysisPipeline(config)
    results = pipeline.run_analysis(sample_data)
    
    # Вывод результатов
    if results:
        print("\n" + "█" * 40 + " РЕЗУЛЬТАТЫ " + "█" * 40)
        
        if not results['top_entries'].empty:
            print("\nТОП РЕКОМЕНДУЕМЫХ ТОЧЕК ВХОДА:")
            print("-" * 80)
            
            formatted_df = results['top_entries'].copy()
            formatted_df['ROI_%'] = formatted_df['ROI_%'].apply(lambda x: f"{x:.1f}%")
            formatted_df['Рейтинг'] = formatted_df['Рейтинг'].apply(lambda x: f"{x:.3f}")
            formatted_df['Риск'] = formatted_df['Риск'].apply(lambda x: f"{x:.3f}")
            formatted_df = formatted_df.rename(columns={
                'Цена_входа': 'Цена',
                'AI_Score': 'AI Потенциал'
            })
            
            print(formatted_df.to_string(index=False))
            
            # Дополнительная статистика
            print("\nСТАТИСТИКА АНАЛИЗА:")
            print("-" * 80)
            metrics = results['analysis_results']['metrics']
            print(f"• Средний ROI прибыльных точек: {metrics['avg_roi']:.1f}%")
            print(f"• Максимальный ROI: {metrics['max_roi']:.1f}%")
            print(f"• Средний уровень риска: {metrics['avg_risk']:.3f}")
            print(f"• Процент прибыльных точек: {metrics['success_rate']:.1f}%")
            
            # Рекомендации
            print("\nРЕКОМЕНДАЦИИ:")
            print("-" * 80)
            best_entry = results['top_entries'].iloc[0]
            print(f"1. Оптимальная цена входа: {best_entry['Цена_входа']:.0f} руб")
            print(f"2. Ожидаемый ROI: {best_entry['ROI_%']:.1f}%")
            print(f"3. Уровень риска: {best_entry['Риск']:.3f} (ниже = лучше)")
            
            if best_entry['Риск'] > 0.3:
                print("4. ⚠️  Высокий риск! Рассмотрите страховку позиции")
            elif best_entry['Риск'] < 0.1:
                print("4. ✅ Низкий риск, можно увеличить объем инвестиций")
            else:
                print("4. ⚖️  Умеренный риск, стандартная стратегия")
                
        else:
            print("\n⚠️  НЕ НАЙДЕНО ПРИБЫЛЬНЫХ ТОЧЕК ВХОДА")
            print("Рекомендации:")
            print("1. Рассмотрите другие ниши")
            print("2. Увеличьте маржинальность")
            print("3. Оптимизируйте логистику")
            print("4. Пересмотрите ценовую политику")
        
        print("\n" + "█" * 92)
        print(f"Отчет сохранен в:")
        print(f"• Статический отчет: market_report_{pipeline.experiment_id}.png")
        print(f"• Интерактивный дашборд: dashboard_{pipeline.experiment_id}.html")
        print(f"• Конфигурация: market_config_{pipeline.experiment_id}.json")
        
        # Сохранение конфигурации
        config.save(f'market_config_{pipeline.experiment_id}.json')
        
        print("\nДля просмотра интерактивного дашборда откройте файл dashboard.html в браузере")
    else:
        print("\n❌ Анализ завершился с ошибкой")
    
    print("=" * 80)