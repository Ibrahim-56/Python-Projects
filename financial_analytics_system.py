# financial_analytics_system/
"""
نظام متكامل لتحليل الأداء المالي والاستثماري
يشمل:
- تحليل المحافظ الاستثمارية
- حساب المؤشرات المالية
- تقييم المخاطر
- التنبؤ بأسعار الأسهم
- تحليل القوائم المالية
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

class FinancialAnalyzer:
    """
    محلل مالي متقدم لتحليل الأسواق والاستثمارات
    """
    
    def __init__(self):
        self.portfolio = {}
        self.stock_data = {}
        self.risk_free_rate = 0.03  # 3% سنوياً
        self.setup_logging()
        
    def setup_logging(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('financial_analytics.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ============= Data Acquisition Module =============
    def download_stock_data(self, symbols: list, start_date: str, end_date: str = None) -> dict:
        """تحميل بيانات الأسهم من Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # إضافة المؤشرات الفنية
                    hist['Returns'] = hist['Close'].pct_change()
                    hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
                    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                    hist['Volume_SMA'] = hist['Volume'].rolling(window=20).mean()
                    
                    # مؤشر القوة النسبية
                    hist['RSI'] = self.calculate_rsi(hist['Close'])
                    
                    # Bollinger Bands
                    hist['BB_upper'], hist['BB_lower'] = self.calculate_bollinger_bands(hist['Close'])
                    
                    data[symbol] = hist
                    self.logger.info(f"تم تحميل بيانات {symbol}: {len(hist)} يوم")
                
            except Exception as e:
                self.logger.error(f"خطأ في تحميل {symbol}: {e}")
        
        self.stock_data = data
        return data
    
    def calculate_rsi(self, prices, period=14):
        """حساب مؤشر القوة النسبية"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """حساب Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    # ============= Portfolio Management Module =============
    def create_portfolio(self, portfolio_dict: dict):
        """إنشاء محفظة استثمارية"""
        self.portfolio = portfolio_dict
        self.logger.info(f"تم إنشاء محفظة بـ {len(portfolio_dict)} أصل")
    
    def calculate_portfolio_returns(self) -> pd.DataFrame:
        """حساب عوائد المحفظة"""
        if not self.portfolio or not self.stock_data:
            raise ValueError("لا توجد بيانات محفظة أو أسهم")
        
        # تجميع عوائد جميع الأسهم
        returns_df = pd.DataFrame()
        for symbol, weight in self.portfolio.items():
            if symbol in self.stock_data:
                returns_df[symbol] = self.stock_data[symbol]['Returns']
        
        # حساب عوائد المحفظة
        portfolio_returns = (returns_df * list(self.portfolio.values())).sum(axis=1)
        
        return portfolio_returns
    
    def calculate_portfolio_metrics(self) -> dict:
        """حساب مقاييس أداء المحفظة"""
        returns = self.calculate_portfolio_returns()
        
        # المقاييس الأساسية
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': (1 + returns).prod() ** (252/len(returns)) - 1,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        return metrics
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """حساب نسبة شارب"""
        excess_returns = returns - self.risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """حساب نسبة سورتينو"""
        excess_returns = returns - self.risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """حساب أقصى انخفاض"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """حساب القيمة المعرضة للخطر"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """حساب القيمة المشروطة المعرضة للخطر"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    # ============= Modern Portfolio Theory =============
    def optimize_portfolio(self, symbols: list, returns_data: pd.DataFrame = None) -> dict:
        """تحسين المحفظة باستخدام نظرية المحفظة الحديثة"""
        if returns_data is None:
            returns_data = pd.DataFrame()
            for symbol in symbols:
                if symbol in self.stock_data:
                    returns_data[symbol] = self.stock_data[symbol]['Returns']
        
        # حساب المتوسطات والتباين
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        num_assets = len(symbols)
        args = (mean_returns, cov_matrix, self.risk_free_rate)
        
        # تحسين النسبة الأفضل (Sharpe Ratio)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        
        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe
        
        optimized = minimize(
            neg_sharpe,
            initial_guess,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if optimized['success']:
            weights = optimized['x']
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            
            return {
                'weights': dict(zip(symbols, weights)),
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_std,
                'sharpe_ratio': sharpe,
                'success': True
            }
        
        return {'success': False}
    
    def calculate_efficient_frontier(self, symbols: list, points: int = 50) -> dict:
        """حساب الحدود الفعالة (Efficient Frontier)"""
        returns_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in self.stock_data:
                returns_data[symbol] = self.stock_data[symbol]['Returns']
        
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        results = {
            'returns': [],
            'volatility': [],
            'weights': []
        }
        
        for _ in range(points):
            weights = np.random.random(len(symbols))
            weights = weights / np.sum(weights)
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results['returns'].append(portfolio_return)
            results['volatility'].append(portfolio_std)
            results['weights'].append(weights)
        
        return results
    
    # ============= Risk Analysis Module =============
    def calculate_beta(self, symbol: str, market_symbol: str = '^GSPC') -> float:
        """حساب معامل بيتا"""
        if symbol not in self.stock_data or market_symbol not in self.stock_data:
            return None
        
        stock_returns = self.stock_data[symbol]['Returns'].dropna()
        market_returns = self.stock_data[market_symbol]['Returns'].dropna()
        
        # محاذاة التواريخ
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        covariance = np.cov(stock_returns, market_returns)[0][1]
        variance = np.var(market_returns)
        
        return covariance / variance
    
    def calculate_correlation_matrix(self, symbols: list = None) -> pd.DataFrame:
        """حساب مصفوفة الارتباط بين الأسهم"""
        if symbols is None:
            symbols = list(self.stock_data.keys())
        
        returns_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in self.stock_data:
                returns_data[symbol] = self.stock_data[symbol]['Returns']
        
        return returns_data.corr()
    
    def calculate_var_covar(self, symbols: list = None) -> dict:
        """حساب مصفوفة التباين-التغاير"""
        if symbols is None:
            symbols = list(self.stock_data.keys())
        
        returns_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in self.stock_data:
                returns_data[symbol] = self.stock_data[symbol]['Returns']
        
        return {
            'covariance_matrix': returns_data.cov().to_dict(),
            'variance': returns_data.var().to_dict(),
            'correlation_matrix': returns_data.corr().to_dict()
        }
    
    # ============= Technical Indicators Module =============
    def calculate_technical_indicators(self, symbol: str) -> dict:
        """حساب المؤشرات الفنية لسهم معين"""
        if symbol not in self.stock_data:
            return {}
        
        df = self.stock_data[symbol]
        
        indicators = {
            'moving_averages': {
                'SMA_20': df['SMA_20'].iloc[-1],
                'SMA_50': df['SMA_50'].iloc[-1],
                'EMA_12': df['Close'].ewm(span=12).mean().iloc[-1],
                'EMA_26': df['Close'].ewm(span=26).mean().iloc[-1]
            },
            'momentum': {
                'RSI': df['RSI'].iloc[-1],
                'MACD': self.calculate_macd(df['Close']),
                'ROC': ((df['Close'].iloc[-1] / df['Close'].iloc[-10]) - 1) * 100
            },
            'volatility': {
                'BB_upper': df['BB_upper'].iloc[-1],
                'BB_lower': df['BB_lower'].iloc[-1],
                'ATR': self.calculate_atr(df),
                'current_price': df['Close'].iloc[-1]
            },
            'volume': {
                'current_volume': df['Volume'].iloc[-1],
                'avg_volume': df['Volume_SMA'].iloc[-1],
                'volume_ratio': df['Volume'].iloc[-1] / df['Volume_SMA'].iloc[-1]
            }
        }
        
        return indicators
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """حساب MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def calculate_atr(self, df, period=14):
        """حساب Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    # ============= Prediction Module =============
    def predict_stock_prices(self, symbol: str, days_ahead: int = 30) -> dict:
        """التنبؤ بأسعار الأسهم باستخدام Random Forest"""
        if symbol not in self.stock_data:
            return {}
        
        df = self.stock_data[symbol].copy()
        
        # إضافة ميزات للتنبؤ
        df['Day'] = df.index.day
        df['Month'] = df.index.month
        df['DayOfWeek'] = df.index.dayofweek
        
        for lag in [1, 2, 3, 5, 10]:
            df[f'Lag_{lag}'] = df['Close'].shift(lag)
        
        df = df.dropna()
        
        # تحضير البيانات
        features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'DayOfWeek'] + [f'Lag_{lag}' for lag in [1, 2, 3, 5, 10]]
        X = df[features]
        y = df['Close']
        
        # تقسيم البيانات
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # تطبيع البيانات
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب النموذج
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # تقييم النموذج
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # التنبؤ بالأيام القادمة
        last_data = X.iloc[-1:].copy()
        predictions = []
        
        for _ in range(days_ahead):
            last_scaled = scaler.transform(last_data)
            pred = model.predict(last_scaled)[0]
            predictions.append(pred)
            
            # تحديث البيانات للتنبؤ التالي
            last_data = last_data.copy()
            for lag in [10, 5, 3, 2, 1]:
                if lag > 1:
                    last_data[f'Lag_{lag}'] = last_data[f'Lag_{lag-1}'].values
            last_data['Lag_1'] = pred
            last_data['Open'] = pred
            last_data['High'] = pred * 1.02
            last_data['Low'] = pred * 0.98
        
        return {
            'predictions': predictions,
            'last_actual_price': y.iloc[-1],
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
    
    # ============= Financial Ratios Module =============
    def calculate_financial_ratios(self, symbol: str) -> dict:
        """حساب النسب المالية لسهم معين"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            ratios = {
                'valuation': {
                    'PE_ratio': info.get('trailingPE', None),
                    'forward_PE': info.get('forwardPE', None),
                    'PB_ratio': info.get('priceToBook', None),
                    'PS_ratio': info.get('priceToSalesTrailing12Months', None),
                    'PEG_ratio': info.get('pegRatio', None)
                },
                'profitability': {
                    'ROE': info.get('returnOnEquity', None),
                    'ROA': info.get('returnOnAssets', None),
                    'profit_margin': info.get('profitMargins', None),
                    'operating_margin': info.get('operatingMargins', None)
                },
                'liquidity': {
                    'current_ratio': info.get('currentRatio', None),
                    'quick_ratio': info.get('quickRatio', None),
                    'debt_to_equity': info.get('debtToEquity', None)
                },
                'dividends': {
                    'dividend_yield': info.get('dividendYield', None),
                    'payout_ratio': info.get('payoutRatio', None),
                    'dividend_rate': info.get('dividendRate', None)
                },
                'growth': {
                    'earnings_growth': info.get('earningsGrowth', None),
                    'revenue_growth': info.get('revenueGrowth', None),
                    'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None)
                }
            }
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"خطأ في حساب النسب المالية لـ {symbol}: {e}")
            return {}
    
    # ============= Monte Carlo Simulation =============
    def monte_carlo_simulation(self, symbol: str, days: int = 252, simulations: int = 1000) -> dict:
        """محاكاة مونت كارلو للأسعار المستقبلية"""
        if symbol not in self.stock_data:
            return {}
        
        df = self.stock_data[symbol]
        returns = df['Returns'].dropna()
        
        last_price = df['Close'].iloc[-1]
        mu = returns.mean()
        sigma = returns.std()
        
        simulation_results = []
        
        for _ in range(simulations):
            prices = [last_price]
            for _ in range(days):
                shock = np.random.normal(mu, sigma)
                prices.append(prices[-1] * np.exp(shock))
            simulation_results.append(prices[1:])
        
        simulation_df = pd.DataFrame(simulation_results).T
        
        results = {
            'final_prices': simulation_df.iloc[-1].tolist(),
            'mean_final_price': simulation_df.iloc[-1].mean(),
            'median_final_price': simulation_df.iloc[-1].median(),
            'std_final_price': simulation_df.iloc[-1].std(),
            'percentile_5': simulation_df.iloc[-1].quantile(0.05),
            'percentile_95': simulation_df.iloc[-1].quantile(0.95),
            'probability_profit': (simulation_df.iloc[-1] > last_price).mean(),
            'expected_return': (simulation_df.iloc[-1].mean() / last_price - 1) * 100,
            'simulation_paths': simulation_df.values.tolist()
        }
        
        return results
    
    # ============= Visualization Module =============
    def plot_stock_prices(self, symbols: list, start_date: str = None, end_date: str = None):
        """رسم أسعار الأسهم"""
        plt.figure(figsize=(15, 8))
        
        for symbol in symbols:
            if symbol in self.stock_data:
                df = self.stock_data[symbol]
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                plt.plot(df.index, df['Close'], label=symbol, linewidth=2)
        
        plt.title('مقارنة أسعار الأسهم', fontsize=16)
        plt.xlabel('التاريخ', fontsize=12)
        plt.ylabel('السعر', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('stock_prices.png')
        plt.close()
    
    def plot_portfolio_allocation(self):
        """رسم توزيع المحفظة"""
        if not self.portfolio:
            return
        
        plt.figure(figsize=(10, 10))
        labels = list(self.portfolio.keys())
        sizes = list(self.portfolio.values())
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('توزيع المحفظة الاستثمارية', fontsize=16)
        plt.axis('equal')
        plt.savefig('portfolio_allocation.png')
        plt.close()
    
    def plot_efficient_frontier(self, symbols: list):
        """رسم الحدود الفعالة"""
        frontier = self.calculate_efficient_frontier(symbols)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(frontier['volatility'], frontier['returns'], c='blue', alpha=0.3)
        
        # إضافة المحفظة المثلى
        optimal = self.optimize_portfolio(symbols)
        if optimal['success']:
            plt.scatter(optimal['expected_volatility'], 
                       optimal['expected_return'], 
                       c='red', s=200, marker='*', label='المحفظة المثلى')
        
        plt.title('الحدود الفعالة (Efficient Frontier)', fontsize=16)
        plt.xlabel('المخاطرة (التقلب السنوي)', fontsize=12)
        plt.ylabel('العائد السنوي المتوقع', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('efficient_frontier.png')
        plt.close()
    
    def plot_correlation_heatmap(self, symbols: list = None):
        """رسم خريطة حرارية للارتباطات"""
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('مصفوفة الارتباط بين الأسهم', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    def plot_monte_carlo(self, symbol: str, simulations: int = 100):
        """رسم محاكاة مونت كارلو"""
        results = self.monte_carlo_simulation(symbol, simulations=simulations)
        
        plt.figure(figsize=(15, 8))
        
        # رسم بعض مسارات المحاكاة
        for i in range(min(50, simulations)):
            plt.plot(results['simulation_paths'][i], alpha=0.1, color='blue')
        
        # رسم المسار المتوسط
        mean_path = np.mean(results['simulation_paths'], axis=0)
        plt.plot(mean_path, color='red', linewidth=2, label='المتوسط')
        
        plt.title(f'محاكاة مونت كارلو لسعر {symbol}', fontsize=16)
        plt.xlabel('اليوم', fontsize=12)
        plt.ylabel('السعر', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'monte_carlo_{symbol}.png')
        plt.close()
    
    # ============= Report Generation Module =============
    def generate_investment_report(self, symbols: list) -> dict:
        """توليد تقرير استثماري شامل"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'market_overview': {},
            'portfolio_analysis': {},
            'risk_metrics': {},
            'recommendations': []
        }
        
        # نظرة عامة على السوق
        for symbol in symbols:
            if symbol in self.stock_data:
                df = self.stock_data[symbol]
                report['market_overview'][symbol] = {
                    'current_price': df['Close'].iloc[-1],
                    'daily_change': df['Returns'].iloc[-1] * 100,
                    'weekly_change': (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else None,
                    'monthly_change': (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else None,
                    'yearly_change': (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else None,
                    'volume': df['Volume'].iloc[-1],
                    'avg_volume': df['Volume'].iloc[-20:].mean(),
                    'RSI': df['RSI'].iloc[-1]
                }
        
        # تحليل المحفظة
        if self.portfolio:
            portfolio_metrics = self.calculate_portfolio_metrics()
            report['portfolio_analysis'] = portfolio_metrics
            
            # تقييم المخاطر
            report['risk_metrics'] = {
                'beta': {},
                'correlation': self.calculate_correlation_matrix(list(self.portfolio.keys())).to_dict()
            }
            
            for symbol in self.portfolio.keys():
                beta = self.calculate_beta(symbol)
                if beta:
                    report['risk_metrics']['beta'][symbol] = beta
        
        # توصيات
        recommendations = self.generate_investment_recommendations(symbols)
        report['recommendations'] = recommendations
        
        return report
    
    def generate_investment_recommendations(self, symbols: list) -> list:
        """توليد توصيات استثمارية"""
        recommendations = []
        
        for symbol in symbols:
            if symbol in self.stock_data:
                df = self.stock_data[symbol]
                current_price = df['Close'].iloc[-1]
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                
                # تحليل بناءً على المؤشرات الفنية
                if current_price < sma_20 and current_price < sma_50:
                    if rsi < 30:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'confidence': 'HIGH',
                            'reason': 'السعر أقل من المتوسطات ومؤشر RSI في منطقة ذروة البيع',
                            'target_price': sma_20 * 1.05
                        })
                    else:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'HOLD',
                            'confidence': 'MEDIUM',
                            'reason': 'السعر أقل من المتوسطات ولكن المؤشرات الفنية محايدة',
                            'target_price': sma_20
                        })
                
                elif current_price > sma_20 and current_price > sma_50:
                    if rsi > 70:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'confidence': 'HIGH',
                            'reason': 'السعر أعلى من المتوسطات ومؤشر RSI في منطقة ذروة الشراء',
                            'target_price': current_price * 0.95
                        })
                    else:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'confidence': 'MEDIUM',
                            'reason': 'اتجاه صعودي قوي مع مؤشرات فنية إيجابية',
                            'target_price': current_price * 1.1
                        })
        
        return recommendations

# main.py
def main():
    # إنشاء المحلل المالي
    analyzer = FinancialAnalyzer()
    
    print("=== نظام تحليل الأداء المالي والاستثماري المتقدم ===\n")
    
    # 1. تحميل بيانات الأسهم
    print("1. تحميل بيانات الأسهم...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', '^GSPC']  # S&P 500
    data = analyzer.download_stock_data(symbols, '2023-01-01')
    
    print(f"تم تحميل بيانات {len(data)} سهم")
    
    # 2. إنشاء محفظة استثمارية
    print("\n2. إنشاء محفظة استثمارية...")
    portfolio = {
        'AAPL': 0.4,
        'GOOGL': 0.3,
        'MSFT': 0.3
    }
    analyzer.create_portfolio(portfolio)
    
    # 3. حساب مقاييس المحفظة
    print("\n3. حساب مقاييس أداء المحفظة...")
    metrics = analyzer.calculate_portfolio_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 4. حساب معامل بيتا
    print("\n4. حساب معامل بيتا...")
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        beta = analyzer.calculate_beta(symbol)
        print(f"{symbol} Beta: {beta:.4f}")
    
    # 5. حساب المؤشرات الفنية
    print("\n5. حساب المؤشرات الفنية...")
    for symbol in ['AAPL']:
        indicators = analyzer.calculate_technical_indicators(symbol)
        print(f"\nمؤشرات {symbol}:")
        print(f"RSI: {indicators['momentum']['RSI']:.2f}")
        print(f"السعر الحالي: {indicators['volatility']['current_price']:.2f}")
    
    # 6. التنبؤ بالأسعار
    print("\n6. التنبؤ بأسعار الأسهم...")
    predictions = analyzer.predict_stock_prices('AAPL', 10)
    print(f"السعر الحالي: {predictions['last_actual_price']:.2f}")
    print(f"التوقعات لـ 10 أيام: {[f'{p:.2f}' for p in predictions['predictions']]}")
    
    # 7. تحسين المحفظة
    print("\n7. تحسين المحفظة...")
    optimal = analyzer.optimize_portfolio(['AAPL', 'GOOGL', 'MSFT'])
    if optimal['success']:
        print("الأوزان المثلى:")
        for symbol, weight in optimal['weights'].items():
            print(f"{symbol}: {weight:.2%}")
        print(f"العائد المتوقع: {optimal['expected_return']:.2%}")
        print(f"المخاطرة المتوقعة: {optimal['expected_volatility']:.2%}")
        print(f"نسبة شارب: {optimal['sharpe_ratio']:.2f}")
    
    # 8. محاكاة مونت كارلو
    print("\n8. محاكاة مونت كارلو...")
    simulation = analyzer.monte_carlo_simulation('AAPL', days=252, simulations=1000)
    print(f"السعر المتوقع بعد سنة: {simulation['mean_final_price']:.2f}")
    print(f"احتمالية الربح: {simulation['probability_profit']:.1%}")
    
    # 9. إنشاء التصورات
    print("\n9. إنشاء التصورات البيانية...")
    analyzer.plot_stock_prices(['AAPL', 'GOOGL', 'MSFT'])
    analyzer.plot_portfolio_allocation()
    analyzer.plot_efficient_frontier(['AAPL', 'GOOGL', 'MSFT'])
    analyzer.plot_correlation_heatmap(['AAPL', 'GOOGL', 'MSFT'])
    analyzer.plot_monte_carlo('AAPL')
    
    # 10. توليد التقرير
    print("\n10. توليد التقرير الاستثماري...")
    report = analyzer.generate_investment_report(['AAPL', 'GOOGL', 'MSFT'])
    
    with open('investment_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n✓ تم إكمال التحليل المالي بنجاح!")
    print("تم إنشاء الملفات التالية:")
    print("- investment_report.json (التقرير المالي)")
    print("- stock_prices.png (رسم أسعار الأسهم)")
    print("- portfolio_allocation.png (توزيع المحفظة)")
    print("- efficient_frontier.png (الحدود الفعالة)")
    print("- correlation_heatmap.png (خريطة الارتباط)")
    print("- monte_carlo_AAPL.png (محاكاة مونت كارلو)")

if __name__ == "__main__":
    main()