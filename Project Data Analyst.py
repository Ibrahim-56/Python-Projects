# sales_analytics_system/
"""
نظام متكامل لتحليل بيانات المبيعات باستخدام Pandas, NumPy, Scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SalesAnalyticsSystem:
    """
    نظام تحليل متقدم للمبيعات يتضمن:
    - تحليل سلاسل زمنية
    - توقعات المبيعات باستخدام ML
    - تحليل العملاء وتقسيمهم
    - تحليل المنتجات الأكثر مبيعاً
    """
    
    def __init__(self, data_path: str = 'sales_data.csv'):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sales_analytics.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_sample_data(self, num_records: int = 10000):
        """توليد بيانات مبيعات عشوائية للاختبار"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='H')
        dates = np.random.choice(dates, num_records)
        
        products = ['لابتوب', 'هاتف', 'جهاز لوحي', 'ساعة ذكية', 'سماعات', 'شاحن']
        categories = ['إلكترونيات', 'إلكترونيات', 'إلكترونيات', 'إكسسوارات', 'إكسسوارات', 'إكسسوارات']
        product_category = dict(zip(products, categories))
        
        regions = ['الرياض', 'جدة', 'الدمام', 'مكة', 'المدينة', 'تبوك']
        
        data = {
            'التاريخ': dates,
            'المنتج': np.random.choice(products, num_records),
            'الكمية': np.random.randint(1, 10, num_records),
            'السعر': np.random.uniform(100, 5000, num_records).round(2),
            'المنطقة': np.random.choice(regions, num_records),
            'نوع_العميل': np.random.choice(['فرد', 'شركة', 'حكومي'], num_records, p=[0.6, 0.3, 0.1])
        }
        
        df = pd.DataFrame(data)
        df['التصنيف'] = df['المنتج'].map(product_category)
        df['الإجمالي'] = df['الكمية'] * df['السعر']
        df['الشهر'] = df['التاريخ'].dt.month
        df['اليوم'] = df['التاريخ'].dt.day
        df['الساعة'] = df['التاريخ'].dt.hour
        df['يوم_الأسبوع'] = df['التاريخ'].dt.dayofweek
        
        df.to_csv(self.data_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"تم توليد {num_records} سجل مبيعات")
        
    def load_and_preprocess_data(self):
        """تحميل وتنظيف البيانات"""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            self.df['التاريخ'] = pd.to_datetime(self.df['التاريخ'])
            
            # تنظيف البيانات
            self.df = self.df.dropna()
            self.df = self.df[self.df['الكمية'] > 0]
            self.df = self.df[self.df['السعر'] > 0]
            
            # إضافة ميزات جديدة
            self.df['الإيرادات_التراكمية'] = self.df['الإجمالي'].cumsum()
            self.df['متوسط_المبيعات_المتحرك'] = self.df['الإجمالی'].rolling(window=7).mean()
            
            self.logger.info(f"تم تحميل {len(self.df)} سجل بعد التنظيف")
            return True
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل البيانات: {e}")
            return False
    
    def analyze_sales_trends(self) -> Dict:
        """تحليل اتجاهات المبيعات"""
        analysis = {}
        
        # المبيعات حسب الشهر
        monthly_sales = self.df.groupby('الشهر')['الإجمالي'].agg(['sum', 'mean', 'count'])
        analysis['monthly_sales'] = monthly_sales.to_dict()
        
        # أفضل 10 منتجات مبيعاً
        top_products = self.df.groupby('المنتج')['الكمية'].sum().sort_values(ascending=False).head(10)
        analysis['top_products'] = top_products.to_dict()
        
        # تحليل المناطق
        region_analysis = self.df.groupby('المنطقة').agg({
            'الإجمالي': ['sum', 'mean'],
            'الكمية': 'sum',
            'نوع_العميل': lambda x: x.mode()[0] if not x.mode().empty else None
        })
        analysis['region_analysis'] = region_analysis.to_dict()
        
        # تحليل العملاء
        customer_analysis = self.df.groupby('نوع_العميل')['الإجمالي'].agg(['sum', 'mean', 'count'])
        analysis['customer_analysis'] = customer_analysis.to_dict()
        
        # تحليل مواسم الذروة
        peak_hours = self.df.groupby('الساعة')['الإجمالي'].sum().sort_values(ascending=False).head(5)
        peak_days = self.df.groupby('يوم_الأسبوع')['الإجمالي'].sum().sort_values(ascending=False)
        analysis['peak_hours'] = peak_hours.to_dict()
        analysis['peak_days'] = peak_days.to_dict()
        
        return analysis
    
    def prepare_ml_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """تحضير الميزات للتعلم الآلي"""
        # تجميع البيانات يومياً
        daily_sales = self.df.groupby(self.df['التاريخ'].dt.date)['الإجمالي'].sum().reset_index()
        daily_sales['التاريخ'] = pd.to_datetime(daily_sales['التاريخ'])
        
        # إنشاء ميزات
        daily_sales['اليوم'] = daily_sales['التاريخ'].dt.day
        daily_sales['الشهر'] = daily_sales['التاريخ'].dt.month
        daily_sales['يوم_الأسبوع'] = daily_sales['التاريخ'].dt.dayofweek
        daily_sales['الأسبوع'] = daily_sales['التاريخ'].dt.isocalendar().week
        
        # ميزات متأخرة (Lags)
        for lag in [1, 2, 3, 7, 14, 30]:
            daily_sales[f'lag_{lag}'] = daily_sales['الإجمالي'].shift(lag)
        
        # إسقاط القيم المفقودة
        daily_sales = daily_sales.dropna()
        
        features = ['اليوم', 'الشهر', 'يوم_الأسبوع', 'الأسبوع'] + [f'lag_{lag}' for lag in [1, 2, 3, 7, 14, 30]]
        X = daily_sales[features].values
        y = daily_sales['الإجمالي'].values
        
        return X, y
    
    def train_prediction_model(self):
        """تدريب نموذج توقع المبيعات"""
        X, y = self.prepare_ml_features()
        
        # تقسيم البيانات
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # تطبيع الميزات
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # تدريب النموذج
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # تقييم النموذج
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.logger.info(f"دقة النموذج على التدريب: {train_score:.4f}")
        self.logger.info(f"دقة النموذج على الاختبار: {test_score:.4f}")
        
        # أهمية الميزات
        feature_importance = dict(zip(
            ['اليوم', 'الشهر', 'يوم_الأسبوع', 'الأسبوع', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30'],
            self.model.feature_importances_
        ))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def predict_future_sales(self, days: int = 30) -> pd.DataFrame:
        """توقع المبيعات المستقبلية"""
        if self.model is None:
            raise ValueError("يجب تدريب النموذج أولاً")
        
        last_date = self.df['التاريخ'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # إنشاء ميزات للتواريخ المستقبلية
        future_data = []
        current_sales = self.df.groupby(self.df['التاريخ'].dt.date)['الإجمالي'].sum()
        
        for i, date in enumerate(future_dates):
            features = {
                'اليوم': date.day,
                'الشهر': date.month,
                'يوم_الأسبوع': date.weekday(),
                'الأسبوع': date.isocalendar().week,
            }
            
            # إضافة الميزات المتأخرة
            for lag in [1, 2, 3, 7, 14, 30]:
                lag_date = date - timedelta(days=lag)
                if lag_date.date() in current_sales.index:
                    features[f'lag_{lag}'] = current_sales[lag_date.date()]
                else:
                    features[f'lag_{lag}'] = current_sales.iloc[-lag] if len(current_sales) > lag else 0
            
            future_data.append(features)
        
        future_df = pd.DataFrame(future_data)
        future_scaled = self.scaler.transform(future_df)
        predictions = self.model.predict(future_scaled)
        
        results = pd.DataFrame({
            'التاريخ': future_dates,
            'المبيعات_المتوقعة': predictions,
            'اليوم': future_df['اليوم'],
            'الشهر': future_df['الشهر'],
            'يوم_الأسبوع': future_df['يوم_الأسبوع']
        })
        
        return results
    
    def generate_executive_report(self) -> Dict:
        """توليد تقرير تنفيذي شامل"""
        analysis = self.analyze_sales_trends()
        
        report = {
            'ملخص_عام': {
                'إجمالي_المبيعات': float(self.df['الإجمالي'].sum()),
                'متوسط_المبيعات_اليومية': float(self.df.groupby(self.df['التاريخ'].dt.date)['الإجمالي'].sum().mean()),
                'إجمالي_المنتجات_المباعة': int(self.df['الكمية'].sum()),
                'عدد_المعاملات': len(self.df),
                'متوسط_قيمة_المعاملة': float(self.df['الإجمالي'].mean()),
                'أعلى_مبيعات_يوم': float(self.df.groupby(self.df['التاريخ'].dt.date)['الإجمالي'].sum().max())
            },
            'تحليل_المنتجات': {
                'المنتج_الأكثر_مبيعاً': list(analysis['top_products'].keys())[0] if analysis['top_products'] else None,
                'إيرادات_المنتج_الأكثر_مبيعاً': float(list(analysis['top_products'].values())[0]) if analysis['top_products'] else 0
            },
            'تحليل_المناطق': {
                'المنطقة_الأكثر_مبيعاً': max(analysis['region_analysis'][('الإجمالي', 'sum')], key=analysis['region_analysis'][('الإجمالي', 'sum')].get) if analysis['region_analysis'] else None,
                'إيرادات_المنطقة_الأكثر_مبيعاً': float(max(analysis['region_analysis'][('الإجمالي', 'sum')].values())) if analysis['region_analysis'] else 0
            },
            'تحليل_العملاء': {
                'أفضل_نوع_عميل': max(analysis['customer_analysis'][('الإجمالي', 'sum')], key=analysis['customer_analysis'][('الإجمالي', 'sum')].get) if analysis['customer_analysis'] else None,
                'إيرادات_أفضل_نوع_عميل': float(max(analysis['customer_analysis'][('الإجمالي', 'sum')].values())) if analysis['customer_analysis'] else 0
            },
            'أوقات_الذروة': {
                'أفضل_ساعة': max(analysis['peak_hours'], key=analysis['peak_hours'].get) if analysis['peak_hours'] else None,
                'أفضل_يوم': max(analysis['peak_days'], key=analysis['peak_days'].get) if analysis['peak_days'] else None
            }
        }
        
        return report
    
    def save_model(self, path: str = 'sales_model.pkl'):
        """حفظ النموذج المدرب"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        self.logger.info(f"تم حفظ النموذج في {path}")
    
    def load_model(self, path: str = 'sales_model.pkl'):
        """تحميل النموذج المدرب"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        self.logger.info(f"تم تحميل النموذج من {path}")

# main.py
def main():
    # إنشاء النظام
    system = SalesAnalyticsSystem('sales_data.csv')
    
    # توليد بيانات تجريبية
    print("1. توليد بيانات تجريبية...")
    system.generate_sample_data(50000)
    
    # تحميل البيانات
    print("\n2. تحميل وتحليل البيانات...")
    if system.load_and_preprocess_data():
        
        # تحليل المبيعات
        print("\n3. تحليل اتجاهات المبيعات...")
        trends = system.analyze_sales_trends()
        print("✓ تم تحليل البيانات بنجاح")
        
        # تدريب نموذج التوقع
        print("\n4. تدريب نموذج التوقع...")
        model_results = system.train_prediction_model()
        print(f"✓ دقة النموذج: {model_results['test_score']:.2%}")
        
        # توقع المبيعات المستقبلية
        print("\n5. توقع المبيعات للأيام القادمة...")
        predictions = system.predict_future_sales(30)
        print(predictions.head(10))
        
        # توليد التقرير التنفيذي
        print("\n6. توليد التقرير التنفيذي...")
        report = system.generate_executive_report()
        print("\n=== التقرير التنفيذي ===")
        for section, data in report.items():
            print(f"\n{section}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        # حفظ النموذج
        print("\n7. حفظ النموذج...")
        system.save_model()
        
        print("\n✓ تم إكمال التحليل بنجاح!")

if __name__ == "__main__":
    main()