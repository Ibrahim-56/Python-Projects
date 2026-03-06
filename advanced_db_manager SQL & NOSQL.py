# advanced_db_manager/
"""
نظام متكامل لإدارة قواعد البيانات مع دعم SQL و NoSQL وتحليل البيانات
"""

import sqlite3
import pymongo
from pymongo import MongoClient
import pandas as pd
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging
from contextlib import contextmanager
import redis
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declarative_base
import pickle
import csv

class AdvancedDatabaseManager:
    """
    مدير قواعد بيانات متقدم يدعم:
    - SQLite للبيانات العلائقية
    - MongoDB للبيانات غير المهيكلة
    - Redis للتخزين المؤقت
    - SQLAlchemy لل ORM
    - تحويل بين أنواع قواعد البيانات
    - نسخ احتياطي واستعادة
    """
    
    def __init__(self, db_name: str = 'advanced_db'):
        self.db_name = db_name
        self.connections = {}
        self.setup_logging()
        self.Base = declarative_base()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('db_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ============= SQLite Connection Manager =============
    @contextmanager
    def sqlite_connection(self):
        """مدير سياق لاتصالات SQLite"""
        conn = sqlite3.connect(f'{self.db_name}.db')
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_sqlite_tables(self):
        """إنشاء جداول SQLite"""
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            
            # جدول المستخدمين
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # جدول المنتجات
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    price REAL NOT NULL,
                    category TEXT,
                    stock INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # جدول الطلبات
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_amount REAL,
                    status TEXT DEFAULT 'pending',
                    payment_method TEXT,
                    shipping_address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # جدول تفاصيل الطلب
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    price_at_time REAL,
                    FOREIGN KEY (order_id) REFERENCES orders (id),
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            ''')
            
            conn.commit()
            self.logger.info("تم إنشاء جداول SQLite بنجاح")
    
    # ============= SQLAlchemy ORM Models =============
    class User(Base):
        __tablename__ = 'users_orm'
        id = Column(Integer, primary_key=True)
        username = Column(String(50), unique=True, nullable=False)
        email = Column(String(100), unique=True, nullable=False)
        password_hash = Column(String(200))
        created_at = Column(DateTime, default=datetime.now)
        
    class Product(Base):
        __tablename__ = 'products_orm'
        id = Column(Integer, primary_key=True)
        name = Column(String(100), nullable=False)
        price = Column(Float, nullable=False)
        category = Column(String(50))
        stock = Column(Integer, default=0)
    
    # ============= MongoDB Connection =============
    def connect_mongodb(self, host='localhost', port=27017):
        """الاتصال بـ MongoDB"""
        try:
            self.mongo_client = MongoClient(host, port)
            self.mongo_db = self.mongo_client[self.db_name]
            self.logger.info("تم الاتصال بـ MongoDB بنجاح")
            return True
        except Exception as e:
            self.logger.error(f"خطأ في الاتصال بـ MongoDB: {e}")
            return False
    
    def create_mongodb_collections(self):
        """إنشاء مجموعات MongoDB"""
        # مجموعة المستخدمين
        users_collection = self.mongo_db['users']
        users_collection.create_index('email', unique=True)
        users_collection.create_index('username')
        
        # مجموعة المنتجات
        products_collection = self.mongo_db['products']
        products_collection.create_index('name')
        products_collection.create_index('category')
        
        # مجموعة السجلات (Logs)
        logs_collection = self.mongo_db['logs']
        logs_collection.create_index('timestamp')
        
        self.logger.info("تم إنشاء مجموعات MongoDB بنجاح")
    
    # ============= Redis Cache Connection =============
    def connect_redis(self, host='localhost', port=6379, db=0):
        """الاتصال بـ Redis للتخزين المؤقت"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            self.redis_client.ping()
            self.logger.info("تم الاتصال بـ Redis بنجاح")
            return True
        except Exception as e:
            self.logger.error(f"خطأ في الاتصال بـ Redis: {e}")
            return False
    
    # ============= Data Operations =============
    def hash_password(self, password: str) -> str:
        """تشفير كلمة المرور"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def add_user_sqlite(self, username: str, email: str, password: str, role: str = 'user'):
        """إضافة مستخدم جديد إلى SQLite"""
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, self.hash_password(password), role))
                conn.commit()
                user_id = cursor.lastrowid
                
                # تخزين مؤقت في Redis
                if hasattr(self, 'redis_client'):
                    user_data = {
                        'id': user_id,
                        'username': username,
                        'email': email,
                        'role': role
                    }
                    self.redis_client.setex(
                        f'user:{user_id}',
                        3600,  # ساعة واحدة
                        json.dumps(user_data)
                    )
                
                self.logger.info(f"تم إضافة المستخدم {username} بنجاح")
                return user_id
            except sqlite3.IntegrityError as e:
                self.logger.error(f"خطأ في إضافة المستخدم: {e}")
                return None
    
    def add_user_mongodb(self, user_data: Dict):
        """إضافة مستخدم إلى MongoDB"""
        if not hasattr(self, 'mongo_db'):
            self.connect_mongodb()
        
        user_data['created_at'] = datetime.now()
        user_data['password_hash'] = self.hash_password(user_data.get('password', ''))
        if 'password' in user_data:
            del user_data['password']
        
        result = self.mongo_db.users.insert_one(user_data)
        self.logger.info(f"تم إضافة المستخدم إلى MongoDB بالمعرف {result.inserted_id}")
        return result.inserted_id
    
    def add_product(self, product_data: Dict):
        """إضافة منتج إلى جميع قواعد البيانات"""
        # SQLite
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO products (name, description, price, category, stock)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                product_data['name'],
                product_data.get('description', ''),
                product_data['price'],
                product_data.get('category', ''),
                product_data.get('stock', 0)
            ))
            conn.commit()
            product_id = cursor.lastrowid
        
        # MongoDB
        if hasattr(self, 'mongo_db'):
            product_data['sqlite_id'] = product_id
            product_data['created_at'] = datetime.now()
            self.mongo_db.products.insert_one(product_data)
        
        # Redis Cache
        if hasattr(self, 'redis_client'):
            self.redis_client.setex(
                f'product:{product_id}',
                1800,  # نصف ساعة
                json.dumps(product_data)
            )
        
        self.logger.info(f"تم إضافة المنتج {product_data['name']} بنجاح")
        return product_id
    
    def search_products(self, query: str, min_price: float = 0, max_price: float = float('inf')):
        """البحث المتقدم عن المنتجات"""
        with self.sqlite_connection() as conn:
            # بحث في SQLite
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM products 
                WHERE (name LIKE ? OR description LIKE ?)
                AND price BETWEEN ? AND ?
                ORDER BY price
            ''', (f'%{query}%', f'%{query}%', min_price, max_price))
            
            sqlite_results = [dict(row) for row in cursor.fetchall()]
        
        # بحث في MongoDB إذا كان متصلاً
        mongo_results = []
        if hasattr(self, 'mongo_db'):
            mongo_results = list(self.mongo_db.products.find({
                '$and': [
                    {'$or': [
                        {'name': {'$regex': query, '$options': 'i'}},
                        {'description': {'$regex': query, '$options': 'i'}}
                    ]},
                    {'price': {'$gte': min_price, '$lte': max_price}}
                ]
            }).limit(50))
            
            # تحويل ObjectId إلى string
            for doc in mongo_results:
                doc['_id'] = str(doc['_id'])
        
        return {
            'sqlite_results': sqlite_results,
            'mongo_results': mongo_results,
            'total_count': len(sqlite_results) + len(mongo_results)
        }
    
    def get_user_orders(self, user_id: int) -> Dict:
        """الحصول على طلبات المستخدم مع التخزين المؤقت"""
        # محاولة الحصول من Redis أولاً
        if hasattr(self, 'redis_client'):
            cached_orders = self.redis_client.get(f'user_orders:{user_id}')
            if cached_orders:
                self.logger.info(f"تم جلب طلبات المستخدم {user_id} من Redis")
                return json.loads(cached_orders)
        
        # إذا لم تكن في Redis، جلب من SQLite
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            
            # معلومات المستخدم
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = dict(cursor.fetchone() or {})
            
            # طلبات المستخدم
            cursor.execute('''
                SELECT o.*, COUNT(oi.id) as items_count,
                       SUM(oi.quantity) as total_items
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                WHERE o.user_id = ?
                GROUP BY o.id
                ORDER BY o.order_date DESC
            ''', (user_id,))
            
            orders = [dict(row) for row in cursor.fetchall()]
            
            # تفاصيل كل طلب
            for order in orders:
                cursor.execute('''
                    SELECT oi.*, p.name, p.category
                    FROM order_items oi
                    JOIN products p ON oi.product_id = p.id
                    WHERE oi.order_id = ?
                ''', (order['id'],))
                order['items'] = [dict(row) for row in cursor.fetchall()]
        
        result = {
            'user': user,
            'orders': orders,
            'total_orders': len(orders),
            'total_spent': sum(o['total_amount'] for o in orders if o['total_amount'])
        }
        
        # تخزين في Redis للمرة القادمة
        if hasattr(self, 'redis_client'):
            self.redis_client.setex(
                f'user_orders:{user_id}',
                300,  # 5 دقائق
                json.dumps(result, default=str)
            )
        
        return result
    
    def generate_sales_report(self, start_date: str, end_date: str) -> Dict:
        """توليد تقرير مبيعات شامل"""
        with self.sqlite_connection() as conn:
            # إجمالي المبيعات
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(total_amount) as total_revenue,
                    AVG(total_amount) as avg_order_value,
                    COUNT(DISTINCT user_id) as unique_customers
                FROM orders
                WHERE order_date BETWEEN ? AND ?
            ''', (start_date, end_date))
            
            summary = dict(cursor.fetchone())
            
            # المبيعات حسب المنتج
            cursor.execute('''
                SELECT 
                    p.name,
                    p.category,
                    SUM(oi.quantity) as quantity_sold,
                    SUM(oi.quantity * oi.price_at_time) as revenue
                FROM order_items oi
                JOIN products p ON oi.product_id = p.id
                JOIN orders o ON oi.order_id = o.id
                WHERE o.order_date BETWEEN ? AND ?
                GROUP BY p.id
                ORDER BY revenue DESC
            ''', (start_date, end_date))
            
            product_sales = [dict(row) for row in cursor.fetchall()]
            
            # المبيعات اليومية
            cursor.execute('''
                SELECT 
                    DATE(order_date) as date,
                    COUNT(*) as orders,
                    SUM(total_amount) as revenue
                FROM orders
                WHERE order_date BETWEEN ? AND ?
                GROUP BY DATE(order_date)
                ORDER BY date
            ''', (start_date, end_date))
            
            daily_sales = [dict(row) for row in cursor.fetchall()]
        
        return {
            'summary': summary,
            'product_sales': product_sales,
            'daily_sales': daily_sales,
            'period': {
                'start': start_date,
                'end': end_date
            }
        }
    
    def backup_database(self, backup_path: str = None):
        """إنشاء نسخة احتياطية كاملة"""
        if not backup_path:
            backup_path = f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        backup_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'db_name': self.db_name
            },
            'sqlite': {},
            'mongodb': {}
        }
        
        # نسخ SQLite
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            
            # الحصول على جميع الجداول
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f'SELECT * FROM {table_name}')
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                table_data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i]
                    table_data.append(row_dict)
                
                backup_data['sqlite'][table_name] = table_data
        
        # نسخ MongoDB إذا كان متصلاً
        if hasattr(self, 'mongo_db'):
            for collection_name in self.mongo_db.list_collection_names():
                collection = self.mongo_db[collection_name]
                documents = list(collection.find())
                
                # تحويل ObjectId إلى string
                for doc in documents:
                    doc['_id'] = str(doc['_id'])
                
                backup_data['mongodb'][collection_name] = documents
        
        # حفظ النسخة الاحتياطية
        with open(f'{backup_path}.json', 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"تم إنشاء نسخة احتياطية في {backup_path}.json")
        return backup_path
    
    def restore_database(self, backup_path: str):
        """استعادة قاعدة البيانات من نسخة احتياطية"""
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        # استعادة SQLite
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            
            for table_name, rows in backup_data['sqlite'].items():
                if rows:
                    # حذف البيانات الموجودة
                    cursor.execute(f'DELETE FROM {table_name}')
                    
                    # إدراج البيانات الجديدة
                    columns = list(rows[0].keys())
                    placeholders = ','.join(['?' for _ in columns])
                    
                    for row in rows:
                        values = [row[col] for col in columns]
                        cursor.execute(f'''
                            INSERT INTO {table_name} ({','.join(columns)})
                            VALUES ({placeholders})
                        ''', values)
            
            conn.commit()
        
        # استعادة MongoDB إذا كان متصلاً
        if hasattr(self, 'mongo_db') and 'mongodb' in backup_data:
            for collection_name, documents in backup_data['mongodb'].items():
                collection = self.mongo_db[collection_name]
                collection.delete_many({})
                if documents:
                    collection.insert_many(documents)
        
        self.logger.info(f"تم استعادة قاعدة البيانات من {backup_path}")
    
    def export_to_csv(self, table_name: str, csv_path: str = None):
        """تصدير جدول إلى CSV"""
        if not csv_path:
            csv_path = f'{table_name}_{datetime.now().strftime("%Y%m%d")}.csv'
        
        with self.sqlite_connection() as conn:
            df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"تم تصدير {table_name} إلى {csv_path}")
        return csv_path
    
    def import_from_csv(self, csv_path: str, table_name: str):
        """استيراد بيانات من CSV"""
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        with self.sqlite_connection() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=False)
        
        self.logger.info(f"تم استيراد {len(df)} سجل إلى {table_name}")
        return len(df)
    
    def get_database_stats(self) -> Dict:
        """إحصائيات قاعدة البيانات"""
        stats = {
            'sqlite': {},
            'mongodb': {},
            'redis': {}
        }
        
        # إحصائيات SQLite
        with self.sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                count = cursor.fetchone()[0]
                stats['sqlite'][table_name] = {
                    'records': count,
                    'name': table_name
                }
        
        # إحصائيات MongoDB
        if hasattr(self, 'mongo_db'):
            for collection_name in self.mongo_db.list_collection_names():
                count = self.mongo_db[collection_name].count_documents({})
                stats['mongodb'][collection_name] = {
                    'records': count,
                    'name': collection_name
                }
        
        # إحصائيات Redis
        if hasattr(self, 'redis_client'):
            stats['redis']['keys'] = self.redis_client.dbsize()
        
        return stats

# main.py
def main():
    # إنشاء مدير قاعدة البيانات
    db_manager = AdvancedDatabaseManager('enterprise_db')
    
    print("=== نظام إدارة قواعد البيانات المتقدم ===\n")
    
    # 1. إنشاء جداول SQLite
    print("1. إنشاء جداول SQLite...")
    db_manager.create_sqlite_tables()
    
    # 2. الاتصال بـ MongoDB
    print("\n2. الاتصال بـ MongoDB...")
    if db_manager.connect_mongodb():
        db_manager.create_mongodb_collections()
    
    # 3. الاتصال بـ Redis
    print("\n3. الاتصال بـ Redis...")
    db_manager.connect_redis()
    
    # 4. إضافة بيانات تجريبية
    print("\n4. إضافة بيانات تجريبية...")
    
    # إضافة مستخدمين
    user1_id = db_manager.add_user_sqlite('ahmed123', 'ahmed@email.com', 'password123', 'admin')
    user2_id = db_manager.add_user_sqlite('sara456', 'sara@email.com', 'pass456', 'user')
    
    # إضافة منتجات
    product1_id = db_manager.add_product({
        'name': 'لابتوب HP',
        'description': 'لابتوب بشاشة 15 بوصة، معالج i7',
        'price': 4500,
        'category': 'إلكترونيات',
        'stock': 50
    })
    
    product2_id = db_manager.add_product({
        'name': 'هاتف Samsung',
        'description': 'هاتف ذكي بشاشتين',
        'price': 2800,
        'category': 'إلكترونيات',
        'stock': 100
    })
    
    # 5. البحث عن منتجات
    print("\n5. البحث عن منتجات...")
    results = db_manager.search_products('لابتوب', 1000, 10000)
    print(f"نتائج البحث: {results['total_count']} منتج")
    
    # 6. الحصول على طلبات المستخدم
    print("\n6. جلب طلبات المستخدم...")
    if user1_id:
        user_orders = db_manager.get_user_orders(user1_id)
        print(f"عدد طلبات المستخدم: {user_orders.get('total_orders', 0)}")
    
    # 7. إحصائيات قاعدة البيانات
    print("\n7. إحصائيات قاعدة البيانات:")
    stats = db_manager.get_database_stats()
    print(f"SQLite: {len(stats['sqlite'])} جداول")
    print(f"MongoDB: {len(stats['mongodb'])} مجموعات")
    print(f"Redis: {stats['redis'].get('keys', 0)} مفتاح")
    
    # 8. إنشاء نسخة احتياطية
    print("\n8. إنشاء نسخة احتياطية...")
    backup_path = db_manager.backup_database()
    
    # 9. تصدير إلى CSV
    print("\n9. تصدير البيانات إلى CSV...")
    db_manager.export_to_csv('users')
    db_manager.export_to_csv('products')
    
    print("\n✓ تم إكمال جميع العمليات بنجاح!")

if __name__ == "__main__":
    main()