# social_media_analytics/
"""
نظام متكامل لتحليل بيانات التواصل الاجتماعي باستخدام:
- Web Scraping
- Natural Language Processing (NLP)
- Sentiment Analysis
- Trend Prediction
- Network Analysis
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import tweepy
import facebook
import instagram_private_api
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class SocialMediaAnalyzer:
    """
    نظام تحليل متكامل لوسائل التواصل الاجتماعي
    """
    
    def __init__(self):
        self.data = pd.DataFrame()
        self.sentiment_scores = []
        self.topics = []
        self.influence_network = nx.Graph()
        self.trends = {}
        self.setup_nltk()
        self.setup_logging()
        
    def setup_nltk(self):
        """تحميل موارد NLTK اللازمة"""
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        self.stop_words = set(stopwords.words('arabic') + stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def setup_logging(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('social_media_analytics.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ============= Web Scraping Module =============
    def scrape_twitter_hashtag(self, hashtag: str, pages: int = 5):
        """سحب تغريدات من تويتر باستخدام hashtag"""
        tweets = []
        base_url = f"https://twitter.com/hashtag/{hashtag}?lang=en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for page in range(pages):
            try:
                response = requests.get(base_url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # استخراج التغريدات (هذا مثال مبسط - في الواقع تحتاج API)
                tweet_cards = soup.find_all('div', {'data-testid': 'tweet'})
                
                for card in tweet_cards:
                    tweet = {
                        'text': card.find('div', {'lang': True}).text if card.find('div', {'lang': True}) else '',
                        'timestamp': datetime.now(),
                        'likes': self.extract_count(card, 'like'),
                        'retweets': self.extract_count(card, 'retweet'),
                        'replies': self.extract_count(card, 'reply'),
                        'source': 'twitter',
                        'hashtag': hashtag
                    }
                    tweets.append(tweet)
                    
            except Exception as e:
                self.logger.error(f"خطأ في سحب صفحة {page}: {e}")
        
        return pd.DataFrame(tweets)
    
    def extract_count(self, card, element_type):
        """استخراج الأرقام من العناصر"""
        try:
            element = card.find('div', {'aria-label': lambda x: x and element_type in x.lower()})
            if element:
                text = element.get('aria-label', '')
                numbers = re.findall(r'\d+', text)
                return int(numbers[0]) if numbers else 0
        except:
            pass
        return 0
    
    # ============= Text Processing Module =============
    def clean_text(self, text: str) -> str:
        """تنظيف النص للتحليل"""
        if not isinstance(text, str):
            return ""
        
        # إزالة الروابط
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # إزالة الإشارات (@username)
        text = re.sub(r'@\w+', '', text)
        
        # إزالة الهاشتاجات
        text = re.sub(r'#\w+', '', text)
        
        # إزالة الرموز التعبيرية
        text = re.sub(r'[^\w\s]', '', text)
        
        # إزالة الأرقام
        text = re.sub(r'\d+', '', text)
        
        # تحويل إلى أحرف صغيرة
        text = text.lower()
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> list:
        """تقطيع النص إلى كلمات مع Lemmatization"""
        tokens = word_tokenize(text)
        
        # إزالة كلمات التوقف
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def analyze_sentiment(self, text: str) -> dict:
        """تحليل المشاعر في النص"""
        if not text:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
        
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def extract_entities(self, text: str) -> dict:
        """استخراج الكيانات المسماة"""
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': []
        }
        
        # استخدام NLTK للتعرف على الكيانات
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        named_entities = nltk.ne_chunk(pos_tags)
        
        for entity in named_entities:
            if hasattr(entity, 'label'):
                entity_text = ' '.join([leaf[0] for leaf in entity.leaves()])
                if entity.label() == 'PERSON':
                    entities['people'].append(entity_text)
                elif entity.label() == 'ORGANIZATION':
                    entities['organizations'].append(entity_text)
                elif entity.label() == 'GPE':
                    entities['locations'].append(entity_text)
                elif entity.label() == 'DATE':
                    entities['dates'].append(entity_text)
        
        return entities
    
    # ============= Topic Modeling Module =============
    def extract_topics(self, texts: list, num_topics: int = 5) -> list:
        """استخراج المواضيع الرئيسية من النصوص"""
        if not texts:
            return []
        
        # تنظيف النصوص
        cleaned_texts = [self.clean_text(text) for text in texts if isinstance(text, str)]
        
        # تحويل النصوص إلى مصفوفة TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(self.stop_words))
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # تطبيق LDA
        lda = LDA(n_components=num_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        # استخراج الكلمات المميزة لكل موضوع
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'weight': float(topic.sum())
            })
        
        return topics
    
    # ============= Trend Analysis Module =============
    def analyze_trends(self, data: pd.DataFrame, time_column: str = 'timestamp') -> dict:
        """تحليل الاتجاهات مع مرور الوقت"""
        if data.empty:
            return {}
        
        # تجميع البيانات حسب الوقت
        data[time_column] = pd.to_datetime(data[time_column])
        data['hour'] = data[time_column].dt.hour
        data['day'] = data[time_column].dt.day
        data['month'] = data[time_column].dt.month
        data['day_of_week'] = data[time_column].dt.dayofweek
        
        trends = {
            'hourly_activity': data.groupby('hour').size().to_dict(),
            'daily_activity': data.groupby('day_of_week').size().to_dict(),
            'monthly_activity': data.groupby('month').size().to_dict(),
            'peak_hours': self.find_peak_times(data),
            'trending_keywords': self.find_trending_keywords(data),
            'engagement_metrics': self.calculate_engagement_metrics(data)
        }
        
        return trends
    
    def find_peak_times(self, data: pd.DataFrame) -> dict:
        """العثور على أوقات الذروة"""
        hourly_engagement = data.groupby('hour').agg({
            'likes': 'sum' if 'likes' in data.columns else lambda x: 1,
            'retweets': 'sum' if 'retweets' in data.columns else lambda x: 1
        })
        
        peak_hours = {
            'most_active_hour': int(hourly_engagement.sum(axis=1).idxmax()),
            'most_engaging_hour': int(hourly_engagement.sum(axis=1).idxmax()),
            'hourly_distribution': hourly_engagement.to_dict()
        }
        
        return peak_hours
    
    def find_trending_keywords(self, data: pd.DataFrame, top_n: int = 20) -> list:
        """العثور على الكلمات المفتاحية الرائجة"""
        if 'text' not in data.columns:
            return []
        
        # تجميع كل النصوص
        all_text = ' '.join(data['text'].dropna().astype(str))
        tokens = self.tokenize_and_lemmatize(all_text)
        
        # حساب تكرار الكلمات
        word_freq = Counter(tokens)
        
        # اختيار أكثر الكلمات تكراراً
        trending = [{'word': word, 'count': count} 
                   for word, count in word_freq.most_common(top_n)]
        
        return trending
    
    def calculate_engagement_metrics(self, data: pd.DataFrame) -> dict:
        """حساب مقاييس التفاعل"""
        metrics = {}
        
        if 'likes' in data.columns:
            metrics['total_likes'] = int(data['likes'].sum())
            metrics['avg_likes'] = float(data['likes'].mean())
        
        if 'retweets' in data.columns:
            metrics['total_shares'] = int(data['retweets'].sum())
            metrics['avg_shares'] = float(data['retweets'].mean())
        
        if 'replies' in data.columns:
            metrics['total_comments'] = int(data['replies'].sum())
            metrics['avg_comments'] = float(data['replies'].mean())
        
        metrics['total_posts'] = len(data)
        metrics['engagement_rate'] = self.calculate_engagement_rate(data)
        
        return metrics
    
    def calculate_engagement_rate(self, data: pd.DataFrame) -> float:
        """حساب معدل التفاعل"""
        if data.empty:
            return 0.0
        
        total_engagement = 0
        if 'likes' in data.columns:
            total_engagement += data['likes'].sum()
        if 'retweets' in data.columns:
            total_engagement += data['retweets'].sum()
        if 'replies' in data.columns:
            total_engagement += data['replies'].sum()
        
        return float(total_engagement / len(data)) if len(data) > 0 else 0.0
    
    # ============= Network Analysis Module =============
    def build_influence_network(self, data: pd.DataFrame):
        """بناء شبكة التأثير بين المستخدمين"""
        G = nx.Graph()
        
        for idx, row in data.iterrows():
            if 'user' in row and 'mentions' in row:
                user = row['user']
                mentions = row['mentions']
                
                # إضافة المستخدم الرئيسي
                G.add_node(user, type='user', posts=1)
                
                # إضافة العلاقات مع المذكورين
                for mentioned in mentions:
                    G.add_node(mentioned, type='user')
                    G.add_edge(user, mentioned, weight=1)
        
        self.influence_network = G
        
        # حساب مقاييس الشبكة
        network_metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'clustering_coefficient': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        # حساب درجة التأثير (PageRank)
        pagerank = nx.pagerank(G)
        top_influencers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
        network_metrics['top_influencers'] = [
            {'user': user, 'influence_score': score} 
            for user, score in top_influencers
        ]
        
        return network_metrics
    
    def detect_communities(self):
        """اكتشاف المجتمعات في الشبكة"""
        from networkx.algorithms.community import greedy_modularity_communities
        
        if self.influence_network.number_of_nodes() < 2:
            return []
        
        communities = greedy_modularity_communities(self.influence_network)
        
        community_info = []
        for i, community in enumerate(communities):
            community_info.append({
                'community_id': i,
                'size': len(community),
                'members': list(community)[:10],  # أول 10 أعضاء فقط
                'density': nx.density(self.influence_network.subgraph(community))
            })
        
        return community_info
    
    # ============= Predictive Analytics Module =============
    def predict_trends(self, historical_data: pd.DataFrame, days_ahead: int = 7) -> dict:
        """التنبؤ بالاتجاهات المستقبلية"""
        predictions = {}
        
        if historical_data.empty:
            return predictions
        
        # تجميع البيانات حسب اليوم
        historical_data['date'] = pd.to_datetime(historical_data['timestamp']).dt.date
        daily_posts = historical_data.groupby('date').size()
        
        if len(daily_posts) < 10:  # تحتاج بيانات كافية
            return predictions
        
        # نموذج بسيط للتنبؤ (يمكن تحسينه باستخدام ARIMA أو LSTM)
        from sklearn.linear_model import LinearRegression
        
        # تحضير البيانات
        X = np.array(range(len(daily_posts))).reshape(-1, 1)
        y = daily_posts.values
        
        # تدريب النموذج
        model = LinearRegression()
        model.fit(X, y)
        
        # التنبؤ بالأيام القادمة
        future_X = np.array(range(len(daily_posts), len(daily_posts) + days_ahead)).reshape(-1, 1)
        future_predictions = model.predict(future_X)
        
        predictions = {
            'predicted_posts': future_predictions.tolist(),
            'trend_direction': 'up' if future_predictions[-1] > y[-1] else 'down',
            'growth_rate': ((future_predictions[-1] - y[-1]) / y[-1]) * 100,
            'confidence': model.score(X, y)
        }
        
        return predictions
    
    # ============= Visualization Module =============
    def create_wordcloud(self, texts: list, title: str = 'Word Cloud'):
        """إنشاء Word Cloud من النصوص"""
        if not texts:
            return None
        
        all_text = ' '.join([str(text) for text in texts if isinstance(text, str)])
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path='arial.ttf',
            stopwords=self.stop_words,
            max_words=100
        ).generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(f'wordcloud_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
        
        return 'wordcloud_generated.png'
    
    def plot_sentiment_trend(self, data: pd.DataFrame):
        """رسم اتجاه المشاعر مع مرور الوقت"""
        if data.empty or 'timestamp' not in data.columns or 'sentiment' not in data.columns:
            return
        
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        daily_sentiment = data.groupby('date')['sentiment'].apply(
            lambda x: (x == 'positive').mean()
        )
        
        plt.figure(figsize=(12, 6))
        daily_sentiment.plot(kind='line', marker='o')
        plt.title('اتجاه المشاعر مع مرور الوقت')
        plt.xlabel('التاريخ')
        plt.ylabel('نسبة المشاعر الإيجابية')
        plt.grid(True, alpha=0.3)
        plt.savefig('sentiment_trend.png')
        plt.close()
    
    def plot_engagement_heatmap(self, data: pd.DataFrame):
        """رسم خريطة حرارية للتفاعل حسب الوقت"""
        if data.empty or 'timestamp' not in data.columns:
            return
        
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        
        engagement_matrix = pd.crosstab(data['day'], data['hour'])
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(engagement_matrix, cmap='YlOrRd', annot=True, fmt='d')
        plt.title('خريطة التفاعل حسب اليوم والساعة')
        plt.xlabel('الساعة')
        plt.ylabel('اليوم (0=الأحد)')
        plt.savefig('engagement_heatmap.png')
        plt.close()
    
    # ============= Main Analysis Pipeline =============
    def analyze_posts(self, posts_df: pd.DataFrame) -> dict:
        """تحليل شامل للمنشورات"""
        results = {
            'metadata': {
                'total_posts': len(posts_df),
                'analysis_time': datetime.now().isoformat()
            }
        }
        
        # تحليل المشاعر
        sentiments = []
        for text in posts_df['text'] if 'text' in posts_df.columns else []:
            if isinstance(text, str):
                sentiments.append(self.analyze_sentiment(text))
            else:
                sentiments.append({'sentiment': 'neutral'})
        
        posts_df['sentiment'] = [s['sentiment'] for s in sentiments]
        posts_df['polarity'] = [s['polarity'] for s in sentiments]
        
        results['sentiment_analysis'] = {
            'positive': (posts_df['sentiment'] == 'positive').sum(),
            'negative': (posts_df['sentiment'] == 'negative').sum(),
            'neutral': (posts_df['sentiment'] == 'neutral').sum(),
            'avg_polarity': posts_df['polarity'].mean()
        }
        
        # استخراج المواضيع
        if 'text' in posts_df.columns:
            topics = self.extract_topics(posts_df['text'].tolist())
            results['topics'] = topics
        
        # تحليل الاتجاهات
        trends = self.analyze_trends(posts_df)
        results['trends'] = trends
        
        # تحليل الشبكة إذا وجدت بيانات المستخدمين
        if 'user' in posts_df.columns:
            network_metrics = self.build_influence_network(posts_df)
            results['network_analysis'] = network_metrics
        
        # التنبؤ بالاتجاهات
        predictions = self.predict_trends(posts_df)
        results['predictions'] = predictions
        
        return results
    
    def generate_report(self, analysis_results: dict, output_format: str = 'json'):
        """توليد تقرير شامل"""
        report = {
            'executive_summary': {
                'total_content_analyzed': analysis_results['metadata']['total_posts'],
                'sentiment_distribution': analysis_results['sentiment_analysis'],
                'top_topics': [t['keywords'][:3] for t in analysis_results.get('topics', [])],
                'engagement_rate': analysis_results.get('trends', {}).get('engagement_metrics', {}).get('engagement_rate', 0),
                'peak_activity_time': analysis_results.get('trends', {}).get('peak_hours', {}).get('most_active_hour', 'N/A')
            },
            'detailed_analysis': analysis_results,
            'recommendations': self.generate_recommendations(analysis_results),
            'generated_at': datetime.now().isoformat()
        }
        
        if output_format == 'json':
            with open('social_media_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        elif output_format == 'html':
            self.generate_html_report(report)
        
        return report
    
    def generate_recommendations(self, analysis: dict) -> list:
        """توليد توصيات بناءً على التحليل"""
        recommendations = []
        
        # توصيات بناءً على المشاعر
        sentiment = analysis.get('sentiment_analysis', {})
        if sentiment.get('negative', 0) > sentiment.get('positive', 0):
            recommendations.append({
                'type': 'sentiment',
                'priority': 'high',
                'message': 'نسبة المشاعر السلبية مرتفعة - ينصح بمراجعة المحتوى وتحسين جودة التفاعل'
            })
        
        # توصيات بناءً على أوقات النشر
        peak_hour = analysis.get('trends', {}).get('peak_hours', {}).get('most_active_hour')
        if peak_hour:
            recommendations.append({
                'type': 'timing',
                'priority': 'medium',
                'message': f'أفضل وقت للنشر هو الساعة {peak_hour} - ينصح بجدولة المنشورات في هذا التوقيت'
            })
        
        # توصيات بناءً على المواضيع الرائجة
        trending_keywords = analysis.get('trends', {}).get('trending_keywords', [])
        if trending_keywords:
            top_keywords = [k['word'] for k in trending_keywords[:5]]
            recommendations.append({
                'type': 'content',
                'priority': 'high',
                'message': f'الكلمات المفتاحية الرائجة: {", ".join(top_keywords)} - ينصح بالتركيز عليها'
            })
        
        return recommendations
    
    def generate_html_report(self, report_data: dict):
        """توليد تقرير HTML تفاعلي"""
        html_template = """
        <!DOCTYPE html>
        <html dir="rtl">
        <head>
            <meta charset="UTF-8">
            <title>تقرير تحليل وسائل التواصل الاجتماعي</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .summary { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .metric-label { color: #7f8c8d; }
                .recommendation { background: #f1c40f; color: #2c3e50; padding: 15px; margin: 10px 0; border-radius: 8px; }
                .high { background: #e74c3c; color: white; }
                .medium { background: #f39c12; color: white; }
                table { width: 100%%; border-collapse: collapse; margin: 20px 0; }
                th { background: #3498db; color: white; padding: 10px; }
                td { padding: 10px; border-bottom: 1px solid #bdc3c7; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 تقرير تحليل وسائل التواصل الاجتماعي</h1>
                <p>تاريخ التقرير: {date}</p>
                
                <div class="summary">
                    <h2>ملخص تنفيذي</h2>
                    <div class="metric">
                        <div class="metric-value">{total_posts}</div>
                        <div class="metric-label">إجمالي المنشورات</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{positive_percent:.1f}%%</div>
                        <div class="metric-label">مشاعر إيجابية</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{engagement_rate:.1f}</div>
                        <div class="metric-label">معدل التفاعل</div>
                    </div>
                </div>
                
                <h2>📈 تحليل المشاعر</h2>
                <div style="display: flex; justify-content: space-around;">
                    <div style="color: green;">إيجابي: {positive}</div>
                    <div style="color: red;">سلبي: {negative}</div>
                    <div style="color: gray;">محايد: {neutral}</div>
                </div>
                
                <h2>💡 التوصيات</h2>
                <div id="recommendations">
                    {recommendations}
                </div>
                
                <h2>🔝 الكلمات المفتاحية الرائجة</h2>
                <table>
                    <tr>
                        <th>الكلمة</th>
                        <th>التكرار</th>
                    </tr>
                    {keywords_table}
                </table>
                
                <h2>📅 أوقات الذروة</h2>
                <div>أفضل ساعة للنشر: {peak_hour}</div>
                <div>أفضل يوم للنشر: {peak_day}</div>
                
                <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
                    تم إنشاء هذا التقرير بواسطة نظام تحليل وسائل التواصل الاجتماعي المتقدم
                </div>
            </div>
        </body>
        </html>
        """
        
        # تنسيق البيانات
        summary = report_data['executive_summary']
        sentiment = summary['sentiment_distribution']
        
        # إنشاء جدول الكلمات المفتاحية
        keywords_table = ""
        for keyword in report_data.get('detailed_analysis', {}).get('trends', {}).get('trending_keywords', [])[:10]:
            keywords_table += f"<tr><td>{keyword['word']}</td><td>{keyword['count']}</td></tr>"
        
        # إنشاء التوصيات
        recommendations_html = ""
        for rec in report_data.get('recommendations', []):
            rec_class = rec.get('priority', 'medium')
            recommendations_html += f'<div class="recommendation {rec_class}">⚠️ {rec["message"]}</div>'
        
        html_content = html_template.format(
            date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            total_posts=summary['total_content_analyzed'],
            positive_percent=(sentiment['positive'] / summary['total_content_analyzed'] * 100) if summary['total_content_analyzed'] > 0 else 0,
            engagement_rate=summary['engagement_rate'],
            positive=sentiment['positive'],
            negative=sentiment['negative'],
            neutral=sentiment['neutral'],
            recommendations=recommendations_html,
            keywords_table=keywords_table,
            peak_hour=summary['peak_activity_time'],
            peak_day='الأحد'  # يمكن تحسين هذا
        )
        
        with open('social_media_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

# main.py
def main():
    # إنشاء المحلل
    analyzer = SocialMediaAnalyzer()
    
    print("=== نظام تحليل وسائل التواصل الاجتماعي المتقدم ===\n")
    
    # 1. سحب بيانات تجريبية (يمكن استبدالها ببيانات حقيقية)
    print("1. سحب بيانات من تويتر...")
    sample_data = {
        'text': [
            'أحب هذا المنتج الرائع! #تقنية',
            'خدمة سيئة جداً لا أنصح بها',
            'الجو جميل اليوم في الرياض',
            'أخبار مثيرة عن التقنية الحديثة',
            'استثمار ذكي في المستقبل'
        ],
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
        'user': ['user1', 'user2', 'user3', 'user4', 'user5'],
        'likes': [100, 5, 50, 200, 75],
        'retweets': [20, 1, 10, 45, 15]
    }
    
    df = pd.DataFrame(sample_data)
    
    # 2. تنظيف وتحليل النصوص
    print("\n2. تنظيف وتحليل النصوص...")
    df['cleaned_text'] = df['text'].apply(analyzer.clean_text)
    df['tokens'] = df['cleaned_text'].apply(analyzer.tokenize_and_lemmatize)
    
    # 3. تحليل المشاعر
    print("\n3. تحليل المشاعر...")
    sentiments = df['text'].apply(analyzer.analyze_sentiment)
    df['sentiment'] = [s['sentiment'] for s in sentiments]
    df['polarity'] = [s['polarity'] for s in sentiments]
    
    print("توزيع المشاعر:")
    print(df['sentiment'].value_counts())
    
    # 4. استخراج المواضيع
    print("\n4. استخراج المواضيع الرئيسية...")
    topics = analyzer.extract_topics(df['text'].tolist())
    for topic in topics:
        print(f"الموضوع {topic['topic_id']}: {', '.join(topic['keywords'][:5])}")
    
    # 5. تحليل الاتجاهات
    print("\n5. تحليل الاتجاهات...")
    trends = analyzer.analyze_trends(df)
    print(f"معدل التفاعل: {trends['engagement_metrics']['engagement_rate']:.2f}")
    
    # 6. إنشاء تصورات
    print("\n6. إنشاء تصورات بيانية...")
    analyzer.plot_sentiment_trend(df)
    analyzer.plot_engagement_heatmap(df)
    analyzer.create_wordcloud(df['text'].tolist())
    
    # 7. تحليل شامل
    print("\n7. إجراء تحليل شامل...")
    results = analyzer.analyze_posts(df)
    
    # 8. توليد التقرير
    print("\n8. توليد التقرير النهائي...")
    report = analyzer.generate_report(results, 'html')
    
    print("\n✓ تم إكمال التحليل بنجاح!")
    print("تم إنشاء الملفات التالية:")
    print("- social_media_report.html (التقرير التفاعلي)")
    print("- social_media_report.json (البيانات الخام)")
    print("- sentiment_trend.png (رسم اتجاه المشاعر)")
    print("- engagement_heatmap.png (خريطة التفاعل)")
    print("- wordcloud_generated.png (سحابة الكلمات)")

if __name__ == "__main__":
    main()