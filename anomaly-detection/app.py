import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Elasticsearch connection
ES_HOST = os.environ.get('ES_HOST', 'elasticsearch')
ES_PORT = os.environ.get('ES_PORT', '9200')
es = Elasticsearch([f'http://{ES_HOST}:{ES_PORT}'])

# Configuration
ANALYSIS_INTERVAL = 300  # 5 minutes
HISTORICAL_WINDOW = 24   # 24 hours
MAX_SAMPLES = 10000      # Max number of data points to process
ANOMALY_THRESHOLD = 0.01  # 1% threshold for anomalies

class AnomalyDetector:
    def __init__(self):
        self.models = {
            'response_time': None,
            'error_rate': None,
            'status_codes': None
        }
        
    def fetch_data(self, hours=HISTORICAL_WINDOW):
        """Fetch API logs from Elasticsearch within a time window"""
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)
        
        query = {
            "size": MAX_SAMPLES,
            "sort": [{"@timestamp": {"order": "asc"}}],
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": start_time.isoformat(), "lte": now.isoformat()}}},
                        {"exists": {"field": "response_time"}}
                    ]
                }
            },
            "_source": ["@timestamp", "service", "endpoint", "status_code", "response_time", 
                      "environment", "request_id", "environment_type"]
        }
        
        try:
            logger.info(f"Fetching data from Elasticsearch for the last {hours} hours")
            result = es.search(index="api-logs-*", body=query)
            hits = result['hits']['hits']
            
            if not hits:
                logger.warning("No data found in Elasticsearch")
                return pd.DataFrame()
                
            # Process the results into a DataFrame
            data = []
            for hit in hits:
                source = hit['_source']
                data.append({
                    'timestamp': source.get('@timestamp'),
                    'service': source.get('service'),
                    'endpoint': source.get('endpoint'),
                    'status_code': source.get('status_code'),
                    'response_time': source.get('response_time'),
                    'environment': source.get('environment'),
                    'request_id': source.get('request_id'),
                    'environment_type': source.get('environment_type')
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['is_error'] = df['status_code'].apply(lambda x: 1 if x >= 400 else 0)
            
            logger.info(f"Fetched {len(df)} records from Elasticsearch")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Elasticsearch: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        """Preprocess the data for anomaly detection"""
        if df.empty:
            return None, None, None
            
        # Group by service and endpoint
        grouped = df.groupby(['service', 'endpoint'])
        
        # Extract features for anomaly detection
        response_time_data = []
        error_rate_data = []
        status_code_data = []
        
        for (service, endpoint), group in grouped:
            # Skip if too few data points
            if len(group) < 10:
                continue
                
            # Calculate statistics per service/endpoint
            avg_response_time = group['response_time'].mean()
            p95_response_time = group['response_time'].quantile(0.95)
            error_rate = group['is_error'].mean()
            status_codes = group['status_code'].value_counts().to_dict()
            
            # Create feature vectors
            response_time_data.append({
                'service': service,
                'endpoint': endpoint,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'count': len(group)
            })
            
            error_rate_data.append({
                'service': service,
                'endpoint': endpoint,
                'error_rate': error_rate,
                'count': len(group)
            })
            
            status_code_data.append({
                'service': service,
                'endpoint': endpoint,
                'status_codes': status_codes,
                'count': len(group)
            })
        
        response_time_df = pd.DataFrame(response_time_data)
        error_rate_df = pd.DataFrame(error_rate_data)
        
        return response_time_df, error_rate_df, status_code_data

    def train_models(self):
        """Train anomaly detection models"""
        # Fetch historical data
        df = self.fetch_data(hours=HISTORICAL_WINDOW)
        if df.empty:
            logger.warning("No data available for training models")
            return
            
        # Preprocess the data
        response_time_df, error_rate_df, status_code_data = self.preprocess_data(df)
        
        if response_time_df is None or len(response_time_df) < 5:
            logger.warning("Insufficient data for training models")
            return
            
        # Train response time model
        try:
            X_rt = response_time_df[['avg_response_time', 'p95_response_time']].values
            scaler_rt = StandardScaler()
            X_rt_scaled = scaler_rt.fit_transform(X_rt)
            
            # Use IsolationForest for response time anomalies
            model_rt = IsolationForest(contamination=ANOMALY_THRESHOLD, random_state=42)
            model_rt.fit(X_rt_scaled)
            
            self.models['response_time'] = {
                'model': model_rt,
                'scaler': scaler_rt,
                'features': ['avg_response_time', 'p95_response_time']
            }
            logger.info("Response time model trained successfully")
        except Exception as e:
            logger.error(f"Error training response time model: {e}")
            
        # Train error rate model
        try:
            X_er = error_rate_df[['error_rate']].values
            scaler_er = StandardScaler()
            X_er_scaled = scaler_er.fit_transform(X_er)
            
            # Use Local Outlier Factor for error rate anomalies
            model_er = LocalOutlierFactor(n_neighbors=5, contamination=ANOMALY_THRESHOLD)
            model_er.fit(X_er_scaled)
            
            self.models['error_rate'] = {
                'model': model_er,
                'scaler': scaler_er,
                'features': ['error_rate']
            }
            logger.info("Error rate model trained successfully")
        except Exception as e:
            logger.error(f"Error training error rate model: {e}")

    def detect_anomalies(self):
        """Detect anomalies in recent data"""
        # Fetch recent data (last 5 minutes)
        recent_df = self.fetch_data(hours=0.1)  # ~6 minutes
        if recent_df.empty:
            logger.warning("No recent data available for anomaly detection")
            return []
            
        # Preprocess the data
        response_time_df, error_rate_df, status_code_data = self.preprocess_data(recent_df)
        
        if response_time_df is None or len(response_time_df) < 1:
            logger.warning("Insufficient recent data for anomaly detection")
            return []
            
        anomalies = []
        
        # Detect response time anomalies
        if self.models['response_time'] is not None:
            try:
                model_info = self.models['response_time']
                X_rt = response_time_df[model_info['features']].values
                X_rt_scaled = model_info['scaler'].transform(X_rt)
                
                # Predict anomalies (-1 for anomalies, 1 for normal)
                predictions = model_info['model'].predict(X_rt_scaled)
                
                # Find anomalies
                for i, pred in enumerate(predictions):
                    if pred == -1:  # Anomaly
                        service = response_time_df.iloc[i]['service']
                        endpoint = response_time_df.iloc[i]['endpoint']
                        avg_rt = response_time_df.iloc[i]['avg_response_time']
                        p95_rt = response_time_df.iloc[i]['p95_response_time']
                        
                        anomalies.append({
                            'type': 'response_time',
                            'service': service,
                            'endpoint': endpoint,
                            'avg_response_time': float(avg_rt),
                            'p95_response_time': float(p95_rt),
                            'timestamp': datetime.utcnow().isoformat(),
                            'severity': 'high' if avg_rt > 1000 else 'medium'
                        })
            except Exception as e:
                logger.error(f"Error detecting response time anomalies: {e}")
                
        # Detect error rate anomalies
        if self.models['error_rate'] is not None:
            try:
                model_info = self.models['error_rate']
                X_er = error_rate_df[model_info['features']].values
                X_er_scaled = model_info['scaler'].transform(X_er)
                
                # Predict anomalies (-1 for anomalies, 1 for normal)
                # LOF returns negative anomaly scores for outliers
                scores = model_info['model'].negative_outlier_factor_
                predictions = np.where(scores < -1, -1, 1)  # Convert scores to binary predictions
                
                # Find anomalies
                for i, pred in enumerate(predictions):
                    if pred == -1:  # Anomaly
                        service = error_rate_df.iloc[i]['service']
                        endpoint = error_rate_df.iloc[i]['endpoint']
                        error_rate = error_rate_df.iloc[i]['error_rate']
                        
                        anomalies.append({
                            'type': 'error_rate',
                            'service': service,
                            'endpoint': endpoint,
                            'error_rate': float(error_rate),
                            'timestamp': datetime.utcnow().isoformat(),
                            'severity': 'high' if error_rate > 0.1 else 'medium'
                        })
            except Exception as e:
                logger.error(f"Error detecting error rate anomalies: {e}")
        
        return anomalies

    def send_alerts(self, anomalies):
        """Send alerts to Elasticsearch for visualization and notification"""
        if not anomalies:
            return
            
        # Index anomalies in Elasticsearch
        for anomaly in anomalies:
            try:
                es.index(index='api-anomalies', document=anomaly)
                logger.info(f"Alert sent: {anomaly['type']} anomaly for {anomaly['service']}/{anomaly['endpoint']}")
            except Exception as e:
                logger.error(f"Error sending alert to Elasticsearch: {e}")

    def run(self):
        """Main execution loop"""
        logger.info("Starting API anomaly detection service")
        
        # Initial model training
        self.train_models()
        
        # Continuous monitoring loop
        while True:
            try:
                # Detect anomalies
                anomalies = self.detect_anomalies()
                
                # Send alerts
                if anomalies:
                    logger.info(f"Detected {len(anomalies)} anomalies")
                    self.send_alerts(anomalies)
                else:
                    logger.info("No anomalies detected")
                
                # Retrain models periodically (every 6 hours)
                if datetime.utcnow().hour % 6 == 0 and datetime.utcnow().minute < 5:
                    logger.info("Retraining anomaly detection models")
                    self.train_models()
                
                # Wait for the next analysis interval
                time.sleep(ANALYSIS_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.run()