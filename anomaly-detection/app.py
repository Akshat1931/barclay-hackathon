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
ANOMALY_THRESHOLD = 0.1  # 10% threshold for anomalies - more sensitive than before

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
            
            # Log sample data for debugging
            if not df.empty:
                logger.info(f"Sample data: \n{df.head(3).to_string()}")
                logger.info(f"Data types: {df.dtypes}")
            
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
            # Skip if too few data points (reduced from 10 to 3)
            if len(group) < 3:
                logger.info(f"Skipping {service}/{endpoint} - only {len(group)} data points")
                continue
                
            # Calculate statistics per service/endpoint
            avg_response_time = group['response_time'].mean()
            p95_response_time = group['response_time'].quantile(0.95)
            error_rate = group['is_error'].mean()
            status_codes = group['status_code'].value_counts().to_dict()
            
            # Log values for debugging
            logger.info(f"{service}/{endpoint}: avg_rt={avg_response_time:.2f}ms, p95={p95_response_time:.2f}ms, error_rate={error_rate:.2f}")
            
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
        
        # Log processed data for debugging
        if not response_time_df.empty:
            logger.info(f"Processed {len(response_time_df)} service/endpoint combinations for response time")
        if not error_rate_df.empty:
            logger.info(f"Processed {len(error_rate_df)} service/endpoint combinations for error rate")
        
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
        
        if response_time_df is None or len(response_time_df) < 2:
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
            model_er = LocalOutlierFactor(n_neighbors=max(2, min(5, len(X_er_scaled) - 1)), 
                                         contamination=ANOMALY_THRESHOLD)
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
                            'severity': 'high' if avg_rt > 1000 else 'medium',
                            'detector': 'ml_model'
                        })
                        logger.info(f"ML model detected response time anomaly: {service}/{endpoint} - {avg_rt}ms")
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
                            'severity': 'high' if error_rate > 0.1 else 'medium',
                            'detector': 'ml_model'
                        })
                        logger.info(f"ML model detected error rate anomaly: {service}/{endpoint} - {error_rate:.2f}")
            except Exception as e:
                logger.error(f"Error detecting error rate anomalies: {e}")
        
        return anomalies
    
    def find_direct_anomalies(self):
        """Find anomalies directly using thresholds"""
        recent_df = self.fetch_data(hours=0.1)
        if recent_df.empty:
            return []
            
        anomalies = []
        
        # Find high response times directly
        high_rt = recent_df[recent_df['response_time'] > 3000]  # 3 seconds
        
        for _, row in high_rt.iterrows():
            service = row['service']
            endpoint = row['endpoint']
            response_time = row['response_time']
            
            anomalies.append({
                'type': 'response_time',
                'service': service,
                'endpoint': endpoint,
                'avg_response_time': float(response_time),
                'p95_response_time': float(response_time * 1.2),
                'timestamp': datetime.utcnow().isoformat(),
                'severity': 'high' if response_time > 5000 else 'medium',
                'detector': 'threshold'
            })
            logger.info(f"Direct threshold detected response time anomaly: {service}/{endpoint} - {response_time}ms")
        
        # Find high error rates directly
        if not recent_df.empty and 'is_error' in recent_df.columns:
            error_groups = recent_df.groupby(['service', 'endpoint']).agg(
                error_count=('is_error', 'sum'),
                total=('is_error', 'count')
            ).reset_index()
            
            error_groups['error_rate'] = error_groups['error_count'] / error_groups['total'].where(error_groups['total'] > 0, 0)
            high_errors = error_groups[error_groups['error_rate'] > 0.2]  # 20% error rate
            
            for _, row in high_errors.iterrows():
                service = row['service']
                endpoint = row['endpoint']
                error_rate = row['error_rate']
                
                anomalies.append({
                    'type': 'error_rate',
                    'service': service,
                    'endpoint': endpoint,
                    'error_rate': float(error_rate),
                    'timestamp': datetime.utcnow().isoformat(),
                    'severity': 'high' if error_rate > 0.3 else 'medium',
                    'detector': 'threshold'
                })
                logger.info(f"Direct threshold detected error rate anomaly: {service}/{endpoint} - {error_rate:.2f}")
        
        return anomalies

    def send_alerts(self, anomalies):
        """Send alerts to Elasticsearch for visualization and notification"""
        if not anomalies:
            return
            
        # Create api-anomalies index if it doesn't exist
        try:
            if not es.indices.exists(index="api-anomalies"):
                logger.info("Creating api-anomalies index")
                es.indices.create(
                    index="api-anomalies",
                    body={
                        "mappings": {
                            "properties": {
                                "timestamp": {"type": "date"},
                                "type": {"type": "keyword"},
                                "service": {"type": "keyword"},
                                "endpoint": {"type": "keyword"},
                                "avg_response_time": {"type": "float"},
                                "p95_response_time": {"type": "float"},
                                "error_rate": {"type": "float"},
                                "severity": {"type": "keyword"},
                                "detector": {"type": "keyword"}
                            }
                        }
                    }
                )
        except Exception as e:
            logger.error(f"Error creating api-anomalies index: {e}")
            
        # Index anomalies in Elasticsearch
        for anomaly in anomalies:
            try:
                result = es.index(index='api-anomalies', document=anomaly)
                logger.info(f"Alert sent: {anomaly['type']} anomaly for {anomaly['service']}/{anomaly['endpoint']} - Result: {result['result']}")
            except Exception as e:
                logger.error(f"Error sending alert to Elasticsearch: {e}")

    def run(self):
        """Main execution loop"""
        logger.info("Starting API anomaly detection service")
        
        # Create test anomaly to verify index and connection
        test_anomaly = {
            'type': 'test',
            'service': 'test-service',
            'endpoint': '/test',
            'avg_response_time': 9999.0,
            'p95_response_time': 12000.0,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'critical',
            'detector': 'startup_test'
        }
        try:
            result = es.index(index='api-anomalies', document=test_anomaly)
            logger.info(f"Test anomaly created successfully: {result}")
        except Exception as e:
            logger.error(f"Error creating test anomaly: {e}")
        
        # Initial model training
        self.train_models()
        
        # Continuous monitoring loop
        while True:
            try:
                # Detect anomalies using ML models
                ml_anomalies = self.detect_anomalies()
                if ml_anomalies:
                    logger.info(f"Detected {len(ml_anomalies)} ML-based anomalies")
                
                # Also detect anomalies using direct thresholds
                direct_anomalies = self.find_direct_anomalies()
                if direct_anomalies:
                    logger.info(f"Found {len(direct_anomalies)} direct threshold anomalies")
                
                # Combine all anomalies
                all_anomalies = ml_anomalies + direct_anomalies
                
                # Send alerts
                if all_anomalies:
                    logger.info(f"Detected {len(all_anomalies)} total anomalies")
                    self.send_alerts(all_anomalies)
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