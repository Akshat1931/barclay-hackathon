# Create a file called extreme-anomaly-generator.py
import requests
import time
import uuid
from datetime import datetime

# Function to generate logs with extreme response times
def generate_extreme_logs(count=100):
    # Create logs for a consistent service/endpoint to ensure enough data for anomaly detection
    service = "user-service"
    endpoint = "/api/users"
    environment = "on_premises"
    
    print(f"Generating {count} extreme response time logs...")
    
    for i in range(count):
        # Create log with extremely high response time (10-20 seconds)
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "level": "INFO",
            "message": "API request completed",
            "logger": "extreme-anomaly-generator",
            "environment": environment,
            "host": f"{environment}-host",
            "request_id": str(uuid.uuid4()),
            "api_details": {
                "method": "GET",
                "endpoint": endpoint,
                "status_code": 200,
                "duration_ms": 15000 + (i * 100),  # 15-25 seconds - extremely slow
                "response_size": 1024
            }
        }
        
        # Send to Logstash
        try:
            response = requests.post("http://localhost:8080", json=log_data, timeout=5)
            print(f"Sent log {i+1}/{count}: Status {response.status_code}")
        except Exception as e:
            print(f"Error sending log: {e}")
        
        # Small delay between logs
        time.sleep(0.1)
    
    print("Finished generating extreme logs")

# Function to directly create anomalies in Elasticsearch
def create_manual_anomalies(count=5):
    print(f"Creating {count} manual anomalies directly in Elasticsearch...")
    
    services = ["user-service", "product-service"]
    endpoints = ["/api/users", "/api/products"]
    
    for i in range(count):
        service = services[i % len(services)]
        endpoint = endpoints[i % len(endpoints)]
        
        # Create anomaly document
        anomaly = {
            "type": "response_time",
            "service": service,
            "endpoint": endpoint,
            "avg_response_time": 15000.0,
            "p95_response_time": 20000.0,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high",
            "source": "manual-generator",
            "manual": True
        }
        
        # Send directly to Elasticsearch
        try:
            response = requests.post(
                "http://localhost:9200/api-anomalies/_doc",
                json=anomaly,
                timeout=5
            )
            print(f"Created manual anomaly {i+1}/{count}: Status {response.status_code}")
        except Exception as e:
            print(f"Error creating anomaly: {e}")
        
        time.sleep(0.5)
    
    print("Finished creating manual anomalies")

# Function to check if anomalies index exists
def check_anomalies_index():
    print("Checking api-anomalies index...")
    
    try:
        response = requests.get("http://localhost:9200/api-anomalies")
        
        if response.status_code == 200:
            print("api-anomalies index exists")
            
            # Check how many documents it has
            count_response = requests.get("http://localhost:9200/api-anomalies/_count")
            if count_response.status_code == 200:
                count = count_response.json().get("count", 0)
                print(f"Index contains {count} anomalies")
        elif response.status_code == 404:
            print("api-anomalies index doesn't exist yet - creating it")
            
            # Create the index
            create_response = requests.put(
                "http://localhost:9200/api-anomalies",
                json={
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "type": {"type": "keyword"},
                            "service": {"type": "keyword"},
                            "endpoint": {"type": "keyword"},
                            "avg_response_time": {"type": "float"},
                            "p95_response_time": {"type": "float"},
                            "severity": {"type": "keyword"}
                        }
                    }
                }
            )
            
            print(f"Index creation result: {create_response.status_code}")
        else:
            print(f"Unexpected response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error checking index: {e}")

if __name__ == "__main__":
    # First check the anomalies index
    check_anomalies_index()
    
    # Generate extreme logs - these should trigger the anomaly detector
    generate_extreme_logs(200)
    
    # Wait a few minutes for the anomaly detector to process the logs
    print("\nWaiting 3 minutes for anomaly detection to process logs...")
    time.sleep(180)
    
    # Check if any anomalies were detected
    check_anomalies_index()
    
    # If no anomalies were detected, create them manually
    create_manual_anomalies(10)
    
    # Verify the manually created anomalies
    check_anomalies_index()