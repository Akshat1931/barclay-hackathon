#!/usr/bin/env python3
"""
API Traffic Generator Script
Generates synthetic API traffic for testing the monitoring system
"""

import requests
import time
import random
import json
import uuid
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API endpoints
SERVICES = {
    "user-service": {
        "endpoints": [
            "/api/users", 
            "/api/users/{id}", 
            "/api/users/authenticate"
        ],
        "environment": "on_premises"
    },
    "product-service": {
        "endpoints": [
            "/api/products", 
            "/api/products/{id}", 
            "/api/products/search"
        ],
        "environment": "aws_cloud"
    },
    "payment-service": {
        "endpoints": [
            "/api/payments", 
            "/api/payments/{id}", 
            "/api/payments/process"
        ],
        "environment": "azure_cloud"
    },
    "notification-service": {
        "endpoints": [
            "/api/notifications", 
            "/api/notifications/send", 
            "/api/notifications/status"
        ],
        "environment": "on_premises"
    }
}

# Configure base URLs for each environment
BASE_URLS = {
    "on_premises": "http://localhost:8000",
    "aws_cloud": "http://localhost:8001",
    "azure_cloud": "http://localhost:8002"
}

# HTTP methods with their relative frequencies
HTTP_METHODS = {
    "GET": 0.7,
    "POST": 0.2, 
    "PUT": 0.05,
    "DELETE": 0.05
}

def generate_request_id():
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def get_random_method():
    """Get a random HTTP method based on configured frequencies"""
    methods, weights = zip(*HTTP_METHODS.items())
    return random.choices(methods, weights=weights, k=1)[0]

def send_request(service_name, endpoint, method, base_url, request_id=None):
    """Send a request to an API endpoint and log the details to Logstash"""
    # Generate a request ID if not provided
    if not request_id:
        request_id = generate_request_id()
        
    # Replace any path parameters
    if "{id}" in endpoint:
        endpoint = endpoint.replace("{id}", str(random.randint(1, 1000)))
        
    # Construct the URL
    url = f"{base_url}{endpoint}"
    
    # Prepare headers
    headers = {
        "X-Request-ID": request_id,
        "Content-Type": "application/json"
    }
    
    # Prepare payload for POST/PUT requests
    payload = None
    if method in ["POST", "PUT"]:
        payload = json.dumps({"timestamp": datetime.now().isoformat(), "data": f"Sample {method} data"})
    
    # Add artificial delay for some requests to simulate slow responses
    slow_response = random.random() < 0.05  # 5% chance of slow response
    
    # Add artificial errors for some requests
    error_response = random.random() < 0.03  # 3% chance of error response
    
    start_time = time.time()
    
    try:
        # Simulate a slow response
        if slow_response:
            time.sleep(random.uniform(1.0, 3.0))
            
        # Simulate an error response
        if error_response and method != "GET":
            # Log directly to Logstash for simulated error
            log_to_logstash(service_name, endpoint, method, request_id, 500, (time.time() - start_time) * 1000, 
                         SERVICES[service_name]["environment"], {"error": "Internal Server Error"})
            logger.info(f"Simulated error for {method} {url}")
            return
            
        # Send the actual request
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=payload,
            timeout=5
        )
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # in milliseconds
        
        # Log the request details to Logstash
        log_to_logstash(service_name, endpoint, method, request_id, response.status_code, response_time, 
                     SERVICES[service_name]["environment"], response.text[:100])
        
        logger.debug(f"{method} {url} - {response.status_code} - {response_time:.2f}ms")
        
    except requests.exceptions.RequestException as e:
        # Log the error to Logstash
        response_time = (time.time() - start_time) * 1000
        log_to_logstash(service_name, endpoint, method, request_id, 500, response_time, 
                     SERVICES[service_name]["environment"], {"error": str(e)})
        logger.error(f"Error sending request to {url}: {e}")

def log_to_logstash(service, endpoint, method, request_id, status_code, response_time, environment, response_body):
    """Send log data to Logstash via HTTP input"""
    logstash_url = "http://localhost:8080"  # Update with your Logstash HTTP input URL
    
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "service": service,
        "level": "INFO",
        "message": f"API request completed",
        "logger": "traffic-generator",
        "environment": environment,
        "host": f"{environment}-host",
        "request_id": request_id,
        "api_details": {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": response_time,
            "response_size": len(str(response_body)) if response_body else 0
        }
    }
    
    try:
        requests.post(logstash_url, json=log_data, timeout=2)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send log to Logstash: {e}")

def generate_distributed_trace(num_services=3):
    """Generate a distributed trace across multiple services"""
    # Select random services to include in the trace
    selected_services = random.sample(list(SERVICES.keys()), min(num_services, len(SERVICES)))
    
    # Generate a single request ID for the entire trace
    trace_id = generate_request_id()
    
    for service_name in selected_services:
        # Select a random endpoint for this service
        endpoint = random.choice(SERVICES[service_name]["endpoints"])
        
        # Get the environment for this service
        environment = SERVICES[service_name]["environment"]
        
        # Select a random HTTP method
        method = get_random_method()
        
        # Send the request as part of the trace
        send_request(service_name, endpoint, method, BASE_URLS[environment], request_id=trace_id)
        
        # Add a small delay between service calls
        time.sleep(random.uniform(0.1, 0.5))

def generate_traffic(rate, duration, distributed_trace_percentage=30):
    """Generate API traffic at the specified rate for the specified duration"""
    logger.info(f"Generating traffic at {rate} requests per second for {duration} seconds")
    
    end_time = time.time() + duration
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        while time.time() < end_time:
            start_batch = time.time()
            
            # Determine if this should be a distributed trace
            if random.random() * 100 < distributed_trace_percentage:
                # Submit a distributed trace job
                executor.submit(generate_distributed_trace, random.randint(2, 4))
            else:
                # Submit a regular single-service request job
                service_name = random.choice(list(SERVICES.keys()))
                endpoint = random.choice(SERVICES[service_name]["endpoints"])
                method = get_random_method()
                environment = SERVICES[service_name]["environment"]
                
                executor.submit(send_request, service_name, endpoint, method, BASE_URLS[environment])
            
            # Calculate sleep time to maintain the request rate
            elapsed = time.time() - start_batch
            sleep_time = max(0, 1.0 / rate - elapsed)
            time.sleep(sleep_time)
    
    logger.info("Traffic generation completed")

def generate_anomaly(service_name, endpoint, duration_seconds=300, rate=5):
    """Generate anomaly traffic for a specific endpoint"""
    logger.info(f"Generating anomaly for {service_name} - {endpoint} for {duration_seconds} seconds")
    
    environment = SERVICES[service_name]["environment"]
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        # For response time anomalies, add high latency
        if random.random() < 0.8:  # 80% of requests during anomaly will be slow
            method = get_random_method()
            
            # Submit request with built-in delay
            time.sleep(random.uniform(2.0, 5.0))  # Simulate processing delay
            send_request(service_name, endpoint, method, BASE_URLS[environment])
        else:
            # For error rate anomalies, generate errors
            method = random.choice(["POST", "PUT", "DELETE"])  # More likely to have errors
            
            # Force error simulation
            request_id = generate_request_id()
            log_to_logstash(service_name, endpoint, method, request_id, 500, 
                         random.uniform(200, 800), environment, {"error": "Simulated anomaly error"})
        
        # Sleep to maintain the anomaly rate
        time.sleep(1.0 / rate)
    
    logger.info(f"Anomaly generation for {service_name} - {endpoint} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API Traffic Generator')
    parser.add_argument('--rate', type=float, default=5.0, help='Requests per second')
    parser.add_argument('--duration', type=int, default=3600, help='Duration in seconds')
    parser.add_argument('--anomaly', action='store_true', help='Generate anomaly traffic')
    args = parser.parse_args()
    
    if args.anomaly:
        # Generate normal traffic in the background
        import threading
        normal_thread = threading.Thread(
            target=generate_traffic, 
            args=(args.rate * 0.5, args.duration)
        )
        normal_thread.daemon = True
        normal_thread.start()
        
        # Generate an anomaly for a random service/endpoint
        service = random.choice(list(SERVICES.keys()))
        endpoint = random.choice(SERVICES[service]["endpoints"])
        generate_anomaly(service, endpoint, duration_seconds=min(300, args.duration))
    else:
        # Generate normal traffic
        generate_traffic(args.rate, args.duration)