"""
Example Flask API with standardized logging and OpenTelemetry integration
"""

import os
import time
import random
import json
from flask import Flask, request, jsonify
import logging
import uuid
import requests

# Import our custom logging configuration
import sys
sys.path.append("../../")  # Add parent directory to path
from api_examples.opentelemetry.tracing import setup_opentelemetry

# Create Flask app
app = Flask(__name__)

# Configure environment
SERVICE_NAME = os.environ.get("SERVICE_NAME", "example-api-service")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "on_premises")  # on_premises, aws_cloud, azure_cloud

# Set up OpenTelemetry
tracer, inject_headers = setup_opentelemetry(app, SERVICE_NAME, ENVIRONMENT)

# Configure logging
class APILogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.hostname = os.environ.get('HOSTNAME', 'localhost')
        
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "service": SERVICE_NAME,
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "environment": ENVIRONMENT,
            "host": self.hostname
        }
        
        # Add request_id if it exists
        request_id = getattr(record, 'request_id', None)
        if request_id:
            log_record["request_id"] = request_id
            
        # Add API call details if they exist
        api_details = getattr(record, 'api_details', {})
        if api_details:
            log_record["api_details"] = api_details
            
        # Add exception info if it exists
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

# Set up logger
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(APILogFormatter())
logger.addHandler(handler)

# Request logging middleware
@app.before_request
def before_request():
    # Generate or extract request ID
    request_id = request.headers.get('X-Request-ID')
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # Save to request context
    request.request_id = request_id
    request.start_time = time.time()
    
    # Log request beginning
    logger.info("API request started", 
               extra={
                   'request_id': request_id,
                   'api_details': {
                       'method': request.method,
                       'endpoint': request.path,
                       'source_ip': request.remote_addr
                   }
               })

@app.after_request
def after_request(response):
    # Calculate request duration
    duration = time.time() - request.start_time
    
    # Add request ID to response headers
    response.headers['X-Request-ID'] = request.request_id
    
    # Log request completion
    logger.info("API request completed",
               extra={
                   'request_id': request.request_id,
                   'api_details': {
                       'method': request.method,
                       'endpoint': request.path,
                       'status_code': response.status_code,
                       'duration_ms': int(duration * 1000),
                       'response_size': len(response.get_data(as_text=True))
                   }
               })
    
    return response

# Sample API endpoints
@app.route('/api/users', methods=['GET'])
def get_users():
    """Get a list of users"""
    # Simulate processing time
    time.sleep(random.uniform(0.05, 0.2))
    
    users = [
        {"id": 1, "name": "John Doe", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
        {"id": 3, "name": "Bob Johnson", "email": "bob@example.com"}
    ]
    
    # Occasionally introduce a slow response
    if random.random() < 0.05:  # 5% chance
        time.sleep(random.uniform(1.0, 3.0))
    
    return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a specific user by ID"""
    # Simulate processing time
    time.sleep(random.uniform(0.05, 0.1))
    
    # Simulate not found for some IDs
    if user_id > 10:
        return jsonify({"error": "User not found"}), 404
    
    user = {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
    return jsonify(user)

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.3))
    
    # Validate request
    if not request.json or 'name' not in request.json:
        return jsonify({"error": "Invalid request"}), 400
    
    # Create user (simulated)
    user = {
        "id": random.randint(100, 999),
        "name": request.json['name'],
        "email": request.json.get('email', f"{request.json['name'].lower().replace(' ', '.')}@example.com")
    }
    
    # Simulate occasional server error
    if random.random() < 0.02:  # 2% chance
        return jsonify({"error": "Internal server error"}), 500
    
    return jsonify(user), 201

@app.route('/api/users/authenticate', methods=['POST'])
def authenticate():
    """Authenticate a user"""
    # Simulate processing time
    time.sleep(random.uniform(0.2, 0.5))
    
    # Validate request
    if not request.json or 'email' not in request.json or 'password' not in request.json:
        return jsonify({"error": "Invalid credentials"}), 400
    
    # Simulate authentication (50% success rate)
    if random.random() < 0.5:
        return jsonify({
            "token": str(uuid.uuid4()),
            "expires_in": 3600
        })
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/external-service', methods=['GET'])
def call_external_service():
    """Call an external service (demonstrates distributed tracing)"""
    with tracer.start_as_current_span("external_service_call"):
        # Inject tracing headers
        headers = inject_headers()
        
        try:
            # Make request to another service (this will fail but shows the code)
            response = requests.get(
                "http://product-service:8000/api/products",
                headers=headers,
                timeout=2
            )
            return jsonify(response.json())
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling external service: {str(e)}", 
                        extra={'request_id': request.request_id})
            return jsonify({"error": "External service unavailable"}), 503

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)