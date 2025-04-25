"""
Example Flask API with standardized logging and OpenTelemetry integration
"""

import sys
import os
import time
import random
import json
import uuid
import logging

import requests
from flask import Flask, request, jsonify

# Comprehensive import workaround for importlib_metadata
try:
    # Try Python 3.8+ standard library first
    from importlib import metadata as importlib_metadata
except ImportError:
    try:
        # Try the backport package
        import importlib_metadata
    except ImportError:
        # If all else fails, create a dummy module
        class DummyMetadata:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        importlib_metadata = DummyMetadata()

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Minimal OpenTelemetry configuration
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.propagate import extract, inject

# Create Flask app
app = Flask(__name__)

# Configure environment
SERVICE_NAME = os.environ.get("SERVICE_NAME", "example-api-service")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "on_premises")  # on_premises, aws_cloud, azure_cloud

# Set up OpenTelemetry
def setup_opentelemetry(app, service_name, environment):
    """
    Set up OpenTelemetry for a Flask application
    """
    # Create a resource with service info
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        "environment": environment
    })
    
    # Create a TracerProvider with the resource
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Get a tracer
    tracer = trace.get_tracer(service_name)
    
    # Instrument Flask
    FlaskInstrumentor().instrument_app(app)
    
    # Instrument requests library
    RequestsInstrumentor().instrument()
    
    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    # Add middleware for cross-service tracing
    @app.before_request
    def before_request():
        # Extract the context from request headers
        context = extract(request.headers)
        
        # Start a span with the extracted context
        with tracer.start_as_current_span("handle_request", context=context):
            # Set attributes on the span
            current_span = trace.get_current_span()
            current_span.set_attribute("http.method", request.method)
            current_span.set_attribute("http.url", request.url)
            current_span.set_attribute("environment", environment)
    
    # Add middleware for outgoing requests
    def inject_headers(headers=None):
        """Inject tracing headers into outgoing requests"""
        if headers is None:
            headers = {}
        
        # Inject the current context into the headers
        inject(headers)
        
        return headers
    
    # Return the tracer and header injector
    return tracer, inject_headers

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
def log_request_start():
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
def log_request_end(response):
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
    with tracer.start_as_current_span("get_users") as span:
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
            span.set_status(Status(StatusCode.ERROR, "Slow response"))
        
        return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a specific user by ID"""
    with tracer.start_as_current_span("get_user") as span:
        # Simulate processing time
        time.sleep(random.uniform(0.05, 0.1))
        
        # Simulate not found for some IDs
        if user_id > 10:
            span.set_status(Status(StatusCode.ERROR, "User not found"))
            return jsonify({"error": "User not found"}), 404
        
        user = {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
        return jsonify(user)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)