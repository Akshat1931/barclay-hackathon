"""
OpenTelemetry integration for API services.
This helps track distributed traces across different environments.
"""

import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.context.context import Context
from flask import request, Flask

def setup_opentelemetry(app, service_name, environment):
    """
    Set up OpenTelemetry for a Flask application
    
    Args:
        app: Flask application
        service_name: Name of the service
        environment: Environment (on_premises, aws_cloud, azure_cloud)
    """
    # Create a resource with service info
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        "environment": environment
    })
    
    # Create a TracerProvider with the resource
    tracer_provider = TracerProvider(resource=resource)
    
    # Set up the OTLP exporter to send spans to Jaeger
    jaeger_host = os.environ.get("JAEGER_HOST", "jaeger")
    jaeger_port = os.environ.get("JAEGER_PORT", "4317")
    otlp_exporter = OTLPSpanExporter(endpoint=f"{jaeger_host}:{jaeger_port}")
    
    # Add the exporter to the TracerProvider
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the TracerProvider as the global tracer provider
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
        carrier = {}
        for key, value in request.headers.items():
            carrier[key] = value
        
        # Extract the context from the carrier
        context = TraceContextTextMapPropagator().extract(carrier=carrier)
        
        # Set the current context
        if context:
            # Create a span with the extracted context as parent
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
            
        # Get the current context
        context = Context()
        
        # Inject the current context into the headers
        TraceContextTextMapPropagator().inject(carrier=headers, context=context)
        
        return headers
    
    # Return the tracer and header injector
    return tracer, inject_headers

# Example usage in a Flask app
if __name__ == "__main__":
    app = Flask(__name__)
    tracer, inject_headers = setup_opentelemetry(app, "example-service", "on_premises")
    
    @app.route("/api/example")
    def example():
        # Start a new span
        with tracer.start_as_current_span("example_operation"):
            # Add attributes to the span
            current_span = trace.get_current_span()
            current_span.set_attribute("custom.attribute", "value")
            
            # Make an outgoing request with tracing headers
            import requests
            headers = inject_headers()
            response = requests.get("https://api.example.com", headers=headers)
            
            return {"status": "ok"}
    
    app.run(host="0.0.0.0", port=8000)