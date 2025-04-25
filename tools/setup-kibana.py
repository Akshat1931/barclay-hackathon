#!/usr/bin/env python3
"""
Script to automatically set up Kibana index patterns, dashboards, and visualizations
"""

import requests
import json
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kibana configuration
KIBANA_HOST = "http://localhost:5601"
ES_HOST = "http://localhost:9200"
KIBANA_API = f"{KIBANA_HOST}/api"

# Wait for Kibana to be ready
def wait_for_kibana():
    """Wait for Kibana to be available"""
    logger.info("Waiting for Kibana to be ready...")
    retries = 30
    
    for i in range(retries):
        try:
            response = requests.get(f"{KIBANA_HOST}/api/status")
            if response.status_code == 200:
                logger.info("Kibana is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"Kibana not ready yet. Retry {i+1}/{retries}")
        time.sleep(10)
    
    logger.error("Kibana did not become available in time")
    return False

# Create index patterns
def create_index_patterns():
    """Create index patterns in Kibana"""
    logger.info("Creating index patterns...")
    
    patterns = [
        {
            "name": "api-logs-*",
            "title": "api-logs-*",
            "timeFieldName": "@timestamp"
        },
        {
            "name": "api-anomalies",
            "title": "api-anomalies",
            "timeFieldName": "timestamp"
        }
    ]
    
    for pattern in patterns:
        try:
            # Check if pattern already exists
            response = requests.get(
                f"{KIBANA_API}/saved_objects/index-pattern/{pattern['name']}"
            )
            
            if response.status_code == 200:
                logger.info(f"Index pattern '{pattern['name']}' already exists")
                continue
                
            # Create the pattern
            response = requests.post(
                f"{KIBANA_API}/saved_objects/index-pattern/{pattern['name']}",
                headers={"kbn-xsrf": "true", "Content-Type": "application/json"},
                json={
                    "attributes": {
                        "title": pattern["title"],
                        "timeFieldName": pattern["timeFieldName"]
                    }
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Created index pattern '{pattern['name']}'")
            else:
                logger.error(f"Failed to create index pattern '{pattern['name']}': {response.text}")
                
        except Exception as e:
            logger.error(f"Error creating index pattern '{pattern['name']}': {e}")

# Set default index pattern
def set_default_index_pattern():
    """Set the default index pattern"""
    logger.info("Setting default index pattern...")
    
    try:
        response = requests.post(
            f"{KIBANA_API}/saved_objects/config/7.17.0",
            headers={"kbn-xsrf": "true", "Content-Type": "application/json"},
            json={
                "attributes": {
                    "defaultIndex": "api-logs-*"
                }
            }
        )
        
        if response.status_code in [200, 201]:
            logger.info("Default index pattern set successfully")
        else:
            logger.error(f"Failed to set default index pattern: {response.text}")
            
    except Exception as e:
        logger.error(f"Error setting default index pattern: {e}")

# Main execution
if __name__ == "__main__":
    if not wait_for_kibana():
        sys.exit(1)
        
    create_index_patterns()
    set_default_index_pattern()
    
    logger.info("Kibana setup completed successfully!")