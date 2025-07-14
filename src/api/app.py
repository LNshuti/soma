import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask
from flask_cors import CORS
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:7860", "http://0.0.0.0:7860"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register blueprints
    try:
        from src.api.routes.health import health_bp
        from src.api.routes.predictions import predictions_bp
        from src.api.routes.recommendations import recommendations_bp
        from src.api.routes.content import content_bp
        
        app.register_blueprint(health_bp)
        app.register_blueprint(predictions_bp, url_prefix='/api')
        app.register_blueprint(recommendations_bp, url_prefix='/api')
        app.register_blueprint(content_bp, url_prefix='/api')
        
        logger.info("All blueprints registered successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import routes: {e}")
        raise
    
    return app

def main():
    """Main application entry point."""
    try:
        app = create_app()
        logger.info("Flask app created successfully")
        
        app.run(
            host=settings.API_HOST,
            port=settings.API_PORT,
            debug=settings.API_DEBUG
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()