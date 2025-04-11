from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hackathon_mental_health_app'
    
    # Configure file upload settings
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
    app.config['UPLOAD_EXTENSIONS'] = ['.webm', '.mp4']
    
    # Register blueprints
    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app 