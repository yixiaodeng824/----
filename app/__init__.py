# 使 app 成为 Python 包
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    
    # 配置
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # 启用CORS
    CORS(app)
    
    # 注册蓝图
    from app.routes.food import food_bp
    app.register_blueprint(food_bp, url_prefix='/api')
    from app.routes.user import user_bp
    app.register_blueprint(user_bp, url_prefix='/api')
    from app.routes.wxlogin import wxlogin_bp
    app.register_blueprint(wxlogin_bp, url_prefix='/api')
    return app