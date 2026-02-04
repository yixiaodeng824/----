from app import create_app

app = create_app()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🍎 膳食健康分析后端服务")
    print("="*50)
    print("API 端点:")
    print("  GET  /api/health      - 健康检查")
    print("  POST /api/detect      - 食物检测")
    print("\n示例请求:")
    print("  curl -X POST -F 'image=@food.jpg' http://192.168.18.66:5000/api/detect")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)