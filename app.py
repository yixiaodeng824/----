import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🍎 膳食健康分析后端服务")
    print("="*50)
    print("\n📋 配置检查:")
    if os.getenv("DEEPSEEK_API_KEY"):
        print("  ✅ DeepSeek API Key 已配置 — 支持联网搜索增强")
    else:
        print("  ⚠️  未设置 DEEPSEEK_API_KEY — 联网搜索功能不可用")
        print("     请创建 .env 文件（参考 .env.example）或设置环境变量")
    print("API 端点:")
    print("  GET  /api/health              - 健康检查")
    print("  POST /api/detect              - 食物检测（YOLOv8）")
    print("  POST /api/detect/deepseek     - 食物检测 + DeepSeek 联网搜索")
    print("  POST /api/detect/deepseek/stream - 同上，流式输出")
    print("  POST /api/deepseek/query       - DeepSeek 文字查询（JSON入参）")
    print("  GET  /api/report/weekly/data    - 周饮食数据")
    print("  GET  /api/report/weekly/canteen - 周食堂统计")
    print("  GET  /api/report/weekly/ai      - AI 周报")
    print("\n示例请求:")
    print("  curl -X POST -F 'image=@food.jpg' http://10.98.211.82:5000/api/detect")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 