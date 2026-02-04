from flask import Blueprint, request, jsonify, current_app
import os
import uuid
from app.services.food_detection_service import FoodDetectionService

food_bp = Blueprint('food', __name__)
detector = FoodDetectionService()

@food_bp.route('/detect', methods=['POST'])
def detect_food():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': '请上传图片文件',
                'code': 400
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': '未选择文件',
                'code': 400
            }), 400
        
        # 生成唯一文件名
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        
        # 保存文件
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, unique_filename)
        file.save(filepath)
        
        print(f"📁 文件保存: {filepath}")
        
        # 检测
        result = detector.detect_from_file(filepath)
        # 适配前端 analysis 页面，返回所有识别结果
        if result['success'] and result['detections']:
            nutrition_map = {
                '香蕉': {'calories': 89, 'protein': 1.1, 'carbs': 22.8, 'fat': 0.3},
                '苹果': {'calories': 52, 'protein': 0.3, 'carbs': 13.8, 'fat': 0.2},
                '三明治': {'calories': 250, 'protein': 8, 'carbs': 30, 'fat': 9},
                '橙子': {'calories': 47, 'protein': 0.9, 'carbs': 11.8, 'fat': 0.1},
                '西兰花': {'calories': 34, 'protein': 2.8, 'carbs': 6.6, 'fat': 0.4},
                '胡萝卜': {'calories': 41, 'protein': 0.9, 'carbs': 9.6, 'fat': 0.2},
                '热狗': {'calories': 290, 'protein': 10, 'carbs': 23, 'fat': 18},
                '披萨': {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10},
                '蛋糕': {'calories': 257, 'protein': 3.8, 'carbs': 38.2, 'fat': 9.9}
            }
            data = []
            for food in result['detections']:
                food_name = food.get('chinese_name', food.get('name', '未知'))
                nutrition = nutrition_map.get(food_name, {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0})
                data.append({
                    'foodName': food_name,
                    'calories': nutrition['calories'],
                    'protein': nutrition['protein'],
                    'carbs': nutrition['carbs'],
                    'fat': nutrition['fat']
                })
            return jsonify({
                'success': True,
                'data': data,
                'filename': unique_filename,
                'code': 200
            })
        else:
            return jsonify({
                'success': False,
                'message': '未识别到食物',
                'code': 400
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'错误: {str(e)}',
            'code': 500
        }), 500


# 推荐接口
@food_bp.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    goal = data.get('goal')
    bmi = data.get('bmi')
    height = data.get('height')
    weight = data.get('weight')
    # 可根据 goal、bmi、height、weight 进行更复杂的推荐逻辑
    recommendations = {
        'gain': [
            "增肌期间建议每日摄入热量为体重的40大卡/公斤。早餐：3个鸡蛋+全麦面包+牛奶；午餐：鸡胸肉150g+糙米饭+蔬菜；加餐：蛋白棒或坚果；晚餐：鱼肉200g+红薯+绿叶菜。保证每天蛋白质摄入量达到2g/公斤体重。",
            "增肌需要充足碳水和蛋白质。推荐：早餐燕麦粥+蛋白粉；午餐牛肉面+额外鸡胸肉；训练后香蕉+蛋白粉；晚餐三文鱼+糙米饭。每日加餐2-3次，可选择希腊酸奶、坚果等。"
        ],
        'lose': [
            "减脂期建议每日摄入热量为体重的25-30大卡/公斤。早餐：2个鸡蛋+蔬菜沙拉；午餐：鸡胸肉120g+藜麦+大量蔬菜；晚餐：清蒸鱼150g+西兰花+豆腐。避免高油高糖，多喝水，保持有氧运动。",
            "减脂饮食要低卡高蛋白。推荐：早餐蔬菜蛋白饼；午餐虾仁炒蔬菜+少量糙米饭；晚餐鸡胸肉沙拉。加餐可选择苹果、黄瓜等低糖水果蔬菜。控制晚餐碳水摄入。"
        ],
        'maintain': [
            "维持期饮食要均衡多样。早餐：全麦面包+鸡蛋+水果；午餐：鱼肉/鸡肉+杂粮饭+多种蔬菜；晚餐：豆腐+蔬菜+少量主食。每天保证12种以上食物，适量运动保持代谢。",
            "健康维持需要均衡营养。推荐五颜六色的餐盘：1/2蔬菜+1/4蛋白质+1/4主食。多吃粗粮、豆制品，适量坚果，少盐少油烹饪。保持饮食规律和适度运动。"
        ]
    }
    import random
    recs = recommendations.get(goal, ["暂无建议"])
    recommendation = random.choice(recs)
    return jsonify({
        'success': True,
        'data': {
            'recommendation': recommendation
        }
    })

@food_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'food-detection',
        'model': 'YOLOv8'
    })