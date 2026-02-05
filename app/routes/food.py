from flask import Blueprint, request, jsonify, current_app
import os
import uuid
from app.services.food_detection_service import FoodDetectionService

from app.services.nutrition_service import get_nutrition_for_foods
from app.services.food_record_service import add_food_record, get_today_records, get_today_nutrition_sum


food_bp = Blueprint('food', __name__)
detector = FoodDetectionService()
# 删除进食记录接口
@food_bp.route('/record/delete', methods=['POST'])
def delete_record():
    data = request.get_json()
    record_id = data.get('id')
    if not record_id:
        return jsonify({'success': False, 'msg': '缺少记录ID'}), 400
    from app.services.food_record_service import delete_food_record
    delete_food_record(record_id)
    return jsonify({'success': True, 'msg': '删除成功'})

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
    goal = data.get('goal')  # 目标：gain/lose/maintain
    height = data.get('height')  # cm
    weight = data.get('weight')  # kg
    foods = data.get('foods', [])  # 识别到的食物名列表

    # 1. 计算推荐热量和蛋白质目标
    bmr = 24 * weight
    if goal == 'gain':
        target_cal = weight * 40
        protein_target = weight * 2
    elif goal == 'lose':
        target_cal = weight * 28
        protein_target = weight * 1.5
    else:
        target_cal = weight * 33
        protein_target = weight * 1.2

    # 2. 统计已摄入营养
    nutrition_list = get_nutrition_for_foods(foods)
    total_cal = sum(item['calories'] for item in nutrition_list)
    total_protein = sum(item['protein'] for item in nutrition_list)
    total_carbs = sum(item['carbs'] for item in nutrition_list)
    total_fat = sum(item['fat'] for item in nutrition_list)

    # 3. 分析营养缺口
    cal_gap = target_cal - total_cal
    protein_gap = protein_target - total_protein

    # 4. 生成个性化建议
    suggestion = f"您的目标为：{goal}。今日推荐热量摄入约{target_cal:.0f}大卡，蛋白质{protein_target:.0f}g。\n"
    suggestion += f"您已摄入：热量{total_cal:.0f}大卡，蛋白质{total_protein:.1f}g，碳水{total_carbs:.1f}g，脂肪{total_fat:.1f}g。\n"
    if cal_gap > 0:
        suggestion += f"还需补充约{cal_gap:.0f}大卡热量。"
    else:
        suggestion += f"热量已达标或超标，请注意控制。"
    if protein_gap > 0:
        suggestion += f"蛋白质还需补充约{protein_gap:.1f}g。"
    else:
        suggestion += f"蛋白质已达标。"
    if goal == 'gain':
        suggestion += "\n建议多摄入高蛋白食物，如鸡胸肉、牛肉、鱼、蛋、奶制品。"
    elif goal == 'lose':
        suggestion += "\n建议多吃蔬菜、瘦肉，控制主食和油脂摄入。"
    else:
        suggestion += "\n保持饮食多样化，均衡营养。"

    return jsonify({
        'success': True,
        'data': {
            'recommendation': suggestion,
            'nutrition': {
                'target_calories': target_cal,
                'target_protein': protein_target,
                'intake_calories': total_cal,
                'intake_protein': total_protein,
                'intake_carbs': total_carbs,
                'intake_fat': total_fat
            }
        }
    })

# 新增：保存进食记录接口
@food_bp.route('/record/add', methods=['POST'])
def add_record():
    data = request.get_json()
    user_id = data.get('user_id', 'default')
    foods = data.get('foods', [])
    for food in foods:
        add_food_record(
            user_id,
            food['name'],
            food.get('calories', 0),
            food.get('protein', 0),
            food.get('carbs', 0),
            food.get('fat', 0)
        )
    return jsonify({'success': True, 'msg': '记录已保存'})

# 新增：查询今日进食记录接口
@food_bp.route('/record/today', methods=['GET'])
def today_record():
    user_id = request.args.get('user_id', 'default')
    records = get_today_records(user_id)
    nutrition_sum = get_today_nutrition_sum(user_id)
    return jsonify({
        'success': True,
        'records': [
            {
                'id': r[0],
                'food_name': r[1],
                'calories': r[2],
                'protein': r[3],
                'carbs': r[4],
                'fat': r[5],
                'time': r[6]
            } for r in records
        ],
        'nutrition_sum': {
            'calories': nutrition_sum[0] or 0,
            'protein': nutrition_sum[1] or 0,
            'carbs': nutrition_sum[2] or 0,
            'fat': nutrition_sum[3] or 0
        }
    })
@food_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'food-detection',
        'model': 'YOLOv8'
    })