from flask import Blueprint, request, jsonify, g
import os
from app.services.food_detection_service import FoodDetectionService
from app.services.auth_service import require_auth
from app.services.upload_service import save_upload, safe_remove
from app.services.nutrition_service import get_nutrition_by_name
from app.services.food_record_service import add_food_record, get_today_records, get_today_nutrition_sum
from app.services.user_service import get_user_info, update_user_info
from app.services.recommendation_service import get_professional_recommendation


food_bp = Blueprint('food', __name__)
detector = FoodDetectionService()
# 删除进食记录接口
@food_bp.route('/record/delete', methods=['POST'])
@require_auth
def delete_record():
    data = request.get_json()
    record_id = data.get('id')
    if not record_id:
        return jsonify({'success': False, 'message': '缺少记录ID'}), 400
    from app.services.food_record_service import delete_food_record
    delete_food_record(record_id, user_id=g.openid)  # 只能删自己的记录
    return jsonify({'success': True, 'message': '删除成功'})

@food_bp.route('/detect', methods=['POST'])
@require_auth
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
        
        # 保存文件（临时，检测后立即删除）
        filepath = save_upload(file)
        print(f"📁 文件保存: {filepath}")

        # 检测
        result = detector.detect_from_file(filepath)

        # 检测完成后删除临时图片，释放磁盘空间
        safe_remove(filepath)
        print(f"🗑️ 临时文件已删除: {filepath}")
        # 适配前端 analysis 页面，返回所有识别结果
        if result['success'] and result['detections']:
            data = []
            for food in result['detections']:
                food_name = food.get('chinese_name', food.get('name', '未知'))
                conf = food.get('confidence', 0)
                nutrition = get_nutrition_by_name(food_name)
                if nutrition is None:
                    nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
                data.append({
                    'foodName': food_name,
                    'confidence': conf,
                    'confidenceRate': f'{conf * 100:.0f}',
                    'calories': nutrition['calories'],
                    'protein': nutrition['protein'],
                    'carbs': nutrition['carbs'],
                    'fat': nutrition['fat'],
                })
            return jsonify({
                'success': True,
                'data': data,
                'filename': os.path.basename(filepath),
                'message': f'识别到 {len(data)} 种食物',
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


@food_bp.route('/recommend', methods=['POST'])
@require_auth
def recommend():
    data = request.get_json()
    goal = data.get('goal')  # 目标：gain/lose/maintain
    user_id = g.openid  # 身份来自登录 token
    # 优先读取数据库中的身高体重
    info = get_user_info(user_id)
    if info and info[0] is not None and info[1] is not None:
        height = info[0]
        weight = info[1]
    else:
        height = data.get('height')
        weight = data.get('weight')
        # 自动保存新用户信息
        if height is not None and weight is not None:
            update_user_info(user_id, height, weight)

    # 1. 计算推荐热量和蛋白质目标
    if weight is not None:
        weight = float(weight)
        if goal == 'gain':
            target_cal = weight * 40
            protein_target = weight * 2
        elif goal == 'lose':
            target_cal = weight * 28
            protein_target = weight * 1.5
        else:
            target_cal = weight * 33
            protein_target = weight * 1.2
    else:
        target_cal = 0
        protein_target = 0

    # 2. 查询当天已摄入营养
    nutrition_sum = get_today_nutrition_sum(user_id)
    intake_cal = nutrition_sum[0] or 0
    intake_protein = nutrition_sum[1] or 0
    intake_carbs = nutrition_sum[2] or 0
    intake_fat = nutrition_sum[3] or 0

    # 3. 分析缺口
    cal_gap = target_cal - intake_cal
    protein_gap = protein_target - intake_protein

    # 4. 生成更专业的推荐内容
    suggestion = get_professional_recommendation(goal, cal_gap, protein_gap, intake_carbs, intake_fat)

    return jsonify({
        'success': True,
        'data': {
            'recommendation': suggestion,
            'nutrition': {
                'target_calories': target_cal,
                'target_protein': protein_target,
                'intake_calories': intake_cal,
                'intake_protein': intake_protein,
                'intake_carbs': intake_carbs,
                'intake_fat': intake_fat
            }
        }
    })

# 新增：保存进食记录接口
@food_bp.route('/record/add', methods=['POST'])
@require_auth
def add_record():
    data = request.get_json()
    user_id = g.openid  # 身份来自登录 token
    meal_type = data.get('meal_type', 'lunch')
    if meal_type not in ('breakfast', 'lunch', 'dinner', 'snack'):
        meal_type = 'lunch'  # 白名单校验，防止脏数据进库导致周报匹配不上
    canteen = data.get('canteen', '未知')
    foods = data.get('foods', [])
    print(user_id)
    for food in foods:
        add_food_record(
            user_id,
            food['name'],
            food.get('calories', 0),
            food.get('protein', 0),
            food.get('carbs', 0),
            food.get('fat', 0),
            meal_type=meal_type,
            canteen=canteen,
        )
    return jsonify({'success': True, 'message': '记录已保存'})

# 新增：查询今日进食记录接口
@food_bp.route('/record/today', methods=['GET'])
@require_auth
def today_record():
    user_id = g.openid  # 身份来自登录 token
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
                'time': r[6],
                'meal_type': r[7] or 'lunch',
                'canteen': r[8] or '未知',
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
