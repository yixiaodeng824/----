from flask import Blueprint, request, jsonify, current_app
import os
import uuid
from app.services.food_detection_service import FoodDetectionService

from app.services.nutrition_service import get_nutrition_for_foods, get_nutrition_by_name
from app.services.food_record_service import add_food_record, get_today_records, get_today_nutrition_sum
from app.services.user_service import get_user_info, update_user_info


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
            data = []
            for food in result['detections']:
                food_name = food.get('chinese_name', food.get('name', '未知'))
                nutrition = get_nutrition_by_name(food_name)
                if nutrition is None:
                    nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
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
def get_professional_recommendation(goal, cal_gap, protein_gap, intake_carbs, intake_fat):
    gap_text = f"营养缺口分析：\n"
    gap_text += f"- 热量缺口：{cal_gap:.0f}大卡\n" if cal_gap > 0 else "- 热量已达标或超标\n"
    gap_text += f"- 蛋白质缺口：{protein_gap:.1f}g\n" if protein_gap > 0 else "- 蛋白质已达标\n"

    if goal == 'gain':
        supplement = (
            "补充建议：\n"
            "- 早餐：鸡蛋3个+燕麦粥+牛奶\n"
            "- 午餐：鸡胸肉150g+糙米饭+蔬菜\n"
            "- 加餐：蛋白棒或坚果\n"
            "- 晚餐：鱼肉200g+红薯+绿叶菜\n"
            "建议多摄入高蛋白食物，如鸡胸肉、牛肉、鱼、蛋、奶制品。"
        )
        notice = "注意事项：保证蛋白质摄入，适量增加碳水，避免高糖高油食物。"
    elif goal == 'lose':
        supplement = (
            "补充建议：\n"
            "- 早餐：2个鸡蛋+蔬菜沙拉\n"
            "- 午餐：鸡胸肉120g+藜麦+大量蔬菜\n"
            "- 晚餐：清蒸鱼150g+西兰花+豆腐\n"
            "建议多吃蔬菜、瘦肉，控制主食和油脂摄入。"
        )
        notice = "注意事项：控制总热量，优先补充蛋白质，减少油脂和精制碳水摄入。"
    else:
        supplement = (
            "补充建议：\n"
            "- 早餐：全麦面包+鸡蛋+水果\n"
            "- 午餐：鱼肉/鸡肉+杂粮饭+多种蔬菜\n"
            "- 晚餐：豆腐+蔬菜+少量主食\n"
            "保持饮食多样化，均衡营养。"
        )
        notice = "注意事项：保证食物多样性，适量运动，控制油盐摄入。"

    replace_text = "食材替换推荐：如不喜欢鸡胸肉，可用牛肉、鱼肉、豆腐等替代；主食可用糙米、红薯、玉米等替换。"

    return f"{gap_text}\n{supplement}\n\n{replace_text}\n\n{notice}"

@food_bp.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    goal = data.get('goal')  # 目标：gain/lose/maintain
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id（openid）'}), 400
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
def add_record():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id（openid）'}), 400
    foods = data.get('foods', [])
    print(user_id)
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
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id（openid）'}), 400
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