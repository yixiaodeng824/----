"""
周报路由 — 提供周饮食数据 + DeepSeek AI 分析报告
"""
from flask import Blueprint, request, jsonify

from app.services.food_record_service import (
    get_weekly_records,
    get_weekly_nutrition_sum_by_day,
    get_canteen_stats,
)
from app.services.weekly_report_service import build_weekly_data, generate_ai_report

weekly_bp = Blueprint("weekly", __name__)

MEAL_TYPE_CN = {
    'breakfast': '早餐',
    'lunch': '午餐',
    'dinner': '晚餐',
    'snack': '加餐',
}

MEAL_TYPE_ORDER = ['breakfast', 'lunch', 'dinner', 'snack']


@weekly_bp.route('/report/weekly/data', methods=['GET'])
def weekly_data():
    """本周原始饮食数据（网格结构）"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id'}), 400

    data = build_weekly_data(user_id)

    # 整理为前端友好的网格格式
    day_list = sorted(data['meal_grid'].keys())
    grid = []
    for day in day_list:
        row = {'date': day, 'meals': []}
        for mt in MEAL_TYPE_ORDER:
            items = data['meal_grid'][day].get(mt, [])
            if items:
                row['meals'].append({
                    'meal_type': mt,
                    'meal_type_cn': MEAL_TYPE_CN.get(mt, mt),
                    'items': items,
                })
        grid.append(row)

    return jsonify({
        'success': True,
        'data': {
            'total_meals': data['total_meals'],
            'grid': grid,
            'daily_summary': data['daily_summary'],
            'canteen_summary': data['canteen_summary'],
        },
    })


@weekly_bp.route('/report/weekly/canteen', methods=['GET'])
def weekly_canteen():
    """本周食堂统计数据"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id'}), 400

    stats = get_canteen_stats(user_id)
    result = []
    for c in stats:
        result.append({
            'name': c[0],
            'meal_count': c[1],
            'avg_calories': c[2] or 0,
            'avg_protein': c[3] or 0,
            'avg_carbs': c[4] or 0,
            'avg_fat': c[5] or 0,
        })

    return jsonify({'success': True, 'data': result})


@weekly_bp.route('/report/weekly/ai', methods=['GET'])
def weekly_ai_report():
    """DeepSeek 生成的 AI 周报"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id'}), 400

    report = generate_ai_report(user_id)
    return jsonify({
        'success': True,
        'data': report,
    })
