from flask import Blueprint, request, jsonify
from app.services.user_service import update_user_info, get_user_info

user_bp = Blueprint('user', __name__)

@user_bp.route('/user/update', methods=['POST'])
def update_info():
    data = request.get_json()
    user_id = data.get('user_id')
    height = data.get('height')
    weight = data.get('weight')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id（openid）'}), 400
    if height is None or weight is None:
        return jsonify({'success': False, 'msg': '缺少身高或体重'}), 400
    update_user_info(user_id, height, weight)
    return jsonify({'success': True, 'msg': '用户信息已保存'})

@user_bp.route('/user/info', methods=['GET'])
def get_info():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'msg': '缺少user_id（openid）'}), 400
    info = get_user_info(user_id)
    if info:
        return jsonify({'success': True, 'height': info[0], 'weight': info[1]})
    else:
        return jsonify({'success': False, 'msg': '未找到用户信息'})
