from flask import Blueprint, request, jsonify, g
from app.services.auth_service import require_auth
from app.services.user_service import update_user_info, get_user_info

user_bp = Blueprint('user', __name__)

@user_bp.route('/user/update', methods=['POST'])
@require_auth
def update_info():
    data = request.get_json()
    user_id = g.openid  # 身份来自登录 token
    height = data.get('height')
    weight = data.get('weight')
    if height is None or weight is None:
        return jsonify({'success': False, 'message': '缺少身高或体重'}), 400
    update_user_info(user_id, height, weight)
    return jsonify({'success': True, 'message': '用户信息已保存'})

@user_bp.route('/user/info', methods=['GET'])
@require_auth
def get_info():
    user_id = g.openid  # 身份来自登录 token
    info = get_user_info(user_id)
    if info:
        return jsonify({'success': True, 'height': info[0], 'weight': info[1]})
    else:
        return jsonify({'success': False, 'message': '未找到用户信息'})
