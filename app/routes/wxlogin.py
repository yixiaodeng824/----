import os
import requests
from flask import Blueprint, request, jsonify

from app.services.auth_service import generate_token

wxlogin_bp = Blueprint('wxlogin', __name__)

# 小程序 AppID 与 AppSecret 从环境变量读取
# AppSecret 敏感，必须放在 .env 中，禁止硬编码在代码里（否则会随 git 历史泄露）
APPID = os.getenv('WX_APP_ID', 'wx4486a9ebe8d3e79b')
SECRET = os.getenv('WX_APP_SECRET', '')


@wxlogin_bp.route('/wxlogin', methods=['POST'])
def wxlogin():
    data = request.get_json()
    code = data.get('code')
    if not code:
        return jsonify({'success': False, 'message': '缺少code'}), 400
    if not SECRET:
        return jsonify({'success': False, 'message': '服务端未配置 WX_APP_SECRET（请检查 .env）'}), 500
    # 请求微信服务器换取 openid
    url = f'https://api.weixin.qq.com/sns/jscode2session?appid={APPID}&secret={SECRET}&js_code={code}&grant_type=authorization_code'
    try:
        resp = requests.get(url, timeout=5)
        wxdata = resp.json()
        if 'openid' in wxdata:
            # 登录成功 → 签发 token，前端后续请求凭此识别身份
            token, expires_at = generate_token(wxdata['openid'])
            return jsonify({
                'success': True,
                'openid': wxdata['openid'],
                'token': token,
                'expires_at': expires_at,
            })
        else:
            return jsonify({'success': False, 'message': wxdata.get('errmsg', '微信认证失败')})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
