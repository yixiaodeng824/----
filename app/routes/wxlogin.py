import requests
from flask import Blueprint, request, jsonify

wxlogin_bp = Blueprint('wxlogin', __name__)

# 替换为你自己的小程序appid和secret
APPID = 'wx4486a9ebe8d3e79b'
SECRET = 'd783cc8d94372ff0e8ac31dfea08f49a'

@wxlogin_bp.route('/wxlogin', methods=['POST'])
def wxlogin():
    data = request.get_json()
    code = data.get('code')
    if not code:
        return jsonify({'success': False, 'msg': '缺少code'}), 400
    # 请求微信服务器换取openid
    url = f'https://api.weixin.qq.com/sns/jscode2session?appid={APPID}&secret={SECRET}&js_code={code}&grant_type=authorization_code'
    try:
        resp = requests.get(url)
        wxdata = resp.json()
        if 'openid' in wxdata:
            return jsonify({'success': True, 'openid': wxdata['openid']})
        else:
            return jsonify({'success': False, 'msg': wxdata.get('errmsg', '微信认证失败')})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)})
