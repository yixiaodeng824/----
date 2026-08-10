"""
鉴权服务 — 登录凭证（token）的签发与校验

设计：wxlogin 换取 openid 后，签发一个随机 token 存入 sessions 表。
前端后续请求带 Authorization: Bearer <token>，后端据此识别用户身份。
token 可撤销（删除行即失效）、有过期时间，无需引入 JWT 等额外依赖。
"""
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

from flask import request, jsonify, g

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/user_info.db')
TOKEN_TTL_DAYS = int(os.getenv('TOKEN_TTL_DAYS', '30'))  # token 有效期 30 天


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA busy_timeout = 10000')
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            openid TEXT NOT NULL,
            created_at TEXT,
            expires_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def generate_token(openid):
    """为 openid 签发一个新 token，返回 (token, 过期时间ISO字符串)"""
    # 顺手清理过期行，避免 sessions 表无限膨胀
    _purge_expired()

    token = secrets.token_urlsafe(32)
    now = datetime.now()
    expires = now + timedelta(days=TOKEN_TTL_DAYS)
    conn = get_db()
    c = conn.cursor()
    c.execute(
        'INSERT INTO sessions (token, openid, created_at, expires_at) VALUES (?, ?, ?, ?)',
        (token, openid, now.strftime('%Y-%m-%d %H:%M:%S'), expires.strftime('%Y-%m-%d %H:%M:%S')),
    )
    conn.commit()
    conn.close()
    return token, expires.isoformat()


def verify_token(token):
    """校验 token，有效返回 openid，无效/过期返回 None"""
    if not token:
        return None
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT openid, expires_at FROM sessions WHERE token = ?', (token,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    openid, expires_at = row
    if expires_at:
        try:
            if datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S') < datetime.now():
                return None  # 已过期
        except ValueError:
            return None
    return openid


def _purge_expired():
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE expires_at < ?', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
    conn.commit()
    conn.close()


def _extract_token(req):
    auth = req.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:].strip()
    return req.headers.get('X-Token') or None


def require_auth(f):
    """鉴权装饰器：校验 Authorization: Bearer <token>，通过后把 openid 注入 flask.g"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = _extract_token(request)
        if not token:
            return jsonify({'success': False, 'message': '缺少登录凭证，请重新进入小程序'}), 401
        openid = verify_token(token)
        if not openid:
            return jsonify({'success': False, 'message': '登录已过期，请重新进入小程序'}), 401
        g.openid = openid
        return f(*args, **kwargs)
    return wrapper
