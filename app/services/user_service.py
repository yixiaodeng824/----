import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/user_info.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_info (
            user_id TEXT PRIMARY KEY,
            height REAL,
            weight REAL,
            update_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def update_user_info(user_id, height, weight):
    # user_id 必须为 openid，不能为 default 或昵称
    conn = get_db()
    c = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO user_info (user_id, height, weight, update_time)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET height=excluded.height, weight=excluded.weight, update_time=excluded.update_time
    ''', (user_id, height, weight, now))
    conn.commit()
    conn.close()

def get_user_info(user_id):
    # user_id 必须为 openid，不能为 default 或昵称
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT height, weight FROM user_info WHERE user_id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    return result
