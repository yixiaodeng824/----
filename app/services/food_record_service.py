# 删除一条进食记录
def delete_food_record(record_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM food_record WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/food_record.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS food_record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            date TEXT,
            food_name TEXT,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            time TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 添加一条进食记录
def add_food_record(user_id, food_name, calories, protein, carbs, fat):
    conn = get_db()
    c = conn.cursor()
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    c.execute('''
        INSERT INTO food_record (user_id, date, food_name, calories, protein, carbs, fat, time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, date, food_name, calories, protein, carbs, fat, time))
    conn.commit()
    conn.close()

# 查询当天所有进食记录
def get_today_records(user_id):
    conn = get_db()
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d')
    c.execute('''
        SELECT id, food_name, calories, protein, carbs, fat, time FROM food_record
        WHERE user_id = ? AND date = ?
        ORDER BY time ASC
    ''', (user_id, date))
    records = c.fetchall()
    conn.close()
    return records

# 查询当天营养总和
def get_today_nutrition_sum(user_id):
    conn = get_db()
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d')
    c.execute('''
        SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat) FROM food_record
        WHERE user_id = ? AND date = ?
    ''', (user_id, date))
    result = c.fetchone()
    conn.close()
    return result

# 初始化数据库
init_db()
