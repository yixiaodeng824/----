import sqlite3
import os
from datetime import datetime, timedelta

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
            time TEXT,
            meal_type TEXT DEFAULT 'lunch',
            canteen TEXT DEFAULT '未知'
        )
    ''')
    # 兼容旧表：尝试添加新字段（表已存在时新增字段）
    for col in ['meal_type', 'canteen']:
        try:
            c.execute(f'ALTER TABLE food_record ADD COLUMN {col} TEXT')
        except sqlite3.OperationalError:
            pass  # 字段已存在
    conn.commit()
    conn.close()


# ── 增 ──

def add_food_record(user_id, food_name, calories, protein, carbs, fat,
                    meal_type='lunch', canteen='未知'):
    """添加一条进食记录"""
    conn = get_db()
    c = conn.cursor()
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    c.execute('''
        INSERT INTO food_record
            (user_id, date, food_name, calories, protein, carbs, fat, time, meal_type, canteen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, date, food_name, calories, protein, carbs, fat, time, meal_type, canteen))
    conn.commit()
    conn.close()


# ── 删 ──

def delete_food_record(record_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM food_record WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()


# ── 查（单日）──

def get_today_records(user_id):
    """查询当天所有进食记录"""
    conn = get_db()
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d')
    c.execute('''
        SELECT id, food_name, calories, protein, carbs, fat, time, meal_type, canteen
        FROM food_record
        WHERE user_id = ? AND date = ?
        ORDER BY time ASC
    ''', (user_id, date))
    records = c.fetchall()
    conn.close()
    return records


def get_today_nutrition_sum(user_id):
    """查询当天营养总和"""
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


# ── 查（周维度）──

def get_weekly_records(user_id):
    """查询最近 7 天的所有记录，按日期、时间排序"""
    conn = get_db()
    c = conn.cursor()
    week_ago = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    c.execute('''
        SELECT id, date, food_name, calories, protein, carbs, fat, time, meal_type, canteen
        FROM food_record
        WHERE user_id = ? AND date >= ?
        ORDER BY date ASC, time ASC
    ''', (user_id, week_ago))
    records = c.fetchall()
    conn.close()
    return records


def get_weekly_nutrition_sum_by_day(user_id):
    """按天统计本周的营养汇总"""
    conn = get_db()
    c = conn.cursor()
    week_ago = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    c.execute('''
        SELECT date,
               SUM(calories), SUM(protein), SUM(carbs), SUM(fat),
               COUNT(*) as meal_count
        FROM food_record
        WHERE user_id = ? AND date >= ?
        GROUP BY date
        ORDER BY date ASC
    ''', (user_id, week_ago))
    result = c.fetchall()
    conn.close()
    return result


def get_canteen_stats(user_id):
    """统计本周各食堂就餐次数及平均营养"""
    conn = get_db()
    c = conn.cursor()
    week_ago = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    c.execute('''
        SELECT canteen,
               COUNT(*) as meal_count,
               ROUND(AVG(calories), 1) as avg_calories,
               ROUND(AVG(protein), 1) as avg_protein,
               ROUND(AVG(carbs), 1) as avg_carbs,
               ROUND(AVG(fat), 1) as avg_fat
        FROM food_record
        WHERE user_id = ? AND date >= ?
        GROUP BY canteen
        ORDER BY meal_count DESC
    ''', (user_id, week_ago))
    result = c.fetchall()
    conn.close()
    return result


# 初始化数据库
init_db()
