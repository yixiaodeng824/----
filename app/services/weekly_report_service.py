"""
周报生成服务

聚合用户一周的饮食记录，调用 DeepSeek 生成 AI 分析报告。
"""
import json
import os
from datetime import datetime
from openai import OpenAI

from app.services.food_record_service import (
    get_weekly_records,
    get_weekly_nutrition_sum_by_day,
    get_canteen_stats,
)

MEAL_TYPE_CN = {
    'breakfast': '早餐',
    'lunch': '午餐',
    'dinner': '晚餐',
    'snack': '加餐',
}


def _get_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30)


def build_weekly_data(user_id):
    """聚合本周饮食数据为结构化字典"""
    records = get_weekly_records(user_id)
    by_day = get_weekly_nutrition_sum_by_day(user_id)
    canteen = get_canteen_stats(user_id)

    # records: (id, date, food_name, calories, protein, carbs, fat, time, meal_type, canteen)
    meal_grid = {}
    for r in records:
        day = r[1]
        if day not in meal_grid:
            meal_grid[day] = {'breakfast': [], 'lunch': [], 'dinner': [], 'snack': []}
        meal_type = r[8] or 'lunch'
        if meal_type not in meal_grid[day]:
            meal_type = 'lunch'
        meal_grid[day][meal_type].append({
            'food_name': r[2],
            'calories': r[3],
            'protein': r[4],
            'carbs': r[5],
            'fat': r[6],
            'time': r[7],
            'canteen': r[9] or '未知',
        })

    # by_day: (date, SUM(calories), SUM(protein), SUM(carbs), SUM(fat), COUNT)
    daily_summary = []
    for d in by_day:
        daily_summary.append({
            'date': d[0],
            'calories': d[1] or 0,
            'protein': d[2] or 0,
            'carbs': d[3] or 0,
            'fat': d[4] or 0,
            'meal_count': d[5] or 0,
        })

    # canteen: (name, meal_count, avg_calories, avg_protein, avg_carbs, avg_fat)
    canteen_summary = []
    for c in canteen:
        canteen_summary.append({
            'name': c[0],
            'meal_count': c[1],
            'avg_calories': c[2] or 0,
            'avg_protein': c[3] or 0,
            'avg_carbs': c[4] or 0,
            'avg_fat': c[5] or 0,
        })

    return {
        'user_id': user_id,
        'total_meals': len(records),
        'meal_grid': meal_grid,
        'daily_summary': daily_summary,
        'canteen_summary': canteen_summary,
    }


WEEKLY_REPORT_SYSTEM_PROMPT = """你是一位专业的营养学分析师。请根据用户一周的饮食记录数据，
分析其饮食习惯，生成一份详细的周报。

请严格按以下 JSON 格式返回（只返回 JSON，不要加 markdown 代码块标记）：

{
  "overall": "本周饮食总体评价（50-100字，中文）",
  "weekly_score": 75,
  "nutrition_trend": "本周热量/蛋白质/碳水/脂肪的摄入趋势分析（80-120字）",
  "canteen_analysis": [
    {
      "name": "食堂名称",
      "meal_count": 5,
      "avg_calories": 650,
      "issues": ["偏油腻", "蛋白质偏低"],
      "rating": "★★★☆☆",
      "suggestion": "建议增加蛋白质选择"
    }
  ],
  "nutrition_issues": ["蔬菜摄入不足", "晚餐碳水偏高", "蛋白质整体偏低"],
  "improvement_suggestions": [
    "建议午餐增加优质蛋白（鸡胸肉、鱼、蛋）",
    "晚餐减少主食比例，增加蔬菜",
    "加餐可选择水果或坚果替代高热量零食"
  ],
  "best_meal": "本周最佳一餐的描述"
}
"""


def generate_ai_report(user_id):
    """
    调用 DeepSeek 生成 AI 周报。
    返回解析后的 JSON 字典，失败时返回包含 error 的字典。
    """
    data = build_weekly_data(user_id)
    total_meals = data['total_meals']

    if total_meals == 0:
        return {'error': '本周暂无饮食记录', 'has_data': False}

    # 构造用户消息
    user_content = f"以下是我本周的饮食记录数据，请帮我分析：\n\n"
    user_content += f"本周共记录 {total_meals} 餐\n\n"

    user_content += "每日营养摄入：\n"
    for d in data['daily_summary']:
        user_content += (
            f"- {d['date']}: 热量{d['calories']}大卡, "
            f"蛋白质{d['protein']}g, 碳水{d['carbs']}g, 脂肪{d['fat']}g "
            f"({d['meal_count']}餐)\n"
        )

    if data['canteen_summary']:
        user_content += "\n食堂就餐统计：\n"
        for c in data['canteen_summary']:
            user_content += (
                f"- {c['name']}: {c['meal_count']}次, "
                f"平均热量{c['avg_calories']}大卡, "
                f"蛋白质{c['avg_protein']}g, 脂肪{c['avg_fat']}g\n"
            )

    user_content += "\n详细饮食记录：\n"
    for day in sorted(data['meal_grid'].keys()):
        user_content += f"\n{day}:\n"
        for meal_type in ('breakfast', 'lunch', 'dinner', 'snack'):
            items = data['meal_grid'][day].get(meal_type, [])
            if items:
                type_cn = MEAL_TYPE_CN.get(meal_type, meal_type)
                for item in items:
                    user_content += (
                        f"  [{type_cn}] {item['food_name']} "
                        f"({item['canteen']}) - {item['calories']}大卡\n"
                    )

    user_content += "\n请根据以上数据，给出专业的饮食分析和建议。"

    try:
        client = _get_deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": WEEKLY_REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=3072,
            extra_body={"enable_search": True},
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
        result = json.loads(content)
        result['has_data'] = True
        return result

    except json.JSONDecodeError:
        return {
            'error': 'AI 返回格式异常',
            'has_data': True,
        }
    except Exception as e:
        return {
            'error': str(e),
            'has_data': True,
        }
