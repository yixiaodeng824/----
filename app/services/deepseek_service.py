"""
DeepSeek API 集成服务

调用 DeepSeek 的联网搜索能力，为识别到的食物提供：
  - 详细的营养功效分析
  - 健康建议与食用禁忌
  - 烹饪做法推荐
  - 食材搭配建议

API 文档: https://api-docs.deepseek.com/zh-cn/
"""
import os
import json
from openai import OpenAI

# 默认 DeepSeek API 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # 最新模型
# DEEPSEEK_MODEL = "deepseek-reasoner"  # 若需思维链可切换


def _get_client() -> OpenAI:
    """获取 DeepSeek 客户端（环境变量 DEEPSEEK_API_KEY）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY 环境变量。\n"
            "请前往 https://platform.deepseek.com/api_keys 获取密钥\n"
            "然后设置: export DEEPSEEK_API_KEY=sk-xxxxx"
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


# ──────────────────────────────────────────────
#  系统提示词 —— 控制 AI 输出的角色与风格
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的营养学专家和美食顾问。请根据用户提供的食物名称（中文），
结合联网搜索获取的最新信息，生成该食物的详细百科。

请严格按以下 JSON 格式返回（只返回 JSON，不要加 markdown 代码块标记）：

{
  "food_name": "食物中文名称",
  "description": "用 2-3 句话简要介绍这是什么食物，包括其起源或主要特点",
  "nutrition_benefits": "详细分析该食物的核心营养价值与健康功效（150-200字）",
  "health_tips": "食用建议，包括适宜人群、注意事项、可能的禁忌（100-150字）",
  "calorie_level": "高热量 / 中热量 / 低热量",
  "cooking_methods": ["做法1", "做法2", "做法3"],
  "pairing_suggestions": "推荐搭配的食材或饮品，以及搭配理由（50-100字）",
  "search_summary": "根据联网搜索结果，补充该食物的最新资讯或有趣的饮食文化知识（50-100字）"
}
"""


def query_food_info(food_name_cn: str, nutrition: dict = None) -> dict:
    """
    调用 DeepSeek（带联网搜索）查询食物的详细信息。

    参数:
        food_name_cn: 食物中文名称（如 "北京烤鸭"）
        nutrition:    本地已有的基础营养数据（可选），会作为上下文提供给 AI

    返回:
        解析后的 JSON 字典，包含 description / nutrition_benefits / health_tips 等字段
        若 API 调用失败，返回包含 error 字段的字典
    """
    client = _get_client()

    # 构造用户消息 —— 将本地已有营养数据也提供给 DeepSeek
    user_content = f"请帮我查询「{food_name_cn}」的详细信息。"
    if nutrition:
        user_content += (
            f"\n\n已知基础营养数据（每 100g）：\n"
            f"- 热量: {nutrition.get('calories', '未知')} 大卡\n"
            f"- 蛋白质: {nutrition.get('protein', '未知')} g\n"
            f"- 碳水化合物: {nutrition.get('carbs', '未知')} g\n"
            f"- 脂肪: {nutrition.get('fat', '未知')} g\n"
        )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=2048,
            # 关键：开启联网搜索能力
            extra_body={"enable_search": True},
        )

        content = response.choices[0].message.content.strip()
        # 尝试解析 JSON（AI 可能用 ```json ... ``` 包裹）
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
        result = json.loads(content)
        result["food_name"] = food_name_cn  # 确保食物名一致
        return result

    except json.JSONDecodeError:
        return {
            "food_name": food_name_cn,
            "description": "",
            "nutrition_benefits": "",
            "health_tips": "",
            "calorie_level": "未知",
            "cooking_methods": [],
            "pairing_suggestions": "",
            "search_summary": "",
            "error": "AI 返回格式异常，无法解析",
            "_raw": content if 'content' in dir() else "",
        }
    except Exception as e:
        return {
            "food_name": food_name_cn,
            "description": "",
            "nutrition_benefits": "",
            "health_tips": "",
            "calorie_level": "未知",
            "cooking_methods": [],
            "pairing_suggestions": "",
            "search_summary": "",
            "error": str(e),
        }


def query_food_info_stream(food_name_cn: str, nutrition: dict = None):
    """
    流式版本 —— 逐块返回 DeepSeek 回复（适合前端打字机效果）。
    使用 yield 产出每个文本块。
    """
    client = _get_client()
    user_content = f"请帮我查询「{food_name_cn}」的详细信息。"
    if nutrition:
        user_content += (
            f"\n\n已知基础营养数据（每 100g）：\n"
            f"- 热量: {nutrition.get('calories', '未知')} 大卡\n"
            f"- 蛋白质: {nutrition.get('protein', '未知')} g\n"
            f"- 碳水化合物: {nutrition.get('carbs', '未知')} g\n"
            f"- 脂肪: {nutrition.get('fat', '未知')} g\n"
        )

    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=2048,
        extra_body={"enable_search": True},
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
