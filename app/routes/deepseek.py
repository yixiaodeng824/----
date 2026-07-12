"""
DeepSeek 增强路由

提供「YOLOv8 识别 + DeepSeek 联网搜索」的完整工作流端点。
"""
import os
import uuid
import json
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context

from app.services.food_detection_service import FoodDetectionService
from app.services.nutrition_service import get_nutrition_by_name
from app.services.deepseek_service import query_food_info, query_food_info_stream

deepseek_bp = Blueprint("deepseek", __name__)
detector = FoodDetectionService()


@deepseek_bp.route("/detect/deepseek", methods=["POST"])
def detect_with_deepseek():
    """
    完整的「YOLOv8 识别 → 本地营养查询 → DeepSeek 联网搜索」工作流。

    请求: multipart/form-data, 字段 image=图片文件
    响应:
    {
      "success": true,
      "yolo_result": {            ← YOLO 识别结果
        "food_name": "北京烤鸭",
        "confidence": 0.92,
        "nutrition": { ... }      ← 本地营养数据
      },
      "deepseek_result": {        ← DeepSeek 生成的百科信息
        "description": "...",
        "nutrition_benefits": "...",
        "health_tips": "...",
        "cooking_methods": ["..."],
        "pairing_suggestions": "...",
        "search_summary": "...",
        "calorie_level": "高热量"
      }
    }
    """
    # ── 1. 检查图片 ──
    if "image" not in request.files:
        return jsonify({"success": False, "message": "请上传图片文件"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "未选择文件"}), 400

    # ── 2. 保存临时文件 ──
    ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # ── 3. YOLOv8 识别 ──
    yolo_result = detector.detect_from_file(filepath)

    # 识别完毕，删除临时图片
    try:
        os.remove(filepath)
    except Exception:
        pass

    if not yolo_result["success"] or not yolo_result["detections"]:
        return jsonify({
            "success": False,
            "message": yolo_result.get("error", "未识别到食物"),
        }), 400

    food = yolo_result["detections"][0]
    food_name = food.get("chinese_name", food.get("name", "未知"))

    # ── 4. 查询本地营养数据 ──
    nutrition = get_nutrition_by_name(food_name)
    if nutrition is None:
        nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

    # ── 5. 调用 DeepSeek（联网搜索） ──
    try:
        deepseek_info = query_food_info(food_name, nutrition)
    except RuntimeError as e:
        # API Key 未配置 —— 降级返回只含 YOLO 的结果
        return jsonify({
            "success": True,
            "message": "未配置 DeepSeek API Key，仅返回 YOLO 识别结果",
            "yolo_result": {
                "food_name": food_name,
                "confidence": food.get("confidence", 0),
                "confidence_rate": f"{(food.get('confidence', 0) * 100):.0f}",
                "nutrition": nutrition,
            },
            "deepseek_result": None,
            "config_error": str(e),
        })

    return jsonify({
        "success": True,
        "message": f"识别到: {food_name}",
        "yolo_result": {
            "food_name": food_name,
            "confidence": food.get("confidence", 0),
            "confidence_rate": f"{(food.get('confidence', 0) * 100):.0f}",
            "nutrition": nutrition,
        },
        "deepseek_result": deepseek_info,
        "_test": "deepseek_ok",
    })


@deepseek_bp.route("/detect/deepseek/stream", methods=["POST"])
def detect_with_deepseek_stream():
    """
    流式版本 —— DeepSeek 部分逐字返回（Server-Sent Events），适合前端打字机效果。

    用法（前端 EventSource 无法传 POST，故仍是普通 POST + fetch stream）：
    ```js
    const resp = await fetch('/api/detect/deepseek/stream', { method:'POST', body: formData });
    const reader = resp.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      // value 是 Uint8Array，解码后按行解析 SSE
    }
    ```
    """
    if "image" not in request.files:
        return jsonify({"success": False, "message": "请上传图片文件"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "未选择文件"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    yolo_result = detector.detect_from_file(filepath)
    if not yolo_result["success"] or not yolo_result["detections"]:
        return jsonify({"success": False, "message": "未识别到食物"}), 400

    food = yolo_result["detections"][0]
    food_name = food.get("chinese_name", food.get("name", "未知"))
    nutrition = get_nutrition_by_name(food_name) or {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

    def generate():
        # 先发送 YOLO 结果（SSE 格式）
        yolo_event = {
            "event": "yolo_result",
            "data": {
                "food_name": food_name,
                "confidence": food.get("confidence", 0),
                "nutrition": nutrition,
            },
        }
        yield f"data: {json.dumps(yolo_event, ensure_ascii=False)}\n\n"

        # 再逐块发送 DeepSeek 流式输出
        yield f"data: {json.dumps({'event': 'deepseek_start'}, ensure_ascii=False)}\n\n"
        try:
            for chunk in query_food_info_stream(food_name, nutrition):
                sse = {"event": "deepseek_chunk", "data": chunk}
                yield f"data: {json.dumps(sse, ensure_ascii=False)}\n\n"
        except RuntimeError as e:
            sse = {"event": "error", "data": str(e)}
            yield f"data: {json.dumps(sse, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'event': 'done'}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@deepseek_bp.route("/deepseek/query", methods=["POST"])
def deepseek_query():
    """
    单独调用 DeepSeek（无需上传图片，只需传入食物名和营养数据）。

    请求 JSON:
    {
      "food_name": "北京烤鸭",
      "nutrition": {"calories": 100, "protein": 10, "carbs": 5, "fat": 7}
    }
    """
    data = request.get_json()
    food_name = data.get("food_name", "")
    if not food_name:
        return jsonify({"success": False, "message": "缺少 food_name"}), 400

    nutrition = data.get("nutrition")
    try:
        deepseek_info = query_food_info(food_name, nutrition)
        return jsonify({
            "success": True,
            "data": deepseek_info,
        })
    except RuntimeError as e:
        return jsonify({
            "success": False,
            "message": str(e),
        })
