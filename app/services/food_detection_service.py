import os
from ultralytics import YOLO
import cv2
import numpy as np

class FoodDetectionService:
    def __init__(self, model_name='yolov8n.pt'):
        # Allow runtime model switching without code edits.
        resolved_model_name = os.getenv('YOLO_MODEL_PATH', model_name)
        print(f"🍎 加载YOLOv8模型: {resolved_model_name}")
        self.model = YOLO(resolved_model_name)
        print("✅ 模型加载成功")
    
    def detect_from_file(self, image_path):
        """从文件检测"""
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': '文件不存在',
                'detections': []
            }
        
        try:
            results = self.model(image_path)
            return self._process_results(results)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detections': []
            }
    
    def _process_results(self, results):
        detections = []
        
        for result in results:
            # Classification model output (no boxes, has probabilities)
            if result.probs is not None:
                top1_id = int(result.probs.top1)
                class_name = result.names.get(top1_id, f'未知{top1_id}')
                detections.append({
                    'name': class_name,
                    'chinese_name': self._get_chinese_name(class_name),
                    'confidence': float(result.probs.top1conf),
                    'bbox': None
                })
                continue

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, f'未知{cls_id}')
                    
                    # 简单的食物类别过滤
                    if self._is_food(class_name):
                        detections.append({
                            'name': class_name,
                            'chinese_name': self._get_chinese_name(class_name),
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        })
        
        return {
            'success': True,
            'detections': detections,
            'count': len(detections),
            'message': f'找到 {len(detections)} 种食物'
        }
    
    def _is_food(self, class_name):
        food_keywords = ['apple', 'banana', 'orange', 'pizza', 'cake', 
                        'sandwich', 'hot dog', 'broccoli', 'carrot']
        return any(keyword in class_name.lower() for keyword in food_keywords)
    
    def _get_chinese_name(self, english_name):
        name_map = {
            'banana': '香蕉',
            'apple': '苹果',
            'sandwich': '三明治',
            'orange': '橙子',
            'broccoli': '西兰花',
            'carrot': '胡萝卜',
            'hot dog': '热狗',
            'pizza': '披萨',
            'cake': '蛋糕'
        }
        return name_map.get(english_name.lower(), english_name)