import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image


class FoodDetectionService:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = os.getenv('YOLO_MODEL_PATH', 'runs/classify/food_cls_l/weights/best.pt')
        resolved = os.getenv('YOLO_MODEL_PATH', model_name)
        print(f"🍎 加载模型: {resolved}")

        ckpt = torch.load(resolved, map_location='cpu', weights_only=False)
        self.model = ckpt['model']
        self.model.eval()
        self.names = ckpt.get('names', {})
        print(f"✅ 模型加载成功 ({len(self.names)} 类)")

        # 与训练一致的预处理
        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_from_file(self, image_path):
        if not os.path.exists(image_path):
            return {'success': False, 'error': '文件不存在', 'detections': []}

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {'success': False, 'error': '图片读取失败', 'detections': []}

            # 锐化增强
            sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img_bgr = cv2.filter2D(img_bgr, -1, sharpen)
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)

            # BGR → RGB → PIL → transforms → 推理
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            x = self.transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                out = self.model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]

            probs = torch.softmax(out, dim=1)[0]
            top1_id = int(probs.argmax())
            top1_conf = float(probs[top1_id])
            food_name = self.names.get(top1_id, f'未知{top1_id}')
            cn_name = self._get_chinese_name(food_name)

            return {
                'success': True,
                'detections': [{
                    'name': cn_name,
                    'chinese_name': cn_name,
                    'confidence': top1_conf,
                    'bbox': None,
                }],
                'count': 1,
                'message': f'识别到: {cn_name}',
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'detections': []}

    def _get_chinese_name(self, english_name):
        name_map = {
            'apple_pie': '苹果派', 'baby_back_ribs': '烤肋排', 'baklava': '果仁蜜饼',
            'beef_carpaccio': '生牛肉薄片', 'beef_tartare': '牛肉鞑靼', 'beet_salad': '甜菜沙拉',
            'beignets': '法式炸甜点', 'bibimbap': '石锅拌饭', 'bread_pudding': '面包布丁',
            'breakfast_burrito': '早餐卷饼', 'bruschetta': '烤面包片', 'caesar_salad': '凯撒沙拉',
            'cannoli': '奶油甜卷', 'caprese_salad': '卡普里沙拉', 'carrot_cake': '胡萝卜蛋糕',
            'ceviche': '酸橘汁腌鱼', 'cheese_plate': '奶酪拼盘', 'cheesecake': '芝士蛋糕',
            'chicken_curry': '咖喱鸡', 'chicken_quesadilla': '鸡肉薄饼', 'chicken_wings': '鸡翅',
            'chocolate_cake': '巧克力蛋糕', 'chocolate_mousse': '巧克力慕斯', 'churros': '吉拿果',
            'clam_chowder': '蛤蜊浓汤', 'club_sandwich': '俱乐部三明治', 'crab_cakes': '蟹饼',
            'creme_brulee': '焦糖布丁', 'croque_madame': '法式三明治', 'cup_cakes': '纸杯蛋糕',
            'deviled_eggs': '魔鬼蛋', 'donuts': '甜甜圈', 'dumplings': '饺子',
            'edamame': '毛豆', 'eggs_benedict': '班尼迪克蛋', 'escargots': '焗蜗牛',
            'falafel': '法拉费', 'filet_mignon': '菲力牛排', 'fish_and_chips': '炸鱼薯条',
            'foie_gras': '鹅肝', 'french_fries': '薯条', 'french_onion_soup': '法式洋葱汤',
            'french_toast': '法式吐司', 'fried_calamari': '炸鱿鱼', 'fried_rice': '炒饭',
            'frozen_yogurt': '冻酸奶', 'garlic_bread': '蒜香面包', 'gnocchi': '意式土豆丸子',
            'greek_salad': '希腊沙拉', 'grilled_cheese_sandwich': '烤芝士三明治', 'grilled_salmon': '烤三文鱼',
            'guacamole': '鳄梨酱', 'gyoza': '锅贴', 'hamburger': '汉堡包',
            'hot_and_sour_soup': '酸辣汤', 'hot_dog': '热狗', 'huevos_rancheros': '墨西哥煎蛋',
            'hummus': '鹰嘴豆泥', 'ice_cream': '冰淇淋', 'lasagna': '千层面',
            'lobster_bisque': '龙虾浓汤', 'lobster_roll_sandwich': '龙虾卷', 'macaroni_and_cheese': '芝士通心粉',
            'macarons': '马卡龙', 'miso_soup': '味噌汤', 'mussels': '青口贝',
            'nachos': '玉米片', 'omelette': '煎蛋卷', 'onion_rings': '洋葱圈',
            'oysters': '生蚝', 'pad_thai': '泰式炒河粉', 'paella': '海鲜饭',
            'pancakes': '松饼', 'panna_cotta': '意式奶冻', 'peking_duck': '北京烤鸭',
            'pho': '越南河粉', 'pizza': '披萨', 'pork_chop': '猪排',
            'poutine': '肉汁薯条', 'prime_rib': '烤牛肋', 'pulled_pork_sandwich': '手撕猪肉三明治',
            'ramen': '拉面', 'ravioli': '意大利饺', 'red_velvet_cake': '红丝绒蛋糕',
            'risotto': '意大利烩饭', 'samosa': '咖喱角', 'sashimi': '刺身',
            'scallops': '扇贝', 'seaweed_salad': '海藻沙拉', 'shrimp_and_grits': '虾仁玉米粥',
            'spaghetti_bolognese': '肉酱意面', 'spaghetti_carbonara': '奶油培根意面', 'spring_rolls': '春卷',
            'steak': '牛排', 'strawberry_shortcake': '草莓蛋糕', 'sushi': '寿司',
            'tacos': '墨西哥卷饼', 'takoyaki': '章鱼小丸子', 'tiramisu': '提拉米苏',
            'tuna_tartare': '金枪鱼鞑靼', 'waffles': '华夫饼',
        }
        return name_map.get(english_name, english_name)
