import os
import math
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image


DEFAULT_FOOD101_MODEL = '/data/zsm/food/runs/classify/food_cls_l_20260616-010619/weights/best.pt'
DEFAULT_OOD_GATE = '/data/zsm/food/runs/ood_gate/manifest_v2_20260616/gate.pt'


class GateMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        if hidden_dim > 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


class FoodDetectionService:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = os.getenv('YOLO_MODEL_PATH', DEFAULT_FOOD101_MODEL)
        resolved = os.getenv('YOLO_MODEL_PATH', model_name)
        self.confidence_threshold = float(os.getenv('FOOD_CONFIDENCE_THRESHOLD', '0.65'))
        self.topk = int(os.getenv('FOOD_TOPK', '5'))
        self.enhance_image = os.getenv('FOOD_ENHANCE_IMAGE', '0') == '1'
        self.gate_enabled = os.getenv('FOOD_OOD_GATE_ENABLED', '1') != '0'
        self.gate_path = os.getenv('FOOD_OOD_GATE_PATH', DEFAULT_OOD_GATE)
        self.gate_policy = os.getenv('FOOD_OOD_GATE_POLICY', 'combo').strip().lower()
        self.gate = None
        self.gate_payload = None
        self.gate_threshold = None
        self.combo_gate_threshold = float(os.getenv('FOOD_OOD_COMBO_GATE_THRESHOLD', '0.535'))
        self.combo_confidence_threshold = float(os.getenv('FOOD_OOD_COMBO_CONFIDENCE_THRESHOLD', '0.755'))
        self.feature_layer = None
        print(f"加载模型: {resolved}")

        ckpt = torch.load(resolved, map_location='cpu', weights_only=False)
        self.model = ckpt['model']
        self.model.eval()
        self.names = self._normalize_names(ckpt.get('names', getattr(self.model, 'names', {})))
        self._validate_food101_checkpoint(resolved)
        self._init_ood_gate()
        print(
            f"模型加载成功 ({len(self.names)} 类, threshold={self.confidence_threshold:.2f}, "
            f"gate={'on' if self.gate_enabled and self.gate is not None else 'off'})"
        )

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

            if self.enhance_image:
                sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                img_bgr = cv2.filter2D(img_bgr, -1, sharpen)
                img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=5)
            # BGR → RGB → PIL → transforms → 推理
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            x = self.transform(img_pil).unsqueeze(0)

            out, embedding = self._forward_with_embedding(x)
            if isinstance(out, (list, tuple)):
                out = out[0]

            probs = self._as_probabilities(out)[0]
            top1_id = int(probs.argmax())
            top1_conf = float(probs[top1_id])
            food_name = self.names.get(top1_id, f'未知{top1_id}')
            cn_name = self._get_chinese_name(food_name)
            candidates = self._top_candidates(probs)
            gate_decision = self._score_gate(out, probs, embedding)

            if top1_conf < self.confidence_threshold:
                return {
                    'success': False,
                    'error': f'低置信度结果，最高仅 {top1_conf:.1%}',
                    'detections': [],
                    'candidates': candidates,
                    'gate': gate_decision,
                    'count': 0,
                    'message': '未能可靠识别食物，请换一张更清晰、主体更完整的照片',
                }

            if not gate_decision['accepted']:
                score_text = ''
                if gate_decision.get('score') is not None:
                    score_text = f"，gate={gate_decision['score']:.3f}"
                return {
                    'success': False,
                    'error': f'OOD gate 拒绝结果{score_text}',
                    'detections': [],
                    'candidates': candidates,
                    'gate': gate_decision,
                    'count': 0,
                    'message': '未能可靠识别为当前支持的 Food-101 食物类别',
                }

            return {
                'success': True,
                'detections': [{
                    'name': cn_name,
                    'english_name': food_name,
                    'chinese_name': cn_name,
                    'confidence': top1_conf,
                    'bbox': None,
                }],
                'candidates': candidates,
                'gate': gate_decision,
                'count': 1,
                'message': f'识别到: {cn_name}',
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'detections': []}

    def _normalize_names(self, raw_names):
        if isinstance(raw_names, dict):
            return {int(k): v for k, v in raw_names.items()}
        if isinstance(raw_names, (list, tuple)):
            return {i: name for i, name in enumerate(raw_names)}
        return {}

    def _validate_food101_checkpoint(self, model_path):
        if os.getenv('ALLOW_NON_FOOD_MODEL', '0') == '1':
            return

        names = set(self.names.values())
        imagenet_markers = {
            'great_white_shark',
            'tiger_shark',
            'hammerhead',
            'goldfish',
            'tench',
        }
        if len(self.names) != 101 or names.intersection(imagenet_markers):
            raise RuntimeError(
                f"{model_path} 看起来不是 Food-101 微调后的 101 类检查点 "
                f"(当前类别数: {len(self.names)})。请设置 YOLO_MODEL_PATH 指向 "
                "runs/classify/food_cls_l/weights/best.pt；如确需加载非 Food-101 模型，"
                "设置 ALLOW_NON_FOOD_MODEL=1。"
            )

    def _init_ood_gate(self):
        if not self.gate_enabled:
            return

        if self.gate_policy not in {'combo', 'gate'}:
            print(f"未知 FOOD_OOD_GATE_POLICY={self.gate_policy}，回退到 combo")
            self.gate_policy = 'combo'

        try:
            payload = torch.load(self.gate_path, map_location='cpu', weights_only=False)
            gate = GateMLP(int(payload['input_dim']), int(payload.get('hidden_dim', 0)))
            gate.load_state_dict(payload['gate_state_dict'])
            gate.eval()
            self.gate = gate
            self.gate_payload = payload
            self.gate_threshold = float(os.getenv('FOOD_OOD_GATE_THRESHOLD', str(payload['threshold'])))
            self.feature_layer = self._find_feature_layer()
            print(
                f"OOD gate 加载成功: {self.gate_path} "
                f"(policy={self.gate_policy}, threshold={self.gate_threshold:.3f}, "
                f"combo_gate={self.combo_gate_threshold:.3f}, "
                f"combo_conf={self.combo_confidence_threshold:.3f})"
            )
        except Exception as e:
            if os.getenv('FOOD_OOD_GATE_REQUIRED', '0') == '1':
                raise
            self.gate_enabled = False
            self.gate = None
            self.gate_payload = None
            self.feature_layer = None
            print(f"OOD gate 加载失败，已禁用: {e}")

    def _find_feature_layer(self):
        module = self.model
        for part in ('model', '9', 'drop'):
            module = getattr(module, part) if not part.isdigit() else module[int(part)]
        return module

    def _forward_with_embedding(self, x):
        if self.feature_layer is None:
            with torch.no_grad():
                return self.model(x), None

        captured = []

        def hook(_module, _inputs, output):
            captured.append(output.detach())

        handle = self.feature_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.model(x)
        finally:
            handle.remove()

        embedding = captured[-1].float().flatten(1) if captured else None
        return out, embedding

    def _score_gate(self, logits, probs, embedding):
        diagnostics = {
            'enabled': bool(self.gate_enabled and self.gate is not None),
            'policy': self.gate_policy,
            'accepted': True,
        }
        if not diagnostics['enabled']:
            diagnostics['reason'] = 'disabled'
            return diagnostics

        if embedding is None:
            diagnostics['reason'] = 'missing_embedding'
            diagnostics['accepted'] = os.getenv('FOOD_OOD_GATE_REQUIRED', '0') != '1'
            return diagnostics

        features = self._gate_features(probs, embedding)
        payload = self.gate_payload
        normalized = (features - payload['feature_mean']) / payload['feature_std'].clamp_min(1e-6)
        with torch.no_grad():
            score = float(torch.sigmoid(self.gate(normalized))[0])

        top1_conf = float(probs.max())
        gate_accept = score >= self.gate_threshold
        combo_accept = gate_accept or top1_conf >= self.combo_confidence_threshold
        accepted = gate_accept if self.gate_policy == 'gate' else combo_accept
        diagnostics.update({
            'accepted': bool(accepted),
            'score': score,
            'threshold': self.gate_threshold,
            'gate_accept': bool(gate_accept),
            'combo_accept': bool(combo_accept),
            'combo_gate_threshold': self.combo_gate_threshold,
            'combo_confidence_threshold': self.combo_confidence_threshold,
            'features': {
                name: float(value)
                for name, value in zip(
                    (
                        'top1_conf',
                        'margin',
                        'normalized_entropy',
                        'top5_mass',
                        'predicted_centroid_cosine',
                        'nearest_centroid_cosine',
                        'nearest_minus_predicted',
                        'embedding_norm',
                    ),
                    features[0].tolist(),
                )
            },
        })
        return diagnostics

    def _gate_features(self, probs, embedding):
        probs_batch = probs.unsqueeze(0)
        topk = min(5, probs_batch.shape[1])
        top_vals, top_ids = torch.topk(probs_batch, k=max(2, topk), dim=1)
        top1 = top_vals[:, 0]
        margin = top_vals[:, 0] - top_vals[:, 1]
        entropy = -(probs_batch.clamp_min(1e-12) * probs_batch.clamp_min(1e-12).log()).sum(dim=1) / math.log(probs_batch.shape[1])
        top5_mass = top_vals[:, :topk].sum(dim=1)
        emb_norm = embedding.norm(dim=1)
        emb_unit = torch.nn.functional.normalize(embedding, dim=1)
        centroids = self.gate_payload['centroids']
        cosine = emb_unit @ centroids.t()
        predicted = top_ids[:, 0]
        pred_cos = cosine.gather(1, predicted.unsqueeze(1)).squeeze(1)
        nearest_cos, _nearest_id = cosine.max(dim=1)
        return torch.stack([
            top1,
            margin,
            entropy,
            top5_mass,
            pred_cos,
            nearest_cos,
            nearest_cos - pred_cos,
            emb_norm,
        ], dim=1)

    def _as_probabilities(self, out):
        probs = out.float()
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)

        row_sums = probs.sum(dim=1)
        sums_to_one = torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-4,
        )
        in_probability_range = bool(torch.all((probs >= 0) & (probs <= 1)).item())
        if in_probability_range and sums_to_one:
            return probs

        return torch.softmax(out, dim=1)

    def _top_candidates(self, probs):
        k = min(max(self.topk, 1), int(probs.numel()))
        confs, ids = torch.topk(probs, k=k)
        candidates = []
        for idx, conf in zip(ids.tolist(), confs.tolist()):
            english_name = self.names.get(int(idx), f'未知{idx}')
            chinese_name = self._get_chinese_name(english_name)
            candidates.append({
                'class_id': int(idx),
                'name': chinese_name,
                'english_name': english_name,
                'chinese_name': chinese_name,
                'confidence': float(conf),
            })
        return candidates

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
