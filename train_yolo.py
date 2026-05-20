import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path

# Windows + conda may load OpenMP runtime twice (e.g. torch + opencv), causing startup failure.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - explicit dependency error path
    raise ImportError("训练脚本需要安装 torch") from exc

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - explicit dependency error path
    raise ImportError("LoRA+ 训练需要安装 torchvision") from exc

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - explicit dependency error path
    raise ImportError("训练进度条需要安装 tqdm") from exc

from ultralytics import YOLO
import ultralytics


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        # 冻结原始线性层，只训练低秩适配器。
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)  # 冻结权重参数
        if self.base_layer.bias is not None: # 如果有偏置参数，也冻结它
            self.base_layer.bias.requires_grad_(False)

        self.rank = rank# 设置低秩矩阵的秩（内部维度）
        self.scaling = alpha / rank # 计算缩放系数，用于控制适配器更新的强度
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 可选的dropout层，用于正则化
        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False) # 降维矩阵：从原始特征维度降至低秩维度
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False) # 升维矩阵：从低秩维度恢复至原始输出维度

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:#前向传播时，先计算原始线性层的输出，再计算LoRA适配器的输出，并将两者相加。适配器输出会乘以缩放系数，以控制其对最终输出的影响。
        # 输出 = 原模型输出 + LoRA 低秩增量。
        return self.base_layer(inputs) + self.lora_up(self.lora_down(self.dropout(inputs))) * self.scaling

    def merged_weight(self) -> torch.Tensor:# 计算 LoRA 模块的权重增量，等效于把 LoRA 的 down 和 up 两部分合成一个矩阵，直接加到原始权重上。
        delta = self.lora_up.weight @ self.lora_down.weight
        return delta * self.scaling

    def to_base_module(self) -> nn.Linear:
        merged = copy.deepcopy(self.base_layer)
        merged.weight.data = merged.weight.data + self.merged_weight().view_as(merged.weight.data)
        return merged


class LoRAConv2d(nn.Module):#卷积层的 LoRA 实现，和线性层类似，但只支持 1x1 卷积，以保持实现简单。
    def __init__(self, base_layer: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        if base_layer.kernel_size != (1, 1):
            raise ValueError("LoRAConv2d only supports 1x1 convolutions")
        if base_layer.groups != 1:
            raise ValueError("LoRAConv2d only supports grouped conv=1")

        # 这里只给 1x1 卷积加 LoRA，避免把所有卷积都复杂化。
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=rank,
            kernel_size=1,
            stride=base_layer.stride,
            padding=0,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            in_channels=rank,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 同样是“原卷积输出 + 低秩残差”。
        return self.base_layer(inputs) + self.lora_up(self.lora_down(self.dropout(inputs))) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        down = self.lora_down.weight.view(self.rank, self.base_layer.in_channels)
        up = self.lora_up.weight.view(self.base_layer.out_channels, self.rank)
        delta = up @ down
        return delta.view(self.base_layer.out_channels, self.base_layer.in_channels, 1, 1) * self.scaling

    def to_base_module(self) -> nn.Conv2d:
        merged = copy.deepcopy(self.base_layer)
        merged.weight.data = merged.weight.data + self.merged_weight().view_as(merged.weight.data)
        return merged


def parse_args() -> argparse.Namespace:# 定义命令行参数，支持不同的训练策略（全量微调、分阶段微调、LoRA+），以及各种训练配置项。
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 classification model using a folder-based dataset"
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="Base model or checkpoint path (e.g. yolov8n-cls.pt or runs/classify/food_cls/weights/last.pt)",
    )
    parser.add_argument(
        "--data",
        default="datasets/food_cls",
        help="Dataset root directory. Must contain train/<class_name>/ and val/<class_name>/",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. 0 / cpu / 0,1")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="food_cls")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument(
        "--strategy",
        choices=["full", "freeze_then_unfreeze", "lora_plus"],
        default="full",
        help="full: full-parameter fine-tuning; freeze_then_unfreeze: staged fine-tuning; lora_plus: parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=10,
        help="Number of layers to freeze in stage 1 when using freeze_then_unfreeze",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=20,
        help="Stage 1 epochs (frozen training) when using freeze_then_unfreeze",
    )
    parser.add_argument(
        "--unfreeze-epochs",
        type=int,
        default=30,
        help="Stage 2 epochs (unfrozen training) when using freeze_then_unfreeze",
    )
    parser.add_argument(
        "--aug-strength",
        choices=["weak", "medium", "strong"],
        default="medium",
        help="Preset data augmentation strength",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous checkpoint. Use --model to point to last.pt.",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="Optional LoRA+ training-state path (lora_training_state.pt). If omitted, defaults to runs/<name>/lora_training_state.pt.",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for lora_plus training")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout applied before LoRA adapters")
    parser.add_argument(
        "--lora-plus-ratio",
        type=float,
        default=16.0,
        help="Learning-rate multiplier for LoRA up-projection parameters",
    )
    return parser.parse_args()


def get_aug_params(aug_strength: str) -> dict:
    if aug_strength == "weak":
        return {
            "hsv_h": 0.01,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
            "degrees": 0.0,
            "translate": 0.05,
            "scale": 0.3,
            "fliplr": 0.3,
        }
    if aug_strength == "strong":
        return {
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.6,
            "degrees": 10.0,
            "translate": 0.15,
            "scale": 0.6,
            "fliplr": 0.5,
        }
    return {
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
    }


def build_image_transforms(imgsz: int, aug_strength: str) -> tuple[transforms.Compose, transforms.Compose]:
    # LoRA+ 训练这里不再依赖 Ultralytics 内部增强，而是显式写出训练/验证变换。
    # 不同增强强度对应不同的 ColorJitter 参数、随机翻转概率和随机裁剪范围。 
    if aug_strength == "weak":
        jitter = transforms.ColorJitter(0.05, 0.05, 0.05, 0.02)
        flip_prob = 0.3
        crop_scale = (0.8, 1.0)
    elif aug_strength == "strong":
        jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        flip_prob = 0.5
        crop_scale = (0.6, 1.0)
    else:
        jitter = transforms.ColorJitter(0.12, 0.12, 0.12, 0.05)
        flip_prob = 0.5
        crop_scale = (0.7, 1.0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 训练变换包含随机裁剪、随机水平翻转、颜色抖动；验证变换则是简单的缩放和中心裁剪。两者最后都要转换成 Tensor 并归一化。
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(imgsz, scale=crop_scale),
            transforms.RandomHorizontalFlip(p=flip_prob),
            jitter,
            transforms.ToTensor(),
            normalize,
        ]
    )
    # 验证变换则是简单的缩放和中心裁剪。两者最后都要转换成 Tensor 并归一化。
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(imgsz * 1.1)),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def resolve_device(device_arg: str | None) -> torch.device:
    # 兼容 cpu、cuda、0、0,1 这几类输入。
    if device_arg is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device_text = str(device_arg).strip().lower()
    if device_text == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_text.startswith("cuda"):
        return torch.device(device_text)
    primary_index = device_text.split(",")[0]
    return torch.device(f"cuda:{primary_index}")


def build_classification_dataloaders(
    data_root: str,
    imgsz: int,
    batch_size: int,
    workers: int,
    aug_strength: str,
) -> tuple[DataLoader, DataLoader]:
    # 直接用 ImageFolder 读取 train/val 目录，类别名就是文件夹名。
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    train_transform, val_transform = build_image_transforms(imgsz, aug_strength)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and val class folders must match exactly")

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": max(0, workers),
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def replace_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> None:
    # 递归扫描整个模型，把可插入的 Linear / 1x1 Conv 替换成 LoRA 模块。
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear):#全连接层直接替换成 LoRALinear 包装器，后者内部会冻结原层参数并添加可训练的适配器权重。
            setattr(module, child_name, LoRALinear(child_module, rank, alpha, dropout))
            continue
        # 1x1 卷积层替换成 LoRAConv2d 包装器，其他类型的层保持不变。递归调用确保整个模型都被扫描到。
        if isinstance(child_module, nn.Conv2d) and child_module.kernel_size == (1, 1) and child_module.groups == 1:
            setattr(module, child_name, LoRAConv2d(child_module, rank, alpha, dropout))
            continue
        replace_with_lora(child_module, rank, alpha, dropout)


def strip_lora_modules(module: nn.Module) -> None:
    # 保存权重前，把 LoRA 合并回普通层，导出成标准 PyTorch checkpoint。
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, LoRALinear):
            setattr(module, child_name, child_module.to_base_module())
            continue
        if isinstance(child_module, LoRAConv2d):
            setattr(module, child_name, child_module.to_base_module())
            continue
        strip_lora_modules(child_module)


def collect_lora_parameter_groups(model: nn.Module, base_lr: float, lora_plus_ratio: float) -> list[dict]:
    # LoRA+ 的核心：LoRA 的 down / up 两部分使用不同学习率。
    down_params = []
    up_params = []
    other_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_down" in name:
            down_params.append(parameter)
        elif "lora_up" in name:
            up_params.append(parameter)
        else:
            other_params.append(parameter)

    parameter_groups = []
    if down_params:
        parameter_groups.append({"params": down_params, "lr": base_lr, "weight_decay": 0.0})
    if up_params:
        parameter_groups.append({"params": up_params, "lr": base_lr * lora_plus_ratio, "weight_decay": 0.0})
    if other_params:
        parameter_groups.append({"params": other_params, "lr": base_lr, "weight_decay": 0.0})

    if not parameter_groups:
        raise ValueError("No trainable parameters found for lora_plus")

    return parameter_groups


def unwrap_model_output(outputs: torch.Tensor | tuple | list | dict) -> torch.Tensor:
    # YOLOv8 分类前向在不同版本里可能返回 Tensor、tuple/list 或 dict。
    # 这里优先取分类 logits / preds；如果没有明确键，再递归找第一个张量。
    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, dict):
        for key in ("logits", "preds", "pred", "output", "outputs"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
        for value in outputs.values():
            if torch.is_tensor(value):
                return value
            if isinstance(value, (tuple, list, dict)):
                try:
                    return unwrap_model_output(value)
                except TypeError:
                    continue

    if isinstance(outputs, (tuple, list)) and outputs:
        for value in outputs:
            if torch.is_tensor(value):
                return value
            if isinstance(value, (tuple, list, dict)):
                try:
                    return unwrap_model_output(value)
                except TypeError:
                    continue

    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    # 手写一个分类训练循环，便于完全控制 LoRA+ 的优化方式。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    skipped_batches = 0

    progress_bar = tqdm(loader, desc=f"[Train][Epoch {epoch}/{total_epochs}]", total=len(loader), leave=False)
    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = unwrap_model_output(model(images))
        loss = criterion(logits, targets)

        if not math.isfinite(loss.item()):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    progress_bar.close()

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    if skipped_batches > 0:
        print(f"  [Train Warning] Skipped {skipped_batches} batches due to NaN loss")
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_batches = 0

    progress_bar = tqdm(loader, desc=f"[Val][Epoch {epoch}/{total_epochs}]", total=len(loader), leave=False)
    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        try:
            logits = unwrap_model_output(model(images))
            loss = criterion(logits, targets)

            if not math.isfinite(loss.item()):
                nan_batches += 1
                continue

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size
        except Exception as e:
            print(f"  [Error in val batch] {e}")
            nan_batches += 1
            continue

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    progress_bar.close()

    if nan_batches > 0:
        print(f"  [Warning] Skipped {nan_batches} validation batches due to NaN")

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def save_training_history(history: list[dict], run_dir: Path) -> None:
    if not history:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "history.csv"
    json_path = run_dir / "history.json"

    fieldnames = list(history[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(history, json_file, ensure_ascii=False, indent=2)

def save_training_history_snapshot(history: list[dict], run_dir: Path) -> None:
    # 训练过程中每轮都刷新一次，避免中途退出时只剩 state 文件没有 CSV。
    save_training_history(history, run_dir)


def resolve_v8_cls_yaml() -> str:
    ul_root = Path(ultralytics.__file__).resolve().parent
    candidates = [
        ul_root / "cfg" / "models" / "v8" / "yolov8-cls.yaml",
        ul_root / "cfg" / "models" / "v8" / "yolo" / "yolov8-cls.yaml",
    ]
    for cls_yaml in candidates:
        if cls_yaml.exists():
            return str(cls_yaml)
    raise FileNotFoundError(f"classification model yaml not found in candidates: {candidates}")


def build_model_for_strategy(model_arg: str, strategy: str) -> YOLO:
    # For Ultralytics built-in training strategies, classification needs a cls model architecture.
    if strategy in {"full", "freeze_then_unfreeze"} and Path(model_arg).name.lower() == "yolov8n.pt":
        cls_yaml = resolve_v8_cls_yaml()
        return YOLO(cls_yaml).load(model_arg)
    return YOLO(model_arg)


def build_classification_yolo(model_arg: str) -> YOLO:
    # LoRA+ 也必须使用分类结构，避免把检测头输出喂给分类损失。
    if Path(model_arg).name.lower() == "yolov8n.pt":
        cls_yaml = resolve_v8_cls_yaml()
        return YOLO(cls_yaml).load(model_arg)
    return YOLO(model_arg)


def resolve_run_artifact_path(raw_path: str, fallback_paths: list[Path]) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    for fallback_path in fallback_paths:
        if fallback_path.exists():
            return fallback_path
    return path


def load_plain_checkpoint_into_lora_model(model: nn.Module, plain_state_dict: dict) -> None:
    current_keys = set(model.state_dict().keys())
    remapped_state_dict: dict[str, torch.Tensor] = {}

    for key, value in plain_state_dict.items():
        if key in current_keys:
            remapped_state_dict[key] = value
            continue
        if key.endswith(".weight"):
            wrapped_key = f"{key[:-7]}.base_layer.weight"
            if wrapped_key in current_keys:
                remapped_state_dict[wrapped_key] = value
                continue
        if key.endswith(".bias"):
            wrapped_key = f"{key[:-5]}.base_layer.bias"
            if wrapped_key in current_keys:
                remapped_state_dict[wrapped_key] = value
                continue
        remapped_state_dict[key] = value

    model.load_state_dict(remapped_state_dict, strict=False)


def build_plain_checkpoint(
    model: nn.Module,
    state_dict: dict,
    metadata: dict,
) -> dict:
    # 复制一份模型并去掉 LoRA 包装，避免后续加载时还依赖这套训练代码。
    restored_model = copy.deepcopy(model)
    restored_model.load_state_dict(state_dict)
    strip_lora_modules(restored_model)
    return {
        "model": restored_model.cpu(),
        **metadata,
    }


def train_with_epoch_progress(model: YOLO, train_kwargs: dict, stage_label: str) -> None:
    epochs = int(train_kwargs.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    progress_bar = tqdm(total=epochs, desc=stage_label, ncols=100, leave=True)

    def on_fit_epoch_end(trainer) -> None:
        current_epoch = min(int(getattr(trainer, "epoch", 0)) + 1, epochs)
        progress_bar.update(1)
        progress_bar.set_postfix(epoch=f"{current_epoch}/{epochs}")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    try:
        model.train(**train_kwargs)
    finally:
        progress_bar.close()
        model.reset_callbacks()


def run_lora_plus_training(args: argparse.Namespace) -> None:
    # 这里就是 LoRA+ 的主流程：加载模型、注入适配器、训练、保存可直接推理的权重。
    if args.lora_rank <= 0:
        raise ValueError("lora-rank must be > 0")
    if args.lora_alpha <= 0:
        raise ValueError("lora-alpha must be > 0")
    if args.lora_plus_ratio <= 0:
        raise ValueError("lora-plus-ratio must be > 0")

    device = resolve_device(args.device)
    output_dir = Path(args.project) / args.name / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_run_artifact_path(
        args.model,
        [
            output_dir / Path(args.model).name,
            Path(args.project) / args.name / Path(args.model).name,
        ],
    )
    default_state_path = Path(args.project) / args.name / "lora_training_state.pt"
    state_path = resolve_run_artifact_path(
        args.resume_state if args.resume_state else str(default_state_path),
        [
            default_state_path,
        ],
    ) if (args.resume or args.resume_state) else default_state_path
    base_yolo = build_classification_yolo(str(model_path))
    model = base_yolo.model
    replace_with_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    model.to(device)

    train_loader, val_loader = build_classification_dataloaders(
        args.data,
        args.imgsz,
        args.batch,
        args.workers,
        args.aug_strength,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        collect_lora_parameter_groups(model, args.lr0, args.lora_plus_ratio),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))


    best_state = None
    last_state = None
    best_acc = -1.0
    history: list[dict] = []
    start_epoch = 1
    best_checkpoint_path = output_dir / "best.pt"

    print("Start LoRA+ fine-tuning with args:")
    print(f"  - device: {device}")
    print(f"  - data: {args.data}")
    print(f"  - model: {args.model}")
    print(f"  - resolved_model: {model_path}")
    print(f"  - imgsz: {args.imgsz}")
    print(f"  - batch: {args.batch}")
    print(f"  - epochs: {args.epochs}")
    print(f"  - lr0: {args.lr0}")
    print(f"  - lora_rank: {args.lora_rank}")
    print(f"  - lora_alpha: {args.lora_alpha}")
    print(f"  - lora_dropout: {args.lora_dropout}")
    print(f"  - lora_plus_ratio: {args.lora_plus_ratio}")
    print(f"  - resume: {args.resume}")
    print(f"  - resume_state: {state_path}")

    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(
                f"--resume was set, but no LoRA+ training state was found at: {state_path}. "
                "Use --resume-state to point to a valid lora_training_state.pt, or remove --resume to start a fresh run."
            )
        resume_state = torch.load(state_path, map_location=device)
        history = list(resume_state.get("history", []))
        finite_history: list[dict] = []
        for record in history:
            if all(
                not isinstance(record.get(key), (int, float)) or math.isfinite(float(record[key]))
                for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")
                if key in record
            ):
                finite_history.append(record)

        has_corruption = len(finite_history) != len(history)
        if has_corruption:
            best_checkpoint_path = output_dir / "best.pt"
            if best_checkpoint_path.exists():
                print(
                    f"Detected non-finite metrics in {state_path}; falling back to {best_checkpoint_path} "
                    "and discarding the corrupted optimizer state."
                )
                best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
                checkpoint_model = best_checkpoint.get("model")
                if hasattr(checkpoint_model, "state_dict"):
                    load_plain_checkpoint_into_lora_model(model, checkpoint_model.state_dict())
                history = finite_history
                best_acc = max((float(record.get("val_acc", -1.0)) for record in finite_history), default=-1.0)
                start_epoch = len(finite_history) + 1
            else:
                if "model_state" in resume_state:
                    model.load_state_dict(resume_state["model_state"])
                best_acc = float(resume_state.get("best_acc", -1.0))
                start_epoch = int(resume_state.get("epoch", 0)) + 1
                print(
                    f"Detected non-finite metrics in {state_path}, but no {best_checkpoint_path} was found; "
                    "continuing with the saved model state."
                )
        else:
            if "model_state" in resume_state:
                model.load_state_dict(resume_state["model_state"])
            if "optimizer_state" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer_state"])
            if "scheduler_state" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state"])
            best_acc = float(resume_state.get("best_acc", -1.0))
            start_epoch = int(resume_state.get("epoch", 0)) + 1

        print(f"Resumed LoRA+ state from: {state_path} (start_epoch={start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            # 每轮先训练，再验证；用验证集精度挑最优模型。
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                args.epochs,
            )
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                args.epochs,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                }
            )

            last_state = copy.deepcopy(model.state_dict())
            if val_acc >= best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "history": history,
                    "best_acc": best_acc,
                    "epoch": epoch,
                },
                state_path,
            )
            save_training_history_snapshot(history, output_dir.parent)
        except Exception as e:
            print(f"[ERROR] Epoch {epoch} failed: {e}")
            if last_state is not None:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "history": history,
                        "best_acc": best_acc,
                        "epoch": epoch,
                    },
                    state_path,
                )
                print(f"[ERROR] Saved recovery checkpoint to {state_path} for --resume")
            raise

    if last_state is None:
        raise RuntimeError("Training did not run any epoch")
    if best_state is None:
        best_state = copy.deepcopy(last_state)

    metadata = {
        "train_args": vars(args),
        "best_val_acc": best_acc,
    }

    last_ckpt = build_plain_checkpoint(model, last_state, metadata)
    best_ckpt = build_plain_checkpoint(model, best_state, metadata)

    # 输出的 best.pt / last.pt 都是普通 checkpoint，方便直接拿去推理。
    torch.save(last_ckpt, output_dir / "last.pt")
    torch.save(best_ckpt, output_dir / "best.pt")
    save_training_history(history, output_dir.parent)
    print(f"LoRA+ training finished. Saved checkpoints to: {output_dir}")


def main() -> None:
    args = parse_args()

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")

    if not os.path.isdir(args.data):
        raise FileNotFoundError(f"dataset directory not found: {args.data}")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val directory not found: {val_dir}")

    if not os.path.exists(args.model) and not args.model.endswith(".pt"):
        raise FileNotFoundError(f"model file not found: {args.model}")

    if args.strategy == "lora_plus":
        # 走 LoRA+ 分支时，不再调用 Ultralytics 的 train，而是走自定义训练循环。
        run_lora_plus_training(args)
        return

    model = build_model_for_strategy(args.model, args.strategy)

    common_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "resume": args.resume,
        "seed": args.seed, 
        "lr0": args.lr0,
        "amp": False,
    }
    common_kwargs.update(get_aug_params(args.aug_strength))

    if args.device is not None:
        common_kwargs["device"] = args.device
    elif torch.cuda.is_available():
        common_kwargs["device"] = 0

    common_kwargs["task"] = "classify"

    if args.strategy == "full":
        train_kwargs = {**common_kwargs, "epochs": args.epochs}

        print("Start full-parameter fine-tuning with args:")
        for k, v in train_kwargs.items():
            print(f"  - {k}: {v}")

        train_with_epoch_progress(model, train_kwargs, "[Full] train")
    else:
        if args.freeze_epochs <= 0 or args.unfreeze_epochs <= 0:
            raise ValueError("freeze-epochs and unfreeze-epochs must be > 0")

        stage1_name = f"{args.name}_freeze"
        stage1_kwargs = {
            **common_kwargs,
            "epochs": args.freeze_epochs,
            "freeze": args.freeze_layers,
            "name": stage1_name,
            "resume": False,
        }

        print("Start staged fine-tuning: stage 1 (freeze backbone) with args:")
        for k, v in stage1_kwargs.items():
            print(f"  - {k}: {v}")

        train_with_epoch_progress(model, stage1_kwargs, "[Freeze] train")

        stage1_last = os.path.join(
            args.project,
            stage1_name,
            "weights",
            "last.pt",
        )
        if not os.path.exists(stage1_last):
            raise FileNotFoundError(f"stage 1 checkpoint not found: {stage1_last}")

        stage2_model = YOLO(stage1_last)
        stage2_name = f"{args.name}_unfreeze"
        stage2_kwargs = {
            **common_kwargs,
            "epochs": args.unfreeze_epochs,
            "name": stage2_name,
            "resume": False,
        }

        print("Start staged fine-tuning: stage 2 (unfreeze all layers) with args:")
        for k, v in stage2_kwargs.items():
            print(f"  - {k}: {v}")

        train_with_epoch_progress(stage2_model, stage2_kwargs, "[Unfreeze] train")

    print("Training finished.")


if __name__ == "__main__":
    main()
