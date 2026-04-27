# Food Classification Dataset

Use this folder for food classification training.

## Required structure

- datasets/food_cls/train/<class_name>/*.jpg
- datasets/food_cls/val/<class_name>/*.jpg

Example:

- datasets/food_cls/train/apple/img001.jpg
- datasets/food_cls/train/banana/img002.jpg
- datasets/food_cls/val/apple/img101.jpg
- datasets/food_cls/val/banana/img102.jpg

## Notes

- No labels txt files are needed for classification.
- Class names come from folder names.
- Keep train and val class folders consistent.

## Train

Full fine-tuning:

python train_yolo.py --data datasets/food_cls --model yolov8n-cls.pt --epochs 50 --imgsz 224 --batch 32 --name food_cls

LoRA+ fine-tuning:

python train_yolo.py --strategy lora_plus --data datasets/food_cls --model yolov8n-cls.pt --epochs 50 --imgsz 224 --batch 32 --name food_cls_lora

The exported checkpoints will be saved to `runs/classify/<name>/weights/best.pt` and `last.pt`.
