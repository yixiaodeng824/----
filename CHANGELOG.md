# 更新日志

## 2026-05-21 — v0.2 架构修复 + 预处理对齐

### 修复
- **分类器输出 1000→101**：模型不再输出 ImageNet 的 1000 类，改为 Food-101 的 101 类
- **分类器解冻**：分类器头不再被 LoRA 冻结，全量更新
- **预处理对齐训练与推理**：验证集 transforms 从 `Resize(246)+CenterCrop(224)` 改为 `Resize(224)+CenterCrop(224)`，与 YOLO 推理保持一致
- **模型导出写入类别名**：`build_plain_checkpoint` 保存 `model.names`，YOLO 加载后 `result.names` 能正确返回食物名
- **检测服务绕过 YOLO 预处理 Bug**：YOLO 的 ClassifyPredictor 硬编码 640x640 且重置 transforms，改用 PyTorch 原生推理保证预处理一致
- **续训 T_max 修复**：CosineAnnealingLR 的 T_max 保存/恢复，续训时学习率不会突变
- **续训 lr 覆盖**：`--resume` 时传 `--lr0` 会覆盖保存的旧学习率
- **错误处理**：训练循环加 try/except，崩溃时自动保存恢复点
- **图片锐化增强**：检测前对图片做锐化 + 对比度增强，改善压缩图片识别

### 新增
- **101 类中文名映射**：所有 Food-101 菜品的中文名 + 营养数据库
- **训练命令速查**：`training_commands.txt`
- **训练报表生成**：`generate_report.py`
- **--prefetch-factor 参数**：缓解 Windows 数据加载瓶颈
- **--amp 参数**：混合精度训练支持（当前关闭，可启用）

### 技术债
- LoRA+ 训练 78 轮 val_acc ≈ 79%，全量微调预期可到 88-92%
- 检测服务使用 PyTorch 原生推理（YOLO 预处理不兼容）
