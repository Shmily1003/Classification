# 多任务EfficientNetV2分类器

基于EfficientNetV2-B1的多任务分类模型，用于同时分类数字(0-99)和图案类别。

## 项目结构

```
project/
├── dataset/                # 数据集目录
│   ├── number/             # 数字类别
│   │   ├── 0/              # 数字0的图片
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── 1/              # 数字1的图片
│   │   └── ...
│   │   └── 99/             # 数字99的图片
│   └── pattern/            # 图案类别
│       ├── H_multimeter/   # 图案类别1
│       │   ├── 1.jpg
│       │   └── ...
│       ├── I_printer/      # 图案类别2
│       ├── J_keyboard/     # 图案类别3
│       └── ...
├── checkpoints/            # 模型保存目录（训练后生成）
├── logger.py               # 日志配置
├── model.py                # 模型定义
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
└── README.md               # 说明文档
```

## 模型架构

### 设计思路
- **Backbone**: EfficientNetV2-B1 预训练模型
- **多任务输出**: 三个分类头
  - `o1`: 2分类 (数字=0, 图案=1)
  - `o2`: 100分类 (数字子类别 0-99)
  - `o3`: 15分类 (图案子类别)

### 条件激活机制
- 当 `o1=0` (数字) 时，激活 `o2`，`o3` 输出为0
- 当 `o1=1` (图案) 时，激活 `o3`，`o2` 输出为0
- 最终输出: `final_output = o1*100 + o2 + o3`

## 安装依赖

```bash
pip install torch torchvision timm pillow matplotlib tqdm
```

## 使用方法

### 1. 训练模型

基本训练：
```bash
python train.py
```

自定义参数训练：
```bash
python train.py --data_dir dataset --batch_size 32 --num_epochs 50 --learning_rate 0.001
```

训练参数说明：
- `--data_dir`: 数据集目录路径
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--save_dir`: 模型保存目录
- `--alpha`, `--beta`, `--gamma`: 三个分类头的损失权重

### 2. 测试模型

快速测试（随机选择数据集中的一张图片）：
```bash
python test.py
```

测试指定图片：
```bash
python test.py --image_path path/to/your/image.jpg
```

测试多个随机样本：
```bash
python test.py --num_random 10
```

可视化结果：
```bash
python test.py --visualize --save_viz result.png
```

测试参数说明：
- `--model_path`: 训练好的模型路径
- `--data_dir`: 数据集目录
- `--image_path`: 指定测试图片路径
- `--num_random`: 随机测试样本数量
- `--visualize`: 是否显示可视化结果
- `--save_viz`: 保存可视化结果的路径

## 输出格式

### 训练输出
```
Epoch 1/50
--------------------------------------------------
Train Loss: 1.2345
Train O1 Acc: 0.8500
Train O2 Acc: 0.7800
Train O3 Acc: 0.8200
Current LR: 0.001000
New best model saved with accuracy: 0.8167
```

### 测试输出
```
============================================================
测试图片: dataset/number/5/1.jpg
============================================================
预测结果:
  O1 (数字/图案): 数字 (置信度: 0.9876)
  O2 (数字类别): 5 (置信度: 0.9234)
  O3 (图案类别): 0 (未激活)
  预测的数字: 5

最终输出 (o1*100 + o2 + o3): 5
```

## 模型特点

1. **多任务学习**: 同时学习类别判断和子类别分类
2. **条件激活**: 根据主分类结果激活对应的子分类头
3. **损失权重**: 支持调整不同任务的损失权重
4. **数据增强**: 训练时使用随机翻转、旋转等数据增强
5. **学习率调度**: 使用StepLR进行学习率衰减

## 性能优化

- 使用预训练的EfficientNetV2-B1作为backbone
- 实现了条件损失计算，只对相关样本计算损失
- 支持GPU加速训练
- 使用数据增强提高模型泛化能力
- 实现了学习率调度和权重衰减

## 故障排除

1. **数据集为空**: 检查数据集目录结构是否正确
2. **模型加载失败**: 确保已完成训练并且模型文件存在
3. **GPU内存不足**: 减少batch_size
4. **收敛困难**: 调整学习率或损失权重
