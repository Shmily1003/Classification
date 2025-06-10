- [多任务EfficientNetV2分类器](#多任务efficientnetv2分类器)
  - [项目结构](#项目结构)
  - [模型架构](#模型架构)
    - [设计思路](#设计思路)
    - [条件激活机制](#条件激活机制)
  - [安装依赖](#安装依赖)
  - [使用方法](#使用方法)
    - [1. 训练模型](#1-训练模型)
    - [2. 测试模型](#2-测试模型)
    - [3.训练技巧](#3训练技巧)
  - [后续](#后续)

# 多任务EfficientNetV2分类器

基于EfficientNetV2-B1的多任务分类模型，用于同时分类数字(0-99)和图案类别。

## 项目结构

```
project/
├── dataset/                # 数据集目录
│   ├── number/             # 数字类别
│   │   ├── A0/              # 数字0的图片
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── A1/              # 数字1的图片
│   │   └── ...
│   │   └── J99/             # 数字99的图片
│   └── pattern/            # 图案类别
│       ├── H_multimeter/   # 图案类别1
│       │   ├── 1.jpg
│       │   └── ...
│       ├── I_printer/      # 图案类别2
│       ├── J_keyboard/     # 图案类别3
│       └── ...
├── config.yaml             # 配置文件
├── checkpoints/            # 模型保存目录（训练后生成）
├── logger.py               # 日志配置
├── model.py                # 模型定义
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
└── README.md               # 说明文档
```
A0、A1、H_multimeter等文件夹名字任意，图片名称任意，其他需保证完全一致（文件夹名字最好也相同）
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
- 输出说明：
```
if final_output < 100:
        my_logger.info(f"  预测的数字: {number_classes[final_output]}")
    else:
        my_logger.info(f"  预测的图案: {pattern_classes[final_output-100]}")
``` 
可自行添加置信度等更多输出
## 安装依赖

```bash
pip install torch torchvision timm pillow matplotlib tqdm
```

## 使用方法

### 1. 训练模型

基本训练：
```bash
python train.py  --config config.yaml
```

### 2. 测试模型

快速测试（随机选择数据集中的一张图片）：
```bash
python test.py  --config config.yaml
```

### 3.训练技巧
- 因为图片的数据集有点少，所以准确率明显高于数字，此时可以稍微增加数字loss的权重，反之同理

## 后续
- 只写了核心部分，其余部分写的有点粗糙，更多功能自行解决
- 该代码使用EfficientNetV2-B1作为backbone，参数量为22M左右，无明显变化，因此帧率应该也差不多
- 没确认过量化部署后art的输出，改写成tf后最好是和你的代码保证一致
- art如果输出是softmax的数据，可直接遍历查找最大的置信度，而不需要使用排序，相较来说应该也能再快一点点（也就是art中sort那块代码）
- 如果确实可行，就尝试使用不同的backbone以及测试集的制作
- 多实地测试不同条件下识别的鲁棒性以及准确性
- 最后的最后，祝省赛一切顺利！国赛一切顺利！
