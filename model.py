import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import timm

import logging
my_logger = logging.getLogger('my_debug_logger')  # 获取已配置的logger

class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.number_classes = []
        self.pattern_classes = []

        # 处理数字类别 (0-99)
        number_dir = self.root_dir / 'number'
        if number_dir.exists():
            self.number_classes = sorted([d.name for d in number_dir.iterdir() if d.is_dir()])
            for idx, number_class in enumerate(self.number_classes):
                class_dir = number_dir / number_class
                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append({
                        'path': img_file,
                        'o1': 0,  # 0表示数字
                        'o2': idx,  # 数字子类别
                        'o3': 0,   # 图案子类别（不使用时为0）
                        'name': number_class
                    })
        
        # 处理图案类别
        pattern_dir = self.root_dir / 'pattern'
        if pattern_dir.exists():
            self.pattern_classes = sorted([d.name for d in pattern_dir.iterdir() if d.is_dir()])
            for idx, pattern_class in enumerate(self.pattern_classes):
                class_dir = pattern_dir / pattern_class
                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append({
                        'path': img_file,
                        'o1': 1,  # 1表示图案
                        'o2': 0,  # 数字子类别（不使用时为0）
                        'o3': idx, # 图案子类别
                        'name': pattern_class
                    })
        my_logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'o1': sample['o1'],
            'o2': sample['o2'],
            'o3': sample['o3'],
            'path': str(sample['path']),
            'class': sample.get('name')
        }

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_number_classes=100, num_pattern_classes=15):
        super(MultiTaskEfficientNet, self).__init__()
        
        # 使用EfficientNetV2-B1作为backbone
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=0)
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        my_logger.info(f"Backbone feature dimension: {self.feature_dim}")
        
        # 三个分类头
        self.o1_head = nn.Linear(self.feature_dim, 2)  # 数字 vs 图案
        self.o2_head = nn.Linear(self.feature_dim, num_number_classes)  # 数字子类别
        self.o3_head = nn.Linear(self.feature_dim, num_pattern_classes)  # 图案子类别
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, return_features=False):
        # 提取特征
        features = self.backbone(x)
        features = self.dropout(features)
        
        my_logger.debug(f"Extracted features shape: {features.shape}")

        # 第一个分类头：数字 vs 图案
        o1_logits = self.o1_head(features)
        my_logger.debug(f"o1_logits shape: {o1_logits.shape}")

        # 第二个分类头：数字子类别
        o2_logits = self.o2_head(features)
        my_logger.debug(f"o2_logits shape: {o2_logits.shape}")

        # 第三个分类头：图案子类别
        o3_logits = self.o3_head(features)
        my_logger.debug(f"o3_logits shape: {o3_logits.shape}")

        if return_features:
            return o1_logits, o2_logits, o3_logits, features
        
        return o1_logits, o2_logits, o3_logits
    
    def predict(self, x):
        """预测函数，实现条件激活逻辑"""
        with torch.no_grad():
            o1_logits, o2_logits, o3_logits = self.forward(x)
            
            # 获取o1的预测结果
            o1_pred = torch.argmax(o1_logits, dim=1)
            o1_prob = F.softmax(o1_logits, dim=1)

            # 获取o2和o3的预测结果和概率
            o2_pred = torch.argmax(o2_logits, dim=1)
            o2_prob = F.softmax(o2_logits, dim=1)
            o3_pred = torch.argmax(o3_logits, dim=1)
            o3_prob = F.softmax(o3_logits, dim=1)
            
            # 实现条件激活：根据o1的结果决定使用哪个分类头
            batch_size = x.size(0)
            final_o2 = torch.zeros_like(o2_pred)
            final_o3 = torch.zeros_like(o3_pred)
            
            # 当o1=0时（数字），激活o2
            number_mask = (o1_pred == 0)
            final_o2[number_mask] = o2_pred[number_mask]
            
            # 当o1=1时（图案），激活o3
            pattern_mask = (o1_pred == 1)
            final_o3[pattern_mask] = o3_pred[pattern_mask]
            
            # 计算最终输出：o1*100 + o2 + o3
            final_output = o1_pred * 100 + final_o2 + final_o3
            
            return {
                'o1': o1_pred,
                'o2': final_o2,
                'o3': final_o3,
                'final_output': final_output,
                'o1_confidence': torch.max(o1_prob, dim=1)[0],
                'o2_confidence': torch.max(o2_prob, dim=1)[0],
                'o3_confidence': torch.max(o3_prob, dim=1)[0]
            }

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha  # o1损失权重
        self.beta = beta    # o2损失权重
        self.gamma = gamma  # o3损失权重
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, o1_logits, o2_logits, o3_logits, targets):
        o1_target = targets['o1']
        o2_target = targets['o2']
        o3_target = targets['o3']
        
        # 计算o1损失（总是计算）
        loss_o1 = self.ce_loss(o1_logits, o1_target)
        
        # 计算o2损失（只对数字样本计算）
        number_mask = (o1_target == 0)
        if number_mask.sum() > 0:
            loss_o2 = self.ce_loss(o2_logits[number_mask], o2_target[number_mask])
        else:
            loss_o2 = torch.tensor(0.0, device=o1_logits.device)
        
        # 计算o3损失（只对图案样本计算）
        pattern_mask = (o1_target == 1)
        if pattern_mask.sum() > 0:
            loss_o3 = self.ce_loss(o3_logits[pattern_mask], o3_target[pattern_mask])
        else:
            loss_o3 = torch.tensor(0.0, device=o1_logits.device)
        
        # 总损失
        total_loss = self.alpha * loss_o1 + self.beta * loss_o2 + self.gamma * loss_o3
        
        return total_loss, loss_o1, loss_o2, loss_o3

def get_transform(mode='train'):
    """获取数据预处理变换"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])