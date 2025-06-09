import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
from model import MultiTaskEfficientNet, MultiTaskDataset, MultiTaskLoss, get_transform

from logger import setup_my_logger
import logging

def calculate_accuracy(outputs, targets, mask=None):
    """计算准确率"""
    if mask is not None:
        if mask.sum() == 0:
            return 0.0
        outputs = outputs[mask]
        targets = targets[mask]
    
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_loss_o1 = 0.0
    total_loss_o2 = 0.0
    total_loss_o3 = 0.0
    
    correct_o1 = 0
    correct_o2 = 0
    correct_o3 = 0
    total_samples = 0
    total_number_samples = 0
    total_pattern_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        targets = {
            'o1': batch['o1'].to(device),
            'o2': batch['o2'].to(device),
            'o3': batch['o3'].to(device)
        }
        
        optimizer.zero_grad()
        
        # 前向传播
        o1_logits, o2_logits, o3_logits = model(images)
        
        # 计算损失
        loss, loss_o1, loss_o2, loss_o3 = criterion(o1_logits, o2_logits, o3_logits, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计损失
        total_loss += loss.item()
        total_loss_o1 += loss_o1.item()
        total_loss_o2 += loss_o2.item()
        total_loss_o3 += loss_o3.item()
        
        # 计算准确率
        batch_size = images.size(0)
        total_samples += batch_size
        
        # o1准确率
        correct_o1 += calculate_accuracy(o1_logits, targets['o1']) * batch_size
        
        # o2准确率（只计算数字样本）
        number_mask = (targets['o1'] == 0)
        number_count = number_mask.sum().item()
        total_number_samples += number_count
        if number_count > 0:
            correct_o2 += calculate_accuracy(o2_logits, targets['o2'], number_mask) * number_count
        
        # o3准确率（只计算图案样本）
        pattern_mask = (targets['o1'] == 1)
        pattern_count = pattern_mask.sum().item()
        total_pattern_samples += pattern_count
        if pattern_count > 0:
            correct_o3 += calculate_accuracy(o3_logits, targets['o3'], pattern_mask) * pattern_count
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'O1_Acc': f'{correct_o1/total_samples:.3f}',
            'O2_Acc': f'{correct_o2/max(1, total_number_samples):.3f}',
            'O3_Acc': f'{correct_o3/max(1, total_pattern_samples):.3f}'
        })
    
    # 计算平均值
    avg_loss = total_loss / len(dataloader)
    avg_loss_o1 = total_loss_o1 / len(dataloader)
    avg_loss_o2 = total_loss_o2 / len(dataloader)
    avg_loss_o3 = total_loss_o3 / len(dataloader)
    
    acc_o1 = correct_o1 / total_samples
    acc_o2 = correct_o2 / max(1, total_number_samples)
    acc_o3 = correct_o3 / max(1, total_pattern_samples)
    
    return {
        'total_loss': avg_loss,
        'loss_o1': avg_loss_o1,
        'loss_o2': avg_loss_o2,
        'loss_o3': avg_loss_o3,
        'acc_o1': acc_o1,
        'acc_o2': acc_o2,
        'acc_o3': acc_o3
    }

def validate(model, dataloader, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0.0
    correct_o1 = 0
    correct_o2 = 0
    correct_o3 = 0
    total_samples = 0
    total_number_samples = 0
    total_pattern_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = {
                'o1': batch['o1'].to(device),
                'o2': batch['o2'].to(device),
                'o3': batch['o3'].to(device)
            }
            
            # 前向传播
            o1_logits, o2_logits, o3_logits = model(images)
            
            # 计算损失
            loss, _, _, _ = criterion(o1_logits, o2_logits, o3_logits, targets)
            total_loss += loss.item()
            
            # 计算准确率
            batch_size = images.size(0)
            total_samples += batch_size
            
            # o1准确率
            correct_o1 += calculate_accuracy(o1_logits, targets['o1']) * batch_size
            
            # o2准确率（只计算数字样本）
            number_mask = (targets['o1'] == 0)
            number_count = number_mask.sum().item()
            total_number_samples += number_count
            if number_count > 0:
                correct_o2 += calculate_accuracy(o2_logits, targets['o2'], number_mask) * number_count
            
            # o3准确率（只计算图案样本）
            pattern_mask = (targets['o1'] == 1)
            pattern_count = pattern_mask.sum().item()
            total_pattern_samples += pattern_count
            if pattern_count > 0:
                correct_o3 += calculate_accuracy(o3_logits, targets['o3'], pattern_mask) * pattern_count
    
    avg_loss = total_loss / len(dataloader)
    acc_o1 = correct_o1 / total_samples
    acc_o2 = correct_o2 / max(1, total_number_samples)
    acc_o3 = correct_o3 / max(1, total_pattern_samples)
    
    return {
        'val_loss': avg_loss,
        'val_acc_o1': acc_o1,
        'val_acc_o2': acc_o2,
        'val_acc_o3': acc_o3
    }

def main():
    parser = argparse.ArgumentParser(description='Train MultiTask EfficientNet')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--alpha', type=float, default=1.0, help='Loss weight for o1')
    parser.add_argument('--beta', type=float, default=1.0, help='Loss weight for o2')
    parser.add_argument('--gamma', type=float, default=1.0, help='Loss weight for o3')
    
    args = parser.parse_args()
    
    my_logger = setup_my_logger(
        stream=dict(enable=False, level=logging.DEBUG),
        file=dict(enable=True, level=logging.INFO),
        cfg=dict(log_root='my_logs')
    )

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_logger.info(f'Using device: {device}')
    
    # 创建数据集
    my_logger.info("Loading dataset...")
    train_dataset = MultiTaskDataset(args.data_dir, transform=get_transform('train'))
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        my_logger.error("Dataset is empty. Please check your dataset directory structure.")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 获取图案类别数量
    num_pattern_classes, num_number_classes = len(train_dataset.pattern_classes), len(train_dataset.number_classes)
    my_logger.info(f"Number of number classes: {num_number_classes}")
    my_logger.info(f"Number of pattern classes: {num_pattern_classes}")
    
    # 创建模型
    my_logger.info("Creating model...")
    model = MultiTaskEfficientNet(
        num_number_classes=num_number_classes, 
        num_pattern_classes=num_pattern_classes
    )
    model.to(device)
    
    # 损失函数和优化器
    criterion = MultiTaskLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 训练循环
    my_logger.info("Starting training...")
    best_acc = 0.0
    
    for epoch in range(args.num_epochs):
        my_logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        my_logger.info("-" * 50)
        
        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 打印训练结果
        my_logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        my_logger.info(f"Train O1 Acc: {train_metrics['acc_o1']:.4f}")
        my_logger.info(f"Train O2 Acc: {train_metrics['acc_o2']:.4f}")
        my_logger.info(f"Train O3 Acc: {train_metrics['acc_o3']:.4f}")
        my_logger.info(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        current_acc = (train_metrics['acc_o1'] + train_metrics['acc_o2'] + train_metrics['acc_o3']) / 3
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_metrics': train_metrics,
                'number_classes': train_dataset.number_classes,
                'pattern_classes': train_dataset.pattern_classes
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_acc:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_metrics': train_metrics,
                'number_classes': train_dataset.number_classes,
                'pattern_classes': train_dataset.pattern_classes
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    my_logger.info("Training completed!")
    my_logger.info(f"Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()