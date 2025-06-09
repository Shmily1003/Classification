import torch
import random
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from model import MultiTaskEfficientNet, MultiTaskDataset, get_transform

from logger import setup_my_logger
import logging

def load_model(model_path, device):
    """加载训练好的模型，支持完整断点checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pattern_classes = checkpoint.get('pattern_classes', [])
        number_classes = checkpoint.get('number_classes', [])
        num_pattern_classes, num_number_classes  = len(pattern_classes), len(number_classes)
        
        model = MultiTaskEfficientNet(
            num_number_classes=num_number_classes,
            num_pattern_classes=num_pattern_classes
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        my_logger.info(f"Model loaded from {model_path}")
        my_logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A') + 1}")
        my_logger.info(f"Best accuracy so far: {checkpoint.get('best_acc', 'N/A')}")
        my_logger.info(f"Number classes: {number_classes}")
        my_logger.info(f"Pattern classes: {pattern_classes}")
        
        return model, pattern_classes, number_classes
    else:
        # 如果 checkpoint 直接是 state_dict
        my_logger.warning("checkpoint does not contain 'model_state_dict' key, assume direct state_dict")
        model = MultiTaskEfficientNet(
            num_number_classes=100,
            num_pattern_classes=15
        )
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, [], []


def test_single_image(model, image_path, transform, device, pattern_classes, number_classes):
    """测试单张图片"""
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model.predict(input_tensor)
    
    o1 = predictions['o1'].item()
    o2 = predictions['o2'].item()
    o3 = predictions['o3'].item()
    final_output = predictions['final_output'].item()
    o1_confidence = predictions['o1_confidence'].item()
    o2_confidence = predictions['o2_confidence'].item()
    o3_confidence = predictions['o3_confidence'].item()
    
    my_logger.info(f"测试图片: {image_path}")
    my_logger.info(f"预测结果:")
    my_logger.info(f"  O1 (数字/图案): {'图案' if o1 == 1 else '数字'} (置信度: {o1_confidence:.4f})")
    if o1 == 0:
        my_logger.info(f"  O2 (数字类别): {o2} (置信度: {o2_confidence:.4f})")
        my_logger.info(f"  O3 (图案类别): {o3} (未激活)")
        my_logger.info(f"  预测的数字: {number_classes[o2]}")
    else:
        my_logger.info(f"  O2 (数字类别): {o2} (未激活)")
        my_logger.info(f"  O3 (图案类别): {o3} (置信度: {o3_confidence:.4f})")
        if o3 < len(pattern_classes):
            my_logger.info(f"  预测的图案: {pattern_classes[o3]}")
        else:
            my_logger.info(f"  预测的图案: Unknown (index {o3})")
    
    return {
        'o1': o1,
        'o2': o2,
        'o3': o3,
        'final_output': final_output,
        'confidences': {
            'o1': o1_confidence,
            'o2': o2_confidence,
            'o3': o3_confidence
        },
        'original_image': original_image
    }


def test_random_samples(model, dataset, transform, device, pattern_classes, number_classes, num_samples=5):
    """测试随机样本"""
    my_logger.info(f"\n{'='*60}")
    my_logger.info(f"随机测试 {num_samples} 个样本")
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    results = []
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_path = sample['path']
        my_logger.info(f"\n样本 {i+1}/{len(indices)}")
        
        result = test_single_image(model, image_path, transform, device, pattern_classes, number_classes)
        results.append(result)
        
    return results


def main():
    parser = argparse.ArgumentParser(description='Test MultiTask EfficientNet')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', 
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a specific image to test')
    parser.add_argument('--num_random', type=int, default=5, help='Number of random samples to test')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_logger.info(f'Using device: {device}')
    
    try:
        model, pattern_classes, number_classes = load_model(args.model_path, device)
    except FileNotFoundError:
        my_logger.error(f"Model file not found at {args.model_path}")
        my_logger.error("Please make sure you have trained the model or specify correct checkpoint path")
        return
    except Exception as e:
        my_logger.error(f"Error loading model: {e}")
        return
    
    transform = get_transform('test')
    
    dataset = MultiTaskDataset(args.data_dir, transform=None)
    if len(dataset) == 0:
        my_logger.error("Dataset is empty. Please check your dataset directory.")
        return
    
    if args.image_path:
        my_logger.info(f"Testing specific image: {args.image_path}")
        try:
            test_single_image(model, args.image_path, transform, device, pattern_classes)
        except FileNotFoundError:
            my_logger.error(f"Image file not found at {args.image_path}")
        except Exception as e:
            my_logger.error(f"Error testing image: {e}")
    else:
        test_random_samples(model, dataset, transform, device, pattern_classes, number_classes, args.num_random)


if __name__ == "__main__":
    import sys
    my_logger = setup_my_logger(
        stream=dict(enable=False, level=logging.DEBUG),
        file=dict(enable=True, level=logging.INFO),
        cfg=dict(log_root='my_logs')
    )
    main()
