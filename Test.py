import torch
import random
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from model import MultiTaskEfficientNet, MultiTaskDataset, get_transform


def load_model(model_path, device):
    """加载训练好的模型，支持完整断点checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pattern_classes = checkpoint.get('pattern_classes', [])
        num_pattern_classes = max(15, len(pattern_classes))
        
        model = MultiTaskEfficientNet(
            num_number_classes=100,
            num_pattern_classes=num_pattern_classes
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best accuracy so far: {checkpoint.get('best_acc', 'N/A')}")
        print(f"Pattern classes: {pattern_classes}")
        
        return model, pattern_classes
    else:
        # 如果 checkpoint 直接是 state_dict
        print("Warning: checkpoint does not contain 'model_state_dict' key, assume direct state_dict")
        model = MultiTaskEfficientNet(
            num_number_classes=100,
            num_pattern_classes=15
        )
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, []


def test_single_image(model, image_path, transform, device, pattern_classes):
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
    
    print(f"\n{'='*60}")
    print(f"测试图片: {image_path}")
    print(f"{'='*60}")
    print(f"预测结果:")
    print(f"  O1 (数字/图案): {'图案' if o1 == 1 else '数字'} (置信度: {o1_confidence:.4f})")
    if o1 == 0:
        print(f"  O2 (数字类别): {o2} (置信度: {o2_confidence:.4f})")
        print(f"  O3 (图案类别): {o3} (未激活)")
        print(f"  预测的数字: {o2}")
    else:
        print(f"  O2 (数字类别): {o2} (未激活)")
        print(f"  O3 (图案类别): {o3} (置信度: {o3_confidence:.4f})")
        if o3 < len(pattern_classes):
            print(f"  预测的图案: {pattern_classes[o3]}")
        else:
            print(f"  预测的图案: Unknown (index {o3})")
    
    print(f"\n最终输出 (o1*100 + o2 + o3): {final_output}")
    
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


def visualize_result(result, save_path=None):
    """可视化预测结果"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(result['original_image'])
    ax.axis('off')
    
    o1 = result['o1']
    o2 = result['o2']
    o3 = result['o3']
    final_output = result['final_output']
    conf = result['confidences']
    
    text = f"预测结果:\n"
    text += f"类型: {'图案' if o1 == 1 else '数字'} (置信度: {conf['o1']:.3f})\n"
    if o1 == 0:
        text += f"数字: {o2} (置信度: {conf['o2']:.3f})\n"
    else:
        text += f"图案索引: {o3} (置信度: {conf['o3']:.3f})\n"
    text += f"最终输出: {final_output}"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('预测结果可视化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    plt.show()


def test_random_samples(model, dataset, transform, device, pattern_classes, num_samples=5):
    """测试随机样本"""
    print(f"\n{'='*60}")
    print(f"随机测试 {num_samples} 个样本")
    print(f"{'='*60}")
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    results = []
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_path = sample['path']
        print(f"\n样本 {i+1}/{len(indices)}: {image_path}")
        
        result = test_single_image(model, image_path, transform, device, pattern_classes)
        results.append(result)
        
        print(f"真实标签:")
        print(f"  O1: {'图案' if sample['o1'] == 1 else '数字'}")
        if sample['o1'] == 0:
            print(f"  O2: {sample['o2']} (数字)")
        else:
            print(f"  O3: {sample['o3']} (图案)")
            if sample.get('pattern_name'):
                print(f"  图案名称: {sample['pattern_name']}")
        
        o1_correct = result['o1'] == sample['o1']
        o2_correct = result['o2'] == sample['o2']
        o3_correct = result['o3'] == sample['o3']
        
        print(f"预测准确性:")
        print(f"  O1: {'✓' if o1_correct else '✗'}")
        print(f"  O2: {'✓' if o2_correct else '✗'}")
        print(f"  O3: {'✓' if o3_correct else '✗'}")
        
    return results


def main():
    parser = argparse.ArgumentParser(description='Test MultiTask EfficientNet')
    parser.add_argument('--model_path', type=str, default='checkpoints/checkpoint_epoch_10.pth', 
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a specific image to test')
    parser.add_argument('--num_random', type=int, default=5, help='Number of random samples to test')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize results')
    parser.add_argument('--save_viz', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    try:
        model, pattern_classes = load_model(args.model_path, device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please make sure you have trained the model or specify correct checkpoint path")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    transform = get_transform('test')
    
    dataset = MultiTaskDataset(args.data_dir, transform=None)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check your dataset directory.")
        return
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    if args.image_path:
        print(f"Testing specific image: {args.image_path}")
        try:
            result = test_single_image(model, args.image_path, transform, device, pattern_classes)
            if args.visualize:
                visualize_result(result, args.save_viz)
        except FileNotFoundError:
            print(f"Error: Image file not found at {args.image_path}")
        except Exception as e:
            print(f"Error testing image: {e}")
    else:
        results = test_random_samples(model, dataset, transform, device, pattern_classes, args.num_random)
        
        if results and args.visualize:
            visualize_result(results[0], args.save_viz)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # 如果没有参数，运行快速测试（随机选择一张图）
        print("快速测试模式：从数据集中随机选择一张图片")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            model, pattern_classes = load_model('checkpoints/checkpoint_epoch_10.pth', device)
        except Exception as e:
            print(f"加载模型失败: {e}")
            sys.exit(1)
        
        transform = get_transform('test')
        dataset = MultiTaskDataset('dataset', transform=None)
        
        if len(dataset) == 0:
            print("数据集为空，请检查路径")
            sys.exit(1)
        
        import random
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        print(f"随机选中样本：{sample['path']}")
        
        result = test_single_image(model, sample['path'], transform, device, pattern_classes)
        visualize_result(result)
    else:
        main()
