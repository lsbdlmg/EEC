import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import shutil

# ==========================================
# 定义反标准化函数 (用于将处理后的Tensor转回可视化的图片)
# ==========================================
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def get_transforms():
    """定义医学影像数据增强策略, 与训练代码一致"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # 灰度图去除 saturation 和 hue
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform

def export_augmented_images(input_dir, output_dir, num_samples=5, num_augs_per_image=4):
    """
    读取 input_dir 下的图片，应用增强后输出到 output_dir
    :param input_dir: 原图片所在目录
    :param output_dir: 增强后图片保存目录
    :param num_samples: 选取多少张原图进行演示
    :param num_augs_per_image: 每张原图生成多少张不同的增强效果
    """
    if not os.path.exists(input_dir):
        print(f"输入路径 {input_dir} 不存在！")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    transform = get_transforms()
    unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 获取支持的图片列表
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not all_images:
        print(f"{input_dir} 目录下没有找到图片！")
        return
        
    # 取前 num_samples 张图测试
    sample_images = all_images[:num_samples]
    
    for img_name in sample_images:
        img_path = os.path.join(input_dir, img_name)
        try:
            # 统一转灰度后铺满RGB三通道，与数据集处理一致
            img = Image.open(img_path).convert('L').convert('RGB')
        except Exception as e:
            print(f"无法读取图片 {img_name}: {e}")
            continue
            
        base_name = os.path.splitext(img_name)[0]
        
        # 将原图也缩放一下并保存一份作为对比
        orig_img_resized = transforms.Resize((224, 224))(img)
        orig_img_resized.save(os.path.join(output_dir, f"{base_name}_0_original.png"))
        
        # 针对同一幅图生成多次不同增强效果
        for i in range(1, num_augs_per_image + 1):
            aug_tensor = transform(img)
            
            # 使用反标准化恢复图像色彩/亮度范围
            unorm_img_tensor = unorm(aug_tensor.clone())
            
            # 限制在 [0, 1] 之间
            unorm_img_tensor = unorm_img_tensor.clamp(0, 1)
            
            save_path = os.path.join(output_dir, f"{base_name}_{i}_aug.png")
            
            # 保存Tensor为图片
            vutils.save_image(unorm_img_tensor, save_path)
            
        print(f"已处理并导出图片: {img_name} -> 生成 {num_augs_per_image} 张增强效果")
        
    print(f"\n所有演示图片已保存至: {output_dir}")

if __name__ == '__main__':
    INPUT_DIR = "E:\\EEC\\EEC\\Hard_Cases_All_Models_Failed\\2026_05_02_01_13_02"   # 原始数据集文件夹（也可以是任意存放图片的文件夹）
    OUTPUT_DIR = "Augmented_Samples"  # 增强后的图片保存文件夹
    
    export_augmented_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=2,          # 从原始数据集中选取多少张图片进行增强演示
        num_augs_per_image=5     # 每张原图生成多少张不同的增强效果
    )
    # python export_augmented_images.py
