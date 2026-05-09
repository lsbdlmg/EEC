"""
使用 Grad-CAM 的独立脚本。为给定的医学图像输出基于预训练模型的热力图 (Attention Map)。
注意：Vision Transformer不能用传统的Grad-CAM(使用 CNN 的最后一层卷积)，应该使用 ViT 特殊的特征支持。
为简单起见，此脚本以最普遍的 ResNet/DenseNet 以及 Vision Transformer 均可用为标准（通过指定 target_layers）。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus, HiResCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch.nn as nn

CLASSES = ['Normal', 'Intestinal', 'Gastric', 'Cancer']

def build_model_for_cam(model_name="resnet50", num_classes=4, weights_path=None):
    """
    为了读取权重构造相同结构的模型，同 train 脚本的一样
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 解析载入模型结构: {model_name}")
    
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        target_layers = [model.features[-1]]
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layers = [model.features[-1]]
        
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        # ViT 需要特殊层作为 target (获取最后的 encoder block)
        target_layers = [model.encoder.layers[-1].ln_1]
        
    elif model_name == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        # Swin T 获取最后一个 block (features[-1][-1] 指向最深的一层 layer)
        target_layers = [model.features[-1][-1].norm2]
        
    else:
        raise ValueError("不支持的模型格式")
        
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[*] 成功加载权重：{weights_path}")
    else:
        print("[!] 警告: 未能加载有效权重。模型使用随机参数初始化。这会导致热力图乱跳没有意义。如果要为论文截图，请放入正确的检查点。")

    model.to(device)
    model.eval()
    return model, target_layers

def reshape_transform_vit(tensor, height=14, width=14):
    """
    针对 ViT 的空间维度适配。
    tensor 结构：[batch_size, 197, 768] (包含cls token)
    剔除cls_token得到 196 -> reshape 至 14x14
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor, height=7, width=7):
    """针对 Swin 的空间特征图适配"""
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_image_for_cam(img_path):
    """加载图片并按照训练标准变换预处理"""
    img = Image.open(img_path).convert('L').convert('RGB')
    
    # 转换为模型输入的 Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    # 获取可视化使用的原图 (需从 0~255 转到 0~1 的 float32)，无需黑白强行处理
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1] # 提取彩图
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    
    return input_tensor, rgb_img

def generate_cam_heatmap(model_name="vit_b_16", weights_path=None, test_img_path="test.jpg", target_class=-1):
    """主热力图生成入口"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, target_layers = build_model_for_cam(model_name, 4, weights_path)
    input_tensor, rgb_img = load_image_for_cam(test_img_path)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        preds = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(preds)

    print(f"[*] 模型对于图片 {os.path.basename(test_img_path)} 判断: {CLASSES[pred_class]} 概率: {preds[pred_class]*100:.2f}%")
    
    # 判断是否为vit/swin 需要特殊的结构变换
    cam_kwargs = {"model": model, "target_layers": target_layers}
    if "vit" in model_name:
        cam_kwargs["reshape_transform"] = reshape_transform_vit
    elif "swin" in model_name:
        cam_kwargs["reshape_transform"] = reshape_transform_swin
        
    # 实例化 GradCAM (或 GradCAM++)
    cam = GradCAM(**cam_kwargs)
    
    # target_class如果是-1，默认画出模型认为最大概率类别的热力情况
    target_list = [ClassifierOutputTarget(target_class)] if target_class >= 0 else None
    
    # 预测并画出热力图
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_list)
    grayscale_cam = grayscale_cam[0, :]
    
    # 覆盖着色到原图
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 创建截图画板
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image (Resized)")
    axes[0].axis('off')
    
    axes[1].imshow(cam_image)
    if target_class == -1:
        axes[1].set_title(f"GradCAM (Predicts: {CLASSES[pred_class]})")
    else:
        axes[1].set_title(f"GradCAM for target: {CLASSES[target_class]}")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 创建文件夹保存论图
    out_dir = "paper_cam_screenshots"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{model_name}_{CLASSES[pred_class]}_{os.path.basename(test_img_path)}"
    
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"[*] 成功保存热力图截图至: {out_path}")

if __name__ == '__main__':
    # 不用这个脚本
    pass
