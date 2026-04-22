"""
带 ViT (Vision Transformer) 和 F1、混淆矩阵画图的完整多分类训练与评估代码
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm
import copy

# ==========================================
# 类别字典定义
# 0: Squamous_Epithelium (正常)
# 1: Intestinal_metaplasia (肠化生)
# 2: Gastric_metaplasia (胃化生)
# 3: Dysplasia_and_Cancer (癌变)
# ==========================================
CLASSES = ['Normal', 'Intestinal', 'Gastric', 'Cancer']

class EsophagusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 统一转灰度后铺满RGB三通道
        image = Image.open(img_path).convert('L').convert('RGB')
        label = int(self.df.iloc[idx]['class_number'])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """定义医学影像数据增强策略, ViT同样使用224的输入尺寸"""
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

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def build_model(model_name="vit_b_16", num_classes=4):
    """
    新增支持 ViT 模型 (Vision Transformer)
    """
    print(f"[*] 正在构建模型: {model_name}")
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "vit_b_16":
        # 引入 Vision Transformer
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_name == "swin_t":
        # 引入 Swin Transformer
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
        
    return model

def plot_confusion_matrix(cm, model_name, acc, f1, timestamp):
    """画混淆矩阵并保存"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                annot_kws={"size": 14})
    plt.title(f'{model_name} Confusion Matrix\nAcc: {acc:.4f} | Macro F1: {f1:.4f}', fontsize=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_dir = os.path.join("evaluation_plots", f"{model_name}_multi", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cm.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] 混淆矩阵已保存至: {save_path}")

def plot_training_curves(history, model_name, timestamp):
    """实时绘画折线图并在拟合光滑曲线"""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    if len(epochs) == 0:
        return
        
    plt.figure(figsize=(14, 6))
    
    # 绘制 Loss
    plt.subplot(1, 2, 1)
    if len(epochs) > 3:
        epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
        
        # 使用单调三次插值(PCHIP)生成平滑曲线，防止产生过度拟合的虚假波峰/波谷
        spline_train_loss = PchipInterpolator(epochs, history['train_loss'])
        spline_val_loss = PchipInterpolator(epochs, history['val_loss'])
        
        # 绘制原始折线图（添加透明度和虚线区分）
        plt.plot(epochs, history['train_loss'], label='Train Loss (Raw)', marker='o', linestyle='--', alpha=0.4, color='blue')
        plt.plot(epochs, history['val_loss'], label='Val Loss (Raw)', marker='o', linestyle='--', alpha=0.4, color='orange')
        
        # 绘制平滑拟合曲线（实线加粗）
        plt.plot(epochs_smooth, spline_train_loss(epochs_smooth), label='Train Loss (Smooth)', color='blue', linewidth=2)
        plt.plot(epochs_smooth, spline_val_loss(epochs_smooth), label='Val Loss (Smooth)', color='orange', linewidth=2)
    else:
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='blue')
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='orange')
        
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制 Accuracy
    plt.subplot(1, 2, 2)
    if len(epochs) > 3:
        epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
        
        spline_train_acc = PchipInterpolator(epochs, history['train_acc'])
        spline_val_acc = PchipInterpolator(epochs, history['val_acc'])
        
        # 绘制原始折线图（添加透明度和虚线区分）
        plt.plot(epochs, history['train_acc'], label='Train Acc (Raw)', marker='o', linestyle='--', alpha=0.4, color='green')
        plt.plot(epochs, history['val_acc'], label='Val Acc (Raw)', marker='o', linestyle='--', alpha=0.4, color='red')
        
        # 绘制平滑拟合曲线（实线加粗）
        plt.plot(epochs_smooth, spline_train_acc(epochs_smooth), label='Train Acc (Smooth)', color='green', linewidth=2)
        plt.plot(epochs_smooth, spline_val_acc(epochs_smooth), label='Val Acc (Smooth)', color='red', linewidth=2)
    else:
        plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', color='green')
        plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o', color='red')
        
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 按文件夹和模型名字存放图案
    save_dir = os.path.join("evaluation_plots", f"{model_name}_multi", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"curves.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def train_and_eval(model_name="vit_b_16"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    BATCH_SIZE = 16
    EPOCHS = 10 # 灰度医学图像需要更多的 Epoch 来收敛
    LR = 3e-5 if ("vit" in model_name or "swin" in model_name) else 1e-4  # Transformer 类对学习率较敏感，通常更小
    IMG_DIR = "TrainingSetImages"
    CSV_FILE = "csv_data/TrainingSet.csv"
    
    df = pd.read_csv(CSV_FILE)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_number'])
    
    train_transform, val_transform = get_transforms()
    train_dataset = EsophagusDataset(train_df, IMG_DIR, transform=train_transform)
    val_dataset = EsophagusDataset(val_df, IMG_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = build_model(model_name).to(device)
    
    # 类别权重处理不平衡
    class_counts = train_df['class_number'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    class_weights = [total_samples / c for c in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # 加入 label_smoothing 防止过拟合和极端预测
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_acc = 0.0
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    # 记录训练过程中的数据用于画图
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # --- Training ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # --- Validation & Evaluation ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        # 计算 F1 Score
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} Macro F1: {val_f1:.4f}")
        
        # 实时绘画折线图（覆盖之前保存的）
        plot_training_curves(history, model_name, run_timestamp)
        
        scheduler.step()
        
        # 保存最优模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            
            model_dir = os.path.join("saved_models", f"{model_name}_multi")
            os.makedirs(model_dir, exist_ok=True)
            best_model_name = os.path.join(model_dir, f"{model_name}_multi_{run_timestamp}.pth")
            torch.save(model.state_dict(), best_model_name)
            
            print(f"[*] 发现更优模型 (Acc: {best_acc:.4f}), 生成并保存混淆矩阵及分析图表...")
            # 画图并保存
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(cm, model_name, val_epoch_acc, val_f1, run_timestamp)
            
            # 报告打印
            print("\n分类详细报告:")
            print(classification_report(all_labels, all_preds, target_names=CLASSES))

    print(f"\n训练完成！最佳模型存储在：{best_model_name}，最好成绩: {best_acc:.4f}")

if __name__ == '__main__':
    # 调用你想训练的模型名称，现已支持 vit_b_16
    # model_name = "resnet18"
    # model_name = "resnet50"
    # model_name = "densenet121"
    model_name = "efficientnet_b0"
    # model_name = "vit_b_16"
    # model_name = "swin_t"
    train_and_eval(model_name=model_name)

    # python train_eval_multi_vit.py

