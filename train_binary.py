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
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm
import copy

# ==========================================
# 二分类模型 (Binary Classification)
# 标签定义说明：
# 0: 健康
# 1: 患病
# ==========================================

class BinaryEsophagusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 统一转化为灰度(L)，再填充为三通道(RGB)，以适配预训练的ResNet/DenseNet等模型
        image = Image.open(img_path).convert('L').convert('RGB')
        
        # 提取标签数据。重构数据集时已将标签二值化为 0 和 1。
        label = int(self.df.iloc[idx]['class_number'])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """定义医学影像数据增强策略"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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

def build_binary_model(model_name="resnet18"):
    """
    修改输出端，num_classes=2
    """
    print(f"[*] 解析并构建二分类模型: {model_name}_binary")
    num_classes = 2
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    else: 
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def plot_training_curves(history, model_name, timestamp):
    """实时绘画折线图拟合光滑曲线"""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    if len(epochs) == 0:
        return
        
    plt.figure(figsize=(14, 6))
    
    # 绘制 Loss
    plt.subplot(1, 2, 1)
    if len(epochs) > 3:
        epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
        
        spline_train_loss = PchipInterpolator(epochs, history['train_loss'])
        spline_val_loss = PchipInterpolator(epochs, history['val_loss'])
        
        plt.plot(epochs, history['train_loss'], label='Train Loss (Raw)', marker='o', linestyle='--', alpha=0.4, color='blue')
        plt.plot(epochs, history['val_loss'], label='Val Loss (Raw)', marker='o', linestyle='--', alpha=0.4, color='orange')
        
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
        
        plt.plot(epochs, history['train_acc'], label='Train Acc (Raw)', marker='o', linestyle='--', alpha=0.4, color='green')
        plt.plot(epochs, history['val_acc'], label='Val Acc (Raw)', marker='o', linestyle='--', alpha=0.4, color='red')
        
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
    save_dir = os.path.join("eval_plots", f"{model_name}_binary", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"curves.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def train_binary_model(model_name="resnet18"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用的设备: {device} [{gpu_name}]")
    else:
        print(f"使用的设备: {device}")
    
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 3e-5 if ("vit" in model_name or "swin" in model_name) else 1e-4
    IMG_DIR = "BinaryTrainSetImages"
    CSV_FILE = "csv_data/mytrain.csv"
    
    df = pd.read_csv(CSV_FILE)
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_transform, val_transform = get_transforms()
    
    # 使用修改过标签逻辑的 Binary dataset
    train_dataset = BinaryEsophagusDataset(train_df, IMG_DIR, transform=train_transform)
    val_dataset = BinaryEsophagusDataset(val_df, IMG_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = build_binary_model(model_name).to(device)
    
    # 计算类别权重：当前数据集中，健康类别与患病类别已达到了几乎1:1的平衡状态，因此类别权重均等设置即可。
    class_weights = torch.FloatTensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_acc = 0.0
    
    # 增加 "_binary_" 标识符
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch（轮次） {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # Train
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
        print(f"Train Loss（训练损失）: {epoch_loss:.4f} Acc（训练准确率）: {epoch_acc:.4f}")
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss（验证损失）: {val_epoch_loss:.4f} Acc（验证准确率）: {val_epoch_acc:.4f}")
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        # 实时绘画折线图（覆盖之前保存的）
        plot_training_curves(history, model_name, run_timestamp)
        
        scheduler.step()
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            model_dir = os.path.join("saved_models", f"{model_name}_binary")
            os.makedirs(model_dir, exist_ok=True)
            best_model_name = os.path.join(model_dir, f"{model_name}_binary_{run_timestamp}.pth")
            torch.save(model.state_dict(), best_model_name)
            print(f"[*] 最佳模型已更新，准确率为: {best_acc:.4f}")
            
    print(f"\n训练完成！模型名称：{model_name}_binary_{run_timestamp}.pth")
    print(f"\n最高准确率: {best_acc:.4f}")

if __name__ == '__main__':
    # model_name = "resnet18"
    # model_name = "resnet50"
    # model_name = "densenet121"
    # model_name = "efficientnet_b0"
    model_name = "vit_b_16"
    # model_name = "swin_t"
    train_binary_model(model_name=model_name)
    # python train_binary.py