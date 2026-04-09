import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
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
        
    else: 
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_binary_model(model_name="resnet18"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用的设备: {device} [{gpu_name}]")
    else:
        print(f"使用的设备: {device}")
    
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
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
    model_name =   "efficientnet_b0"
    train_binary_model(model_name=model_name)