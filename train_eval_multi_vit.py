
import os # 操作系统接口模块，用于创建文件路径和文件夹
import time # 时间处理模块，用于生成时间戳命名文件
import pandas as pd # pandas 库，用于读取和组织 CSV 数据表
import numpy as np # 数值计算库，用于处理数组和数值运算
import torch # PyTorch 核心模块，支持张量和运算
import torch.nn as nn # 神经网络模块，包含各种神经网络层和损失函数
import torch.optim as optim # 优化器模块，包含 Adam, SGD 等优化算法
from torch.utils.data import Dataset, DataLoader # 数据集与数据加载器，协助批量吐出训练数据
from torchvision import transforms, models # 提供预训练模型和丰富的图像处理工具
from PIL import Image # Python Imaging Library，用于加载和解析图像文件
from sklearn.model_selection import train_test_split # 用于划分训练集和验证集
from sklearn.metrics import f1_score, confusion_matrix, classification_report # 评估指标和报告工具
import matplotlib.pyplot as plt # 绘图库，用于画图和可视化
import seaborn as sns # Seaborn 库，基于 Matplotlib 的高级数据可视化工具，特别适合画统计图表
from scipy.interpolate import PchipInterpolator # 用于平滑曲线拟合的插值工具，保持单调性防止过拟合的虚假波峰/波谷
from tqdm import tqdm # 进度条库，用于显示训练和验证过程的进度

# ==========================================
# 类别字典定义
# 0: Squamous_Epithelium (正常)
# 1: Intestinal_metaplasia (肠化生)
# 2: Gastric_metaplasia (胃化生)
# 3: Dysplasia_and_Cancer (癌变)
# ==========================================
CLASSES = ['Normal', 'Intestinal', 'Gastric', 'Cancer'] # 类别名称列表，供画图和报告使用

"""定义数据集类，负责加载图像和标签，并应用数据增强"""
class EsophagusDataset(Dataset):
    # 初始化函数，接收 DataFrame、图像目录和数据增强变换
    def __init__(self, df, img_dir, transform=None):
        self.df = df                # 保存了文件名称和标签类的数据表csv
        self.img_dir = img_dir      # 真实的本地图片所在根目录
        self.transform = transform  # 指向预设好的图像增强函数
    
    # 返回数据集大小
    def __len__(self):
        # 数据集大小即为csv表格中的行数，每行对应一张图片和一个标签
        return len(self.df) # 返回数据集大小，即csv表格中的行数
    
    # 获取指定索引的数据项，加载图像并应用变换
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_filename'] # 从csv表格中获取指定行的图片文件名
        img_path = os.path.join(self.img_dir, img_name) # 构建图片的完整路径
        
        # 统一转灰度后铺满RGB三通道
        image = Image.open(img_path).convert('L').convert('RGB') # 加载图像并转换为灰度图再转换回RGB三通道，保持与数据增强一致的输入格式
        # 获取与该图像配对的医学类别 (0,1,2,3)
        label = int(self.df.iloc[idx]['class_number']) # 从csv表格中获取指定行的标签，并转换为整数类型

        # 如果定义了数据增强变换，则对图像应用这些变换，增强训练数据的多样性和模型的泛化能力
        if self.transform:
            image = self.transform(image) # 如果定义了数据增强变换，则对图像应用这些变换，增强训练数据的多样性和模型的泛化能力

        # 返回处理后的图像和对应的标签，供训练或评估使用
        return image, label

def get_transforms():
    """定义医学影像数据增强策略, ViT同样使用224的输入尺寸"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),              # 统一调整图像大小为256x256，后续随机裁剪到224x224
        transforms.RandomCrop(224),                 # 第一层增强：随机裁剪到224x224，增加位置变化的多样性
        transforms.RandomHorizontalFlip(p=0.5),     # 第二层增强：50% 概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),       # 第三层增强：50% 概率垂直翻转
        transforms.RandomRotation(degrees=45),      # 第四层增强：随机旋转±45度，增加旋转不变性
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # 第五层增强：随机调整亮度和对比度，增加光照变化
        transforms.ToTensor(),                 # 将 PIL 图像转换为 PyTorch 的 Tensor 格式，准备输入模型
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), # 第六层增强：随机擦除，模拟局部遮挡，增加模型的鲁棒性
        # 虽然医学图像与自然图像不同，但使用 ImageNet 的统计数据进行标准化通常能帮助预训练模型更好地适应新任务
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 第七层增强：标准化，使用 ImageNet 的均值和标准差，适应预训练模型的输入要求
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 验证阶段直接调整到模型输入尺寸，不进行随机变换，保持评估的一致性和稳定性
        transforms.ToTensor(), # 将 PIL 图像转换为 PyTorch 的 Tensor 格式，准备输入模型
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化，使用 ImageNet 的均值和标准差，适应预训练模型的输入要求
    ])
    
    # 返回训练和验证阶段的变换函数，供数据集类使用
    return train_transform, val_transform

"""定义模型构建函数，支持多种预训练模型架构，包括 ViT 和 Swin Transformer"""
def build_model(model_name="vit_b_16", num_classes=4):
    print(f"[*] 正在构建模型: {model_name}")
    # 根据传入的模型名称，加载对应的预训练模型，并替换最后的分类层以适应当前任务的类别数
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # 替换 DenseNet 的分类层，DenseNet 的分类层是 model.classifier，而不是 model.fc
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # 替换 EfficientNet 的分类层，EfficientNet 的分类层也是 model.classifier，而不是 model.fc
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "vit_b_16":
        # 引入 Vision Transformer
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # 替换 ViT 的分类层，ViT 的分类层是 model.heads.head，而不是 model.fc
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_name == "swin_t":
        # 引入 Swin Transformer
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        # 替换 Swin Transformer 的分类层，Swin 的分类层是 model.head，而不是 model.fc
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
    # 返回构建好的模型实例，供训练和评估使用
    return model

"""画混淆矩阵并保存"""
def plot_confusion_matrix(cm, model_name, acc, f1, timestamp):
    # 使用 Seaborn 绘制混淆矩阵热力图，显示每个类别的预测情况，并在标题中标注准确率和 F1 分数
    plt.figure(figsize=(10, 8))# 设置图像大小
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',# 绘制热力图，annot=True 显示数字，fmt='d' 格式化为整数，cmap='Blues' 使用蓝色渐变
                xticklabels=CLASSES, yticklabels=CLASSES,
                annot_kws={"size": 14})
    # 在标题中标注模型名称、准确率和 F1 分数，字体大小为15
    plt.title(f'{model_name} Confusion Matrix\nAcc: {acc:.4f} | Macro F1: {f1:.4f}', fontsize=15)
    plt.ylabel('True Label', fontsize=12)       # 设置坐标轴标签，字体大小为12
    plt.xlabel('Predicted Label', fontsize=12)  # 设置坐标轴标签，字体大小为12
    plt.tight_layout()                          # 调整布局以防止标签重叠
    
    # 按文件夹和模型名字存放图案
    save_dir = os.path.join("eval_plots", f"{model_name}_multi", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)# 创建保存目录，如果不存在则创建
    save_path = os.path.join(save_dir, f"cm.png")# 构建保存路径，文件名为 cm.png
    plt.savefig(save_path, dpi=300)# 保存图像，dpi=300 提高分辨率以便打印和展示
    plt.close()# 关闭图像释放内存
    print(f"[*] 混淆矩阵已保存至: {save_path}")# 打印保存路径提示用户

"""实时绘画折线图并在拟合光滑曲线"""
def plot_training_curves(history, model_name, timestamp):
    # 从历史记录中提取训练和验证的损失与准确率数据，并生成对应的折线图，使用 PCHIP 插值进行平滑处理以更清晰地展示趋势
    epochs = np.arange(1, len(history['train_loss']) + 1)
    # 如果没有数据则直接返回，避免画图函数出错
    if len(epochs) == 0:
        return
    # 设置图像大小为14x6英寸，适合展示两个子图（损失和准确率）
    plt.figure(figsize=(14, 6))
    
    # 绘制 Loss
    plt.subplot(1, 2, 1)
    # 如果 epoch 数量大于3，才进行平滑处理，否则直接画原始折线图，避免过度拟合的虚假波峰/波谷
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
        # 如果 epoch 数量不多，则直接绘制原始折线图，避免过度拟合的虚假波峰/波谷
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='blue')
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='orange')
    

    plt.title('Training and Validation Loss', fontsize=14)  # 设置标题，字体大小为14
    plt.xlabel('Epochs', fontsize=12)                       # 设置 x 轴标签，字体大小为12
    plt.ylabel('Loss', fontsize=12)                         # 设置 y 轴标签，字体大小为12
    plt.legend()                                            # 显示图例，区分训练和验证曲线
    plt.grid(True, linestyle='--', alpha=0.6)               # 添加网格线，使用虚线样式和适当的透明度以增强可读性
    
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
    save_dir = os.path.join("eval_plots", f"{model_name}_multi", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"curves.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close()

""""训练和评估函数，支持多模型循环训练"""
def train_and_eval(model_name="vit_b_16"):
    # 动态检测是否存在 NVIDIA GPU（cuda）。有显卡用显卡，没显卡自动 fallback 到 CPU 防止闪退。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)  # 获取如 "RTX 3060" 之类的硬件字串供打印信息提示
        print(f"使用的设备: {device} [{gpu_name}]")
    else:
        print(f"使用的设备: {device}")
    
    BATCH_SIZE = 16  # 每次训练送入显卡的图片数量。
    EPOCHS = 50  # 整个训练集被完整送入模型的次数。
    # Transformer 类对学习率较敏感，通常更小。ViT 和 Swin Transformer 通常需要更小的学习率（如 3e-5）
    # 而 ResNet、DenseNet、EfficientNet 等 CNN 模型可以使用稍大的学习率（如 1e-4）
    LR = 3e-5 if ("vit" in model_name or "swin" in model_name) else 1e-4  # Transformer 类对学习率较敏感，通常更小
    IMG_DIR = "TrainingSetImages" # 图片所在的根目录，csv表格中的文件名会与这个目录结合成完整路径供数据集类加载图像使用
    CSV_FILE = "csv_data/TrainingSet.csv" # 包含图片文件名和对应标签的 CSV 文件路径，数据集类会读取这个文件来获取训练数据的信息
    
    df = pd.read_csv(CSV_FILE) # 读取 CSV 文件，加载包含图片文件名和对应标签的数据表，供后续划分训练集和验证集使用
    
    # 使用 sklearn 的 train_test_split 函数将数据表划分为训练集和验证集 0.2 的数据作为验证集
    # 保持类别分布一致（stratify），并设置随机种子以确保结果可复现
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_number'])
    
    # 获取训练和验证阶段的数据增强变换函数，供数据集类使用
    train_transform, val_transform = get_transforms()
    # 创建训练和验证数据集实例，传入对应的数据表、图片目录和数据增强变换函数
    train_dataset = EsophagusDataset(train_df, IMG_DIR, transform=train_transform) 
    # 创建训练和验证数据集实例，传入对应的数据表、图片目录和数据增强变换函数
    val_dataset = EsophagusDataset(val_df, IMG_DIR, transform=val_transform)
    
    # 创建训练和验证数据加载器，设置批量大小、是否打乱数据（训练集打乱，验证集不打乱）以及使用多线程加载数据以加速训练过程
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 构建模型实例，并将其移动到指定的设备上进行训练和评估
    model = build_model(model_name).to(device)
    
    # 类别权重处理不平衡
    # 计算每个类别的样本数量，按照类别索引排序以确保与类别标签对应
    class_counts = train_df['class_number'].value_counts().sort_index().values  
    # 计算总样本数量，供后续计算类别权重使用
    total_samples = sum(class_counts)  
    # 计算类别权重，样本数量较少的类别权重较大，样本数量较多的类别权重较小，以平衡训练过程中的类别不平衡问题                                        
    class_weights = [total_samples / c for c in class_counts]      
    # 将类别权重转换为 PyTorch 的 FloatTensor，并移动到指定的设备上，以供损失函数使用            
    class_weights = torch.FloatTensor(class_weights).to(device)                 
    
    # 加入 label_smoothing 防止过拟合和极端预测
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # 定义优化器，使用 AdamW 优化器，设置学习率和权重衰减以帮助模型更好地收敛并防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    # 定义学习率调度器，使用 Cosine Annealing Warm Restarts 调度器，适合训练过程中动态调整学习率以帮助模型更好地收敛
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # 记录最佳验证准确率，以便在训练过程中保存最优模型
    best_acc = 0.0
    # 生成当前训练的时间戳，用于命名保存的模型和图表文件，确保每次训练的结果都能独立存储且易于区分
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    # 记录训练过程中的数据用于画图
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 训练和评估循环，遍历指定的训练轮数，每轮进行一次完整的训练和验证过程，并记录相关指标以供后续分析和可视化
    for epoch in range(EPOCHS):
        # 打印当前的训练轮数和总轮数，提供训练进度的提示信息
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # --- Training ---
        model.train() # 设置模型为训练模式，启用 Dropout 和 BatchNorm 等训练特有的行为
        running_loss = 0.0 # 累积训练损失，用于计算每轮的平均损失
        running_corrects = 0 # 累积正确预测的样本数量，用于计算每轮的准确率
        
        # 使用 tqdm 包装训练数据加载器，显示训练过程的进度条，提供每轮训练的实时反馈
        for inputs, labels in tqdm(train_loader, desc="Training"):
            # 将输入数据和标签移动到指定的设备上进行训练，确保计算在 GPU 上进行以加速训练过程
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 每个训练步骤开始前清零优化器的梯度，防止梯度累积导致更新错误
            optimizer.zero_grad() # 每个训练步骤开始前清零优化器的梯度，防止梯度累积导致更新错误
            outputs = model(inputs)# 前向传播，计算模型的输出
            loss = criterion(outputs, labels)# 计算损失，使用定义好的损失函数（包含类别权重和标签平滑）
            _, preds = torch.max(outputs, 1)# 获取预测结果，取输出中概率最大的类别作为预测标签
            
            loss.backward()# 反向传播，计算梯度
            optimizer.step()# 更新模型参数
            
            running_loss += loss.item() * inputs.size(0)# 累积训练损失，乘以当前批次的样本数量，以便后续计算平均损失
            running_corrects += torch.sum(preds == labels.data)# 累积正确预测的样本数量，比较预测标签和真实标签，统计正确的数量以便后续计算准确率
            
        epoch_loss = running_loss / len(train_dataset)# 计算每轮的平均训练损失，除以训练集的总样本数量
        epoch_acc = running_corrects.double() / len(train_dataset)# 计算每轮的训练准确率，除以训练集的总样本数量，并转换为浮点数格式以便后续使用
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")# 打印当前轮的训练损失和准确率，提供训练过程的反馈信息
        
        history['train_loss'].append(epoch_loss)# 记录每轮的训练损失，供后续画图使用
        history['train_acc'].append(epoch_acc.item())# 记录每轮的训练准确率，供后续画图使用
        
        # --- Validation & Evaluation ---
        model.eval() # 设置模型为评估模式，禁用 Dropout 和 BatchNorm 等训练特有的行为，确保评估的一致性和稳定性
        val_loss = 0.0 # 累积验证损失，用于计算每轮的平均损失
        val_corrects = 0 # 累积正确预测的样本数量，用于计算每轮的准确率
        all_preds = [] # 累积所有的预测标签，用于计算 F1 分数和混淆矩阵
        all_labels = [] # 累积所有的真实标签，用于计算 F1 分数和混淆矩阵
        
        with torch.no_grad(): # 在验证阶段禁用梯度计算，节省内存和加速评估过程
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device) # 将输入数据和标签移动到指定的设备上进行评估，确保计算在 GPU 上进行以加速评估过程
                
                outputs = model(inputs) # 前向传播，计算模型的输出
                loss = criterion(outputs, labels) # 计算损失，使用定义好的损失函数（包含类别权重和标签平滑）
                _, preds = torch.max(outputs, 1) # 获取预测结果，取输出中概率最大的类别作为预测标签
                
                val_loss += loss.item() * inputs.size(0) # 累积验证损失，乘以当前批次的样本数量，以便后续计算平均损失
                val_corrects += torch.sum(preds == labels.data) # 累积正确预测的样本数量，比较预测标签和真实标签，统计正确的数量以便后续计算准确率
                
                all_preds.extend(preds.cpu().numpy()) # 累积所有的预测标签，移动到 CPU 并转换为 NumPy 数组后添加到列表中，供后续计算 F1 分数和混淆矩阵使用
                all_labels.extend(labels.cpu().numpy()) # 累积所有的真实标签，移动到 CPU 并转换为 NumPy 数组后添加到列表中，供后续计算 F1 分数和混淆矩阵使用
                
        val_epoch_loss = val_loss / len(val_dataset) # 计算每轮的平均验证损失，除以验证集的总样本数量
        val_epoch_acc = val_corrects.double() / len(val_dataset) # 计算每轮的验证准确率，除以验证集的总样本数量，并转换为浮点数格式以便后续使用

        history['val_loss'].append(val_epoch_loss) # 记录每轮的验证损失，供后续画图使用
        history['val_acc'].append(val_epoch_acc.item()) # 记录每轮的验证准确率，供后续画图使用
        
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
    # model_name = "efficientnet_b0"
    # model_name = "vit_b_16"
    # model_name = "swin_t"
    # 循环3次 每次执行不同的模型
    for model_name in [ "vit_b_16", "swin_t","resnet18", "resnet50", "densenet121", "efficientnet_b0"]:
        train_and_eval(model_name=model_name)

    # python train_eval_multi_vit.py

