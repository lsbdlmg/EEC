"""
食道癌早期检测模型训练脚本 (多分类版)

【后续优化该模型的方法】
1. 数据层面 (Data Level):
   - 裁边去伪影 (Masking): 胃镜图片往往带有黑色边框和反光，可以使用 OpenCV 识别圆形轮廓，将黑色区域置为纯零以防模型将其误认为特征。
   - 色彩标准化 (Stain/Color Normalization): 使用直方图均衡化（CLAHE）等算法统一各类图片的亮度与色调。
   - 外部验证集融合: 发现模型在自建集表现不佳时，需要将一定比例的自建集（或目标环境数据）混入训练，以消除领域偏移（Domain Shift）。

2. 模型架构与损失评估 (Model & Loss):
   - Focal Loss (焦点损失): 替换目前的 CrossEntropyLoss。这会让模型将注意力完全放在“难以区分的边缘样本”上（比如肠化生与胃化生）。
   - 注意力聚合 (Attention Mechanisms): 在当前 ResNet/DenseNet 之上嵌入 CBAM 或 SENet 模块。

3. 训练策略层面 (Training Strategy):
   - K-Fold 交叉验证 (K-Fold Cross Validation): 替代单纯的 train_test_split，有效压榨这几千张小数据的潜力。
   - 模型集成 (Ensemble): 训练三个架构（如 ResNet, DenseNet, EfficientNet），最后将它们的 Softmax 概率加权平均作为最后决定，以显著提升泛化能力。
"""

import os                                            # 操作系统接口模块，用于创建文件路径和文件夹
import time                                          # 时间处理模块，用于生成时间戳命名文件
import pandas as pd                                  # pandas 库，用于读取和组织 CSV 数据表
import torch                                         # PyTorch 核心模块，支持张量和运算
import torch.nn as nn                                # 神经网络模块，包含各种神经网络层和损失函数
import torch.optim as optim                          # 优化器模块，包含 Adam, SGD 等优化算法
from torch.utils.data import Dataset, DataLoader     # 数据集与数据加载器，协助批量吐出训练数据
from torchvision import transforms, models           # 提供预训练模型和丰富的图像处理工具
from PIL import Image                                # Python Imaging Library，用于加载和解析图像文件
from sklearn.model_selection import train_test_split # 来自 scikit-learn 库，用于打乱和切割数据集
from tqdm import tqdm                                # 一个极好的进度条库，让长时间的训练可视化
import copy                                          # 提供深拷贝功能，保证保存最佳模型参数时不被后续污染

# ==========================================
# 类别字典定义 (Medical Class Definitions)
# 0: Squamous_Epithelium (鳞状上皮-健康/正常)
# 1: Intestinal_metaplasia (肠上皮化生-初期的异常/发炎警报)
# 2: Gastric_metaplasia (胃上皮化生-巴雷特食管 / 癌前病变第一步)
# 3: Dysplasia_and_Cancer (异型增生和癌症-最危险)
# ==========================================

class EsophagusDataset(Dataset):
    """
    基础数据集类。只要是使用 PyTorch，任何自定义格式都需要包裹在这个类里。
    它负责接收图片清单，并在模型需要时把“图片+对应标签”一组组返回。
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df                     # 保存了文件名称和标签类的 pandas dataframe
        self.img_dir = img_dir           # 真实的本地图片所在根目录
        self.transform = transform       # 指向预设好的图像增强函数

    def __len__(self):
        # 配合 DataLoader 使用，用以决定整个 Epoch 会被切割为多少个 Batch
        return len(self.df)

    def __getitem__(self, idx):
        # 通过索引逐个提取 dataframe 中的图片名
        img_name = self.df.iloc[idx]['image_filename']
        # 把大目录路径和子图片名完美接合成绝对路径
        img_path = os.path.join(self.img_dir, img_name)
        
        # 使用 PIL 将其统一转为标准的 L 灰度图，然后再强制铺成模型的 RGB 三通道(但去除了色彩干扰)
        image = Image.open(img_path).convert('L').convert('RGB')
        # 获取与该图像配对的医学类别 (0,1,2,3)
        label = int(self.df.iloc[idx]['class_number'])

        # 如果设定了 transform（一定会设），就在此时对这张图片执行裁剪、翻转等重塑操作
        if self.transform:
            image = self.transform(image)

        # 返回处理完毕的可放入显卡的纯数据格式
        return image, label

def get_transforms():
    """
    定义训练集和验证集的图像增强与预处理流程。
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),              # 统一将图片拉伸或压扁到 256x256 (预大尺度缩放)
        transforms.RandomCrop(224),                 # 第一层增强：随机在 256 里挖一块 224x224 出来，这等于让模型每天看的位置略有差异
        transforms.RandomHorizontalFlip(p=0.5),     # 第二层增强：50% 概率照片水平翻转，由于细胞无方向限制，相当于增加了一张新图
        transforms.RandomVerticalFlip(p=0.5),       # 第三层增强：50% 概率上下翻转
        transforms.RandomRotation(degrees=45),      # 第四层增强：随机旋转图片，角度在 -45 到 +45 度之间，进一步增加模型对不同拍摄角度的鲁棒性
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),# 第五层增强：随机调整亮度、对比度、饱和度和色调，模拟不同内镜设备和拍摄环境下的色彩变化
        transforms.ToTensor(),                      # 把 PIL 格式图像转换成 PyTorch 能看懂的数值矩阵 (并将像素值从 0~255 压缩为 0~1 的浮点数)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),# 第六层增强：20% 的概率在图片上随机挖一个黑色矩形（模拟遮挡），迫使模型学会从不完整信息中提取特征
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 第七层增强：使用 ImageNet 的均值和标准差对图像进行归一化，确保输入数据的分布与预训练模型的期望一致
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),              # 验证集代表真实世界表现，不进行随机裁剪，直接缩放到 224x224
        transforms.ToTensor(),                      # 同样转化为张量格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 确保色彩打分基准一致
    ])
    
    return train_transform, val_transform

def build_model(model_name="resnet50", num_classes=4):
    """
    根据用户指定的模型名称，构建一个预训练的网络骨架，并将其输出层改为适合我们四分类任务的结构。
    """
    print(f"[*] 正在构建模型: {model_name}")
    
    # 按照输入名称进行实例化，并加载 ImageNet 预训练权重
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features               # 获取该网络最后一层全连接层(fc)输入了多少个特征
        model.fc = nn.Linear(num_ftrs, num_classes)   # 将原本输出维度的 1000 斩断，换为我们需要判定的类别数（4类别）
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # DenseNet 结构特殊，它的分类层叫 classifier 而不是 fc
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # EfficientNet 将分类器封装在一个名叫 classifier 的 Sequential 序列数组的第二个位置即索引[1]里
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    else:  
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model # 返回构建好的模型实例，等待被送入显卡和训练循环中

def train_model(model_name="densenet121"):
    # ================= 1. 环境与参数设置 =================
    # 动态检测是否存在 NVIDIA GPU（cuda）。有显卡用显卡，没显卡自动 fallback 到 CPU 防止闪退。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)  # 获取诸如 "RTX 3060" 之类的硬件字串供打印信息提示
        print(f"使用的设备: {device} [{gpu_name}]")
    else:
        print(f"使用的设备: {device}")
    
    BATCH_SIZE = 16  # 每次训练送入显卡的图片数量。过大可能显存爆炸，过小则训练效率低下。
    EPOCHS = 15      # 整个训练集被完整送入模型的次数。过多可能过拟合，过少可能欠拟合。
    LR = 1e-4        # 学习率，控制模型权重更新的步伐。过大可能训练不稳定，过小可能收敛过慢。
    IMG_DIR = "TrainingSetImages"
    CSV_FILE = "csv_data/TrainingSet.csv"
    
    # ================= 2. 数据处理 =================
    df = pd.read_csv(CSV_FILE)
    
    # 从总数据中拆出 20% 作为考试题(验证集)，不给网络看。
    # 这里使用 stratify 参数确保每个类别在训练集和验证集中都保持相似的分布比例，防止某个类别过度稀缺导致模型无法学习。
    # train_test_split 参数详解：test_size=0.2 代表 20% 的数据划分为验证集；random_state=42 固定随机种子以保证每次运行结果一致。
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_number'])
    
    # 获取前面写好的图片形变函数
    train_transform, val_transform = get_transforms()
    
    # 实例化数据集对接器
    train_dataset = EsophagusDataset(train_df, IMG_DIR, transform=train_transform)
    val_dataset = EsophagusDataset(val_df, IMG_DIR, transform=val_transform)
    
    # DataLoader 控制发给机器的流速。
    # num_workers=4 代表四核心CPU协同去给显卡递送图片；pin_memory=True 能加速从内存到显存的传输效率
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # ================= 3. 模型构建 =================
    # 将指定好的网络模型搬上显卡(如 .to(cuda))
    model = build_model(model_name=model_name, num_classes=4).to(device)
    
    # 核心：计算数据集里样本的倾向偏斜从而提供不同权重
    # 数量严重分层: [健康0: 1469, 肠化生1: 3177, 胃化生2: 1206, 癌症3: 3594]
    # 我们用最大数(3594) 除以各自类别数目（惩罚反比）。
    # 算出的 class_weights 会严重偏袒总数仅 1000 出头的类2（倍率为2.98），强迫模型把目光分给不常见的病情
    class_weights = torch.FloatTensor([2.45, 1.13, 2.98, 1.0]).to(device)
    
    # 初始化交叉熵损失（附带了针对弱势群体的权重关照）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 选择最先进的自适应优化器 AdamW
    # weight_decay (权重衰减项): 当个别参数神经元被赋予过高权重时，定期砍它一刀，防止偏执(过拟合)现象
    # 参数详解：model.parameters() 代表所有可训练的权重；lr=LR 代表学习率；weight_decay=1e-3 代表每次更新后权重衰减的程度
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    # 余弦退火学习调度器。让它前两周高速学习，后面缓慢减小；到第5周直接重开（重启学习率），强迫它重新审视错过的局部坑洼
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # ================= 4. 模型训练 =================
    best_acc = 0.0      # 记录变量，最高准确率记录器
    best_model_name = ""
    
    # 记录脚本运行的这一秒钟。用来把模型文件命名出差异化，防覆盖。
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    # 正式进入训练循环，外层循环控制 Epoch 轮次，内层循环控制每个 Batch 的训练和验证过程
    for epoch in range(EPOCHS):
        print(f"\nEpoch (轮次) {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # ---- 训练阶段 (Training) ----
        model.train()           # 解锁神经元，打开 Dropout 和 BatchNorm 组件的更新权限
        running_loss = 0.0
        running_corrects = 0
        
        # tqdm 让这个循环变成一个带进度条的可视化过程，desc 参数设置了进度条前的描述文本
        for inputs, labels in tqdm(train_loader, desc="Training (训练)"):
            # 将每次拿到的 16 张图与正确的病情答案，送入到显卡内存
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # 每次训练前先把之前的梯度清零，防止它们叠加影响当前的权重更新
            
            outputs = model(inputs)            # 把这批图像送入模型，得到它们的预测输出（4类的概率分布）
            loss = criterion(outputs, labels)  # 计算这批预测与真实标签之间的损失值（越大说明模型越不满意自己的表现）
            
            _, preds = torch.max(outputs, 1)   # 从输出的概率分布中选出概率最高的那个类别索引作为模型的预测结果
            
            loss.backward()  # 反向传播，计算每个权重的梯度（告诉模型哪些神经连接需要加强，哪些需要削弱）
            optimizer.step() # 更新权重，根据计算出的梯度和学习率调整模型的参数，使其在下一轮训练中表现更好
            
            # 累计损失和正确预测的数量，用于计算整个 Epoch 的平均损失和准确率
            running_loss += loss.item() * inputs.size(0)
            # 计算总共对了几张图
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)                  # 算本轮失误率 
        epoch_acc = running_corrects.double() / len(train_dataset)      # 算本轮最终学习的准确率
        print(f"Train Loss (训练损失): {epoch_loss:.4f} Acc (准确率): {epoch_acc:.4f}")
        
        # ---- 验证阶段 ----
        model.eval()   # 锁定神经元，关闭 Dropout 和 BatchNorm 组件的更新权限，进入评估模式
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): # 在验证阶段不需要计算梯度，节省显存和计算资源
            for inputs, labels in tqdm(val_loader, desc="Validation (验证)"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)   # 计算验证集的损失值，评估模型在未见过的数据上的表现
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        # 计算整个验证集的平均损失和准确率
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss (验证损失): {val_epoch_loss:.4f} Acc (准确率): {val_epoch_acc:.4f}")
        
        # 更新学习率调度器，让它根据预设的余弦退火策略调整学习率，以便在训练后期更细致地优化模型权重
        scheduler.step()
        
        # 如果当前验证准确率超过了之前记录的最佳准确率，就更新最佳模型文件名
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc  # 更新最佳准确率记录器
            
            # 创建各模型专属的保存文件夹路径 (例如：saved_models/resnet18_multi)
            model_dir = os.path.join("saved_models", f"{model_name}_multi")
            os.makedirs(model_dir, exist_ok=True)
            # 生成文件名，包含模型名称、以及训练开始的时间戳，方便区分和管理多个模型版本
            best_model_name = os.path.join(model_dir, f"{model_name}_multi_{run_timestamp}.pth")
            # 将最佳模型的权重参数保存到指定路径，文件格式为 .pth，这是 PyTorch 推荐的模型权重保存格式
            torch.save(model.state_dict(), best_model_name)
            print(f"[*] 最佳模型已更新，准确率为: {best_acc:.4f}")
    print(f"\n训练完成！模型名称：{model_name}_multi_{run_timestamp}.pth")
    print(f"\n最高准确率: {best_acc:.4f}")
    return best_model_name

if __name__ == '__main__':
    # model_name = "resnet18"
    # model_name = "resnet50"
    model_name = "densenet121"
    # model_name = "efficientnet_b0"
    train_model(model_name=model_name)
