"""
食道癌早期检测模型测试脚本
该脚本旨在加载经过训练的疾病分类模型，并在完全陌生的测试集数据上进行推理预测，
最终将预测结果按照规定格式生成 CSV 结果文件，以供官方评测或提交。
"""

import os               # 操作系统接口模块，用于创建文件路径和处理文件名
import torch            # PyTorch 核心模块，支持张量和运算
from torch.utils.data import Dataset, DataLoader # 数据集与数据加载器，批量读取测试数据
from PIL import Image   # Python Imaging Library，用于加载和解析图像文件
from torchvision import transforms # 图像处理和增强工具
import pandas as pd     # pandas 库，用于读取和生成保存最终结果的 CSV 文件
from tqdm import tqdm   # 进度条可视化库

class TestEsophagusDataset(Dataset):
    """
    针对纯测试任务的数据集加载类。
    与训练集不同，测试集通常只有图片名称，不包含真实标签(或者我们不应该在这里查看标签)，
    因此 __getitem__ 只返回“图片张量”和它的“文件名”。
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df                     # 包含待测试图片名称列表的数据表
        self.img_dir = img_dir           # 测试集图片的真实本地存储目录
        self.transform = transform       # 指向为测试集制定的预处理函数 

    def __len__(self):
        # 返回测试集的样本总数
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图片名称和它的完整物理路径
        img_name = self.df.iloc[idx]['image_filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 核心的数据对齐策略：
        # 测试集的图片来源可能更杂，甚至包含显微镜直接直出的纯黑白图片。
        # 这里统一先转设为 'L' (灰度图，消除RGB色阶干扰)，然后再转换为 'RGB' 以匹配预训练模型所需的三通道输入格式。
        # 这是为了彻底消除测试集和训练集在色彩分布上可能存在的领域偏移 (Domain Shift)。
        image = Image.open(img_path).convert('L').convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # 返回处理好的图片张量和它的文件名，供后续推理和结果记录使用
        return image, img_name

def run_test(model_path, csv_template="csv_data/test_data_order.csv", img_dir="TestSetImages"):
    """
    执行测试(验证)并输出结果表格的主函数。
    
    参数:
    model_path (str): 之前训练保存下来的 `.pth` 权重文件路径。
    csv_template (str): 测试数据集的文件名清单(表格模板，通常要求输出结果顺序与此模板一致)。
    img_dir (str): 测试图片的物理目录。
    """
    # 动态检测是否存在可用显卡并打印提示
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 开始测试阶段 ===")
    print(f"加载模型: {model_path}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)  # 获取诸如 "RTX 3060" 之类的硬件字串供打印信息提示
        print(f"使用的设备: {device} [{gpu_name}]")
    else:
        print(f"使用的设备: {device}")
    
    # ---------------- 1. 解析模型名称以规划其所属的测试结果目录 ----------------
    filename = os.path.basename(model_path) # 从长路径中抠出文件名 (如 densenet121_2026_03_27_16.pth)
    # 利用字符串替换去除后缀等字符，粗略提取出时间戳
    timestamp = filename.replace(".pth", "")
    
    is_binary = "binary" in filename
    type_str = "binary" if is_binary else "multi"  # 根据字符串判断这是个二分类模型还是四分类模型
    
    # 根据保存的文件名里的标识判断它原本套用的具体什么预训练结构
    if "resnet18" in filename: model_str = "resnet18"
    elif "densenet121" in filename: model_str = "densenet121"
    elif "efficientnet" in filename: model_str = "efficientnet_b0"
    else: model_str = "resnet50"
        
    # 定义并将该测试结果文件存在哪个专属目录下，防止与别的网络测试结果混合
    result_dir = os.path.join("eval_results", f"{model_str}_{type_str}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 合成完整的测试结果CSV的输出路径文件名
    output_csv = os.path.join(result_dir, f"{timestamp}.csv")
    
    # ---------------- 2. 读取要测试的数据模板 ----------------
    df = pd.read_csv(csv_template)
    
    # ---------------- 3. 数据预处理构建 (严格遵循验证集配置) ----------------
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),              # 简单地把所有长图方图压成规定的 224 正方形
        transforms.ToTensor(),                      # 把图片转成 PyTorch 能处理的数据格式矩阵
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet 归一化参数，保持与训练相同
    ])
    
    # 将包含表格数据与预处理方式的类实例化为 test_dataset，并装载进数据流加载器 DataLoader 里
    test_dataset = TestEsophagusDataset(df, img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # ---------------- 4. 动态反向推理加载模型结构 ----------------
    # 提取网络名以便调用初始化函数
    if "resnet18" in model_path:
        model_name = "resnet18"
    elif "densenet121" in model_path:
        model_name = "densenet121"
    elif "efficientnet" in model_path:
        model_name = "efficientnet_b0"
    else:
        model_name = "resnet50"
        
    # 判断该权重模型是 2个输出节点 还是 4个输出节点
    is_binary = "binary" in model_path
    num_classes = 2 if is_binary else 4
        
    print(f"[*] 自动匹配网络结构检测为: {model_name} (Classes: {num_classes})")
    
    # 根据检测结果从对应的上游文件提取相应的模型构建器骨架
    if is_binary:
        from train_binary import build_binary_model
        model = build_binary_model(model_name=model_name)
    else:
        from train_multi import build_model
        model = build_model(model_name=model_name, num_classes=num_classes)
        
    # 使用 torch.load 原封不动读取硬盘里的模型记忆参数，将其赋予新建出来的一张白纸的骨架上。
    # map_location 是为了防止原本是在其他型号的显卡练的强行在本机加载引发错误，将其统一引导至本机的设备流。
    # weights_only=True 是为了兼容 PyTorch 2.0 以后的版本，如果你用的是更老的版本，可能需要去掉这个参数。
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # 把模型投入显卡，准备好进行高速推理计算
    model.eval()      # 切换到评测模式，禁用 Dropout 和 BatchNorm 等训练专用层的随机行为，确保推理结果稳定可靠
    
    predictions = []  # 定义一个空的列表收集模型预测的答案
    
    # ---------------- 5. 开始实战推理循环 ----------------
    # 使用 torch.no_grad() 上下文管理器来禁止梯度计算，节省显存和加速推理过程，因为我们在测试阶段不需要反向传播。
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="测试中"):
            inputs = inputs.to(device)         # 将图片投入显卡
            outputs = model(inputs)            # 向前传播产生一系列的概率矩阵
            
            # 使用 argmax 取出这一列里数值最大的那个位置（比如输出[0.1, 0.8, 0.05, 0.05]，则输出的类别索引就是 1）
            _, preds = torch.max(outputs, 1)
            
            # 由于可能一次送进去了16张图(因 batch_size 决定)，所以要把这16个结果拆开，连同名字组成字典，逐个挂在集合列表中
            for img_name, pred_class in zip(img_names, preds.cpu().numpy()):
                predictions.append({
                    "image_filename": img_name,
                    "class_number": pred_class
                })
                
    # ---------------- 6. 整理预测答案并制表输出 ----------------
    # 将字典列表转为表格数据
    results_df = pd.DataFrame(predictions)
    
    # 将预测结果的类别数字写回到原始的 DataFrame 中，覆盖掉原本的 class_number 列
    df['class_number'] = results_df['class_number']
    # 导出到本方法开头计算好的物理表格里，并且不自带左侧毫无意义的序号索引(index=False)
    df.to_csv(output_csv, index=False)
    
    print(f"\n测试完成.")
    print(f"[*] 测试结果已保存至: {output_csv}")
    
    # 返回这个保存好的报表绝对路径，以便调用者可以直接拿去评测或提交
    return output_csv

if __name__ == '__main__':
    pass
