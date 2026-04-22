import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from train_eval_multi_vit import build_model as build_model_vit
from train_multi import build_model as build_model_cnn

class TestEsophagusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_filename']
        img_path = os.path.join(self.img_dir, img_name)
        # 对应灰度图像预处理
        image = Image.open(img_path).convert('L').convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, img_name

def load_model(model_name_type, model_path, device, num_classes=4):
    """
    加载模型架构和权重
    """
    if model_name_type in ["vit_b_16", "swin_t"]:
        model = build_model_vit(model_name=model_name_type, num_classes=num_classes)
    else:
        model = build_model_cnn(model_name=model_name_type, num_classes=num_classes)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    print("=====================================================")
    print("    多模型软投票集成测试 (Soft Voting Ensemble)     ")
    print("=====================================================\n")

    # 1. 指定要融合的模型文件名字 (请根据你的 saved_models 实际文件名修改)
    # 比如这里我们融合一个 CNN 和一个 Transformer
    model1_filename = "efficientnet_b0_multi_2026_04_09_14_51_40.pth"
    model1_type = "efficientnet_b0"
    
    model2_filename = "swin_t_multi_2026_04_09_13_54_23.pth" 
    model2_type = "swin_t"

    model1_path = os.path.join("saved_models", f"{model1_type}_multi", model1_filename)
    model2_path = os.path.join("saved_models", f"{model2_type}_multi", model2_filename)

    if not os.path.exists(model1_path):
        print(f"找不到模型1: {model1_path}")
        sys.exit(1)
    if not os.path.exists(model2_path):
        print(f"找不到模型2: {model2_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 2. 加载数据
    csv_template = "csv_data/test_data_order.csv"
    img_dir = "TestSetImages"
    df = pd.read_csv(csv_template)

    # 验证集的标准预处理 (注意这里我们统一使用 CNN/ViT 公用的 224 输入尺寸，如果你之前改了大小，这里也要对等修改)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestEsophagusDataset(df, img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # 3. 实例化两个模型
    print(f"[*] 正在加载模型 1: {model1_type}")
    model1 = load_model(model1_type, model1_path, device)
    
    print(f"[*] 正在加载模型 2: {model2_type}")
    model2 = load_model(model2_type, model2_path, device)

    # 可以为不同模型设置权重，比如认为 CNN 更可靠，则 weight_cnn=0.6, weight_trans=0.4
    w1, w2 = 0.5, 0.5 

    predictions = []

    # 4. 预测与融合
    print("\n>>> 开始集成推理...")
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Ensemble Testing"):
            inputs = inputs.to(device)
            
            # 分别获取两个模型的 logits 输出
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            
            # 转化为概率分布 (Softmax)
            probs1 = F.softmax(outputs1, dim=1)
            probs2 = F.softmax(outputs2, dim=1)
            
            # Soft Voting: 概率加权平均
            ensemble_probs = (probs1 * w1) + (probs2 * w2)
            
            # 使用 argmax 取综合概率最大的作为最终结果
            _, preds = torch.max(ensemble_probs, 1)
            
            for img_name, pred_class in zip(img_names, preds.cpu().numpy()):
                predictions.append({
                    "image_filename": img_name,
                    "class_number": pred_class
                })

    # 5. 保存结果
    results_df = pd.DataFrame(predictions)
    df['class_number'] = results_df['class_number']
    
    # 将集成结果保存到专门的文件夹
    result_dir = os.path.join("eval_results", "ensemble_multi")
    os.makedirs(result_dir, exist_ok=True)
    output_csv = os.path.join(result_dir, f"ensemble_{model1_type}_{model2_type}.csv")
    
    df.to_csv(output_csv, index=False)
    print(f"\n[*] 测试完成! 集成预测结果已保存至: {output_csv}")

if __name__ == '__main__':
    main()