"""
自建数据集测试
"""

import os               # 操作系统接口，处理文件与路径判断
import sys              # 系统模块，用于控制脚本强制退出
import pandas as pd     # pandas库，处理和比对预测标签表与真实标签表
from test import run_test # 导入通用的底层测试执行器方法

def main():
    """
    主程序执行入口：负责寻找权重、调用测试组件、并进行结果比对与评估报表生成。
    """
    print("=====================================================")
    print("    测试并对比 '自建数据集-0或1' 中的数据          ")
    print("=====================================================\n")
    
    # ---------------- 1. 定位要测试的模型权重文件 ----------------
    # 用户需手动修改为你想要测试的具体模型的文件名
    model_name = 'vit_b_16_binary_2026_04_22_14_39_15.pth' 
    # python test_binary.py
    best_model_path = model_name
    # 解析预训练架构类别
    if "resnet18" in model_name: m_str = "resnet18"
    elif "densenet121" in model_name: m_str = "densenet121"
    elif "efficientnet" in model_name: m_str = "efficientnet_b0"
    elif "vit" in model_name: m_str = "vit_b_16"
    elif "swin" in model_name: m_str = "swin_t"
    else: m_str = "resnet50"
        
    possible_path = os.path.join("saved_models", f"{m_str}_binary", model_name)
    if os.path.exists(possible_path):
        print(f"[*] 自动匹配到模型路径: {possible_path}")
        best_model_path = possible_path
    else:
        print(f"\n 找不到模型文件: {model_name} 或 {possible_path}")
        sys.exit(1)

    # ---------------- 2. 检查评测专用验证表是否存在 ----------------
    # 确认记录了真实验证数据的目录已经生成
    csv_file = os.path.join("csv_data", "mytest.csv")
    if not os.path.exists(csv_file):
        print(f"\n 找不到 {csv_file}，请先运行数据分割相关脚本生成此问卷")
        sys.exit(1)

    # ---------------- 3. 推理与打分流程 ----------------
    try:
        print("\n>>> 步骤 1/2: 开始对自建数据集推理")
        
        # 委托 test.py 中的通用函数为我们完成显卡推理运算。
        # 传递指定的新表(mytest.csv)和专属的新图文件夹。
        output_csv_path = run_test(best_model_path, csv_template=csv_file, img_dir="BinaryTestSetImages")
        
        if not output_csv_path or not os.path.exists(output_csv_path):
            print(" 预测结果 CSV 生成失败！")
            sys.exit(1)
            
        print("\n>>> 步骤 2/2: 计算系统正确率")
        # 加载正确答案表
        true_df = pd.read_csv(csv_file)
        # 加载 AI 写出的答案表
        pred_df = pd.read_csv(output_csv_path)

        total_samples = len(true_df) # 总题数
        correct_count = 0            # 答对题数
        
        # 收集每道题的评卷结果字串
        validation_status = []

        # 逐行阅卷
        for idx in range(total_samples):
            true_label = true_df.iloc[idx]['class_number']
            pred_label = pred_df.iloc[idx]['class_number']

            # 根据真实标签和预测标签的对比结果，统计正确数量，并在 validation_status 中记录 '错误' 或 '' 以便后续生成成绩单时使用
            if (true_label == 0 and pred_label == 0) or (true_label == 1 and pred_label == 1):
                correct_count += 1
                validation_status.append("")       # 答对了，在此单元格留空以便后续人工复核时快速定位正确题目
            else:
                validation_status.append("错误")     # 答错了，在此单元格标注 '错误' 以便后续人工复核时快速定位问题题目

        # ---------------- 4. 制作用于展示的综合成绩单 ----------------
        accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
        new_column_name = f"正确率：{accuracy:.2f}%"

        # 在刚才那张预测表的最右侧追加出这个阅卷列，作为总结展示
        pred_df[new_column_name] = validation_status
        # 使用 utf-8-sig 非常重要，它能强行带上BOM头，防止通过微软Excel软件点开时产生中文乱码
        pred_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig') 

        print(f"\n=====================================================")
        print(f"    自建数据集的综合检测正确率为: {accuracy:.2f}%")
        print(f"=====================================================")

    except Exception as e:
        print(f"\n(流程运行出错): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
