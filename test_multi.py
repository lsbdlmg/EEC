"""
官方测试集测试
"""

import os               # 操作系统接口操作相关
import sys              # 捕获异常或未找到文件时强行退出脚本
from test import run_test # 导入通用的推理核心引擎

def main():
    print("=====================================================")
    print("    测试          ")
    print("=====================================================\n")
    
    # ---------------- 1. 定位要测试的基准模型 ----------------
    # 用户需手动修改为你想要测试的具体模型的文件名
    model_name = 'efficientnet_b0_multi_2026_04_22_13_01_38.pth' 
    # python test_multi.py
    best_model_path = model_name

    # 解析预训练架构类别
    if "resnet18" in model_name: m_str = "resnet18"
    elif "densenet121" in model_name: m_str = "densenet121"
    elif "efficientnet" in model_name: m_str = "efficientnet_b0"
    elif "vit" in model_name: m_str = "vit_b_16"
    elif "swin" in model_name: m_str = "swin_t"
    else: m_str = "resnet50"
        
    possible_path = os.path.join("saved_models", f"{m_str}_multi", model_name)
    if os.path.exists(possible_path):
        print(f"[*] 自动匹配到模型路径: {possible_path}")
        best_model_path = possible_path
    else:
        print(f"\n找不到模型文件: {model_name} 或 {possible_path}")
        sys.exit(1)

    # ---------------- 2. 调取预测引擎 ----------------
    try:
        print("\n>>> 开始测试")
        # 委托 test.py 中的通用函数为我们完成显卡推理运算。
        run_test(best_model_path, csv_template="csv_data/test_data_order.csv", img_dir="TestSetImages")
        
        print("\n=====================================================")
        print("    测试结果推断已全部完成，输出待查收   ")
        print("=====================================================")
        
    except Exception as e:
        print(f"\n流程运行出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
