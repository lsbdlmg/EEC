import os
import pandas as pd
import glob
import time
# 找到所有的带有 binary 的目录以及下面的 CSV
csv_files = []
for root, _, files in os.walk('eval_results'):
    for file in files:
        if 'binary' in root or 'binary' in file:
            if file.endswith('.csv') and 'mytest' in file:
                # 只针对 mytest 的预测结果 (包含新数据测试)，或者如果您用了别的前缀，请自行修改
                pass
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

print(f"找到以下相关的评测结果 ({len(csv_files)} 个):")
for f in csv_files:
    print(" -", f)

# 我们只取最新的四大网络 (resnet18, resnet50, densenet121, efficientnet_b0)
# 每个网络取最新的一个 csv

model_types = ['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vit_b_16', 'swin_t']
latest_csvs = {}

for m in model_types:
    m_files = [f for f in csv_files if m in f.lower() and 'binary' in f.lower()]
    if m_files:
        # 按照文件名排序（时间戳在文件名中）取最后一个
        m_files.sort()
        latest_csvs[m] = m_files[-1]

print("\n--- 提取最新日期的六个模型结果 ---")
for k, v in latest_csvs.items():
    print(k, ":", v)

if len(latest_csvs) < 6:
    print("警告：没凑齐6个二分类模型的结果！")

# 这里假设 test.py 最后输出的 csv 表格里有一列是 '正确率...' 或者 '错误' 或者能对比真实标签
# 更保险的做法：拿 csv_data/mytest.csv 作为真实标签，但有可能是别的模板。
# 我们直接提取 CSV 中带有判定结论的列，由于代码使用 "错误" 这个中文字符作为标志。

error_images_per_model = {}

for m, f_path in latest_csvs.items():
    try:
        df = pd.read_csv(f_path)
        # 寻找那一列判断是否错误的，列名可能包含 "正确率"
        acc_col = [col for col in df.columns if "正确" in col or "错误" in col]
        if acc_col:
            result_col = acc_col[0]
            # 找到结果那一列填了"错误"或者非空的
            # 兼容 only_mytest.py 里的 '错误' 以及 only_mytest2.py 里的 '错误(原...' 
            error_rows = df[df[result_col].astype(str).str.contains("错误")]
            img_names = error_rows['image_filename'].tolist()
            error_images_per_model[m] = set(img_names)
        else:
            print(f"在 {f_path} 中找不到带有 正确/错误 字眼的判断列。")
    except Exception as e:
        print(f"读取 {f_path} 出错: {e}")

import shutil

if error_images_per_model:
    print(f"\n--- 各自的预测错误数量 ---")
    for m, err_set in error_images_per_model.items():
        print(f"{m}: {len(err_set)} 张图片预测错误")
    
    from collections import Counter
    
    # 统计每张错误图片在所有模型中出错过几次
    sets_list = list(error_images_per_model.values())
    all_error_images = []
    for err_set in sets_list:
        all_error_images.extend(err_set)
        
    error_counts = Counter(all_error_images)
    # 取出至少有 5 个模型都预测错的图片
    common_errors = {img for img, count in error_counts.items() if count >= 4}
    
    print(f"\n>>> 至少有 5 个模型预测错误的图片数量: {len(common_errors)} 张 <<<")
    
    # 将这些预测失败的共性难点图片导出到一个新的专属文件夹供人工审查
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    export_dir = os.path.join("Hard_Cases_All_Models_Failed", run_timestamp)
    os.makedirs(export_dir, exist_ok=True)
        
    print(f"\n[操作] 正在将这 {len(common_errors)} 张图片拷贝到 '{export_dir}' 目录下...")
    base_img_dir = "BinaryTestSetImages"
    
    for i, img in enumerate(sorted(common_errors)):
        # 找出预测正确的模型
        correct_models = [m for m, err_set in error_images_per_model.items() if img not in err_set]
        
        # 构建新的文件名
        if correct_models:
            correct_str = "_".join(correct_models)
            new_img_name = f"对_{correct_str}_{img}"
        else:
            new_img_name = f"全错_{img}"
            
        # 打印部分列表
        if i < 23:
            print(f" {i+1}. {img} -> {new_img_name}")
            
        # 复制操作
        src_path = os.path.join(base_img_dir, img)
        dst_path = os.path.join(export_dir, new_img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f" [警告] 找不到源图片文件: {src_path}")
            
    print("\n[完成] 所有的疑难杂症图片已被成功聚合完毕，请前往审查。")
else:
    print("没有找到错误判断的数据。")
    # python find_common_errors.py
