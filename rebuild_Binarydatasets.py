"""
数据集重构与混合域自建脚本

该脚本用于解决模型在跨医院、跨光照环境（领域偏移 Domain Shift）下的泛化能力下降问题。
核心逻辑：
1. 从自建的二分类新数据（`自建数据集-0或1`）以及旧版的 0 和 3 类别中，强行各抽取 250 张正负样本（总计 1000 张）作为【混合盲测集】，用于客观反映模型真实跨域落地能力。
2. 将自建数据集中剩下的所有的图片，以及旧数据抽取剩下的部分交融在一起，组合成一个完美平衡（约 3000 多张级别）的新训练集，解决旧数据集不平衡与领域偏移的问题。
"""

import os               # 处理路径和系统文件遍历
import shutil           # 处理文件的物理拷贝与转移
import random           # 随机数发生器，用于打乱和随机抽样
import pandas as pd     # pandas库，建立新映射表
from tqdm import tqdm   # 提供控制台进度条显示

def main():
    print("==================================================")
    print(" 开始执行混合域数据集重组架构 (Domain Mix Building)")
    print("==================================================\n")
    # 随机种子固定，确保每次运行结果一致，便于调试和复现
    random.seed(42)

    # ---- 原始数据输入路径配置 ----
    CUSTOM_EEC_DIR = os.path.join("自建数据集-0或1", "eec")            # 新自建集的患病文件夹
    CUSTOM_NO_EEC_DIR = os.path.join("自建数据集-0或1", "no-eec")      # 新自建集的健康文件夹
    ORIGINAL_CSV = "TrainingSet.csv"
    ORIGINAL_IMG_DIR = "TrainingSetImages"                           # 官方/早期主训练图片库

    # ---- 最终的目标输出路径配置 ----
    TEST_DIR = "BinaryTestSetImages"       # 这个文件夹将被重建为全新的跨域测试集，包含来自自建数据和旧数据的混合样本，且标签已经二值化为 0 和 1，方便后续的模型评估和对比分析。
    TRAIN_DIR = "BinaryTrainSetImages"     # 这个文件夹将被重建为全新的训练集，包含来自自建数据和旧数据的混合样本，且标签已经二值化为 0 和 1，方便后续的模型训练和对比分析。
    
    # 第一步：清空旧的测试集和训练集文件夹，确保它们是干净的空目录，准备接收新的混合数据
    print("第一步：正在清空旧的 BinaryTestSetImages 和 BinaryTrainSetImages 文件夹...")
    for d in [TEST_DIR, TRAIN_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d) # 创建崭新的空目录
    print(" 文件夹已清空。\n")

    # 第二步：定义两个列表缓存，分别用于记录测试集和训练集的文件名及其对应的二值化标签，后续会将这些列表转换成 DataFrame 并导出为 CSV 文件，方便模型读取和使用
    list_test = []
    list_train = []

    # ===============================================
    # 1. 处理自建数据集的 eec (患病 -> Label 1)
    # ===============================================
    print("正在处理 自建数据集 eec (患病)...")
    # 把该目录下所有图片名列出来
    eec_files = [f for f in os.listdir(CUSTOM_EEC_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(eec_files) # 将这批患病新图内部彻底随机打乱
    
    # 抽取配额 - 前250张作为跨域测试图，1000张剩余的混入训练堆
    eec_test = eec_files[:250]     # 切割前250张作为跨域测试图
    eec_train = eec_files[250:1250]    # 切割第250以后的1000张作为训练集，混入训练堆

    # 1.1 拷贝 患病(eec) 测试集
    # 注意：这里的标签已经被直接映射成了二分类的患病(1)，不再保留原来的多分类标签，以适应新的二分类训练和评估需求
    for f in tqdm(eec_test, desc="  -> 抽取 250 张至测试集"):
        new_name = f"test_eec_{f}"# 给每个文件加上前缀，明确它们的来源和标签，方便后续分析和排查
        shutil.copy(os.path.join(CUSTOM_EEC_DIR, f), os.path.join(TEST_DIR, new_name))
        list_test.append((new_name, 1)) # 把目标文件名及其二值化标签[1]记录到测试表缓存中
        
    # 1.2 拷贝 患病(eec) 剩余训练集
    for f in tqdm(eec_train, desc="  -> 剩余图片放入训练集"):
        new_name = f"train_custom_eec_{f}"# 给每个文件加上前缀，明确它们的来源和标签，方便后续分析和排查
        shutil.copy(os.path.join(CUSTOM_EEC_DIR, f), os.path.join(TRAIN_DIR, new_name))
        list_train.append((new_name, 1))

    # ===============================================
    # 2. 处理自建数据集的 no-eec (健康 -> Label 0)
    # ===============================================
    print("\n正在处理 自建数据集 no-eec (健康)...")
    no_eec_files = [f for f in os.listdir(CUSTOM_NO_EEC_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(no_eec_files)
    
    no_eec_test = no_eec_files[:250] # 切割前250张作为跨域测试图
    no_eec_train = no_eec_files[250:1100] # 切割第250以后的850张作为训练集，混入训练堆

    # 2.1 拷贝 健康(no-eec) 测试集，和上面同样的操作，但标签归宿定为 [0]
    for f in tqdm(no_eec_test, desc="  -> 抽取 250 张至测试集"):
        new_name = f"test_noeec_{f}"# 给每个文件加上前缀，明确它们的来源和标签，方便后续分析和排查
        shutil.copy(os.path.join(CUSTOM_NO_EEC_DIR, f), os.path.join(TEST_DIR, new_name))
        list_test.append((new_name, 0))

    # 2.2 拷贝 健康(no-eec) 剩余训练集
    for f in tqdm(no_eec_train, desc="  -> 剩余图片放入训练集"):
        new_name = f"train_custom_noeec_{f}"# 给每个文件加上前缀，明确它们的来源和标签，方便后续分析和排查
        shutil.copy(os.path.join(CUSTOM_NO_EEC_DIR, f), os.path.join(TRAIN_DIR, new_name))
        list_train.append((new_name, 0))

    # ===============================================
    # 3. 处理旧版数据集的健康(0)和癌症(3)，并将它们混入测试集和训练集
    # ===============================================
    print("\n正在向训练集混入原数据集...")
    ORIGINAL_CSV = os.path.join("csv_data", "TrainingSet.csv")
    if not os.path.exists(ORIGINAL_CSV):
        print(f"[警告] 找不到 {ORIGINAL_CSV}，请确认数据是否完整！")
        return
        
    orig_df = pd.read_csv(ORIGINAL_CSV)
    
    # 用 Pandas 的布尔切片条件定位出全部旧版的健康(0)图片名，和最严重癌症(3)图片名
    orig_class0 = orig_df[orig_df['class_number'] == 0]['image_filename'].tolist()
    orig_class3 = orig_df[orig_df['class_number'] == 3]['image_filename'].tolist()

    # 将原有数据集的两类也全面打乱，保证抽取的随机性
    random.shuffle(orig_class0)
    random.shuffle(orig_class3)

    # 3.1 划分旧版健康(0)：抽取250张去测试集，其余全去训练集
    orig_class0_test = orig_class0[:250]
    orig_class0_train = orig_class0[250:1400]# 这1150张剩余的健康图全都放入训练集，追求二分类的均衡分布

    for f in tqdm(orig_class0_test, desc="  -> 抽取 250 张旧版健康(0)至测试集"):
        new_name = f"test_orig_class0_{f}"
        src_path = os.path.join(ORIGINAL_IMG_DIR, f)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(TEST_DIR, new_name))
            list_test.append((new_name, 0))

    for f in tqdm(orig_class0_train, desc="  -> 导入剩余旧版健康(0)至训练集"):
        new_name = f"train_orig_class0_{f}"
        src_path = os.path.join(ORIGINAL_IMG_DIR, f)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(TRAIN_DIR, new_name))
            list_train.append((new_name, 0))

    # 3.2 划分旧版癌症(3)：抽取250张去测试集，再截取1000张去训练集（共计动用1250张以追求二分类均衡）
    orig_class3_test = orig_class3[:250]
    orig_class3_train = orig_class3[250:1250]
    orig_class3_manual = orig_class3[1250:1300] # 这50张剩余的癌症图先放一边，单独放到一个手动测试文件夹里，留给后续的人工排查和模型微调使用

    for f in tqdm(orig_class3_test, desc="  -> 抽取 250 张旧版癌症(3)至测试集"):
        new_name = f"test_orig_class3_{f}"# 给每个文件加上前缀，明确它们的来源和标签，方便后续分析和排查
        src_path = os.path.join(ORIGINAL_IMG_DIR, f)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(TEST_DIR, new_name))
            list_test.append((new_name, 1))

    for f in tqdm(orig_class3_train, desc="  -> 导入抽样 950 张旧版癌症(3)至训练集"):
        new_name = f"train_orig_class3_{f}"
        src_path = os.path.join(ORIGINAL_IMG_DIR, f)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(TRAIN_DIR, new_name))
            list_train.append((new_name, 1))

    MANUAL_DIR = "ManualTest_Class3_Remaining"
    if os.path.exists(MANUAL_DIR):
        shutil.rmtree(MANUAL_DIR)
    os.makedirs(MANUAL_DIR)
    
    for f in tqdm(orig_class3_manual, desc="  -> 导出剩余的50张旧版癌症(3)至手动测试文件夹"):
        src_path = os.path.join(ORIGINAL_IMG_DIR, f)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(MANUAL_DIR, f))
 # 注意老版中的类别3在这里被归属降维打击映射成了二分类的患病(1)

    # ===============================================
    # 4. 生成统一标准的描述清单映射表 (CSV)
    # ===============================================
    # 将列表转换成 DataFrame 格式，方便直出表格
    test_df = pd.DataFrame(list_test, columns=["image_filename", "class_number"])
    train_df = pd.DataFrame(list_train, columns=["image_filename", "class_number"])
    
    # frac=1 表示按 100% 比例二次抽样(即实现彻底洗牌乱序)
    # reset_index 则是抹去乱序后歪斜的原本行号重新排号，防止网络读取索引错乱
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    # 统包导出
    os.makedirs("csv_data", exist_ok=True)
    test_df.to_csv(os.path.join("csv_data", "mytest.csv"), index=False)
    train_df.to_csv(os.path.join("csv_data", "mytrain.csv"), index=False)

    print("\n==================================================")
    print(" 混合打包完成！")
    print(f" [测试集] 生成数量: {len(test_df)} 张 -> csv_data/mytest.csv 和 BinaryTestSetImages/")
    print(f" [训练集] 生成数量: {len(train_df)} 张 -> csv_data/mytrain.csv 和 BinaryTrainSetImages/")
    print("==================================================")

if __name__ == '__main__':
    main()
    # python rebuild_Binarydatasets.py