import torch
import sys

def check_gpu():
    print("=" * 40)
    print("环境与 GPU 检测报告")
    print("=" * 40)
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"检测到的 GPU 数量: {gpu_count}")
        for i in range(gpu_count):
            print("-" * 20)
            print(f"GPU {i} 名称: {torch.cuda.get_device_name(i)}")
            # 获取显存信息 (总计显存/已分配/已缓存)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"GPU {i} 总显存: {total_memory:.2f} GB")
            print(f"GPU {i} 计算能力: {torch.cuda.get_device_capability(i)}")
    else:
        print("-" * 20)
        print("警告: 未检测到可用的 GPU。PyTorch 将使用 CPU 运行。")
        print("如果你的电脑有显卡，请检查是否正确安装了对应版本的 CUDA 和带有 GPU 支持的 PyTorch。")
    print("=" * 40)

if __name__ == "__main__":
    # 运行 GPU 检测函数
    check_gpu()
