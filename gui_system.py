import os
import time
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from collections import Counter

# ================================
# 全局配置配置参数
# ================================
ARCHITECTURES = ['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vit_b_16', 'swin_t']

CLASS_NAMES_BINARY = {
    0: "健康 (Normal)", 
    1: "患病 (Disease)"
}

CLASS_NAMES_MULTI = {
    0: "正常 (Squamous_Epithelium)", 
    1: "肠化生 (Intestinal_metaplasia)", 
    2: "胃化生 (Gastric_metaplasia)", 
    3: "癌变 (Dysplasia_and_Cancer)"
}

SYSTEM_MODELS_DIR = "system_models"
EXPORT_DIR = "exports"

os.makedirs(os.path.join(SYSTEM_MODELS_DIR, "binary"), exist_ok=True)
os.makedirs(os.path.join(SYSTEM_MODELS_DIR, "multi"), exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ================================
# 模型相关辅助函数
# ================================
def infer_arch(filename):
    """从文件名中推断模型架构"""
    filename = filename.lower()
    for arch in ARCHITECTURES:
        if arch in filename:
            return arch
    return None

def build_eval_model(model_name, num_classes):
    """根据架构名称和分类数量构建模型骨架"""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"无法识别该架构: {model_name}")
    return model

def get_image_transform():
    """定义图片预测时的预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ================================
# GUI 类
# ================================
class CADSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("食道癌智能诊断辅助系统")
        self.root.geometry("1350x750")
        
        self.current_type = tk.StringVar(value="binary")
        self.loaded_image_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_prediction_results = ""  # 存储最新预测结果用于导出
        
        self.setup_ui()
        self.refresh_model_lists()

    def setup_ui(self):
        # 整体布局: 左侧模型管理，中间预测设置，右侧结果输出
        left_frame = tk.Frame(self.root, width=250, bd=2, relief="groove")
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        mid_frame = tk.Frame(self.root, width=400, bd=2, relief="groove")
        mid_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        right_frame = tk.Frame(self.root, bd=2, relief="groove")
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # --- 左侧：模型管理 ---
        tk.Label(left_frame, text="1. 分类任务切换", font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Radiobutton(left_frame, text="二分类 (正常 vs 患病)", variable=self.current_type, 
                       value="binary", command=self.refresh_model_lists).pack(anchor="w", padx=10)
        tk.Radiobutton(left_frame, text="四分类 (细粒度诊断)", variable=self.current_type, 
                       value="multi", command=self.refresh_model_lists).pack(anchor="w", padx=10)
        
        tk.Label(left_frame, text="2. 模型库管理", font=("Arial", 12, "bold")).pack(pady=(20,10))
        
        # 模型列表 (添加横向与纵向滚动条，并加宽显示)
        listbox_frame = tk.Frame(left_frame)
        listbox_frame.pack(fill="x", padx=10)
        
        scroll_y = tk.Scrollbar(listbox_frame, orient="vertical")
        scroll_x = tk.Scrollbar(listbox_frame, orient="horizontal")
        
        self.listbox_models = tk.Listbox(listbox_frame, height=15, width=45, exportselection=False,
                                         yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.config(command=self.listbox_models.yview)
        scroll_x.config(command=self.listbox_models.xview)
        
        scroll_x.pack(side="bottom", fill="x")
        self.listbox_models.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="导入新模型", command=self.upload_model, width=12).pack(side="left", padx=5)
        tk.Button(btn_frame, text="删除选中", command=self.delete_model, width=12, fg="red").pack(side="left", padx=5)
        
        # --- 中间：图像与预测设定 ---
        tk.Label(mid_frame, text="3. 图像上传与预测", font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Button(mid_frame, text="上传待测图片", command=self.upload_image, width=20, bg="#e0e0e0").pack(pady=5)
        
        # 使用固定大小的Frame来包裹Label，防止其被图像或空字元撑破/压缩导致失调
        img_frame = tk.Frame(mid_frame, width=280, height=210, bg="black")
        img_frame.pack(pady=10)
        img_frame.pack_propagate(False)
        self.img_label = tk.Label(img_frame, text="[图像预览区]", bg="black", fg="white")
        self.img_label.pack(expand=True, fill="both")
        
        self.filename_label = tk.Label(mid_frame, text="暂未选择图片", fg="gray", font=("Arial", 9), wraplength=300, justify="center")
        self.filename_label.pack(pady=(0, 10))
        
        tk.Label(mid_frame, text="4. 并行模型选择 (最多6个)", font=("Arial", 11, "bold")).pack(pady=5)
        self.combo_boxes = []
        for i in range(6):
            frame = tk.Frame(mid_frame)
            frame.pack(fill="x", padx=20, pady=2)
            tk.Label(frame, text=f"模型 {i+1}:", width=8).pack(side="left")
            cb = ttk.Combobox(frame, state="readonly", width=30)
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", self.on_combobox_selected)
            self.combo_boxes.append(cb)
            
        action_frame = tk.Frame(mid_frame)
        action_frame.pack(pady=20)
        tk.Button(action_frame, text="立即开始预测", command=self.run_prediction, width=15, bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=10)
        tk.Button(action_frame, text="导出预测报告", command=self.export_results, width=15, bg="#2196F3", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=10)

        # --- 右侧：结果展示区 ---
        tk.Label(right_frame, text="5. 综合诊断输出", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 滚动文本框 (设置宽度)
        self.text_result = tk.Text(right_frame, height=35, width=70, font=("Consolas", 10))
        scroll = tk.Scrollbar(right_frame, command=self.text_result.yview)
        self.text_result.configure(yscrollcommand=scroll.set)
        
        # 配置颜色 Tag
        self.text_result.tag_config("color_0", foreground="#00aa00") # 绿色
        self.text_result.tag_config("color_1_binary", foreground="red")
        self.text_result.tag_config("color_1_multi", foreground="#cccc00") # 黄色 (用较暗的黄保证白底可读)
        self.text_result.tag_config("color_2_multi", foreground="orange")
        self.text_result.tag_config("color_3_multi", foreground="red")
        self.text_result.tag_config("normal", foreground="black")
        
        self.text_result.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        
    def on_combobox_selected(self, event):
        """当下拉框选择变化时，动态更新其他下拉框的可用选项"""
        self.update_combobox_options()

    def update_combobox_options(self):
        """更新下拉框可选模型，防止重复选取"""
        c_type = self.current_type.get()
        folder = os.path.join(SYSTEM_MODELS_DIR, c_type)
        
        models_available = []
        if os.path.exists(folder):
            models_available = [f for f in os.listdir(folder) if f.endswith('.pth')]
            
        selected_models = [cb.get() for cb in self.combo_boxes if cb.get() not in ("无", "")]
        
        for cb in self.combo_boxes:
            current_val = cb.get()
            candidates = ["无"]
            for m in models_available:
                # 只有当该模型没有被其他下拉框选去，或者是该下拉框原本保持选中的模型时才可以加入菜单
                if m not in selected_models or m == current_val:
                    candidates.append(m)
            cb['values'] = candidates

    def refresh_model_lists(self):
        """刷新模型列表和下拉框，依据当前分类选择"""
        c_type = self.current_type.get()
        folder = os.path.join(SYSTEM_MODELS_DIR, c_type)
        
        self.listbox_models.delete(0, tk.END)
        self.display_to_filename = {}
        
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.pth')]
            for i, file in enumerate(files):
                display_text = f"[{i+1}] {file}"
                self.listbox_models.insert(tk.END, display_text)
                self.display_to_filename[display_text] = file
                
        for cb in self.combo_boxes:
            cb.set("无")
            
        self.update_combobox_options()
            
    def upload_model(self):
        """将用户模型拷贝到系统目录中持久化"""
        c_type = self.current_type.get()
        filepaths = filedialog.askopenfilenames(title="选择 PyTorch 模型", filetypes=[("PyTorch 模型", "*.pth")])
        if not filepaths: return
        
        folder = os.path.join(SYSTEM_MODELS_DIR, c_type)
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            
            # 检测架构是否明晰
            if not infer_arch(filename):
                messagebox.showwarning("架构识别失败", f"文件名 [{filename}] 中未包含支持的架构名称。\n支持的架构: {', '.join(ARCHITECTURES)}")
                continue
                
            dest = os.path.join(folder, filename)
            shutil.copy(filepath, dest)
            
        self.refresh_model_lists()
        messagebox.showinfo("成功", "模型已成功导入至本地模型库。")

    def delete_model(self):
        """删除选中的模型"""
        selection = self.listbox_models.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先在列表中选中要删除的模型。")
            return
            
        display_text = self.listbox_models.get(selection[0])
        filename = self.display_to_filename.get(display_text)
        
        if not filename: return
        
        c_type = self.current_type.get()
        confirm = messagebox.askyesno("确认删除", f"确定要彻底删除模型 [{filename}] 吗？")
        if confirm:
            file_path = os.path.join(SYSTEM_MODELS_DIR, c_type, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            self.refresh_model_lists()

    def upload_image(self):
        filepath = filedialog.askopenfilename(title="选择待测图片", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if filepath:
            self.loaded_image_path = filepath
            # 显示图片名称
            filename = os.path.basename(filepath)
            self.filename_label.config(text=f"图片: {filename}", fg="black")
            
            # 显示预览 (转为灰度图并预览，与模型实际吃入的数据形态保持一致)
            img = Image.open(filepath).convert('L')
            # Resize保持比例填充至 280x210 等
            img.thumbnail((280, 210))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img, text="")

    def run_prediction(self):
        if not self.loaded_image_path:
            messagebox.showwarning("操作错误", "请先上传一张待测图片。")
            return
            
        selected_files = [cb.get() for cb in self.combo_boxes if cb.get() != "无" and cb.get() != ""]
        if not selected_files:
            messagebox.showwarning("操作错误", "请至少选择一个模型用于预测。")
            return
            
        # 检测是否重复
        if len(selected_files) != len(set(selected_files)):
            messagebox.showwarning("操作错误", "您在选择框中选择了重复的模型，请移除重复项。")
            return

        self.text_result.delete("1.0", tk.END)
        self.text_result.insert(tk.END, ">>> 正在加载并运行预测...\n\n")
        self.root.update()
        
        c_type = self.current_type.get()
        num_classes = 2 if c_type == "binary" else 4
        class_names = CLASS_NAMES_BINARY if c_type == "binary" else CLASS_NAMES_MULTI
        
        def get_color_tag(c_type, p_class):
            if c_type == "binary":
                return "color_0" if p_class == 0 else "color_1_binary"
            else:
                if p_class == 0: return "color_0"
                if p_class == 1: return "color_1_multi"
                if p_class == 2: return "color_2_multi"
                if p_class == 3: return "color_3_multi"
            return "normal"
            
        def insert_text(text, tag="normal"):
            self.text_result.insert(tk.END, text + "\n", tag)
            results_text.append(text)
        
        # 图像预处理
        transform = get_image_transform()
        img_pil = Image.open(self.loaded_image_path).convert('L').convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)
        
        folder = os.path.join(SYSTEM_MODELS_DIR, c_type)
        
        # 清空文本进行彩色打印重构
        self.text_result.delete("1.0", tk.END)
        
        results_text = [] # 用于纯文本导出
        
        insert_text(f"图片路径: {self.loaded_image_path}")
        insert_text(f"图片名称: {os.path.basename(self.loaded_image_path)}")
        insert_text(f"预测模式: {'二分类' if c_type=='binary' else '四分类'}\n")
        insert_text("-" * 50)
        
        final_votes = []
        
        # 对每一个模型遍历执行
        for m_file in selected_files:
            arch = infer_arch(m_file)
            m_path = os.path.join(folder, m_file)
            
            insert_text(f"【模型名称】: {m_file}")
            
            try:
                # 动态加载模型
                model = build_eval_model(arch, num_classes)
                model.load_state_dict(torch.load(m_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    
                final_votes.append(pred_class)
                
                # 记录单独预测值
                for i in range(num_classes):
                    tag = get_color_tag(c_type, i) if i == pred_class else "normal"
                    insert_text(f"   -> 类别 [{class_names[i]}] 的预测概率: {probs[i]:.4f}", tag)
                
                # 单点汇总也赋予颜色
                insert_text(f"   => 单模型推断结果：{class_names[pred_class]}\n", get_color_tag(c_type, pred_class))
                
            except Exception as e:
                insert_text(f"   => 运行失败: {str(e)}\n")
        
        insert_text("-" * 50)
        insert_text("【最终系统汇总诊断】")
        
        # 多数表决机制处理
        if final_votes:
            counter = Counter(final_votes)
            most_common = counter.most_common()
            
            max_votes = most_common[0][1]
            total_votes = len(final_votes)
            
            # 如果最大票数不超过总数的一半 (例如 6 个模型，最大票数是 3，则一半一半) 
            # 意味着模型之间分歧严重
            if max_votes <= total_votes / 2:
                final_res = "【需要人工检测】 (各模型预测存在较大分歧，无法得出多数肯定结论)"
                insert_text(final_res, "normal")
            else:
                final_res_class = most_common[0][0]
                final_res = f"综合肯定预测: 【{class_names[final_res_class]}】"
                insert_text(final_res, get_color_tag(c_type, final_res_class))
        
        # 更新输出纯文本
        self.last_prediction_results = "\n".join(results_text)


    def export_results(self):
        if not self.last_prediction_results or not self.loaded_image_path:
            messagebox.showwarning("导出失败", "无预测结果可导出，请先执行一次成功的预测。")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_folder = os.path.join(EXPORT_DIR, f"report_{timestamp}")
        os.makedirs(export_folder, exist_ok=True)
        
        # 保存图片副本
        img_filename = os.path.basename(self.loaded_image_path)
        dest_img_path = os.path.join(export_folder, img_filename)
        shutil.copy(self.loaded_image_path, dest_img_path)
        
        # 保存结果文本
        report_path = os.path.join(export_folder, "prediction_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== 食道癌诊断系统检测报告 ===\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(self.last_prediction_results)
            
        messagebox.showinfo("导出成功", f"结果已成功导出至目录:\n{export_folder}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CADSystemGUI(root)
    root.mainloop()
