import torch
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # 关键修复：设置后端
import matplotlib.pyplot as plt

# 配置参数
MODEL_PATH = "runs/train/exp2/weights/best.pt"  # 直接修改模型路径
CONF_THRES = 0.25
OUTPUT_DIR = "predict_results"

def select_image():
    """修复版：确保弹出文件选择窗口"""
    root = tk.Tk()
    root.attributes('-topmost', True)  # 确保窗口在最前
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    root.destroy()  # 关闭临时窗口
    return file_path

def main():
    # 加载模型
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    
    # 选择图片
    print("🖼️ 请选择图片...")
    img_path = select_image()
    if not img_path:
        print("❌ 未选择图片")
        return
    
    # 预测并显示
    results = model(img_path)
    results.show()
    results.save(OUTPUT_DIR)  # 保存结果
    
    # 显示结果
    plt.imshow(results.render()[0])
    plt.axis('off')
    plt.title("检测结果")
    plt.show()
    print(f"✅ 结果已保存至: {Path(OUTPUT_DIR).resolve()}")

if __name__ == "__main__":
    main()