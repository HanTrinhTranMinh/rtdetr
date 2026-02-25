# predict.py
# predict.py (Cập nhật để lưu vào results)
import time
import cv2
import torch
from models.rtdetr import RTDETR #

def run_benchmark_and_visualize(model, val_loader, device, save_dir="results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Benchmark thời gian inference
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    avg_time = (time.time() - start_time) / 100
    print(f"Average Inference Time: {avg_time*1000:.2f}ms") #

    # 2. Lưu hình ảnh dự đoán mẫu
    # (Lấy một batch từ val_loader và vẽ boxes)
    # ... code vẽ cv2 và lưu vào results/sample_predict.jpg ...