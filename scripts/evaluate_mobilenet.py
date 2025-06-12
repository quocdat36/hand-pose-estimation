# FILE: scripts/evaluate_mobilenet.py

import sys
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các công cụ cần thiết
from src.model_hourglass.config import cfg
from src.model_hourglass.data.build import build_dataset
from app import load_mobilenet_model # Import trực tiếp hàm load
from src.model_mobilenet.handler import extract_mobilenet_coords # Import trực tiếp hàm xử lý
from src.utils import load_mobilenet_model

def main():
    print("Đang tải model MobileNetV2...")
    net = load_mobilenet_model()

    print("Đang tải bộ dữ liệu đánh giá FreiHAND...")
    cfg.merge_from_file("configs/eval_FreiHAND_dataset.yaml")
    dataset = build_dataset(cfg.EVAL.DATASET)
    
    print(f"Bắt đầu đánh giá trên {len(dataset)} ảnh...")
    all_errors = []

    for i in tqdm(range(len(dataset))):
        image_tensor, _, _, _, image_id = dataset[i]
        ground_truth_uv = dataset.keypoints_list[image_id.item()].numpy()

        image_np_chw = image_tensor.numpy()
        image_np_hwc = np.transpose(image_np_chw, (1, 2, 0))
        # Chuyển đổi màu sắc đúng cách cho OpenCV
        image_np_bgr = cv2.cvtColor((image_np_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        predicted_uv = extract_mobilenet_coords(net, image_np_bgr)

        if predicted_uv is not None:
            predicted_uv = predicted_uv[:21]
            # Loại bỏ các cặp có chứa NaN trước khi tính lỗi
            valid_indices = ~np.isnan(predicted_uv).any(axis=1) & ~np.isnan(ground_truth_uv).any(axis=1)
            if np.any(valid_indices):
                errors = np.linalg.norm(predicted_uv[valid_indices] - ground_truth_uv[valid_indices], axis=1)
                all_errors.extend(errors)

    if all_errors:
        mpjpe = np.mean(all_errors)
        print("="*50)
        print(f"Đánh giá MobileNetV2 trên FreiHAND Test Set hoàn tất.")
        print(f"MPJPE (2D Pixel Error): {mpjpe:.2f} pixels")
        print("="*50)
    else:
        print("Không có dự đoán hợp lệ nào được tạo ra.")

# GỌI HÀM MAIN TRỰC TIẾP
main()