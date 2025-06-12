# FILE: src/model_hourglass/model/pose_network.py (FINAL, CORRECTED VERSION)

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import cv2
import numpy as np

# --- CÁC ĐƯỜNG DẪN IMPORT ĐÃ ĐƯỢC XÁC THỰC ---
from .detectors import HandDetector, crop_frame
from .net_hg import Net_HM_HG
from ..util.net_util import load_net_model
from ..util.image_util import BHWC_to_BCHW, normalize_image
from ..util.heatmap_util import compute_uv_from_heatmaps

# Định nghĩa các hằng số kích thước
RESIZE_DIM = (256, 256)
HEATMAP_DIM = (64, 64)

class PoseNetwork(nn.Module):
    def __init__(self, cfg):
        super(PoseNetwork, self).__init__()
        # Lưu lại config để sử dụng trong toàn bộ class
        self.cfg = cfg
        
        # Truyền config vào HandDetector khi khởi tạo
        self.detector = HandDetector(self.cfg)

        num_joints = self.cfg.MODEL.NUM_JOINTS
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=self.cfg.MODEL.HOURGLASS.NUM_STAGES,
                                num_modules=self.cfg.MODEL.HOURGLASS.NUM_MODULES,
                                num_feats=self.cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)
        self.device = torch.device(self.cfg.MODEL.DEVICE)

    def load_model(self):
        # Hàm này không cần tham số, nó sẽ dùng self.cfg đã được lưu
        load_net_model(self.cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, self.net_hm)

    def to(self, *args, **kwargs):
        super(PoseNetwork, self).to(*args, **kwargs)
        self.net_hm.to(*args, **kwargs)
        return self

    def forward(self, input_data, detect_hand=False):
        """
        Hàm forward linh hoạt, xử lý cả Tensor (từ dataloader) và ảnh Numpy (từ app).
        """
        if not detect_hand:
            input_tensor = input_data.to(self.device)
            input_tensor = BHWC_to_BCHW(input_tensor)
            input_tensor = normalize_image(input_tensor)
            est_hm_list, _ = self.net_hm(input_tensor)
            final_heatmap = est_hm_list[-1]
            est_pose_uv = compute_uv_from_heatmaps(final_heatmap, (224, 224))
            
            # Trả về heatmap và tọa độ 2D
            return final_heatmap, est_pose_uv[:, :, :2]
        else:
            # --- CHẾ ĐỘ DEMO THỜI GIAN THỰC (REAL-TIME MODE) ---
            # input_data ở đây là một ảnh numpy BGR
            input_image = input_data
            
            hands_bbox = self.detector.detect(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            
            if hands_bbox is not None:
                coord, frame_cropped = crop_frame(input_image, hands_bbox, ratio=0.4)
                
                if coord is not None:
                    frame_resized = cv2.resize(frame_cropped, RESIZE_DIM)
                    
                    # Chuyển đổi sang tensor
                    frame_tensor = torch.from_numpy(frame_resized.astype(np.float32)).to(self.device)
                    frame_tensor = frame_tensor.reshape((-1, RESIZE_DIM[1], RESIZE_DIM[0], 3))
                    frame_tensor = BHWC_to_BCHW(frame_tensor)
                    frame_tensor = normalize_image(frame_tensor)

                    # Dự đoán heatmap
                    est_hm_list, _ = self.net_hm(frame_tensor)
                    final_heatmap = est_hm_list[-1]

                    # Tính toán tọa độ từ heatmap
                    est_pose_uv = compute_uv_from_heatmaps(final_heatmap, HEATMAP_DIM)
                    
                    # Trả về tất cả thông tin cần thiết
                    return coord, final_heatmap, est_pose_uv[:, :, :2]
                    
            # Trả về None nếu không phát hiện được tay
            return None, None, None