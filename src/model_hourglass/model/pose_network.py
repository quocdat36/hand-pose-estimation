# FILE: src/model_hourglass/model/pose_network.py (PHIÊN BẢN CUỐI CÙNG)

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import cv2
import numpy as np

# Import HandDetector và crop_frame từ cùng một file detectors.py
from .detectors import HandDetector, crop_frame 

# Import các hàm còn lại từ đúng vị trí của chúng
from .net_hg import Net_HM_HG
from ..util.net_util import load_net_model
from ..util.image_util import BHWC_to_BCHW, normalize_image
from ..util.heatmap_util import compute_uv_from_heatmaps

# Định nghĩa hằng số kích thước ở đây để dễ quản lý
INPUT_DIM = (256, 256)
HEATMAP_DIM = (64, 64) # Kích thước đầu ra của mạng Hourglass

class PoseNetwork(nn.Module):
    def __init__(self, cfg):
        super(PoseNetwork, self).__init__()
        # LƯU LẠI OBJECT CFG ĐỂ DÙNG SAU
        self.cfg = cfg 
        
        # TRUYỀN `cfg` XUỐNG CHO HandDetector
        self.detector = HandDetector(self.cfg)

        num_joints = self.cfg.MODEL.NUM_JOINTS
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=cfg.MODEL.HOURGLASS.NUM_STAGES,
                                num_modules=cfg.MODEL.HOURGLASS.NUM_MODULES,
                                num_feats=cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)
        self.device = self.cfg.MODEL.DEVICE

    def load_model(self): 
    # Nó sẽ dùng self.cfg mà đã được lưu trong hàm __init__
        load_net_model(self.cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, self.net_hm)

    def to(self, *args, **kwargs):
        # Đảm bảo tất cả các model con đều được chuyển sang đúng thiết bị
        super(PoseNetwork, self).to(*args, **kwargs)
        self.net_hm.to(*args, **kwargs)
        return self

    def forward(self, input_image, detect_hand=False):
        if not detect_hand:
            # Chế độ này ít được dùng, giữ nguyên logic gốc
            input_tensor = torch.from_numpy(input_image).to(self.device)
            input_tensor = BHWC_to_BCHW(input_tensor)
            input_tensor = normalize_image(input_tensor)
            est_hm_list, _ = self.net_hm(input_tensor)
            est_pose_uv = compute_uv_from_heatmaps(est_hm_list[-1], HEATMAP_DIM)
            return est_hm_list[-1], est_pose_uv[:, :, :2]
        else:
            # Chế độ chính cho ứng dụng web (detect_hand=True)
            hands_bbox = self.detector.detect(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            
            if hands_bbox is not None:
                coord, frame_cropped = crop_frame(input_image, hands_bbox, ratio=0.4)
                
                if coord is not None:
                    frame_resized = cv2.resize(frame_cropped, INPUT_DIM)
                    
                    # Chuyển đổi sang tensor
                    frame_tensor = torch.from_numpy(frame_resized.astype(np.float32))
                    frame_tensor = frame_tensor.to(self.device)
                    frame_tensor = frame_tensor.reshape((-1, INPUT_DIM[1], INPUT_DIM[0], 3))
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