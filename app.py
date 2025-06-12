# FILE: app.py (FINAL VERSION WITH 2 IMPROVEMENTS)

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import yaml
import torch
import mediapipe as mp
import pandas as pd
import time
import sys
import os

# --- THÊM ĐƯỜNG DẪN CỦA DỰ ÁN VÀO PYTHON PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# --- CÁC DÒNG IMPORT TÙY CHỈNH ---
from src.model_hourglass.config import cfg
from src.model_hourglass.model.pose_network import PoseNetwork
from src.model_hourglass.util.vis_pose_only import draw_2d_skeleton
from src.model_mobilenet.handler import process_frame_with_mobilenet

# --- CÀI ĐẶT TRANG ---
st.set_page_config(page_title="Hand Pose Estimation Demo", page_icon="👋", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_posenetwork_model():
    """Tải và cấu hình mô hình PoseNetwork (Hourglass)."""
    print("Đang tải model PoseNetwork...")
    config_file = "configs/eval_webcam.yaml"
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    model = PoseNetwork(cfg)
    device = cfg.MODEL.DEVICE
    model.load_model()
    model.to(device)
    model = model.eval()
    print("Đã tải model PoseNetwork thành công!")
    return model

@st.cache_resource
def load_mobilenet_model():
    """Tải mô hình MobileNetV2 (Caffe)."""
    print("Đang tải model MobileNetV2...")
    protoFile = "weights/mobilenet/pose_deploy.prototxt"
    weightsFile = "weights/mobilenet/pose_iter_102000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    print("Đã tải model MobileNetV2 thành công!")
    return net


posenetwork_model = load_posenetwork_model()
mobilenet_model = load_mobilenet_model()

# --- TIÊU ĐỀ VÀ KHỞI TẠO MEDIAPIPE ---
st.title("So sánh các mô hình Ước tính Tư thế Bàn tay")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- CÁC HÀM XỬ LÝ ---
def process_image_with_mediapipe(image_bgr):
    # (Nội dung hàm này giữ nguyên)
    annotated_image = image_bgr.copy()
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image

def process_with_posenetwork(image_bgr, model):
    annotated_image = image_bgr.copy()
    RESIZE_DIM = (256, 256)
    HEATMAP_DIM = (64, 64)
    with torch.no_grad():
        coord, _, est_pose_uv = model(annotated_image, detect_hand=True)
    if est_pose_uv is not None:
        est_pose_uv = est_pose_uv.to('cpu')
        crop_width = coord[2] - coord[0]
        crop_height = coord[3] - coord[1]
        scale_x = crop_width / HEATMAP_DIM[0]
        scale_y = crop_height / HEATMAP_DIM[1]
        keypoints_np = est_pose_uv[0].detach().numpy()
        keypoints_np[:, 0] = keypoints_np[:, 0] * scale_x + coord[0]
        keypoints_np[:, 1] = keypoints_np[:, 1] * scale_y + coord[1]
        cv2.rectangle(annotated_image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)
        annotated_image = draw_2d_skeleton(annotated_image, keypoints_np)
    return annotated_image

def extract_posenetwork_coords(model, image_bgr):
    """Trích xuất tọa độ thô từ PoseNetwork và trả về mảng Numpy."""
    with torch.no_grad():
        coord, _, est_pose_uv = model(image_bgr, detect_hand=True)
    if est_pose_uv is not None:
        est_pose_uv_cpu = est_pose_uv.to('cpu')
        HEATMAP_DIM = (64, 64)
        crop_width = coord[2] - coord[0]
        crop_height = coord[3] - coord[1]
        scale_x = crop_width / HEATMAP_DIM[0]
        scale_y = crop_height / HEATMAP_DIM[1]
        keypoints_np = est_pose_uv_cpu[0].detach().numpy()
        keypoints_np[:, 0] = keypoints_np[:, 0] * scale_x + coord[0]
        keypoints_np[:, 1] = keypoints_np[:, 1] * scale_y + coord[1]
        return keypoints_np
    return None

def extract_mobilenet_coords(net, image_bgr, threshold=0.1):
    """Trích xuất tọa độ thô từ MobileNet và trả về mảng Numpy."""
    image_height, image_width, _ = image_bgr.shape
    aspect_ratio = image_width / image_height
    in_height = 368
    in_width = int(((aspect_ratio * in_height) * 8) // 8)
    inpBlob = cv2.dnn.blobFromImage(image_bgr, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    points = np.full((22, 2), np.nan, dtype=np.float32)
    for i in range(22):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (image_width, image_height))
        _, prob, _, point = cv2.minMaxLoc(probMap)
        if prob > threshold:
            points[i, 0], points[i, 1] = point[0], point[1]
    if np.all(np.isnan(points)): return None
    return points

def draw_posenetwork_skeleton(image, keypoints_np):
    """Vẽ bộ xương cho PoseNetwork."""
    if keypoints_np is None: return image
    return draw_2d_skeleton(image, keypoints_np)

def draw_mobilenet_skeleton(image, keypoints_np):
    """Vẽ bộ xương cho MobileNet."""
    if keypoints_np is None: return image
    POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    for pair in POSE_PAIRS:
        partA, partB = pair[0], pair[1]
        if not np.isnan(keypoints_np[partA, 0]) and not np.isnan(keypoints_np[partB, 0]):
            ptA = (int(keypoints_np[partA, 0]), int(keypoints_np[partA, 1]))
            ptB = (int(keypoints_np[partB, 0]), int(keypoints_np[partB, 1]))
            cv2.line(image, ptA, ptB, (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(image, ptA, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(image, ptB, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return image
    
# --- LỚP XỬ LÝ WEBCAM (ĐÃ TÍCH HỢP CẢI TIẾN) ---
class HandPoseTransformer(VideoTransformerBase):
    def __init__(self):
        # Khởi tạo các thuộc tính ở trạng thái "chưa sẵn sàng"
        self.model_choice = None
        self.posenetwork_model = None
        self.mobilenet_model = None
        self.threshold = 0.1
        self.last_processed_image = None # Dùng cho frame skipping

        # Khởi tạo các biến thu thập số liệu
        self._reset_stats()

    def _reset_stats(self):
        """Hàm nội bộ để reset tất cả các số liệu thống kê."""
        self.frame_count = 0
        self.detection_count = 0
        self.jitter_values = []
        self.last_wrist_point = None
        self.fps_values = []
        self.prev_time = 0
        self.frame_counter_for_skipping = 0

    def update_config(self, model_choice, posenetwork_model_obj, mobilenet_model_obj, threshold, frame_skip_rate=0):
        # Reset số liệu khi đổi model
        if self.model_choice != model_choice:
            self._reset_stats()
            print(f"Switched to model: {model_choice}. Resetting stats.")

        # Cập nhật các config
        self.model_choice = model_choice
        self.posenetwork_model = posenetwork_model_obj
        self.mobilenet_model = mobilenet_model_obj
        self.threshold = threshold
        # (Frame skipping sẽ được xử lý trong logic chính của app)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Kiểm tra điều kiện sẵn sàng
        if not self.model_choice or \
           (self.model_choice == 'PoseNetwork (Hourglass)' and not self.posenetwork_model) or \
           (self.model_choice == 'MobileNetV2 (Lightweight)' and not self.mobilenet_model):
            return img

        # --- BẮT ĐẦU THU THẬP SỐ LIỆU ---
        self.frame_count += 1
        
        # Tính toán FPS
        if self.prev_time > 0:
            try:
                fps = 1.0 / (time.time() - self.prev_time)
                self.fps_values.append(fps)
            except ZeroDivisionError:
                pass
        self.prev_time = time.time()
        
        # Xử lý chính
        processed_img = img.copy()
        raw_keypoints = None

        if self.model_choice == 'MediaPipe':
            processed_img = process_image_with_mediapipe(img)
        
        elif self.model_choice == 'PoseNetwork (Hourglass)':
            raw_keypoints = extract_posenetwork_coords(self.posenetwork_model, img)
            processed_img = draw_posenetwork_skeleton(img.copy(), raw_keypoints) if raw_keypoints is not None else img

        elif self.model_choice == 'MobileNetV2 (Lightweight)':
            raw_keypoints = extract_mobilenet_coords(self.mobilenet_model, img, self.threshold)
            processed_img = draw_mobilenet_skeleton(img.copy(), raw_keypoints) if raw_keypoints is not None else img

        # Thu thập Jitter và Detection Count
        if raw_keypoints is not None:
            self.detection_count += 1
            wrist_point = raw_keypoints[0] # Lấy khớp cổ tay
            
            if self.last_wrist_point is not None and not np.isnan(wrist_point[0]) and not np.isnan(self.last_wrist_point[0]):
                jitter = np.linalg.norm(wrist_point - self.last_wrist_point)
                self.jitter_values.append(jitter)
            self.last_wrist_point = wrist_point
        
        return processed_img

# --- GIAO DIỆN VÀ LOGIC CHÍNH ---
with st.sidebar:
    st.title("Tùy chọn")
    source_choice = st.radio("Chọn Nguồn Dữ liệu", ("Webcam Thời gian thực", "Tải lên Ảnh", "Tải lên Video"))
    model_choice = st.radio(
        "Chọn Model", 
        ("MediaPipe", "PoseNetwork (Hourglass)", "MobileNetV2 (Lightweight)"),
        key="model_selector"
    )
    st.sidebar.header("Cải tiến")
    confidence_threshold = st.sidebar.slider(
        "Ngưỡng tin cậy (MobileNetV2)", 
        min_value=0.0, max_value=1.0, value=0.1, step=0.05
    )

if source_choice == "Tải lên Ảnh":
    uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if model_choice == 'MediaPipe':
            processed_image = process_image_with_mediapipe(cv_image)
        elif model_choice == 'PoseNetwork (Hourglass)':
            processed_image = process_with_posenetwork(cv_image, posenetwork_model)
        else: # MobileNetV2
            processed_image = process_frame_with_mobilenet(mobilenet_model, cv_image, threshold=confidence_threshold)
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1: st.image(pil_image, caption="Ảnh Gốc", use_column_width=True)
        with col2: st.image(processed_image_rgb, caption=f"Kết quả từ: {model_choice}", use_column_width=True)

elif source_choice == "Tải lên Video":
    uploaded_file = st.file_uploader("Chọn một file video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if model_choice == 'MediaPipe':
                processed_frame = process_image_with_mediapipe(frame)
            elif model_choice == 'PoseNetwork (Hourglass)':
                processed_frame = process_with_posenetwork(frame, posenetwork_model)
            else: # MobileNetV2
                processed_frame = process_frame_with_mobilenet(mobilenet_model, frame, threshold=confidence_threshold)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(processed_frame_rgb, channels="RGB", use_column_width=True)
        cap.release()

elif source_choice == "Webcam Thời gian thực":
    st.header("Webcam Feed và Phân tích Thời gian thực")
    st.info("Đưa tay của bạn trước camera và giữ tương đối yên trong vài giây để đo độ ổn định (Jitter).")
    
    # Chia layout
    video_col, stats_col = st.columns([3, 1])

    with video_col:
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=HandPoseTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    with stats_col:
        st.subheader(f"Phân tích: {model_choice}")
        
        # Tạo các placeholder
        fps_placeholder = st.empty()
        jitter_placeholder = st.empty()
        detection_placeholder = st.empty()
        chart_placeholder = st.empty()

        if ctx.video_processor:
            # Cập nhật config cho worker
            ctx.video_processor.update_config(model_choice, posenetwork_model, mobilenet_model, confidence_threshold)

            # Vòng lặp để cập nhật giao diện số liệu
            while ctx.state.playing:
                if model_choice != 'MediaPipe':
                    # Lấy dữ liệu từ worker
                    stats = {
                        "fps": np.mean(ctx.video_processor.fps_values) if ctx.video_processor.fps_values else 0,
                        "jitter": np.mean(ctx.video_processor.jitter_values) if ctx.video_processor.jitter_values else 0,
                        "detection_rate": (ctx.video_processor.detection_count / ctx.video_processor.frame_count * 100) if ctx.video_processor.frame_count > 0 else 0,
                        "fps_history": ctx.video_processor.fps_values
                    }

                    # Cập nhật các thẻ hiển thị
                    fps_placeholder.metric("FPS Trung bình", f"{stats['fps']:.1f}")
                    jitter_placeholder.metric("Độ ổn định (Jitter)", f"{stats['jitter']:.2f} px/frame", help="Càng nhỏ càng tốt. Đo sự rung giật của khớp cổ tay.")
                    detection_placeholder.metric("Tỷ lệ Phát hiện", f"{stats['detection_rate']:.1f} %")

                    # Cập nhật biểu đồ
                    if stats["fps_history"]:
                        chart_data = pd.DataFrame(stats["fps_history"], columns=["FPS"])
                        with chart_placeholder:
                            st.area_chart(chart_data.tail(100)) # Dùng area_chart cho đẹp
                else:
                    # Xóa số liệu cũ nếu là MediaPipe
                    fps_placeholder.empty()
                    jitter_placeholder.empty()
                    detection_placeholder.empty()
                    chart_placeholder.empty()
                
                # Chờ một chút trước khi cập nhật lại để giảm tải
                time.sleep(0.5)
        else:
            st.info("Nhấn 'START' trên video feed để bắt đầu phân tích.")