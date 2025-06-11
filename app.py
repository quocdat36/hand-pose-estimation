# FILE: app.py 

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import yaml
import torch
import mediapipe as mp
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

# --- LỚP XỬ LÝ WEBCAM ---
class HandPoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.model_choice = "MediaPipe"
        self.threshold = 0.1 # Giá trị mặc định
        
        # --- CẢI TIẾN FRAME SKIPPING ---
        self.frame_counter = 0
        self.skip_frames = 2 # Sẽ xử lý 1 frame, bỏ qua 2 frame tiếp theo
        self.last_known_image = None # Lưu lại ảnh đã xử lý cuối cùng
        # --- KẾT THÚC CẢI TIẾN ---

    def update_config(self, model_choice, posenetwork_model_obj, mobilenet_model_obj, threshold):
        self.model_choice = model_choice
        self.posenetwork_model = posenetwork_model_obj
        self.mobilenet_model = mobilenet_model_obj
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # --- LOGIC FRAME SKIPPING ---
        if self.model_choice == 'MobileNetV2 (Lightweight)':
            self.frame_counter += 1
            if self.frame_counter % self.skip_frames != 0 and self.last_known_image is not None:
                # Nếu đây là frame cần bỏ qua, trả về ảnh đã xử lý gần nhất
                return self.last_known_image
        else:
            # Reset bộ đếm nếu chuyển sang model khác
            self.frame_counter = 0
        # --- KẾT THÚC LOGIC ---

        if self.model_choice == 'MediaPipe':
            processed_img = process_image_with_mediapipe(img)
        elif self.model_choice == 'PoseNetwork (Hourglass)':
            processed_img = process_with_posenetwork(img, self.posenetwork_model)
        else: # MobileNetV2
            processed_img = process_frame_with_mobilenet(self.mobilenet_model, img, threshold=self.threshold)
        self.last_known_image = processed_img
        return processed_img

# --- GIAO DIỆN VÀ LOGIC CHÍNH ---
with st.sidebar:
    st.title("Tùy chọn")
    source_choice = st.radio("Chọn Nguồn Dữ liệu", ("Webcam Thời gian thực", "Tải lên Ảnh", "Tải lên Video"))
    model_choice = st.radio(
        "Chọn Model", 
        ("MediaPipe", "PoseNetwork (Hourglass)", "MobileNetV2 (Lightweight)"), 
        help="Chọn model để thực hiện ước tính tư thế bàn tay."
    )
    st.sidebar.header("Cải tiến")
    confidence_threshold = st.sidebar.slider(
        "Ngưỡng tin cậy (Threshold) cho MobileNetV2", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1,
        step=0.05
    )
    st.sidebar.info("Áp dụng cho MobileNetV2. Kéo lên để kết quả chính xác hơn, kéo xuống để thấy nhiều điểm hơn.")

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
    st.header("Webcam Feed")
    st.write("Nhấn 'START' để bật camera và so sánh các model.")
    ctx = webrtc_streamer(
        key="webcam",
        video_processor_factory=HandPoseTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.update_config(model_choice, posenetwork_model, mobilenet_model, confidence_threshold)