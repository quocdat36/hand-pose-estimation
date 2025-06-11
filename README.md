# Real-time Hand Pose Estimation: A Comparative Study

This project presents a comprehensive web application for real-time 2D hand pose estimation, allowing for a direct comparison between different deep learning models and architectures. The application is built with Streamlit, providing an interactive and user-friendly interface.

[![Project Demo GIF](demo.gif)](https://drive.google.com/drive/folders/1lE4-MoDBO1UR-2NBPmp2nVe-3_IhjrNG?usp=sharing)

## 🌟 Key Features

- **Multi-Model Comparison:** Switch seamlessly between three distinct models in real-time:
    1.  **MediaPipe Hands:** Google's high-performance, industry-standard solution, serving as the benchmark.
    2.  **PoseNetwork (Hourglass):** A deep, accuracy-focused model based on the Stacked Hourglass Network architecture, running on GPU via PyTorch.
    3.  **MobileNetV2:** A lightweight, performance-focused model designed for efficiency, running on CPU via OpenCV's DNN module.

- **Multiple Input Sources:** Supports:
    - Live Webcam Feed
    - Image Upload
    - Video Upload

- **Implemented Improvements:**
    - **Configurable Confidence Threshold:** An interactive slider allows users to adjust the detection confidence threshold for the MobileNetV2 model, directly demonstrating the precision-recall tradeoff.
    - **Performance Optimization:** Input frames for the MobileNetV2 model are resized to a smaller dimension, significantly reducing computational load and latency for a smoother real-time experience.

## 🛠️ Project Structure

The project is organized with a clean and modular structure to separate concerns, making it easy to maintain and extend.
- **[Download Weights from Google Drive]**([https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j?usp=sharing](https://drive.google.com/drive/folders/1lE4-MoDBO1UR-2NBPmp2nVe-3_IhjrNG?usp=sharing))

After downloading and unzipping, ensure your project structure looks like this:
```
hand-pose-estimation/
├── app.py              # The main Streamlit web application
├── configs/            # Configuration files (.yaml) for models
├── data/               # Placeholder for FreiHAND dataset (not included in repo)
├── notebooks/          # Jupyter notebooks for experimentation
├── scripts/            # Scripts for training and evaluation
├── src/                # Main source code package
│   ├── __init__.py
│   ├── model_hourglass/
│   └── model_mobilenet/
├── weights/            # Pre-trained model weights
├── .gitignore          # Specifies files to be ignored by Git
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## 🚀 Setup & Installation

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA drivers installed (for PoseNetwork GPU acceleration)
- Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/quocdat36/hand-pose-estimation.git
    cd hand-pose-estimation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (on Windows)
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project requires specific versions of libraries like PyTorch for GPU support. It's recommended to install them manually first, then install the rest.

    ```bash
    # 1. Install PyTorch with GPU support (example for CUDA 11.8)
    # Please visit the official PyTorch website for the command matching your system.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # 2. Install the remaining packages
    pip install -r requirements.txt
    ```

## 🖥️ Usage

### Running the Web Application

To launch the interactive comparison tool, simply run the following command in your terminal:

```bash
streamlit run app.py
```

Navigate to the local URL provided (usually `http://localhost:8501`) in your web browser.

### Using the Interface

- **Select Input Source:** Use the radio buttons in the sidebar to choose between Webcam, Image Upload, or Video Upload.
- **Select Model:** Switch between `MediaPipe`, `PoseNetwork (Hourglass)`, and `MobileNetV2 (Lightweight)` to see their performance Unterschiede in real-time.
- **Adjust Confidence Threshold:** When `MobileNetV2` is selected, use the slider to see how changing the confidence threshold affects the number and accuracy of detected keypoints.

## 🔬 Model Training & Evaluation (Optional)

The scripts for training the PoseNetwork model are included.

- **Configuration:** Modify the `.yaml` files in the `configs/` directory to adjust hyperparameters like learning rate, batch size, etc.
- **Training:** To fine-tune or retrain the PoseNetwork model on the FreiHAND dataset, run:
    ```bash
    python scripts/train_posenetwork.py
    ```
- **Evaluation:** To evaluate the model on the FreiHAND evaluation set, use:
    ```bash
    python scripts/evaluate_freihand.py
    ```

---
*This project was developed as part of a Computer Vision course, demonstrating the integration, comparison, and improvement of various deep learning models for hand pose estimation.*
