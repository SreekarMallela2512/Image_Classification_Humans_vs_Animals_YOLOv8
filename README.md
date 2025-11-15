# YOLOv8 Object Detection (Custom Model)

This repository contains a custom-trained YOLOv8 model (`best_1.pt`)
and scripts for running inference on images and webcam video.

---

## 🚀 Features
- Run YOLOv8 inference on images
- Live webcam object detection
- Easy Python script (`inference.py`)
- Supports CPU and GPU (CUDA)

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

Install dependencies:

pip install -r requirements.txt

🖼️ Run Inference on an Image
python inference.py --mode image --input path/to/image.jpg --weights best_1.pt


📸 Run Webcam Detection
python inference.py --mode webcam --weights best_1.pt


Press Q to exit.