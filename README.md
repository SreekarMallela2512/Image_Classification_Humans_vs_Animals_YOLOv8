# Image Classification: Humans vs Animals (and Objects) using YOLOv8

This project uses **YOLOv8** for image classification and detection in the context of the **SPIDAS** (Smart Perimeter Intrusion Detection & Alert System) project.

It currently supports two modes of operation:

1. **Custom binary model (`best_1.pt`)** – Classify **Humans vs Animals**
2. **Pretrained YOLOv8 (COCO)** – Group detections into **Human / Animal / Object** using a simple custom mapping

---

## 🔧 Features

- ✅ Custom YOLOv8 model trained to classify **Humans vs Animals**
- ✅ Pretrained YOLOv8 (COCO) mapped to **3 classes**: Human / Animal / Object
- ✅ Run inference on **single images**
- ✅ Real-time detection on **webcam**
- ✅ Simple, minimal Python scripts
- ✅ Works on **CPU** and **GPU (CUDA)**

---

## 📁 Repository Structure

```text
Image_Classification_Humans_vs_Animals_YOLOv8/
├─ best_1.pt                       # Custom trained YOLOv8 model (Humans vs Animals)
├─ Image_model.ipynb               # Notebook for training / experimenting with custom model
├─ inference.py                    # Inference script for custom binary model
├─ Infrence_on_YOLOV8_3_classes.py # (NEW) 3-class Human/Animal/Object script using YOLOv8 COCO
├─ Three_Class_YOLOv8_Inference.ipynb  # (NEW) Notebook for 3-class experiments (optional)
├─ requirements.txt
└─ README.md
🚀 Installation
Clone the repository

bash
Copy code
git clone https://github.com/Shivansh0047/Image_Classification_Humans_vs_Animals_YOLOv8.git
cd Image_Classification_Humans_vs_Animals_YOLOv8
(Optional but recommended) create a virtual environment

bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
🧠 1) Custom Model – Humans vs Animals (best_1.pt)
This uses your trained YOLOv8 model to classify Human vs Animal.

▶ Run on an image
bash
Copy code
python inference.py --mode image --input path/to/image.jpg --weights best_1.pt
🎥 Run on webcam
bash
Copy code
python inference.py --mode webcam --weights best_1.pt
A window will open with live detections

Press Q to exit

🧠 2) Pretrained YOLOv8 – Human / Animal / Object (COCO Mapping)
The script Infrence_on_YOLOV8_3_classes.py uses a COCO-pretrained YOLOv8 model (e.g. yolov8n.pt) and then maps COCO classes into three high-level groups:

Human → person

Animal → cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe, ...

Object → everything else (cars, trees, chairs, etc.)

🎥 Default: Run on webcam
By default, this script runs in webcam mode:

bash
Copy code
python Infrence_on_YOLOV8_3_classes.py
Options:

Use a different camera (e.g., external webcam):

bash
Copy code
python Infrence_on_YOLOV8_3_classes.py --camera 1
Use a different YOLOv8 model (larger & more accurate):

bash
Copy code
python Infrence_on_YOLOV8_3_classes.py --weights yolov8s.pt
Adjust confidence threshold:

bash
Copy code
python Infrence_on_YOLOV8_3_classes.py --conf 0.5
Press q to quit the webcam window.

🖼 Run on a single image
bash
Copy code
python Infrence_on_YOLOV8_3_classes.py --mode image --input path/to/image.jpg
📒 Notebooks
Image_model.ipynb – Training / experimenting with the custom Humans vs Animals model (best_1.pt)

Three_Class_YOLOv8_Inference.ipynb – (Optional) Playground for 3-class Human/Animal/Object logic using YOLOv8

You can open these in Jupyter or VS Code.

🧩 Requirements
Core libraries used:

ultralytics – YOLOv8 framework

opencv-python – image & webcam handling (cv2)

numpy

matplotlib (for notebooks)

torch (PyTorch backend for YOLOv8)

See requirements.txt for the exact list.

🧪 Project Context – SPIDAS
This repo is part of the SPIDAS (Smart Perimeter Intrusion Detection & Alert System) project:

Distinguish between Humans and Animals near a perimeter

Optionally categorize other detections as generic Objects

Can be integrated with sensors / alarms / IoT devices

Future work ideas:

Better animal species classification

Lighting / night vision robustness

Edge deployment (Jetson, Raspberry Pi, etc.)

🙌 Acknowledgements
Ultralytics YOLOv8

PyTorch & OpenCV communities

SPIDAS project collaborators

yaml
Copy code

---

## 3️⃣ After you paste the new README and add new files

From inside the cloned repo folder:

```bash
git status