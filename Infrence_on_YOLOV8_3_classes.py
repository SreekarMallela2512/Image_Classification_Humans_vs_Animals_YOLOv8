import argparse
import os

import cv2
from ultralytics import YOLO


# --------- Class mapping: COCO -> Human / Animal / Object ---------

ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe"
}


def map_to_three_classes(class_name: str) -> str:
    """Convert COCO class -> Human / Animal / Object."""
    if class_name == "person":
        return "Human"
    elif class_name in ANIMAL_CLASSES:
        return "Animal"
    else:
        return "Object"


# ------------------------ IMAGE INFERENCE ------------------------

def run_on_image(model, image_path: str, conf_thresh: float = 0.3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    results = model(image_path)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        label = map_to_three_classes(class_name)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label box
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 (Image) - Human / Animal / Object", img)
    print("Press any key to close image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------ WEBCAM INFERENCE ------------------------

def run_on_webcam(model, camera_index=0, conf_thresh=0.3):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index {camera_index})")

    print("Webcam active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            label = map_to_three_classes(class_name)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("YOLOv8 (Webcam) - Human / Animal / Object", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------ ARGUMENT PARSER ------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Human/Animal/Object classifier"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "webcam"],
        default="webcam",      # <-- DEFAULT = WEBCAM
        help="Inference mode. Defaults to webcam."
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Image path (required for --mode image)"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 weights."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold"
    )

    return parser.parse_args()


# ------------------------- MAIN -------------------------

def main():
    args = parse_args()

    print(f"[INFO] Loading model {args.weights} ...")
    model = YOLO(args.weights)

    if args.mode == "image":
        if not args.input:
            raise ValueError("You must provide --input for image mode")
        run_on_image(model, args.input, args.conf)
    else:
        run_on_webcam(model, args.camera, args.conf)


if __name__ == "__main__":
    main()
