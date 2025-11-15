import cv2
from ultralytics import YOLO
import argparse


def run_image(model_path, image_path, output_path="output.jpg"):
    model = YOLO(model_path)

    results = model(
        image_path,
        conf=0.45,        # confidence threshold
        iou=0.45,         # IoU threshold for NMS
        max_det=1,        # allow only 1 detection (change if needed)
        agnostic_nms=False
    )

    annotated = results[0].plot()
    cv2.imwrite(output_path, annotated)
    print(f"Saved output to {output_path}")


def run_webcam(model_path, cam_index=0):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=0.45,
            iou=0.45,
            max_det=1,
            agnostic_nms=False
        )

        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Live", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["image", "webcam"])

    parser.add_argument("--weights", type=str, default="best_1.pt")
    parser.add_argument("--input", type=str)

    args = parser.parse_args()

    if args.mode == "image":
        if not args.input:
            print("Error: --input required for image mode.")
        else:
            run_image(args.weights, args.input)

    else:
        run_webcam(args.weights)
