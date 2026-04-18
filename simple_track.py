from ultralytics import YOLO
import cv2
import os
from collections import defaultdict, deque
from motion_compensation import MotionCompensator

model = YOLO("models/yolov8n.pt")

os.makedirs("outputs", exist_ok=True)
data_path = "data"

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing {folder}...")

    track_history = defaultdict(lambda: deque(maxlen=20))
    motion = MotionCompensator()

    results = model.track(
        source=folder_path,
        persist=True,
        classes=[0],
        stream=True,
        imgsz=1280,
        conf=0.15,
        iou=0.5
    )

    for i, r in enumerate(results):
        print(f"Processing frame {i}")

        frame = r.orig_img

        display_frame = cv2.resize(
            frame, (640, 360), interpolation=cv2.INTER_AREA
        )

        # Motion compensation
        if i % 3 == 0:
            try:
                display_frame = motion.compensate(display_frame)
            except:
                pass

        annotated = display_frame.copy()

        if r.boxes is not None and r.boxes.xyxy is not None:
            boxes = r.boxes.xyxy.cpu().numpy()

            orig_h, orig_w = frame.shape[:2]
            new_h, new_w = display_frame.shape[:2]

            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            if r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
            else:
                ids = [None] * len(boxes)
                confs = [0] * len(boxes)

            for idx, (box, track_id) in enumerate(zip(boxes, ids)):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1,
                    lineType=cv2.LINE_AA
                )

                if track_id is not None:
                    label = f"ID:{track_id} {confs[idx]:.2f}"

                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                        lineType=cv2.LINE_AA
                    )

                    # Center
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    track_history[track_id].append((cx, cy))

                    for j in range(1, len(track_history[track_id])):
                        cv2.line(
                            annotated,
                            track_history[track_id][j - 1],
                            track_history[track_id][j],
                            (255, 0, 0),
                            1,
                            lineType=cv2.LINE_AA
                        )

        # Save with HIGH QUALITY
        folder_output = os.path.join("outputs", folder)
        os.makedirs(folder_output, exist_ok=True)

        cv2.imwrite(
            f"{folder_output}/frame_{i}.jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

print(" Multi-dataset tracking completed!")