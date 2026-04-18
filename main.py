import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque


DATA_PATH = r"C:\Users\91939\Downloads\Aerial_Guardian\VisDrone2019-MOT-val\sequences"
OUTPUT_PATH = "outputs"
VIDEO_PATH = "videos"
MODEL_PATH = "yolov8n.pt"  

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VIDEO_PATH, exist_ok=True)


class MotionCompensator:
    def __init__(self):
        self.prev_gray = None

    def compensate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])

        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        self.prev_gray = gray
        return stabilized


model = YOLO(MODEL_PATH)


for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\n Processing: {folder}")

    track_history = defaultdict(lambda: deque(maxlen=20))
    motion = MotionCompensator()

    results = model.track(
        source=folder_path,
        persist=True,
        stream=True,
        imgsz=1280,
        conf=0.15,
        iou=0.5
    )

    output_folder = os.path.join(OUTPUT_PATH, folder)
    os.makedirs(output_folder, exist_ok=True)

    frames_list = []

    for i, r in enumerate(results):
        print(f"Frame {i}")

        frame = r.orig_img
        display_frame = cv2.resize(frame, (640, 360))

        # Motion compensation every 3 frames
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

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

                if track_id is not None:
                    label = f"ID:{track_id} {confs[idx]:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    # Track trail
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    track_history[track_id].append((cx, cy))

                    for j in range(1, len(track_history[track_id])):
                        cv2.line(
                            annotated,
                            track_history[track_id][j - 1],
                            track_history[track_id][j],
                            (255, 0, 0),
                            1
                        )

        # Save frame
        frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames_list.append(frame_path)

    
    print(f"🎬 Creating video for {folder}")

    if len(frames_list) == 0:
        continue

    first_frame = cv2.imread(frames_list[0])
    h, w = first_frame.shape[:2]

    video_file = os.path.join(VIDEO_PATH, f"{folder}.mp4")
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

    for img_path in sorted(frames_list):
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)

    out.release()

    print(f" Video saved: {video_file}")

print("\n FULL PIPELINE COMPLETED!")
