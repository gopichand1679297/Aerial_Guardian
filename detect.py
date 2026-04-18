from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")

img = cv2.imread("test.jpg")

results = model(img)

results[0].show()