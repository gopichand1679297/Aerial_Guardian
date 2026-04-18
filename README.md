# Aerial Guardian – Drone-Based Multi-Object Tracking

## Overview
This project implements a lightweight computer vision pipeline to detect and track multiple persons from aerial drone footage (VisDrone dataset).

Drone-based vision introduces several challenges:
- Small object sizes due to high altitude
- Camera motion (ego-motion)
- Frequent occlusions and ID switching

The system addresses these challenges using optimized detection, tracking, and motion compensation techniques.

---

## Architecture

### Detection
- Model: YOLOv8 (Ultralytics)
- Optimizations:
  - Higher input resolution (imgsz = 640 / 1280)
  - Lower confidence threshold (conf = 0.15)
- Target class: Person (class 0)

---

### Tracking
- Algorithm: ByteTrack (via YOLOv8 tracking API)
- Features:
  - Maintains consistent IDs across frames
  - Handles occlusions and re-identification
  - Efficient and lightweight

---

### Motion Compensation
- Technique: Farneback Optical Flow
- Purpose:
  - Reduce camera motion caused by drone movement
  - Improve tracking stability
  - Reduce ID switching

---

## Features
- Person detection in aerial footage  
- Multi-object tracking with unique IDs  
- Confidence score display  
- Trajectory (tail) visualization  
- Motion compensation  
- Multi-dataset processing  
- Video generation pipeline  

---

## Optimization and Trade-offs

To ensure efficient performance on limited hardware:
- Used YOLOv8n (nano model) for lightweight inference
- Reduced resolution when required to prevent memory issues
- Applied frame skipping to avoid crashes
- Balanced accuracy and speed for real-time feasibility

---

## FPS and Hardware

- Hardware: CPU (Intel i5 class system)
- Performance: Approximately 5–10 FPS
- Note: Performance varies depending on resolution and dataset size

---

## Edge Deployment (Jetson Adaptation)

To deploy this system on edge devices such as NVIDIA Jetson:
- Use YOLOv8n model
- Convert model to TensorRT
- Reduce input resolution (e.g., 640)
- Optimize inference pipeline
- Disable heavy components if necessary

---
## Installation and Setup

2. Create a virtual environment  
```bash
python -m venv venv

Activate the environment

venv\Scripts\activate   # Windows

Install dependencies
pip install -r requirements.txt

Run the project

Step 1: Run tracking
python src/simple_track.py

Step 2: Generate video

python src/make_output_video.py

Project Structure

AerialGuardian/
├── src/
│   ├── simple_track.py
│   ├── make_output_video.py
│   └── motion_compensation.py
├── models/
├── README.md
├── .gitignore

