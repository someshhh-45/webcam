# YOLOv8 Webcam Object Detection & Tracking

This project uses Ultralytics YOLOv8 with built-in ByteTrack tracking for real-time object detection and tracking from a webcam feed.

## Features
- Real-time object detection using YOLOv8
- Object tracking with ByteTrack (unique IDs per object)
- Polygon zone visualization for region-based logic
- Bounding boxes and object IDs displayed on video stream

## Requirements
See `requirements.txt` for all dependencies.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python test.py
```

## Description
- Captures frames from your webcam
- Detects and tracks objects in real-time
- Displays bounding boxes and object IDs
- Visualizes a polygon zone for region-based logic

## Applications
- Surveillance
- People counting
- Activity monitoring
- Region-based event detection

## Customization
- Adjust detection confidence threshold in `test.py`
- Modify polygon zone coordinates for your use case
- Change model type (e.g., yolov8n.pt, yolov8l.pt) for speed/accuracy tradeoff

---
Created with Ultralytics YOLOv8 and Supervision
