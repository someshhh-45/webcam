import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Tune these for accuracy vs speed
    CONF_THRESHOLD = 0.6  # higher = fewer, more confident detections

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # yolov8n.pt for fastest speed
    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator()

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=args.webcam_resolution)

    while True:
        ret, frame = cap.read()
        # Use YOLOv8 built-in tracking (ByteTrack)
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, tracker="bytetrack.yaml")
        result = results[0]

        # Build Supervision Detections object manually from YOLOv8 result
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        # Get track IDs if available
        if hasattr(result.boxes, "id") and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = None
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )

        # Annotate boxes and show track IDs
        frame = box_annotator.annotate(
            frame,
            detections
        )
        # Draw track IDs on each box
        if track_ids is not None:
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int)
                label = f"ID:{track_id}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Polygon zone logic (no color)
        zone.trigger(detections=detections)
        frame = sv.draw_polygon(frame, zone_polygon, color=sv.Color(0, 255, 0))

        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()