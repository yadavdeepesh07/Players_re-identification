from ultralytics import YOLO

model = YOLO(r"models\best.pt")   # replce this path with downloaded model path

def detect_players(frame):
    results = model(frame)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((int(x1), int(y1), int(x2), int(y2)))
    return detections
