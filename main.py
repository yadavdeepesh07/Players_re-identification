# main.py
import cv2
import os
from detection.yolo_detector import detect_players
from reid.tracker import TrackManager

# Input/Output paths
video_path = r"video_input\broadcast.mp4"
output_path = r"video_output/broadcast.mp4"
os.makedirs("video_output", exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"❌ Failed to open video file: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize tracker
tracker = TrackManager(frame_size=(width, height), fps=fps)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect players (returns list of bounding boxes)
    detections = detect_players(frame)

    # Update tracker
    tracker.update(frame_num, frame, detections)
    tracked_frame = tracker.draw_tracks(frame)

    # Save frame to output
    out.write(tracked_frame)

    # Optional display
    # cv2.imshow("Tracked", tracked_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n Tracking complete. Output saved to:", output_path)
print(" Logs available at: logs/reid_log.txt")
