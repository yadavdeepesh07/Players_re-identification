# tracker.py
import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import os

class SimpleExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        weights_path = "models/resnet18-f37072fd.pth"    
        self.model = models.resnet18(weights=None)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            feature = self.model(tensor)
        return feature.cpu().numpy().flatten()

class TrackManager:
    def __init__(self, frame_size, fps=30, max_disappeared_frames=30):
        self.next_id = 0
        self.tracks = {}
        self.extractor = SimpleExtractor()
        self.disappeared_count = defaultdict(int)
        self.frame_size = frame_size
        self.fps = fps
        self.max_disappeared_frames = max_disappeared_frames
        self.log_path = 'logs/reid_log.txt'
        os.makedirs('logs', exist_ok=True)
        open(self.log_path, 'w').close()


    def _log(self, frame_num, event, track_id, details):
        with open(self.log_path, 'a') as f:
            f.write(f"{frame_num},{event},{track_id},{details}\n")

    def update(self, frame_num, frame, detections):
        new_features = []
        new_boxes = []
        for (x1, y1, x2, y2) in detections:
            crop = frame[y1:y2, x1:x2]
            feat = self.extractor.extract(crop)
            new_features.append(feat)
            new_boxes.append((x1, y1, x2, y2))

        assigned = set()
        updated_tracks = {}

        for tid, track in self.tracks.items():
            best_match = -1
            best_score = float('inf')
            for i, feat in enumerate(new_features):
                score = cosine(track['feature'], feat)
                if score < 0.5 and i not in assigned and score < best_score:
                    best_score = score
                    best_match = i

            if best_match >= 0:
                updated_tracks[tid] = {
                    'bbox': new_boxes[best_match],
                    'feature': new_features[best_match]
                }
                assigned.add(best_match)
                self.disappeared_count[tid] = 0
                self._log(frame_num, "ReID", tid, f"score={best_score:.2f}")
            else:
                self.disappeared_count[tid] += 1
                if self.disappeared_count[tid] > self.max_disappeared_frames:
                    self._log(frame_num, "Removed", tid, "Max disappearance")
                else:
                    updated_tracks[tid] = track

        for i, feat in enumerate(new_features):
            if i not in assigned:
                tid = self.next_id
                self.next_id += 1
                updated_tracks[tid] = {
                    'bbox': new_boxes[i],
                    'feature': feat
                }
                self.disappeared_count[tid] = 0
                self._log(frame_num, "NewID", tid, f"assigned from unmatched box {i}")

        self.tracks = updated_tracks

    def draw_tracks(self, frame):
        for tid, track in self.tracks.items():
            if self.disappeared_count[tid] <= self.max_disappeared_frames:
                x1, y1, x2, y2 = track['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
