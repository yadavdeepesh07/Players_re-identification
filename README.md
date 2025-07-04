# Player Re-Identification System

This project implements a computer vision pipeline for tracking and re-identifying players in sports footage using YOLO-based detection and CNN-based feature extraction.

## Features

* **Player Detection**: Detect players per frame using YOLO.
* **Tracking & Re-Identification**:

  * Tracks players over time
  * Handles temporary occlusion using Kalman-like disappear count
  * Uses ResNet18 to extract CNN features and re-identify players upon re-entry
* **Video Output**: Annotated output saved to `video_output/` folder
* **Logging**: Events like new IDs, re-identification, and disappearance are logged to `logs/reid_log.txt`

## Folder Structure

```
.
├── main.py
├── tracker.py
├── detection/
│   └── yolo_detector.py
├── video_input/
│   └── 15sec_input_720p.mp4
├── video_output/
├── logs/
├── docs/
`

## Step 1: Download ing Pre-trained Models from given link 

 # Link : https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

## Step 2:  paste the model to the models folder 

## Step 3 :  now replace the model path in the yolo_detector.py file with the path of the model you downloaded

## Setup

```bash
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python main.py
```

## Requirements

* Python 3.8+
* GPU optional (recommended for faster feature extraction)

## Output

* `video_output/output_with_ids.mp4`
* `logs/reid_log.txt`: contains event logs per frame for evaluation
